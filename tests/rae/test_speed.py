"""sbm_rae.speed on CPU: the hooks it installs in GuidedCBMLoss (here with the eager twin, so the wiring is checked without
torch.compile), BatchedMuon and the foreach EMA, each against the stock code path.

    .venv/bin/python -m pytest tests/rae/test_speed.py -q
"""
import copy

import pytest
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from sbm_rae.guided_cbm import GuidedCBMLoss
from sbm_rae.models.dit import DiT
from sbm_rae.models.tangent_only import use_tangent_only_jvp
from sbm_rae.speed import BatchedMuon, _fwd, _jvp_detached, ema_multi_avg_fn, plain_twin
from test_guided_cbm import LABELS, batch, grads, our_cfg


def tiny_repa_dit(seed=0):
    torch.manual_seed(seed)
    m = DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=3, num_heads=4, label_dim=LABELS,
            attn_func="default", head_hidden_size=96, head_depth=2, head_num_heads=4,
            repa_enabled=True, repa_encoder_depth=1, repa_z_dim=4 * 2 * 2, repa_projector_dim=32)
    with torch.no_grad():   # off the zero init, so every term is non-trivial
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.float()


@pytest.mark.parametrize("repa", [False, True])
def test_student_hooks_match_the_stock_calls(repa):
    """With the student's JVP and endpoint forward routed through the plain-layer twin (what sbm_rae.speed compiles),
    the guided step's losses and every parameter gradient match the stock eager calls on the tangent-only model."""
    x, y = batch(n=8)
    model = use_tangent_only_jvp(tiny_repa_dit())     # as TrainingModule prepares the student
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.02 * torch.randn_like(p))
    over = {"repa": {"enabled": True, "weight": 0.5}} if repa else {}
    loss = GuidedCBMLoss(our_cfg(**over), label_dim=LABELS)
    loss.teacher = teacher
    if repa:   # patch-2 token layout for the test model (as in test_guided_cbm.test_repa_with_head)
        loss._repa_loss = lambda proj, x_, self=loss: GuidedCBMLoss._repa_loss(
            self, proj, torch.nn.functional.pixel_unshuffle(x_, 2))
    out = []
    for hooked in (False, True):
        if hooked:
            twin = plain_twin(model)
            assert all(a is b for a, b in zip(twin.parameters(), model.parameters()))   # shared, not copied
            loss.student_jvp = lambda m, I, tan, lab, om: _jvp_detached(twin, I, tan, lab, om)
            loss.student_fwd = lambda m, I, lab, om, return_repa=False: _fwd(twin, I, lab, om, return_repa)
        torch.manual_seed(11)
        losses, _ = loss(model, None, x, y, 5)
        out.append(({k: v.detach() for k, v in losses.items()}, grads(model, losses)))
    (l0, g0), (l1, g1) = out
    assert set(l0) == set(l1) == ({"transport", "endpoint", "repa"} if repa else {"transport", "endpoint"})
    for k in l0:
        torch.testing.assert_close(l0[k], l1[k], rtol=1e-5, atol=1e-6)
    assert len(g0) == len(g1)
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_batched_muon_matches_muon():
    """BatchedMuon: the stock torch.optim.Muon update, Newton-Schulz batched over same-shaped matrices."""
    torch.manual_seed(0)
    shapes = [(48, 32), (32, 32), (32, 48), (48, 32), (32, 32), (64, 16)]
    p_new = [torch.nn.Parameter(torch.randn(s) * 0.02) for s in shapes]
    p_ref = [torch.nn.Parameter(p.detach().clone()) for p in p_new]
    kw = dict(lr=5e-3, weight_decay=0.01, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.7750, 2.0315),
              eps=1e-7, ns_steps=5)
    o_ref, o_new = torch.optim.Muon(p_ref, **kw), BatchedMuon(p_new, **kw)
    for _ in range(3):
        for p, q in zip(p_ref, p_new):
            g = torch.randn_like(p)
            p.grad, q.grad = g.clone(), g.clone()
        before = [p.detach().clone() for p in p_ref]
        o_ref.step()
        o_new.step()
        for p, q, b0 in zip(p_ref, p_new, before):   # bf16 Newton-Schulz: equal to a small fraction of the update
            assert float((p - q).norm() / (p - b0).norm()) < 2e-2


def test_foreach_ema_matches_per_parameter_ema():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 8))
    ref = AveragedModel(m, avg_fn=get_ema_avg_fn(0.999))
    new = AveragedModel(m, multi_avg_fn=ema_multi_avg_fn(0.999))
    for _ in range(5):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        ref.update_parameters(m)
        new.update_parameters(m)
    for a, b in zip(ref.parameters(), new.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
