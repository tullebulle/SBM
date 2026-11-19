"""loss.guided_cbm.guide.scale_cond (guidance-scale conditioning).

    .venv/bin/python -m pytest tests/rae/test_scale_cond.py -q
"""
import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from sbm_rae.guided_cbm import GuidedCBMLoss, guidance_scale_cond
from sbm_rae.models.dit import DiT
from test_guided_cbm import LABELS, batch, our_cfg, tiny_dit

AG = {"enabled": True, "ckpt": "x.ckpt", "weight": 0.0909}
TINY = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2, num_heads=4, label_dim=LABELS,
            attn_func="default")


def perturbed(model, scale=0.02, seed=3):
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


def with_scale_input(model):
    """The same network with a guidance-scale input whose embedding outputs zero (a non-zero input layer)."""
    m = DiT(guidance_enabled=True, **TINY).float()
    missing, unexpected = m.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("w_embedder.") for k in missing)
    with torch.no_grad():
        m.w_embedder.mlp[0].weight.normal_(0, 0.5)
        m.w_embedder.mlp[2].weight.zero_()
        m.w_embedder.mlp[2].bias.zero_()
    return m


def run(loss, model, x, y, seed=7):
    torch.manual_seed(seed)
    losses, aux = loss(model, None, x, y, 3)
    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None and not n.startswith("w_embedder")}
    return {k: v.detach() for k, v in losses.items()}, aux, g


def test_fixed_scale_at_w0_reproduces_the_plain_guide():
    """lo = hi = w0 on a network whose scale embedding outputs zero: the same losses and gradients as the guide without
    scale conditioning (kappa and the autoguide weight recomputed from w0, so equal up to float rounding)."""
    x, y = batch(n=8)
    plain = tiny_dit()
    teacher, weak = perturbed(plain, seed=4), perturbed(plain, scale=0.05, seed=5)
    off = GuidedCBMLoss(our_cfg(guide={"kappa": 0.7273, "autoguide": AG}), label_dim=LABELS)
    w0 = guidance_scale_cond(our_cfg(guide={"kappa": 0.7273, "autoguide": AG}))[3]
    on = GuidedCBMLoss(our_cfg(guide={"kappa": 0.7273, "autoguide": AG,
                                      "scale_cond": {"enabled": True, "lo": w0, "hi": w0}}), label_dim=LABELS)
    off.teacher, off.weak = teacher, weak
    on.teacher, on.weak = with_scale_input(teacher), with_scale_input(weak)
    l0, _, g0 = run(off, plain, x, y)
    l1, aux, g1 = run(on, with_scale_input(plain), x, y)
    for k in l0:
        torch.testing.assert_close(l1[k], l0[k], rtol=1e-5, atol=1e-6)
    assert set(g0) == set(g1)
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], rtol=1e-4, atol=1e-6)
    assert abs(float(aux["guide_scale_mean"]) - w0) < 1e-5


def test_kappa_and_weight_follow_the_scale():
    """kappa / (1 - kappa - lambda) = w - 1 (the CFG increment at the fixed point) and lambda / (1 - kappa - lambda) = a
    at every w; at w0 they are the configured kappa and autoguide weight."""
    cfg = our_cfg(guide={"kappa": 0.7273, "autoguide": AG, "scale_cond": {"enabled": True}})
    loss = GuidedCBMLoss(cfg, label_dim=LABELS)
    _, _, _, w0, a = guidance_scale_cond(cfg)
    assert abs(a - 0.5) < 1e-3 and abs(w0 - 5.0) < 1e-2
    w = torch.tensor([2.0, w0, 8.0], dtype=torch.float64)
    k, lam = loss._guide_weights(w, torch.zeros(3, dtype=torch.float64))
    rest = 1 - k - lam
    torch.testing.assert_close(k / rest, w - 1)
    torch.testing.assert_close(lam / rest, torch.full_like(w, a))
    assert abs(float(k[1]) - 0.7273) < 1e-9 and abs(float(lam[1]) - 0.0909) < 1e-9
    assert bool(((k + lam) < 1).all())


class Spy(nn.Module):
    def __init__(self, inner, log):
        super().__init__()
        self.inner, self.log = inner, log

    def forward(self, x, y, omega=None, w=None, **kw):
        self.log.append(w)
        return self.inner(x, y, omega=omega, w=w, **kw)


def test_scale_reaches_student_teacher_and_weak():
    """Drawn per example in [lo, hi]; the student's calls, the teacher's b(y) / b(null) reads and the weak model's read
    all get the same w."""
    x, y = batch(n=8)
    model = with_scale_input(tiny_dit())
    loss = GuidedCBMLoss(our_cfg(guide={"kappa": 0.7273, "autoguide": AG, "scale_cond": {"enabled": True, "lo": 2.0,
                                                                                         "hi": 8.0}}), label_dim=LABELS)
    s_log, t_log, w_log = [], [], []
    loss.teacher, loss.weak = Spy(perturbed(model), t_log), Spy(perturbed(model, seed=9), w_log)
    torch.manual_seed(7)
    loss(Spy(model, s_log), None, x, y, 3)
    ws = [w for w in s_log + t_log + w_log]
    assert ws and all(w is not None for w in ws)
    ref = ws[0]
    assert bool(((ref >= 2.0) & (ref <= 8.0)).all()) and float(ref.std()) > 0
    for w in ws:
        assert torch.equal(w, ref)
    assert len(t_log) == 4 and len(w_log) == 2   # b(null), b(y) on the tangent and at the endpoint; weak twice


def test_off_by_default_calls_without_w():
    x, y = batch(n=8)
    log = []
    loss = GuidedCBMLoss(our_cfg(), label_dim=LABELS)
    loss.teacher = Spy(perturbed(tiny_dit()), log)
    torch.manual_seed(7)
    loss(Spy(tiny_dit(), log), None, x, y, 3)
    assert log and all(w is None for w in log)


def test_config_is_checked():
    with pytest.raises(ValueError, match="scale_cond"):
        GuidedCBMLoss(our_cfg(guide={"kappa": 0.8, "autoguide": {**AG, "weight": 0.25},
                                     "scale_cond": {"enabled": True}}), label_dim=LABELS)
    with pytest.raises(ValueError, match="scale_cond"):
        GuidedCBMLoss(our_cfg(guide={"scale_cond": {"enabled": True, "lo": 0.5, "hi": 8.0}}), label_dim=LABELS)
