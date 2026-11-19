"""sbm_rae.guided_cbm against the reference implementation (CBMLoss of the `sbm` package, src/sbm/loss.py), on a tiny
DiT: same draws, same losses, same gradients. Also: the tangent-only JVP changes nothing, the time shift and the freeze
balancer run, and the trainer-side checkpoint state round-trips.

    .venv/bin/python -m pytest tests/rae/test_guided_cbm.py -q
"""
import copy
import os
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.guided_cbm import GuidedCBMLoss
from sbm_rae.models.dit import DiT
from sbm_rae.models.tangent_only import use_tangent_only_jvp

# The reference implementation: `sbm.loss.CBMLoss` of this repo's sbm package; SBM_REFERENCE_SRC points at another
# source root.
REFERENCE_SRC = os.environ.get("SBM_REFERENCE_SRC", str(Path(__file__).resolve().parents[2] / "src"))
LABELS = 10


def tiny_dit(seed=0, head=False):
    torch.manual_seed(seed)
    extra = dict(head_hidden_size=96, head_depth=2, head_num_heads=4) if head else {}
    m = DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2, num_heads=4, label_dim=LABELS,
            attn_func="default", **extra)
    with torch.no_grad():   # off the zero init, so every term is non-trivial
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.float()


def reference_cfg(start_step=0):
    """A full loss block of the reference implementation (the SD-VAE XL recipe), the guide on from start_step."""
    return OmegaConf.create({
        "transport": {"p": 1.0, "c": 0.01, "p_within": None, "bins": 16, "bin_ema": 0.99, "weight": 2000},
        "endpoint": {"p": 0.0, "c": 0.01, "weight": 1, "omega_skew": 1.0,
                     "late_weight": {"t0": 0.6, "power": 1.0, "cap": 8.0}},
        "anchor_alpha": 0.65, "restrict_transport": False,
        "time": {"p_mean": -1.2, "p_std": 1.0, "eps": 1e-4, "p_mean_end": None, "anneal_start": 0, "anneal_steps": 0,
                 "mix_frac": 0.5, "mix_lo": 0.3},
        "balance": {"mode": "none", "parameter_path": "final_layer", "ema": 0.99},
        "signed": {"alpha": 0.0, "paired": False},
        "cfg_distill": {"enabled": False, "strength": 0.25, "t_max": 0.5, "fixed_weight": None},
        "gcond": {"enabled": False},
        "frechet": {"enabled": False, "strength": 1.0, "n_samples": 8, "omega": 0.8, "ema": 0.99, "w_every": 10,
                    "fd_ref": None, "start_step": 0, "save_state": False, "anneal": True, "global_norm": False,
                    "fixed_weight": None, "stats": None, "dino_hub": None, "vae": None},
        "aux": {"alpha": 0.0, "align": 0.0, "pool": []},
        "guide": {"enabled": True, "w": 1.0, "kappa": 0.8, "t_lo": 0.35, "t_hi": 0.6, "negative": "uncond",
                  "start_step": start_step, "endpoint": True, "diag_frac": 0.5, "omega_cond": False, "t_cap": 1.0,
                  "teacher_decay": 1.0, "teacher_refresh": 10000, "v_weight": 1.0, "calib": {"enabled": False}},
        "label_dropout": 0.1, "diag_every": 0,
    })


def our_cfg(**over):
    base = {
        "time_shift": 1.0,
        "time": {"p_mean": -1.2, "p_std": 1.0, "eps": 1e-4, "mix_frac": 0.5, "mix_lo": 0.3},
        "anchor_alpha": 0.65, "label_dropout": 0.1,
        "transport": {"p": 1.0, "c": 0.01, "weight": 2000.0},
        "endpoint": {"p": 0.0, "c": 0.01, "weight": 1.0, "late_weight": {"t0": 0.6, "power": 1.0, "cap": 8.0},
                     "late_weight_start_step": 0},
        "balance": {"mode": "none", "parameter_path": "final_layer", "ema": 0.999},
        "guide": {"enabled": True, "kappa": 0.8, "t_lo": 0.35, "t_hi": 0.6, "endpoint": True, "diag_frac": 0.5,
                  "start_step": 0, "teacher_refresh": 10000},
        "tangent_only_jvp": True,
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(over))


def batch(seed=1, n=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, 8, 8, generator=g), torch.randint(0, LABELS, (n,), generator=g)


def grads(model, losses):
    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


def _reference_cbm_loss():
    sys.path.insert(0, REFERENCE_SRC)
    try:
        from sbm.loss import CBMLoss
    finally:
        sys.path.remove(REFERENCE_SRC)
    return CBMLoss


def test_matches_reference_loss():
    """Golden cross-check against the reference CBMLoss (skipped if the reference is not importable)."""
    try:
        CBMLoss = _reference_cbm_loss()
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"reference CBMLoss not importable from {REFERENCE_SRC}: {exc}")
    model, x, y = tiny_dit(), *batch()
    teacher = copy.deepcopy(model)
    with torch.no_grad():   # a different teacher, so reading the wrong network would show
        for p in teacher.parameters():
            p.add_(0.02 * torch.randn_like(p))
    ref = CBMLoss(reference_cfg(), label_dim=LABELS)
    ours = GuidedCBMLoss(our_cfg(), label_dim=LABELS)
    ours.teacher = teacher
    for step in (0, 5):
        torch.manual_seed(1234 + step)
        l_ref, a_ref = ref(model, x, y, step, teacher=teacher)
        g_ref = grads(model, l_ref)
        torch.manual_seed(1234 + step)
        l_our, a_our = ours(model, None, x, y, step)
        g_our = grads(model, l_our)
        for k in ("transport", "endpoint"):
            assert torch.equal(l_ref[k], l_our[k]), (k, l_ref[k].item(), l_our[k].item())
        for k in ("guide_frac", "guide_shift_rms", "guide_endpoint_frac", "guide_endpoint_shift_rms",
                  "endpoint_late_weight_mean"):
            assert torch.equal(a_ref[k], a_our[k]), k
        assert all(torch.equal(a, b) for a, b in zip(g_ref, g_our))
        assert float(a_our["guide_frac"]) > 0 and float(a_our["guide_endpoint_frac"]) > 0


@pytest.mark.parametrize("head", [False, True])
def test_tangent_only_jvp_changes_nothing(head):
    stock, x, y = tiny_dit(head=head), *batch()
    fast = use_tangent_only_jvp(copy.deepcopy(stock))
    ours = GuidedCBMLoss(our_cfg(), label_dim=LABELS)
    out = []
    for m in (stock, fast):
        ours.teacher = copy.deepcopy(stock)
        torch.manual_seed(7)
        losses, _ = ours(m, None, x, y, 3)
        out.append(({k: v.detach() for k, v in losses.items()}, grads(m, losses)))
    (l0, g0), (l1, g1) = out
    for k in l0:
        torch.testing.assert_close(l0[k], l1[k], rtol=1e-5, atol=1e-6)
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_time_shift_and_freeze_balance():
    model, x, y = tiny_dit(), *batch()
    ours = GuidedCBMLoss(our_cfg(time_shift=6.93, balance={"mode": "freeze"}, transport={"weight": 1.0},
                                 guide={"start_step": 3}, endpoint={"late_weight_start_step": None}), label_dim=LABELS)
    ours.teacher = copy.deepcopy(model)
    seen = []
    for step in range(5):
        torch.manual_seed(step)
        losses, aux = ours(model, None, x, y, step)
        assert all(torch.isfinite(v) for v in losses.values())
        if step < 3:   # balancer phase: no guide, no late band, the ratio EMA tracked
            assert "guide_frac" not in aux and "endpoint_late_weight_mean" not in aux
            assert float(aux["balance_ratio_ema"]) > 0
            # the actual time sits below the nominal one (more noise) under the shift
            assert float(aux["interpolant_t_mean"]) < float(aux["nominal_t_mean"])
        else:
            assert "guide_frac" in aux and "endpoint_late_weight_mean" in aux
            seen.append(float(aux["balance_transport_weight"]))
        grads(model, losses)
    assert seen[0] == seen[1] == ours.state_dict()["frozen_weight"] > 0
    restored = GuidedCBMLoss(our_cfg(balance={"mode": "freeze"}, guide={"start_step": 3}), label_dim=LABELS)
    restored.load_state_dict(ours.state_dict())
    assert restored.state_dict() == ours.state_dict()


def test_guided_step_without_teacher_raises():
    model, x, y = tiny_dit(), *batch()
    ours = GuidedCBMLoss(our_cfg(), label_dim=LABELS)
    with pytest.raises(RuntimeError, match="teacher"):
        ours(model, None, x, y, 0)


def test_ddt_head_is_wired():
    """The wide head changes the output, is conditioned per token by the trunk, and is zero-initialised like the trunk
    (the final layer starts at zero, so the untrained model outputs zero either way)."""
    x, y = batch()
    omega = torch.full((x.shape[0],), 0.7)
    fresh = DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2, num_heads=4, label_dim=LABELS,
                attn_func="default", head_hidden_size=96, head_depth=2, head_num_heads=4)
    assert torch.count_nonzero(fresh(x, y, omega=omega)) == 0
    m = tiny_dit(head=True)
    out = m(x, y, omega=omega)
    assert out.shape == x.shape and torch.isfinite(out).all()
    with torch.no_grad():   # the trunk reaches the output only through the head's per-token conditioning
        m.s_projector.weight.zero_(); m.s_projector.bias.zero_()
    assert not torch.allclose(m(x, y, omega=omega), out)


def test_marginal_tangent():
    """guide.marginal_tangent: the JVP tangent is the teacher's b(y) on nominal t in [t_lo, t_hi], plus kappa (b(y) -
    b(null)) inside the guide band, and V outside; off (the default) leaves the tangent untouched."""
    model, x, y = tiny_dit(), *batch(n=8)
    teacher = copy.deepcopy(model)
    on = GuidedCBMLoss(our_cfg(guide={"start_step": 0, "marginal_tangent": {"enabled": True, "t_hi": 0.95}}),
                       label_dim=LABELS)
    off = GuidedCBMLoss(our_cfg(guide={"start_step": 0}), label_dim=LABELS)
    torch.manual_seed(1)
    z = torch.randn_like(x)
    t = torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 0.97, 0.3, 0.55])[: x.shape[0]]
    t_a = on._warp(t)
    I_t, v = on._interp(z, x, t_a)
    with torch.no_grad():
        s = (1.0 - t_a).view(-1, *([1] * (x.ndim - 1)))
        b_y = (teacher(I_t, y, omega=t) - I_t) / s
        b_n = (teacher(I_t, torch.full_like(y, LABELS), omega=t) - I_t) / s
    tan_on, aux = on._guided_tangent(teacher, I_t, v, t, t_a, y, x)
    tan_off, _ = off._guided_tangent(teacher, I_t, v, t, t_a, y, x)
    for i in range(x.shape[0]):
        ti = float(t[i])
        in_band = on.g_tlo <= ti <= on.g_thi
        in_marg = on.m_tlo <= ti <= on.m_thi
        base = b_y[i] if in_marg else v[i]
        want = base + on.g_kappa * (b_y[i] - b_n[i]) if in_band else base
        torch.testing.assert_close(tan_on[i], want, rtol=1e-4, atol=1e-5)
        want_off = v[i] + off.g_kappa * (b_y[i] - b_n[i]) if in_band else v[i]
        torch.testing.assert_close(tan_off[i], want_off, rtol=1e-4, atol=1e-5)
    assert 0 < float(aux["marginal_frac"]) < 1


def test_repa_with_head():
    """loss.guided_cbm.repa: the endpoint forward returns the block-k projection of the wide-head model, the loss gains a
    finite 'repa' term whose gradient reaches the projector and the trunk blocks up to k; off, no 'repa' term."""
    x, y = batch(n=8)
    torch.manual_seed(0)
    m = DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=3, num_heads=4, label_dim=LABELS,
            attn_func="default", head_hidden_size=96, head_depth=2, head_num_heads=4,
            repa_enabled=True, repa_encoder_depth=1, repa_z_dim=4 * 2 * 2, repa_projector_dim=32)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    # patch 2 here: the clean-latent tokens are the 2x2 patches flattened, so compare against them
    on = GuidedCBMLoss(our_cfg(guide={"start_step": 10 ** 9}, repa={"enabled": True, "weight": 0.5}), label_dim=LABELS)
    off = GuidedCBMLoss(our_cfg(guide={"start_step": 10 ** 9}), label_dim=LABELS)
    on._repa_loss = lambda proj, x_, self=on: GuidedCBMLoss._repa_loss(
        self, proj, torch.nn.functional.pixel_unshuffle(x_, 2))   # patch-2 token layout for the test model
    torch.manual_seed(3)
    losses, aux = on(m, None, x, y, 0)
    assert "repa" in losses and torch.isfinite(losses["repa"]) and float(losses["repa"]) > 0
    assert -1.0 <= float(aux["repa_cosine"]) <= 1.0
    m.zero_grad(set_to_none=True)
    losses["repa"].backward()
    assert m.repa_projector[0].weight.grad is not None and m.repa_projector[0].weight.grad.abs().sum() > 0
    assert m.blocks[0].attn.qkv.weight.grad.abs().sum() > 0          # block 1 <= k feeds the projection
    assert m.blocks[2].attn.qkv.weight.grad is None or m.blocks[2].attn.qkv.weight.grad.abs().sum() == 0
    torch.manual_seed(3)
    losses_off, _ = off(m, None, x, y, 0)
    assert "repa" not in losses_off
    for k in ("transport", "endpoint"):   # the other terms are untouched by the REPA forward
        torch.testing.assert_close(losses_off[k], losses[k], rtol=1e-5, atol=1e-6)
