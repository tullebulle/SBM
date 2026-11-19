"""iMF's Omega-conditioning on the map (loss.guide.omega_cond; off in the paper's maps): the scale and interval inputs,
iMF's draws, the gated self-reads, the guided tangent and endpoint, the checkpoint fill and the evaluator read."""
from __future__ import annotations

import math

import pytest
import torch

from test_golden import CBMLoss, HERE, loss_cfg
from sbm import eval as E
from sbm.models import build_model, fill_gw_keys

BASE = {"label_dropout": 0.1, "anchor_alpha": 0.65}
R2 = {"enabled": True, "w": 1.0, "kappa": 0.0, "endpoint": True, "omega_cond": True, "w_min": 1.0, "w_max": 8.0, "beta": 1.0}


@pytest.fixture(scope="module")
def b2():
    return torch.load(HERE / "golden" / "b2.pt", weights_only=False)


def tiny(omega_cond=False):
    torch.manual_seed(0)
    kw = {"gw_embedder": True, "gw_input": "kappa", "gw_interval": True} if omega_cond else {}
    return build_model("DiT-tiny", input_size=8, label_dim=10, **kw)


def _perturb(model, seed=3, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return model


def _parent_and_child(b2):
    parent = _perturb(tiny())
    parent.load_state_dict({**parent.state_dict(), **b2["model"]})
    _perturb(parent)
    child = tiny(True)
    sd = dict(parent.state_dict())
    added = fill_gw_keys(child.state_dict(), sd)
    child.load_state_dict(sd, strict=True)
    return parent, child, added


def test_defaults_have_no_interval_input():
    m = tiny()
    assert m.gi_lo_embedder is None and m.gw_embedder is None
    with pytest.raises(ValueError):
        m(torch.zeros(2, 4, 8, 8), torch.zeros(2, dtype=torch.long), s_lo=0.1, s_hi=0.9)


def test_fill_and_zero_init_is_the_parent(b2):
    parent, child, added = _parent_and_child(b2)
    assert {k.split(".")[0] for k in added} == {"gw_embedder", "gi_lo_embedder", "gi_hi_embedder"}
    x, y = b2["x"], b2["labels"].long()
    ref = parent(x, y, omega=0.8)
    B = x.shape[0]
    for w, lo, hi in [(None, None, None), (3.0, 0.2, 0.7), (torch.linspace(1, 8, B), torch.rand(B) * 0.5, 0.5 + torch.rand(B) * 0.5)]:
        kw = {} if lo is None else {"s_lo": lo, "s_hi": hi}
        assert torch.allclose(child(x, y, omega=0.8, w=w, **kw), ref, atol=1e-6)


def test_draws_follow_imf_laws():
    loss = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)
    torch.manual_seed(0)
    w, lo, hi = loss._draw_omega(torch.zeros(20000, 4, 8, 8))
    assert float(w.min()) >= 1.0 and float(w.max()) <= 8.0 and 0 <= float(lo.min()) and float(lo.max()) <= 0.5
    assert 0.5 <= float(hi.min()) and float(hi.max()) <= 1.0
    # beta 1: log w uniform on [0, log 8], mean log 8 / 2
    assert abs(float(w.log().mean()) - math.log(8) / 2) < 0.02


def test_unit_scale_reduces_to_btm(b2):
    """w == 1 everywhere: the tangent is V and the endpoint target x, so the terms equal the unguided recipe's."""
    parent, child, _ = _parent_and_child(b2)
    torch.manual_seed(0)
    ref, _ = CBMLoss(loss_cfg(**BASE), label_dim=10)(parent, b2["x"], b2["labels"], 0)
    torch.manual_seed(0)
    got, aux = CBMLoss(loss_cfg(**BASE, guide={**R2, "w_max": 1.0}), label_dim=10)(child, b2["x"], b2["labels"], 0)
    for k in ref:
        assert torch.allclose(ref[k], got[k], rtol=1e-5, atol=1e-6), k
    assert float(aux["guide_shift_rms"]) == 0.0 and float(aux["guide_endpoint_shift_rms"]) == 0.0


def test_tangent_and_endpoint_match_hand(b2):
    _, child, _ = _parent_and_child(b2)
    _perturb(child, seed=9, scale=0.05)   # nonzero embedders: the reads depend on w
    loss = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)
    x, labels = b2["x"], b2["labels"].long()
    B = x.shape[0]
    t = torch.linspace(0.2, 0.9, B)
    w = torch.linspace(1.5, 6.0, B)
    lo, hi = torch.full((B,), 0.3), torch.full((B,), 0.7)
    I_t, v = loss._interp(torch.randn(x.shape, generator=torch.Generator().manual_seed(4)), x, t)
    tangent, aux, inside = loss._guided_tangent_omega(child, I_t, v, t, labels, w, lo, hi)
    want_in = (t >= 0.35) & (t >= 0.3) & (t <= 0.7)
    assert torch.equal(inside, want_in)
    w_g = torch.where(want_in, w, torch.ones_like(w))
    z0, o1 = torch.zeros(B), torch.ones(B)
    with torch.no_grad():
        d_y = child(I_t, labels, omega=t, w=w_g, s_lo=z0, s_hi=o1)
        d_0 = child(I_t, torch.full_like(labels, 10), omega=t, w=o1, s_lo=z0, s_hi=o1)
    want = v + (1 - 1 / w_g).view(-1, 1, 1, 1) * (d_y - d_0) / (1 - t).view(-1, 1, 1, 1)
    assert torch.allclose(tangent, want, rtol=1e-5, atol=1e-6) and not tangent.requires_grad
    target, _ = loss._guided_endpoint_omega(child, I_t, x, t, labels, w, lo, hi)
    want_e = x + (1 - 1 / w_g).view(-1, 1, 1, 1) * (d_y - d_0)
    assert torch.allclose(target, want_e, rtol=1e-5, atol=1e-6)


def test_t_cap_bounds_the_guided_tangent(b2):
    """t_cap: no guided tangent above it (the read's 1 / (1 - t) amplification), the endpoint target untouched."""
    _, child, _ = _parent_and_child(b2)
    _perturb(child, seed=9, scale=0.05)
    x, labels = b2["x"], b2["labels"].long()
    B = x.shape[0]
    t = torch.linspace(0.4, 0.98, B)
    w = torch.full((B,), 4.0)
    lo, hi = torch.zeros(B), torch.ones(B)
    I_t, v = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)._interp(torch.randn(x.shape, generator=torch.Generator().manual_seed(4)), x, t)
    capped = CBMLoss(loss_cfg(**BASE, guide={**R2, "t_cap": 0.7}), label_dim=10)
    tangent, _, inside = capped._guided_tangent_omega(child, I_t, v, t, labels, w, lo, hi)
    assert torch.equal(inside, t <= 0.7 + 1e-6) and bool(inside.any()) and not bool(inside.all())
    assert torch.equal(tangent[~inside], v[~inside])
    full, _, inside_full = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)._guided_tangent_omega(child, I_t, v, t, labels, w, lo, hi)
    assert bool(inside_full.all()) and torch.allclose(tangent[inside], full[inside], rtol=1e-5, atol=1e-6)
    e_cap, _ = capped._guided_endpoint_omega(child, I_t, x, t, labels, w, lo, hi)
    e_full, _ = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)._guided_endpoint_omega(child, I_t, x, t, labels, w, lo, hi)
    assert torch.allclose(e_cap, e_full)


def test_diagonal_rows_see_the_full_interval(b2):
    _, child, _ = _parent_and_child(b2)
    seen = []
    orig = child.forward

    def spy(x, y, omega=None, w=None, s_lo=None, s_hi=None, **kw):
        if s_lo is not None and torch.is_grad_enabled():
            seen.append((s_lo.detach().clone(), s_hi.detach().clone()))
        return orig(x, y, omega=omega, w=w, s_lo=s_lo, s_hi=s_hi, **kw)

    child.forward = spy
    torch.manual_seed(0)
    CBMLoss(loss_cfg(**BASE, guide={**R2, "diag_frac": 1.0}), label_dim=10)(child, b2["x"], b2["labels"], 0)
    end_lo, end_hi = seen[-1]   # the endpoint forward (the transport JVP's inputs come first)
    assert torch.all(end_lo == 0) and torch.all(end_hi == 1)


def test_full_step_trains_the_embedders(b2):
    _, child, _ = _parent_and_child(b2)
    _perturb(child, seed=11, scale=0.05)
    torch.manual_seed(0)
    got, aux = CBMLoss(loss_cfg(**BASE, guide=R2), label_dim=10)(child, b2["x"], b2["labels"], 0)
    assert all(torch.isfinite(v) for v in got.values()) and 1.0 <= float(aux["guide_w_mean"]) <= 8.0
    emb = list(child.gw_embedder.parameters()) + list(child.gi_lo_embedder.parameters())
    g = torch.autograd.grad(sum(got.values()), emb, allow_unused=True)
    assert any(x is not None and x.abs().sum() > 0 for x in g)


def test_config_guards():
    with pytest.raises(ValueError):
        CBMLoss(loss_cfg(**BASE, guide={**R2, "endpoint": False}), label_dim=10)
    with pytest.raises(ValueError):
        CBMLoss(loss_cfg(**BASE, guide={**R2, "w_min": 0.5}), label_dim=10)


def test_evaluator_reads_at_scale_and_interval(b2):
    _, child, _ = _parent_and_child(b2)
    _perturb(child, seed=13, scale=0.05)
    labels = b2["labels"].long()
    p = {"steps": [1], "gw": 3.0, "gw_lo": 0.35, "gw_hi": 0.6}
    g = torch.Generator().manual_seed(1)
    out = E._sample_b2(child, labels, p, torch.device("cpu"), g, 0.8)[1]
    x = torch.randn((len(labels), 4, 8, 8), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(out, child(x, labels, omega=0.8, w=3.0, s_lo=0.35, s_hi=0.6), atol=1e-5)
