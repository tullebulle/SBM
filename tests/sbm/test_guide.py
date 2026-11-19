"""iMF-style guided tangent and endpoint (loss.guide): the SBM recipes' self-guidance."""
from __future__ import annotations

import pytest
import torch

from test_golden import CBMLoss, HERE, loss_cfg, tiny

BASE = {"label_dropout": 0.1, "anchor_alpha": 0.65}


@pytest.fixture(scope="module")
def b2():
    return torch.load(HERE / "golden" / "b2.pt", weights_only=False)


def _model(b2):
    m = tiny()
    m.load_state_dict(b2["model"])
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():   # nonzero readout, so the denoiser velocity is not trivially -I_t / (1 - t)
        for p in m.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return m


def _losses(b2, model, **over):
    torch.manual_seed(0)
    return CBMLoss(loss_cfg(**{**BASE, **over}), label_dim=10)(model, b2["x"], b2["labels"], 0)


def test_w1_and_empty_interval_are_btm(b2):
    """w = 1 (two-term form: V + 0 * b) and an interval no transport time falls in give BTM's terms exactly."""
    model = _model(b2)
    ref, _ = _losses(b2, model)
    for guide in ({"enabled": True, "w": 1.0}, {"enabled": True, "w": 3.0, "t_lo": 0.35, "t_hi": 0.35 + 1e-7}):
        got, _ = _losses(b2, model, guide=guide)
        for k in ref:
            assert torch.allclose(ref[k], got[k], rtol=1e-6, atol=1e-7), (guide, k)


@pytest.mark.parametrize("kappa", [0.0, 0.5])   # 0: iMF's two-term form (the default); 0.5: MeanFlow's third term
def test_tangent_matches_hand_computation(b2, kappa):
    model = _model(b2)
    w = 2.0
    loss = CBMLoss(loss_cfg(**BASE, guide={"enabled": True, "w": w, "kappa": kappa}), label_dim=10)
    x, labels = b2["x"], b2["labels"].long()
    t = torch.linspace(0.2, 0.7, x.shape[0])
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(5))
    I_t, v = loss._interp(z, x, t)
    tangent, aux, _ = loss._guided_tangent(model, I_t, v, t, labels, x)
    assert not tangent.requires_grad
    s = (1 - t).view(-1, 1, 1, 1)
    with torch.no_grad():
        b_u = (model(I_t, torch.full_like(labels, 10), omega=t) - I_t) / s
        b_c = (model(I_t, labels, omega=t) - I_t) / s
    want = w * v + kappa * b_c + (1 - w - kappa) * b_u
    inside = ((t >= 0.35) & (t <= 0.6)).view(-1, 1, 1, 1)
    assert torch.allclose(tangent, torch.where(inside, want, v), rtol=1e-5, atol=1e-6)
    assert 0 < float(aux["guide_frac"]) < 1


def test_null_labels_unguided(b2):
    model = _model(b2)
    loss = CBMLoss(loss_cfg(**BASE, guide={"enabled": True, "w": 3.0}), label_dim=10)
    x = b2["x"]
    labels = torch.full((x.shape[0],), 10, dtype=torch.long)
    t = torch.full((x.shape[0],), 0.5)
    I_t, v = loss._interp(torch.randn_like(x), x, t)
    tangent, _, _ = loss._guided_tangent(model, I_t, v, t, labels)
    assert torch.equal(tangent, v)


def test_guided_loss_differs_and_trains(b2):
    model = _model(b2)
    ref, _ = _losses(b2, model)
    got, aux = _losses(b2, model, guide={"enabled": True, "w": 3.0})
    assert not torch.allclose(ref["transport"], got["transport"]) and torch.isfinite(got["transport"])
    assert torch.allclose(ref["endpoint"], got["endpoint"])
    g = torch.autograd.grad(got["transport"], [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    assert any(x is not None and x.abs().sum() > 0 for x in g)


@pytest.mark.parametrize("over", [
    {"guide": {"enabled": True, "w": 2.0}, "label_dropout": 0.0},                       # no null class
    {"guide": {"enabled": True, "w": 2.0}, "anchor_alpha": 0.4},                         # anchors stop at 0.6 > t_lo
    {"guide": {"enabled": True, "w": 2.0, "negative": "nothing"}},
])
def test_config_validation(over):
    with pytest.raises(ValueError):
        CBMLoss(loss_cfg(**{**BASE, **over}), label_dim=10)


def test_shift_form_diagnostics(b2):
    """w = 1, kappa = g - 1 is the shift form V + (g - 1)(b_y - b_null); its step logs the weight spread and denoiser MSE."""
    model = _model(b2)
    got, aux = _losses(b2, model, guide={"enabled": True, "w": 1.0, "kappa": 2.0})
    assert torch.isfinite(got["transport"]) and "guide_weight_cv" in aux and "guide_denoiser_mse" in aux
    assert float(aux["guide_weight_cv"]) >= 0 and float(aux["guide_denoiser_mse"]) > 0


def test_guided_endpoint_reduces_and_matches_hand(b2):
    """Endpoint target: x at w = 1, kappa = 0; inside the interval w x + kappa D(y) + (1 - w - kappa) D(null) else x."""
    model = _model(b2)
    ref, _ = _losses(b2, model)
    got, _ = _losses(b2, model, guide={"enabled": True, "w": 1.0, "kappa": 0.0, "endpoint": True})
    for k in ref:
        assert torch.allclose(ref[k], got[k], rtol=1e-6, atol=1e-7), k
    loss = CBMLoss(loss_cfg(**BASE, guide={"enabled": True, "w": 1.0, "kappa": 2 / 3, "endpoint": True}), label_dim=10)
    x, labels = b2["x"], b2["labels"].long()
    om = torch.linspace(0.3, 0.7, x.shape[0])
    I_end, _ = loss._interp(torch.randn(x.shape, generator=torch.Generator().manual_seed(7)), x, om)
    target, aux = loss._guided_endpoint(model, I_end, x, om, labels)
    with torch.no_grad():
        d_y = model(I_end, labels, omega=om)
        d_0 = model(I_end, torch.full_like(labels, 10), omega=om)
    inside = ((om >= 0.35) & (om <= 0.6)).view(-1, 1, 1, 1)
    want = torch.where(inside, x + (2 / 3) * (d_y - d_0), x)
    assert torch.allclose(target, want, rtol=1e-5, atol=1e-6) and not target.requires_grad
    assert 0 < float(aux["guide_endpoint_frac"]) < 1


def test_diag_rows_and_full_recipe_step(b2):
    """The recipes' configuration (bootstrapped tangent + guided endpoint + diagonal rows) runs, every term finite and trained."""
    model = _model(b2)
    got, aux = _losses(b2, model, guide={"enabled": True, "w": 1.0, "kappa": 0.8, "endpoint": True, "diag_frac": 0.5})
    assert all(torch.isfinite(v) for v in got.values()) and "guide_endpoint_shift_rms" in aux
    g = torch.autograd.grad(sum(got.values()), [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    assert any(x is not None and x.abs().sum() > 0 for x in g)


def test_start_step_gates_the_guide(b2):
    """Before loss.guide.start_step only the diagonal endpoint rows differ from BTM (they are drawn from step 0); with
    diag_frac 0 the step before start_step is BTM's exactly."""
    model = _model(b2)
    ref, _ = _losses(b2, model)
    g = {"enabled": True, "w": 1.0, "kappa": 0.8, "endpoint": True, "diag_frac": 0.0, "start_step": 35000}
    early, aux = _losses(b2, model, guide=g)
    for k in ref:
        assert torch.equal(ref[k], early[k]), k
    assert "guide_frac" not in aux
    torch.manual_seed(0)
    late, aux_late = CBMLoss(loss_cfg(**BASE, guide=g), label_dim=10)(model, b2["x"], b2["labels"], 35000)
    assert "guide_frac" in aux_late and not torch.equal(ref["transport"], late["transport"])
