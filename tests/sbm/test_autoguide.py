"""Autoguidance (loss.guide.autoguide; Karras et al. 2024, arXiv 2406.02507): on the guided band the tangent gains
+ weight (b(y) - b_weak(y)) and the endpoint target + weight (D(y) - D_weak(y)), b_weak / D_weak read by a frozen weak model.
Off, every number is bit for bit the plain guided step's."""
from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

from test_golden import DEFAULTS, HERE, CBMLoss, loss_cfg, tiny

BASE = {"label_dropout": 0.1, "anchor_alpha": 0.65}
GUIDE = {"enabled": True, "w": 1.0, "kappa": 0.8, "endpoint": True, "diag_frac": 0.5, "start_step": 0}   # the SBM recipes' guide
NULL = 10               # label_dim of the tiny model


def ag(weight, **more):
    return {**GUIDE, "autoguide": {"enabled": True, "weight": weight, "ckpt": "unused-in-the-loss.ckpt", **more}}


@pytest.fixture(scope="module")
def b2():
    return torch.load(HERE / "golden" / "b2.pt", weights_only=False)


def _model(b2, seed=3, scale=0.02):
    m = tiny()
    m.load_state_dict(b2["model"])
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():   # nonzero readout, so the denoiser velocity is not trivially -I_t / (1 - t)
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


def _copy(model, seed, scale=0.05):
    """A perturbed frozen copy: the teacher, or the weak model."""
    other = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    other.requires_grad_(False)
    return other.eval()


def _run(loss_cls, cfg, model, b2, teacher=None, weak=None, seed=0, step=0):
    torch.manual_seed(seed)
    kw = {} if weak is None else {"weak": weak}
    losses, aux = loss_cls(cfg, label_dim=NULL)(model, b2["x"], b2["labels"], step, teacher=teacher, **kw)
    return losses, aux


# ---------------------------------------------------------------------- off: bit-identical to the plain guided step
@pytest.mark.parametrize("case", ["plain", "guided", "guided_teacher"])
def test_disabled_is_bit_identical(b2, case):
    """`enabled: false` with a ckpt and weight set gives the default config's terms and logged scalars exactly."""
    model = _model(b2)
    teacher = _copy(model, seed=5) if case == "guided_teacher" else None
    over = {} if case == "plain" else {"guide": GUIDE}
    ref, ref_aux = _run(CBMLoss, loss_cfg(**BASE, **over), model, b2, teacher=teacher)
    off_guide = {**(GUIDE if case != "plain" else {}), "autoguide": {"enabled": False, "weight": 0.3, "ckpt": "some.ckpt"}}
    got, got_aux = _run(CBMLoss, loss_cfg(**BASE, guide=off_guide), model, b2, teacher=teacher)
    assert ref.keys() == got.keys() and ref_aux.keys() == got_aux.keys()
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    for k in ref_aux:
        assert torch.equal(ref_aux[k], got_aux[k]), k
    if case == "guided":
        assert float(got_aux["guide_frac"]) > 0 and float(got_aux["guide_endpoint_frac"]) > 0   # the guided path was exercised


def test_weight_zero_reproduces_disabled(b2):
    """Enabled with weight 0: the weak model is read, the correction is exactly zero, the terms are the disabled ones."""
    model = _model(b2)
    teacher, weak = _copy(model, seed=5), _copy(model, seed=11)
    ref, _ = _run(CBMLoss, loss_cfg(**BASE, guide=GUIDE), model, b2, teacher=teacher)
    got, aux = _run(CBMLoss, loss_cfg(**BASE, guide=ag(0.0)), model, b2, teacher=teacher, weak=weak)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert "autoguide_shift_rms" in aux and float(aux["autoguide_shift_rms"]) == 0.0


def test_weak_equal_to_teacher_is_a_no_op(b2):
    """b(y) - b_weak(y) = 0 exactly when the weak model IS the teacher: the terms are the disabled ones."""
    model = _model(b2)
    teacher = _copy(model, seed=5)
    ref, _ = _run(CBMLoss, loss_cfg(**BASE, guide=GUIDE), model, b2, teacher=teacher)
    got, aux = _run(CBMLoss, loss_cfg(**BASE, guide=ag(0.1)), model, b2, teacher=teacher, weak=teacher)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert float(aux["autoguide_shift_rms"]) == 0.0


# ---------------------------------------------------------------------- on: finite, moving, no gradient into the weak model
def test_enabled_is_finite_moves_both_terms_and_trains(b2):
    model = _model(b2)
    teacher, weak = _copy(model, seed=5), _copy(model, seed=11)
    ref, _ = _run(CBMLoss, loss_cfg(**BASE, guide=GUIDE), model, b2, teacher=teacher)
    got, aux = _run(CBMLoss, loss_cfg(**BASE, guide=ag(0.1)), model, b2, teacher=teacher, weak=weak)
    assert all(torch.isfinite(v) for v in got.values())
    assert not torch.equal(ref["transport"], got["transport"]) and not torch.equal(ref["endpoint"], got["endpoint"])
    assert float(aux["autoguide_shift_rms"]) > 0 and float(aux["guide_frac"]) > 0 and float(aux["guide_endpoint_frac"]) > 0
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.autograd.grad(sum(got.values()), params, allow_unused=True)
    assert all(x is None or torch.isfinite(x).all() for x in g) and any(x is not None and x.abs().sum() > 0 for x in g)
    assert all(p.grad is None and not p.requires_grad for p in weak.parameters())   # read without gradient, frozen
    # before guide.start_step nothing is guided, so nothing is autoguided either, and no weak model is needed
    early, e_aux = _run(CBMLoss, loss_cfg(**BASE, guide={**ag(0.1), "start_step": 10 ** 9}), model, b2, teacher=teacher)
    assert all(torch.isfinite(v) for v in early.values()) and "autoguide_shift_rms" not in e_aux


def test_matches_hand_computation(b2):
    """Tangent inside [t_lo, t_hi] (non-null labels): V + kappa (b_y - b_null) + weight (b_y - b_weak), the three reads at
    (I_t, ., omega = t), V elsewhere. Endpoint target inside: x + kappa (D_y - D_null) + weight (D_y - D_weak) at
    (I_omega', ., omega = omega'), x elsewhere. Rows with the null label are never guided."""
    model = _model(b2)
    weak = _copy(model, seed=11)
    kappa, weight = 0.8, 0.1
    loss = CBMLoss(loss_cfg(**BASE, guide=ag(weight)), label_dim=NULL)
    x = b2["x"]
    labels = b2["labels"].long().clone()
    labels[0] = NULL   # one null row inside the band
    t = torch.linspace(0.36, 0.59, x.shape[0])
    t[-1] = 0.7        # one row outside the band
    I_t, v = loss._interp(torch.randn(x.shape, generator=torch.Generator().manual_seed(5)), x, t)
    tangent, aux, inside = loss._guided_tangent(model, I_t, v, t, labels, x, weak=weak)
    s = (1 - t).view(-1, 1, 1, 1)
    with torch.no_grad():
        b_0 = (model(I_t, torch.full_like(labels, NULL), omega=t) - I_t) / s
        b_y = (model(I_t, labels, omega=t) - I_t) / s
        b_w = (weak(I_t, labels, omega=t) - I_t) / s
    band = ((t >= 0.35) & (t <= 0.6) & (labels != NULL))
    assert torch.equal(inside, band) and bool(band[1]) and not bool(band[0]) and not bool(band[-1])
    want = torch.where(band.view(-1, 1, 1, 1), v + kappa * (b_y - b_0) + weight * (b_y - b_w), v)
    assert torch.allclose(tangent, want, rtol=1e-5, atol=1e-6) and not tangent.requires_grad
    assert torch.equal(tangent[0], v[0]) and torch.equal(tangent[-1], v[-1])
    rms = (weight * (b_y - b_w)).flatten(1).pow(2).sum(1)[band].mean().sqrt()
    assert torch.allclose(aux["autoguide_shift_rms"], rms, rtol=1e-5, atol=1e-6)
    om = t.clone()
    I_end, _ = loss._interp(torch.randn(x.shape, generator=torch.Generator().manual_seed(7)), x, om)
    target, _ = loss._guided_endpoint(model, I_end, x, om, labels, weak=weak)
    with torch.no_grad():
        d_y, d_0, d_w = model(I_end, labels, omega=om), model(I_end, torch.full_like(labels, NULL), omega=om), weak(I_end, labels, omega=om)
    want_e = torch.where(band.view(-1, 1, 1, 1), x + kappa * (d_y - d_0) + weight * (d_y - d_w), x)
    assert torch.allclose(target, want_e, rtol=1e-5, atol=1e-6) and not target.requires_grad
    assert torch.equal(target[0], x[0]) and torch.equal(target[-1], x[-1])


# ---------------------------------------------------------------------- validation
@pytest.mark.parametrize("guide", [
    {**GUIDE, "kappa": 0.8, "autoguide": {"enabled": True, "weight": 0.2, "ckpt": "w.ckpt"}},     # kappa + weight = 1: not < 1
    {**GUIDE, "kappa": 0.9, "autoguide": {"enabled": True, "weight": 0.1, "ckpt": "w.ckpt"}},     # kappa + weight = 1
    {**GUIDE, "autoguide": {"enabled": True, "weight": 0.1}},                                      # no ckpt
    {**GUIDE, "autoguide": {"enabled": True, "weight": -0.1, "ckpt": "w.ckpt"}},                   # negative weight
    {**GUIDE, "enabled": False, "autoguide": {"enabled": True, "weight": 0.1, "ckpt": "w.ckpt"}},  # guide off
    {**GUIDE, "omega_cond": True, "autoguide": {"enabled": True, "weight": 0.1, "ckpt": "w.ckpt"}},   # iMF's omega_cond path
])
def test_config_validation(guide):
    with pytest.raises(ValueError):
        CBMLoss(loss_cfg(**BASE, guide=guide), label_dim=NULL)
    CBMLoss(loss_cfg(**BASE, guide={**GUIDE, "kappa": 0.8, "autoguide": {"enabled": True, "weight": 0.1, "ckpt": "w.ckpt"}}),
            label_dim=NULL)   # 0.9 < 1 passes


def test_missing_weak_model_raises(b2):
    model = _model(b2)
    with pytest.raises(RuntimeError, match="no weak model"):
        _run(CBMLoss, loss_cfg(**BASE, guide=ag(0.1)), model, b2, teacher=_copy(model, seed=5))


# ---------------------------------------------------------------------- the trainer: a frozen copy loaded once, no submodule
def _lightning_ckpt(path, active, slow, raw):
    """A checkpoint as sbm.ema writes it: state_dict = the active EMA copy, current_model_state = raw, the others in ema_state_dicts."""
    p = lambda m: {"model." + k: v.clone() for k, v in m.state_dict().items()}
    torch.save({"state_dict": p(active), "current_model_state": p(raw), "ema_state_dicts": {"0_9999": p(slow)},
                "ema_decays": {"0_999": 0.999, "0_9999": 0.9999}, "ema_active_decay": 0.999, "global_step": 7}, path)


def _module_cfg(ckpt, which="ema", enabled=True):
    return OmegaConf.merge(DEFAULTS, OmegaConf.from_dotlist([
        "model.name=DiT-tiny", "model.input_size=8", f"model.label_dim={NULL}", "loss.label_dropout=0.1", "loss.anchor_alpha=0.65",
        "loss.guide.enabled=true", "loss.guide.kappa=0.8", "loss.guide.endpoint=true", "loss.guide.diag_frac=0.5",
        f"loss.guide.autoguide.enabled={str(enabled).lower()}", "loss.guide.autoguide.weight=0.1",
        f"loss.guide.autoguide.ckpt={ckpt}", f"loss.guide.autoguide.which={which}"]))


def test_trainer_builds_a_frozen_weak_model_once(b2, tmp_path):
    from sbm.train import Module
    model = _model(b2)
    active, slow, raw = _copy(model, seed=11), _copy(model, seed=12), _copy(model, seed=13)
    ck = tmp_path / "step0000007.ckpt"
    _lightning_ckpt(ck, active, slow, raw)
    mod = Module(_module_cfg(ck), tiny())
    weak = mod._guide_weak()
    assert weak is mod._guide_weak()                                                   # loaded once, then reused
    assert not weak.training and all(not p.requires_grad for p in weak.parameters())   # eval, frozen
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert same(weak, active) and not same(weak, slow) and not same(weak, raw)         # which=ema: the active EMA, as init.weights=ema
    assert same(Module(_module_cfg(ck, which="ema_0_9999"), tiny())._guide_weak(), slow)
    assert same(Module(_module_cfg(ck, which="raw"), tiny())._guide_weak(), raw)
    assert Module(_module_cfg(ck, enabled=False), tiny())._guide_weak() is None
    # the weak model is no submodule: the module's checkpoint layout is unchanged, so a resume's strict load is untouched
    plain = Module(_module_cfg(ck, enabled=False), tiny())
    assert list(mod.state_dict()) == list(plain.state_dict()) and not any("weak" in k for k in mod.state_dict())
    mod.load_state_dict(plain.state_dict(), strict=True)
    assert same(mod._guide_weak(), active)                                             # a load does not touch it
    # the loss runs with it: finite terms, the correction logged
    torch.manual_seed(0)
    losses, aux = mod.loss(mod.model, b2["x"], b2["labels"], 0, teacher=_copy(model, seed=5), weak=mod._guide_weak())
    assert all(torch.isfinite(v) for v in losses.values()) and float(aux["autoguide_shift_rms"]) > 0
