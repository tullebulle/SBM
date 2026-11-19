"""The XL recipe's switches: the frozen teacher's reads, the late-path time mixture, t_cap, the
EMA balancer, v_weight and the late endpoint weight. Off, each is the recipe bit for bit."""
from __future__ import annotations

import copy

import torch

from sbm.loss import CBMLoss
from test_golden import HERE, loss_cfg, tiny


def _golden_model():
    model = tiny()
    model.load_state_dict(torch.load(HERE / "golden" / "b2.pt", weights_only=False)["model"])
    return model


def test_guide_teacher_reads():
    """loss.guide.teacher_decay: the fixed-kappa guide reads from the teacher; teacher = the online map is the recipe."""
    torch.manual_seed(0)
    model = _golden_model()
    cfg = loss_cfg(label_dropout=0.1, anchor_alpha=0.65,
                   guide={"enabled": True, "kappa": 2.0 / 3.0, "endpoint": True, "diag_frac": 0.5, "start_step": 0})
    loss = CBMLoss(cfg, label_dim=10)
    x, y = torch.randn(8, 4, 8, 8), torch.randint(10, (8,))

    def run(teacher):
        torch.manual_seed(1)
        losses, aux = loss(model, x, y, 0, teacher=teacher)
        return {k: float(v) for k, v in losses.items()}, aux

    base, _ = run(None)
    same, _ = run(model)
    assert base == same
    other = copy.deepcopy(model)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.05 * torch.randn_like(p))
    moved, aux = run(other)
    assert moved["transport"] != base["transport"] and moved["endpoint"] != base["endpoint"]
    assert float(aux.get("guide_frac", 0)) > 0


def test_time_mix_stays_below_the_anchor():
    """loss.time.mix_frac: the late transport times are uniform on [mix_lo, omega); off, the draws are the recipe's."""
    model = _golden_model()
    x, y = torch.randn(64, 4, 8, 8), torch.randint(10, (64,))
    seen = {}

    def run(**time):
        loss = CBMLoss(loss_cfg(anchor_alpha=0.65, time=time), label_dim=10)
        orig = loss._interp
        loss._interp = lambda z, x_, t: (seen.setdefault("t", []).append(t.clone()), orig(z, x_, t))[1]
        torch.manual_seed(3)
        losses, aux = loss(model, x, y, 0)
        return {k: float(v) for k, v in losses.items()}, float(aux["interpolant_t_mean"])

    base, t_base = run()
    off, t_off = run(mix_frac=0.0)
    assert base == off and t_base == t_off
    seen.clear()
    _, t_on = run(mix_frac=1.0, mix_lo=0.3)
    t = seen["t"][0]                                    # the transport draw (the endpoint's interpolant comes second)
    assert float(t.min()) >= 0.3 and float(t.max()) <= 1.0 and t_on > t_base


def test_guide_t_cap_bounds_the_tangent_only():
    """loss.guide.t_cap: the guided tangent stops at t_cap, the guided endpoint target runs to t_hi; default = no cap."""
    model = _golden_model()
    x, y = torch.randn(256, 4, 8, 8), torch.randint(10, (256,))
    base = {"enabled": True, "kappa": 0.8, "endpoint": True, "diag_frac": 0.5, "start_step": 0}

    def run(**guide):
        loss = CBMLoss(loss_cfg(label_dropout=0.1, anchor_alpha=0.65, time={"p_mean": 0.5}, guide={**base, **guide}), label_dim=10)
        torch.manual_seed(5)
        losses, aux = loss(model, x, y, 0)
        return {k: float(v) for k, v in losses.items()}, {k: float(v) for k, v in aux.items() if k.startswith("guide")}

    plain, a0 = run()
    same, a1 = run(t_cap=1.0)
    assert plain == same and a0 == a1                                   # default: unchanged
    wide, aw = run(t_hi=0.9)                                             # tangent and endpoint guided to 0.9
    capped, ac = run(t_hi=0.9, t_cap=0.6)                                # tangent to 0.6, endpoint to 0.9
    assert ac["guide_frac"] == a0["guide_frac"] < aw["guide_frac"]       # tangent interval as [0.35, 0.6]
    assert ac["guide_endpoint_frac"] == aw["guide_endpoint_frac"] > a0["guide_endpoint_frac"]


def test_balance_ema_smooths_the_ratio():
    """loss.balance.mode=ema: the final-layer norms are smoothed before the ratio is taken; the raw norms are still logged,
    and the smoothed state rides in the loss state_dict."""
    torch.manual_seed(0)
    model = tiny()
    x, y = torch.randn(4, 4, 8, 8), torch.randint(10, (4,))
    per_step = CBMLoss(loss_cfg(anchor_alpha=0.65), label_dim=10)
    ema = CBMLoss(loss_cfg(anchor_alpha=0.65, balance={"mode": "ema", "ema": 0.5}), label_dim=10)
    assert ema._balance_norms is None

    def terms(seed):
        torch.manual_seed(seed)
        out = model(x, y, omega=torch.full((4,), 0.9))
        return (out - x).pow(2).mean() * (1 + seed), (out * 3 - x).pow(2).mean() / (1 + seed)

    for seed in range(3):
        t_, e_ = terms(seed)
        w_t_step, w_e_step, aux_step = per_step._balance(model, t_, e_)
        t_, e_ = terms(seed)
        w_t_ema, w_e_ema, aux_ema = ema._balance(model, t_, e_)
        # same raw norms logged by both; the first ema step equals the per-step ratio, later ones differ
        assert torch.allclose(aux_step["final_layer_grad_norm_transport"], aux_ema["final_layer_grad_norm_transport"])
        if seed == 0:
            assert torch.allclose(w_t_step, w_t_ema) and torch.allclose(w_e_step, w_e_ema)
        else:
            assert not (torch.allclose(w_t_step, w_t_ema) and torch.allclose(w_e_step, w_e_ema))
    m_e, m_t = ema._balance_norms
    assert torch.allclose(w_t_ema * w_e_ema, torch.clamp_min(torch.maximum(m_e, m_t) / (torch.minimum(m_e, m_t) + 1e-8), 1.0))
    state = ema.state_dict()
    fresh = CBMLoss(loss_cfg(anchor_alpha=0.65, balance={"mode": "ema", "ema": 0.5}), label_dim=10)
    fresh.load_state_dict(state)
    assert all(torch.allclose(a, b) for a, b in zip(fresh._balance_norms, ema._balance_norms))
    reset = CBMLoss(loss_cfg(anchor_alpha=0.65,
                             balance={"mode": "ema", "ema": 0.5, "reset_on_resume": True}), label_dim=10)
    reset.load_state_dict(state)
    assert reset._balance_norms is None


def test_balance_none_with_fixed_weights():
    """loss.balance.mode=none + transport.weight / endpoint.weight: the XL line's frozen ratio (2000 / 1); the unweighted
    terms equal the balanced run's and the weights multiply them."""
    torch.manual_seed(0)
    model = _golden_model()
    x, y = torch.randn(4, 4, 8, 8), torch.randint(10, (4,))
    torch.manual_seed(1)
    bal, aux_b = CBMLoss(loss_cfg(anchor_alpha=0.65), label_dim=10)(model, x, y, 0)
    torch.manual_seed(1)
    fixed, aux_f = CBMLoss(loss_cfg(anchor_alpha=0.65, balance={"mode": "none"}, transport={"weight": 2000}, endpoint={"weight": 1}),
                           label_dim=10)(model, x, y, 0)
    assert torch.equal(aux_b["transport_unweighted"], aux_f["transport_unweighted"])
    assert torch.equal(aux_b["endpoint_unweighted"], aux_f["endpoint_unweighted"])
    assert torch.allclose(fixed["transport"], 2000 * aux_f["transport_unweighted"]) and torch.equal(fixed["endpoint"], aux_f["endpoint_unweighted"])
    assert "balance_transport_weight" not in aux_f and "balance_transport_weight" in aux_b


def test_guide_v_weight_swaps_the_sample_velocity():
    """loss.guide.v_weight: 1 is the recipe (bit for bit); 0 replaces the sample velocity x - z inside the guided interval by
    the conditional read b_y (iMF's predicted tangent), so the guided tangent no longer depends on the drawn x there."""
    torch.manual_seed(0)
    model = _golden_model()
    g = {"enabled": True, "kappa": 0.8, "t_lo": 0.35, "t_hi": 0.6, "start_step": 0}
    stock = CBMLoss(loss_cfg(anchor_alpha=0.65, label_dropout=0.1, guide=g), label_dim=10)
    same = CBMLoss(loss_cfg(anchor_alpha=0.65, label_dropout=0.1, guide={**g, "v_weight": 1.0}), label_dim=10)
    pred = CBMLoss(loss_cfg(anchor_alpha=0.65, label_dropout=0.1, guide={**g, "v_weight": 0.0}), label_dim=10)
    half = CBMLoss(loss_cfg(anchor_alpha=0.65, label_dropout=0.1, guide={**g, "v_weight": 0.5}), label_dim=10)
    B = 6
    z, x, y = torch.randn(B, 4, 8, 8), torch.randn(B, 4, 8, 8), torch.randint(10, (B,))
    t = torch.full((B,), 0.5)
    I_t = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
    v = x - z
    tan_stock, _, inside = stock._guided_tangent(model, I_t, v, t, y, x)
    tan_same, _, _ = same._guided_tangent(model, I_t, v, t, y, x)
    tan_pred, aux_pred, _ = pred._guided_tangent(model, I_t, v, t, y, x)
    tan_half, _, _ = half._guided_tangent(model, I_t, v, t, y, x)
    assert inside.all()
    assert torch.equal(tan_stock, tan_same)
    with torch.no_grad():
        s = (1 - t)[:, None, None, None]
        b_y = (model(I_t, y, omega=t) - I_t) / s
        b_0 = (model(I_t, torch.full_like(y, 10), omega=t) - I_t) / s
    assert torch.allclose(tan_stock, v + 0.8 * (b_y - b_0), atol=1e-5)
    assert torch.allclose(tan_pred, b_y + 0.8 * (b_y - b_0), atol=1e-5)
    assert torch.allclose(tan_half, 0.5 * (tan_stock + tan_pred), atol=1e-5)
    # the predicted tangent is independent of the drawn x at fixed I_t
    other_x = x + torch.randn_like(x)
    tan_pred2, _, _ = pred._guided_tangent(model, I_t, other_x - z, t, y, other_x)
    assert torch.allclose(tan_pred, tan_pred2, atol=1e-5)
    assert float(aux_pred["guide_vnoise_rms"]) > 0


def test_endpoint_late_weight():
    """loss.endpoint.late_weight: rows at omega' > t0 weighted ((1 - t0)/(1 - omega'))^power up to cap, 1 below; off = the recipe."""
    torch.manual_seed(0)
    model = tiny()
    x, y = torch.randn(4, 4, 8, 8), torch.randint(10, (4,))
    stock = CBMLoss(loss_cfg(anchor_alpha=0.65), label_dim=10)
    late = CBMLoss(loss_cfg(anchor_alpha=0.65, endpoint={"late_weight": {"t0": 0.6, "power": 2.0, "cap": 64.0}}), label_dim=10)
    assert stock._late_weight(torch.rand(4)) is None
    op = torch.tensor([0.4, 0.6, 0.8, 0.99])
    w = late._late_weight(op)
    assert torch.allclose(w, torch.tensor([1.0, 1.0, 4.0, 64.0]))
    xl = CBMLoss(loss_cfg(anchor_alpha=0.65, endpoint={"late_weight": {"t0": 0.6, "power": 1.0, "cap": 8.0}}), label_dim=10)
    assert torch.allclose(xl._late_weight(op), torch.tensor([1.0, 1.0, 2.0, 8.0]))   # the XL line's {0.6, 1, 8}
    torch.manual_seed(1)
    l0, _ = stock(model, x, y, 0)
    torch.manual_seed(1)
    l1, aux = late(model, x, y, 0)
    assert torch.equal(l0["transport"], l1["transport"])
    assert float(aux["endpoint_late_weight_mean"]) >= 1.0
