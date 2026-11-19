"""The engineering switches: every default leaves the computation unchanged, the exact switches (tangent-only JVP, foreach
EMA, sync-free scalars) are bit-identical to it, and the read-outs (FID monitor, eval prefixes, step timer) agree with the
evaluator. CPU, fp32 (and CPU bf16 autocast where a switch touches the JVP), tiny DiT."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))
from helpers import VAEStub  # noqa: E402

from sbm import cli  # noqa: E402
from sbm.ema import EMA  # noqa: E402
from sbm.loss import CBMLoss  # noqa: E402
from sbm.models import TangentOnlyConv2d, TangentOnlyLinear, build_model  # noqa: E402
from sbm.monitor import FIDMonitor, StepTimer, fid_inf  # noqa: E402

DEFAULTS = OmegaConf.load(HERE.parents[1] / "configs" / "sbm" / "train.yaml")
GUIDED = {"label_dropout": 0.1, "anchor_alpha": 0.65, "time": {"mix_frac": 0.5}, "balance": {"mode": "ema"},
          "guide": {"enabled": True, "kappa": 0.8, "endpoint": True, "diag_frac": 0.5, "start_step": 0}}   # the SBM recipe's loss


def tiny(input_size=8, label_dim=10, **kw):
    torch.manual_seed(5)
    m = build_model("DiT-tiny", input_size=input_size, label_dim=label_dim, **kw)
    with torch.no_grad():                       # the zero-initialised layers perturbed, so every gradient is non-trivial
        g = torch.Generator().manual_seed(7)
        for _, p in sorted(m.named_parameters()):
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return m


def pair():
    """(stock, tangent-only) tiny DiTs with the same weights."""
    a = tiny()
    b = tiny(tangent_only_jvp=True)
    b.load_state_dict(a.state_dict())
    return a, b


def loss_cfg(**over):
    return OmegaConf.merge(DEFAULTS.loss, OmegaConf.create(over))


def data(B=8, size=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 4, size, size, generator=g), torch.arange(B) % 10


def step(model, loss, x, y, seed=1234, s=0):
    torch.manual_seed(seed)
    losses, aux = loss(model, x, y, s)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(sum(losses.values()), params)
    return losses, aux, grads, float(torch.rand(()))


# ---------------------------------------------------------------------- defaults
def test_defaults_are_off():
    d = DEFAULTS
    assert d.model.tangent_only_jvp is False and d.model.gw_embedder is False
    assert d.train.ema.foreach is False and d.train.sync_free is False
    assert d.train.step_timer == 0 and d.train.monitor.every == 0
    assert d.loss.guide.enabled is False and d.loss.balance.mode == "match_smaller_to_larger" and d.loss.time.mix_frac == 0.0
    assert d.loss.endpoint.late_weight is None and d.loss.transport.weight == 1.0 and d.loss.label_dropout == 0.0
    assert not any(isinstance(m, (TangentOnlyLinear, TangentOnlyConv2d)) for m in build_model("DiT-tiny", input_size=8,
                                                                                               label_dim=10).modules())


# ---------------------------------------------------------------------- tangent-only JVP
def test_tangent_only_layers_and_state_dict():
    a, b = pair()
    assert list(a.state_dict()) == list(b.state_dict())
    kinds = [type(m) for m in b.modules()]
    assert kinds.count(TangentOnlyConv2d) == 1 and kinds.count(TangentOnlyLinear) == 4 * len(b.blocks) + 1
    assert type(b.blocks[0].adaLN_modulation[-1]) is torch.nn.Linear      # conditioning path: no tangent, untouched


@pytest.mark.parametrize("autocast", [False, True])
def test_tangent_only_jvp_is_bit_identical(autocast):
    a, b = pair()
    x, y = data()
    v = torch.randn_like(x)
    omega = torch.full((len(y),), 0.8)
    out = {}
    for name, m in (("stock", a), ("fast", b)):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            with FlopCounterMode(display=False) as fc:
                pred, jvp = torch.func.jvp(lambda i: m(i, y, omega=omega), (x,), (v,))
            loss = (pred.float() - (pred.detach() + jvp.detach()).float()).pow(2).sum() + pred.float().pow(2).mean()
        grads = torch.autograd.grad(loss, [p for p in m.parameters() if p.requires_grad])
        out[name] = (pred, jvp, grads, fc.get_total_flops())
    (p0, j0, g0, f0), (p1, j1, g1, f1) = out["stock"], out["fast"]
    assert torch.equal(p0, p1) and torch.equal(j0, j1)
    assert all(torch.equal(u, w) for u, w in zip(g0, g1))
    assert f1 < f0                                                        # the zero-tangent GEMMs are gone


@pytest.mark.parametrize("over", [{}, GUIDED])
def test_tangent_only_recipes_are_bit_identical(over):
    a, b = pair()
    x, y = data()
    la, _, ga, ra = step(a, CBMLoss(loss_cfg(**over), label_dim=10), x, y)
    lb, _, gb, rb = step(b, CBMLoss(loss_cfg(**over), label_dim=10), x, y)
    assert {k: float(v) for k, v in la.items()} == {k: float(v) for k, v in lb.items()}
    assert all(torch.equal(u, w) for u, w in zip(ga, gb)) and ra == rb


# ---------------------------------------------------------------------- foreach EMA, sync-free scalars
class _Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = tiny()
        self.register_buffer("latents_scale", torch.full((1, 4, 1, 1), 0.18215))


def test_foreach_ema_is_bit_identical():
    mods = {False: _Lit(), True: _Lit()}
    mods[True].load_state_dict(mods[False].state_dict())
    emas = {f: EMA((0.99, 0.999, 0.9999), 0.999, min_step=0, foreach=f) for f in (False, True)}
    for f in (False, True):
        emas[f].setup(None, mods[f], "fit")
    g = torch.Generator().manual_seed(0)
    for s in range(1, 6):
        noise = [0.01 * torch.randn(p.shape, generator=g) for p in mods[False].parameters()]
        for f in (False, True):
            with torch.no_grad():
                for p, n in zip(mods[f].parameters(), noise):
                    p.add_(n)
            emas[f].on_train_batch_end(SimpleNamespace(global_step=s), mods[f], None, None, 0)
    for label in emas[False]._models:
        sa, sb = emas[False]._models[label].state_dict(), emas[True]._models[label].state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), label


def test_frozen_copy_label():
    """A decay of 1.0 (the guide's frozen teacher) is tracked under the label '1' and never moves on its own."""
    ema = EMA((0.999, 1.0), 0.999, min_step=0)
    assert set(ema._models) == set() and "1" in ema._decay_of and ema._decay_of["1"] == 1.0
    mod = _Lit()
    ema.setup(None, mod, "fit")
    before = {k: v.clone() for k, v in ema._models["1"].module.state_dict().items()}
    ema.on_train_batch_end(SimpleNamespace(global_step=1), mod, None, None, 0)    # first update: copies the current weights
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(1.0)
    ema.on_train_batch_end(SimpleNamespace(global_step=2), mod, None, None, 0)
    after = ema._models["1"].module.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("model."))


@pytest.mark.parametrize("over", [{}, GUIDED])
def test_sync_free_scalars_match(over):
    x, y = data()
    m = tiny()
    la, xa, ga, ra = step(m, CBMLoss(loss_cfg(**over), label_dim=10), x, y)
    lb, xb, gb, rb = step(m, CBMLoss(loss_cfg(**over), label_dim=10, sync_free=True), x, y)
    assert {k: float(v) for k, v in la.items()} == {k: float(v) for k, v in lb.items()}
    assert set(xa) == set(xb) and all(torch.equal(xa[k].float(), xb[k].float()) for k in xa)
    assert all(torch.equal(u, w) for u, w in zip(ga, gb)) and ra == rb


# ---------------------------------------------------------------------- monitor, eval prefixes, timer
class _Features(torch.nn.Module):
    """(B, 3, H, W) images -> 16-d features, float64 (stands in for Inception)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(21)
        self.w = torch.randn(3 * 16, 16, generator=g)

    def forward(self, imgs):
        return (torch.nn.functional.adaptive_avg_pool2d(imgs, 4).flatten(1) @ self.w).tanh().double()


def _ref_stats():
    g = torch.Generator().manual_seed(22)
    a = torch.randn(16, 16, generator=g, dtype=torch.float64)
    return {"val": (torch.randn(16, generator=g, dtype=torch.float64), a @ a.T / 16 + 0.1 * torch.eye(16, dtype=torch.float64))}


def test_fid_inf():
    f, k = fid_inf(2000, 30.0 + 3e4 / 2000, 8000, 30.0 + 3e4 / 8000)
    assert f == pytest.approx(30.0) and k == pytest.approx(3e4)


def _stub_eval(monkeypatch):
    import torchmetrics.image.fid as tfid

    from sbm import eval as E
    feats = _Features()

    class Metric:
        def __init__(self, *a, **k):
            self.fake_features_sum = torch.zeros(16, dtype=torch.float64)
            self.fake_features_cov_sum = torch.zeros(16, 16, dtype=torch.float64)
            self.fake_features_num_samples = torch.tensor(0)

        def to(self, device):
            return self

        def update(self, imgs, real):
            f = feats(imgs)
            self.fake_features_sum += f.sum(0)
            self.fake_features_cov_sum += f.t().mm(f)
            self.fake_features_num_samples += imgs.shape[0]

    monkeypatch.setattr(tfid, "FrechetInceptionDistance", Metric)
    monkeypatch.setattr(E, "_vae", lambda device: VAEStub())
    monkeypatch.setattr(E, "_ensure_inception_weights", lambda: None)
    monkeypatch.setattr(E, "_ref", lambda name, device: _ref_stats()["val"])
    return feats


def test_eval_prefixes_equal_separate_reads(monkeypatch):
    from sbm import eval as E
    _stub_eval(monkeypatch)
    model = tiny(label_dim=1000).eval()          # the protocol's labels are arange(n) % 1000
    p = {**E.PROTOCOLS["b2_adm50k"], "refs": ["val"], "n": 750, "prefixes": [250, 500]}
    out, _ = E._run_b2(model, p, torch.device("cpu"), 0.18215)
    for k in (250, 500):
        alone, _ = E._run_b2(model, {**p, "n": k, "prefixes": []}, torch.device("cpu"), 0.18215)
        assert out["prefix_fid_val"]["1"][str(k)] == alone["fid_val"]["1"]
    assert list(out)[0] == "fid_val"


def test_eval_label_scale_moves_the_read(monkeypatch):
    """label_scale != 1 reads a copy with the class rows extrapolated from the null row; the model itself is untouched."""
    from sbm import eval as E
    _stub_eval(monkeypatch)
    model = tiny(label_dim=1000).eval()
    table = model.y_embedder.embedding_table.weight.detach().clone()
    p = {**E.PROTOCOLS["b2_adm50k"], "refs": ["val"], "n": 250}
    plain, _ = E._run_b2(model, p, torch.device("cpu"), 0.18215)
    scaled, _ = E._run_b2(model, {**p, "label_scale": 1.5}, torch.device("cpu"), 0.18215)
    assert scaled["fid_val"]["1"] != plain["fid_val"]["1"]
    assert torch.equal(model.y_embedder.embedding_table.weight, table)


def test_monitor_rank0_read_equals_eval(monkeypatch, tmp_path):
    from sbm import eval as E
    feats = _stub_eval(monkeypatch)
    model = tiny(label_dim=1000).eval()
    mon = FIDMonitor(tmp_path, every=2, n_per_rank=500, refs=("val",),
                     nets=(VAEStub(), feats), ref_stats=_ref_stats())
    module = SimpleNamespace(model=model, device=torch.device("cpu"), latents_scale=torch.full((1, 4, 1, 1), 0.18215))
    trainer = SimpleNamespace(global_step=2, global_rank=0, world_size=1, callbacks=[], logger=None)
    mon.on_train_batch_end(trainer, module, None, None, 0)
    mon.on_train_batch_end(trainer, module, None, None, 0)          # same step: no second read
    rows = [json.loads(line) for line in (tmp_path / "monitor" / "fid_monitor.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 2
    want, _ = E._run_b2(model, {**E.PROTOCOLS["b2_adm50k"], "refs": ["val"], "n": 500}, torch.device("cpu"), 0.18215)
    assert rows[0]["raw_fid_val_n500"] == want["fid_val"]["1"]


def test_step_timer(tmp_path):
    timer = StepTimer(tmp_path, every=2, images_per_step=64)
    for s in range(1, 6):
        timer.on_train_batch_end(SimpleNamespace(global_step=s, world_size=1, is_global_zero=True, logger=None),
                                 None, None, None, 0)
    rows = [json.loads(line) for line in (tmp_path / "monitor" / "perf.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 5] and all(r["sec_per_step"] > 0 for r in rows)


# ---------------------------------------------------------------------- end to end
def test_pretrain_run_with_every_switch(tmp_path, monkeypatch):
    """`sbm train` with every engineering switch on (the monitor's networks stubbed), then resumed: the loss state rides
    in the checkpoint, the monitor and timer write their files."""
    monkeypatch.setenv("SBM_STORE", str(tmp_path / "store"))
    feats = _Features()

    def build(self, device):
        self._nets, self._ref_stats = (VAEStub(), feats), _ref_stats()
    monkeypatch.setattr(FIDMonitor, "_build", build)
    pack = tmp_path / "packed" / "train"
    pack.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(pack / "part_000.npy", rng.standard_normal((32, 2, 8, 8, 8)).astype(np.float32))
    (pack / "part_000.json").write_text(json.dumps({"labels": [i % 10 for i in range(32)], "files": [f"{i}.pt" for i in range(32)]}))
    over = {"model.name": "DiT-tiny", "model.input_size": 8, "model.label_dim": 1000, "wandb.mode": "disabled",
            "train.stage_checkpoints_locally": False, "data.num_workers": 0, "name": "tiny-switches",
            "data.latent_pack_dir": str(tmp_path / "packed"), "train.precision": "32-true", "train.global_batch": 8,
            "train.per_device_batch": 8, "train.max_steps": 4, "train.checkpoint_every": 2, "train.val_every": "null",
            "train.ema.min_step": 0, "model.tangent_only_jvp": True, "train.ema.foreach": True, "train.sync_free": True,
            "train.step_timer": 1, "train.monitor.every": 2, "train.monitor.n_per_rank": 250, "train.monitor.refs": "[val]",
            "train.monitor.label_scales": "[1.0,1.5]", "loss.balance.mode": "ema"}
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.from_dotlist([f"{k}={v}" for k, v in over.items()]))
    path = tmp_path / "exp.yaml"
    OmegaConf.save(cfg, path)
    cli.main(["train", str(path)])
    run_dir = next((tmp_path / "store" / "runs").iterdir())
    ck = torch.load(run_dir / "ckpt" / "step0000004.ckpt", map_location="cpu", weights_only=False)
    assert set(ck["cbm_loss_state"]) == {"balance_norms"} and all(torch.isfinite(n) for n in ck["cbm_loss_state"]["balance_norms"])
    rows = [json.loads(line) for line in (run_dir / "monitor" / "fid_monitor.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4] and all(np.isfinite(r["ema0_999_fid_val_n250"]) for r in rows)
    assert all("ema0_999_ls1_5_fid_val_n250" in r for r in rows)
    assert (run_dir / "monitor" / "perf.jsonl").exists()
    frozen = OmegaConf.load(run_dir / "run.yaml")
    frozen.train.max_steps = 6
    OmegaConf.save(frozen, run_dir / "run.yaml")
    cli.main(["train", str(run_dir)])
    assert (run_dir / "ckpt" / "step0000006.ckpt").exists()


def test_monitor_long_run_readouts(monkeypatch, tmp_path):
    """label_scales, epoch and schedule: the scale-1 read is the recipe's, the table is restored, the gap is logged."""
    feats = _stub_eval(monkeypatch)
    model = tiny(label_dim=1000).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))          # a non-zero map (the tiny DiT's output layer starts at zero)
    table = model.y_embedder.embedding_table.weight.detach().clone()
    kw = dict(every=2, n_per_rank=500, refs=("val",), nets=(VAEStub(), feats), ref_stats=_ref_stats())
    module = SimpleNamespace(model=model, device=torch.device("cpu"), latents_scale=torch.full((1, 4, 1, 1), 0.18215))
    trainer = SimpleNamespace(global_step=2, global_rank=0, world_size=1, callbacks=[], logger=None)
    plain = FIDMonitor(tmp_path / "a", **kw).read(trainer, module, 2)
    mon = FIDMonitor(tmp_path / "b", label_scales=[1.0, 1.5], steps_per_epoch=4,
                     schedule={"metric": "raw_ls1_5_fid_val_pooled", "points": [[0, 10.0], [4, 6.0]]}, **kw)
    row = mon.read(trainer, module, 2)
    assert row["raw_fid_val_n500"] == plain["raw_fid_val_n500"]
    assert row["raw_ls1_5_fid_val_n500"] != row["raw_fid_val_n500"]
    assert torch.equal(model.y_embedder.embedding_table.weight, table)
    assert row["epoch"] == 0.5 and row["schedule"] == 8.0
    assert row["gap_to_schedule"] == pytest.approx(row["raw_ls1_5_fid_val_n500"] - 8.0)
    assert row["raw_trace_ratio_val"] > 0 and row["raw_mean_val"] >= 0


def test_monitor_guidance_reads(monkeypatch, tmp_path):
    """guidance_reads: a map with a guidance input is also read at [w, s_lo, s_hi]; the plain read is unchanged."""
    feats = _stub_eval(monkeypatch)
    model = tiny(label_dim=1000, gw_embedder=True, gw_input="kappa", gw_interval=True).eval()
    kw = dict(every=2, n_per_rank=500, refs=("val",), nets=(VAEStub(), feats), ref_stats=_ref_stats())
    module = SimpleNamespace(model=model, device=torch.device("cpu"), latents_scale=torch.full((1, 4, 1, 1), 0.18215))
    trainer = SimpleNamespace(global_step=2, global_rank=0, world_size=1, callbacks=[], logger=None)
    plain = FIDMonitor(tmp_path / "a", **kw).read(trainer, module, 2)
    row = FIDMonitor(tmp_path / "b", guidance_reads=[[3.0, 0.35, 0.6]], **kw).read(trainer, module, 2)
    assert row["raw_fid_val_n500"] == plain["raw_fid_val_n500"]
    assert "raw_w3_fid_val_n500" in row and row["raw_w3_fid_val_n500"] != row["raw_fid_val_n500"]


def test_time_importance_weights():
    """loss.time.importance: the recipe law's density integrates to one for every anchor, the weights are bounded by 1 / eps
    with mean ~1 under the p_eps draw, the term carries them when on, and off is the default (no new draw: the goldens pin that path)."""
    assert DEFAULTS.loss.time.importance.enabled is False
    L = CBMLoss(loss_cfg(time={"p_mean": 0.0, "mix_frac": 0.5, "mix_lo": 0.3, "importance": {"enabled": True, "eps": 0.1}}, anchor_alpha=0.65), label_dim=10)
    tt = (torch.arange(200000) + 0.5) / 200000
    for om in (0.35, 0.8, 1.0):
        assert abs(float(L._time_density(tt, torch.full_like(tt, om)).mean()) - 1.0) < 2e-3, om
    torch.manual_seed(0)
    x = torch.zeros(200000, 1)
    t, omega = L._time(x), L._omega(x)
    late = torch.rand_like(t) < L.t_mix
    t = torch.where(late, L.t_mix_lo + (omega - L.t_mix_lo).clamp_min(0.0) * torch.rand_like(t), t)
    unif = torch.rand_like(t) < L.t_imp_eps
    t = torch.where(unif, torch.rand_like(t), t)
    w = L._importance_weight(t, omega)
    assert float(w.max()) <= 1.0 / L.t_imp_eps + 1e-6 and abs(float(w.mean()) - 1.0) < 0.02
    m, (xb, yb) = tiny(), data()
    g = dict(GUIDED); g["time"] = {"mix_frac": 0.5, "importance": {"enabled": True, "eps": 0.1}}
    torch.manual_seed(1); off, aux_off = CBMLoss(loss_cfg(**GUIDED), label_dim=10)(m, xb, yb, 0)
    torch.manual_seed(1); on, aux_on = CBMLoss(loss_cfg(**g), label_dim=10)(m, xb, yb, 0)
    assert torch.isfinite(on["transport"]) and float(on["transport"]) != float(off["transport"])
    assert "transport_importance_mean" in aux_on and "transport_importance_mean" not in aux_off
    with pytest.raises(ValueError):
        CBMLoss(loss_cfg(restrict_transport=True, time={"importance": {"enabled": True}}), label_dim=10)
