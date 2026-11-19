"""The guidance-strength input (model.gw_embedder): the zero-initialised embedder loads weights and whole training states
written without it (model, raw, every EMA copy, optimizer) with the function unchanged at every w, and the evaluator
reads at a given w. CPU, fp32, tiny DiT."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from test_switches import DEFAULTS, _stub_eval, data, tiny

from sbm import cli
from sbm.models import build_model, fill_gw_keys, load_weights
from sbm.optim import build_optimizer, extend_optimizer_state, split_params

GW_KEYS = ["gw_embedder.mlp.0.weight", "gw_embedder.mlp.0.bias", "gw_embedder.mlp.2.weight", "gw_embedder.mlp.2.bias"]


def tiny_pair(label_dim=10, perturb_gw=False):
    """(parent, the same weights with the guidance input). perturb_gw moves the embedder off zero (so w matters)."""
    base = tiny(label_dim=label_dim)
    g = build_model("DiT-tiny", input_size=8, label_dim=label_dim, gw_embedder=True)
    sd = dict(base.state_dict())
    assert fill_gw_keys(g.state_dict(), sd) == GW_KEYS
    g.load_state_dict(sd)
    if perturb_gw:
        gen = torch.Generator().manual_seed(9)
        with torch.no_grad():
            for _, p in sorted(g.gw_embedder.named_parameters()):
                p.add_(0.05 * torch.randn(p.shape, generator=gen))
    return base, g


# ---------------------------------------------------------------------- defaults, model, loading
def test_defaults_are_off():
    assert DEFAULTS.model.gw_embedder is False
    m = build_model("DiT-tiny", input_size=8, label_dim=10)
    assert m.gw_embedder is None and not any(k.startswith("gw_embedder") for k in m.state_dict())
    with pytest.raises(ValueError, match="no guidance-strength input"):
        m(*data(), omega=0.8, w=1.0)


def test_embedder_comes_last_and_moves_nothing():
    """Every other parameter keeps its init (same seed) and its place in the state dict and in each optimizer."""
    torch.manual_seed(0)
    a = build_model("DiT-tiny", input_size=8, label_dim=10)
    torch.manual_seed(0)
    b = build_model("DiT-tiny", input_size=8, label_dim=10, gw_embedder=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sb) == list(sa) + GW_KEYS and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert float(sb["gw_embedder.mlp.2.weight"].abs().sum() + sb["gw_embedder.mlp.2.bias"].abs().sum()) == 0.0
    for o in (DEFAULTS.optim, OmegaConf.merge(DEFAULTS.optim, {"name": "adamw"})):
        for ga, gb in zip(split_params(a, o), split_params(b, o), strict=True):
            names = [n for n, _ in gb]
            assert names[:len(ga)] == [n for n, _ in ga] and set(names[len(ga):]) <= set(GW_KEYS)


@pytest.mark.parametrize("w", [None, 0.0, 1.0, 3.7, "per_example"])
@pytest.mark.parametrize("autocast", [False, True])
def test_zero_init_is_the_parent_function(w, autocast):
    base, g = tiny_pair()
    x, y = data()
    v = torch.randn_like(x)
    omega = torch.full((len(y),), 0.8)
    ww = torch.linspace(0.0, 4.0, len(y)) if w == "per_example" else w
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        p0, j0 = torch.func.jvp(lambda i: base(i, y, omega=omega), (x,), (v,))
        p1, j1 = torch.func.jvp(lambda i: g(i, y, omega=omega, w=ww), (x,), (v,))
    assert torch.equal(p0, p1) and torch.equal(j0, j1)


def test_load_weights_fills_only_the_embedder(tmp_path):
    base, _ = tiny_pair()
    f = tmp_path / "old.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in base.state_dict().items()}}, f)
    g = load_weights(build_model("DiT-tiny", input_size=8, label_dim=10, gw_embedder=True), f)
    x, y = data()
    assert torch.equal(g(x, y, omega=0.8, w=2.5), base(x, y, omega=0.8))
    torch.save({"state_dict": {f"model.{k}": v for k, v in base.state_dict().items() if not k.startswith("y_embedder")}}, f)
    with pytest.raises(RuntimeError, match="y_embedder"):
        load_weights(build_model("DiT-tiny", input_size=8, label_dim=10, gw_embedder=True), f)


@pytest.mark.parametrize("name", ["muon", "adamw"])
def test_optimizer_state_is_extended(name):
    """A state saved without the embedder loads into the optimizer built with it: old parameters keep their momentum /
    moments exactly, the new ones start fresh, and the optimizer steps."""
    o = OmegaConf.merge(DEFAULTS.optim, {"name": name})
    base, g = tiny_pair()
    opt = build_optimizer(base, o)
    gen = torch.Generator().manual_seed(1)
    for p in base.parameters():
        p.grad = torch.randn(p.shape, generator=gen)
    opt.step()
    subs = opt.optimizers if name == "muon" else [opt]
    old = {n: copy.deepcopy(subs[i].state[p]) for i, grp in enumerate(split_params(base, o)) for n, p in grp}
    saved = extend_optimizer_state(copy.deepcopy(opt.state_dict()), split_params(g, o), GW_KEYS)
    opt2 = build_optimizer(g, o)
    opt2.load_state_dict(saved)
    subs2 = opt2.optimizers if name == "muon" else [opt2]
    for i, grp in enumerate(split_params(g, o)):
        for n, p in grp:
            st = subs2[i].state.get(p, {})
            if n in GW_KEYS:
                assert not st, n
            else:
                assert set(st) == set(old[n]) and all(torch.equal(torch.as_tensor(st[k]), torch.as_tensor(old[n][k]))
                                                      for k in st), n
    for p in g.parameters():
        p.grad = torch.randn(p.shape, generator=gen)
    opt2.step()
    with pytest.raises(ValueError, match="expected"):
        extend_optimizer_state(copy.deepcopy(opt.state_dict()), split_params(g, o), GW_KEYS[:2])


# ---------------------------------------------------------------------- evaluator
def test_eval_samples_at_w(monkeypatch):
    from sbm import eval as E
    _stub_eval(monkeypatch)
    base, g = tiny_pair(label_dim=1000, perturb_gw=True)
    g.eval()
    p = {**E.PROTOCOLS["b2_adm50k"], "refs": ["val"], "n": 250, "gw": 1.5}
    labels = torch.arange(8)
    gen = lambda: torch.Generator().manual_seed(5)
    got = E._sample_b2(g, labels, p, torch.device("cpu"), gen(), torch.tensor(0.8))[1]
    z = torch.randn((8, 4, 8, 8), generator=gen())
    assert torch.equal(got, g(z, labels, omega=0.8, w=1.5))
    f = {w: E._run_b2(g, {**p, "gw": w}, torch.device("cpu"), 0.18215)[0]["fid_val"]["1"] for w in (0.0, 1.5)}
    assert f[0.0] != f[1.5]
    _, g0 = tiny_pair(label_dim=1000)                                           # zero-init: every w reads as the parent
    assert E._run_b2(g0.eval(), {**p, "gw": 3.0}, torch.device("cpu"), 0.18215)[0] == \
        E._run_b2(base.eval(), {**p, "gw": None}, torch.device("cpu"), 0.18215)[0]
    assert E._resolve_gw(g0, {**p, "gw": None})["gw"] == 1.0
    with pytest.raises(ValueError, match="no guidance-strength input"):
        E._resolve_gw(base, dict(p))


# ---------------------------------------------------------------------- end to end: an old run's state -> a run with the input
def _exp(tmp_path, name, **over):
    pack = tmp_path / "packed" / "train"
    if not pack.exists():
        pack.mkdir(parents=True)
        rng = np.random.default_rng(0)
        np.save(pack / "part_000.npy", rng.standard_normal((32, 2, 8, 8, 8)).astype(np.float32))
        (pack / "part_000.json").write_text(json.dumps({"labels": [i % 10 for i in range(32)], "files": [f"{i}.pt" for i in range(32)]}))
    base = {"model.name": "DiT-tiny", "model.input_size": 8, "model.label_dim": 1000, "wandb.mode": "disabled",
            "train.stage_checkpoints_locally": False, "data.num_workers": 0, "name": name,
            "data.latent_pack_dir": str(tmp_path / "packed"), "train.precision": "32-true", "train.global_batch": 8,
            "train.per_device_batch": 8, "train.checkpoint_every": 2, "train.val_every": "null", "train.ema.min_step": 0}
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.from_dotlist([f"{k}={v}" for k, v in {**base, **over}.items()]))
    path = tmp_path / f"{name}.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def test_old_run_resumes_with_the_input_and_evaluates_at_w(tmp_path, monkeypatch):
    from sbm.train import Module
    monkeypatch.setenv("SBM_STORE", str(tmp_path / "store"))
    cli.main(["train", _exp(tmp_path, "parent", **{"train.max_steps": 2})])
    runs = tmp_path / "store" / "runs"
    ck = next(runs.glob("*_parent")) / "ckpt" / "step0000002.ckpt"
    # the patch alone: every weight copy of the parent computes what it did, at any w; the optimizer state loads
    cfg = OmegaConf.merge(OmegaConf.load(_exp(tmp_path, "probe")), OmegaConf.from_dotlist(["model.gw_embedder=true"]))
    mod = Module(cfg, build_model(**cfg.model))
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    old = copy.deepcopy(raw)
    mod.on_load_checkpoint(raw)
    x, y = data()
    for key in ("state_dict", "current_model_state", "ema:0_99", "ema:0_9999"):
        get = (lambda c: c["ema_state_dicts"][key[4:]]) if key.startswith("ema:") else (lambda c: c[key])
        strip = lambda sd: {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        new_m = build_model(**cfg.model)
        new_m.load_state_dict(strip(get(raw)), strict=True)
        old_m = build_model(**{**cfg.model, "gw_embedder": False})
        old_m.load_state_dict(strip(get(old)), strict=True)
        assert torch.equal(new_m(x, y, omega=0.8, w=2.5), old_m(x, y, omega=0.8)), key
    build_optimizer(mod.model, cfg.optim).load_state_dict(raw["optimizer_states"][0])
    # the resumed run trains
    cli.main(["train", _exp(tmp_path, "child", **{"train.max_steps": 4, "init.from": str(ck), "init.resume_state": True,
                                                  "model.gw_embedder": True})])
    child = next(runs.glob("*_child"))
    c4 = torch.load(child / "ckpt" / "step0000004.ckpt", map_location="cpu", weights_only=False)
    assert c4["global_step"] == 4
    for sd in (c4["state_dict"], c4["current_model_state"], *c4["ema_state_dicts"].values()):
        assert all(f"model.{k}" in sd for k in GW_KEYS)
    # and is read at a chosen w through the CLI
    _stub_eval(monkeypatch)
    cli.main(["eval", str(child), "ckpt=step0000004.ckpt", "n=250", "gw=1.5"])
    rec = json.loads((child / "evals" / "b2_adm50k_step0000004_gw1.5_n250.json").read_text())
    assert rec["settings"]["gw"] == 1.5 and np.isfinite(rec["fid"])
