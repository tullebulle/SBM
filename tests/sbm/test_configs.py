"""Every experiment config composes, names its run, resolves its store paths and asks for no missing input; the recipe
files carry the paper maps' settings; the released exports (when present in the store) load into the model (strict)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sbm import cli
from sbm import run as R
from sbm.models import build_model, load_weights

EXPS = sorted(p.stem for p in (Path(cli.CONFIGS) / "exp").glob("*.yaml") if not p.stem.startswith("_"))


def test_exp_list():
    assert EXPS == ["b2_btm", "b2_sbm", "xl_finetune", "xl_guided", "xl_sbm"]


@pytest.mark.parametrize("exp", EXPS)
def test_compose(exp):
    cfg = cli.compose_exp(exp, [])
    assert cfg.name == exp and cfg.exp_name == exp
    assert not OmegaConf.missing_keys(cfg)
    assert cfg.optim.name == "muon" and cfg.train.global_batch == 256
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["data"]["latent_pack_dir"] == f"{R.store()}/data/imagenet_latents_packed_adm"
    assert "${" not in str(resolved)                      # every interpolation resolves (the store resolver included)
    text = OmegaConf.to_yaml(cfg)
    assert ": /" not in text and "\"/" not in text   # no absolute paths in the configs themselves: the store is ${store:}


def test_store_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("SBM_STORE", str(tmp_path))
    assert R.store() == tmp_path
    assert OmegaConf.create({"p": "${store:}/x"}).p == f"{tmp_path}/x"
    monkeypatch.delenv("SBM_STORE")
    assert R.store() == R.REPO / "store"


def test_b2_sbm_recipe():
    """exp/b2_sbm.yaml: the SBM-B/2 recipe (from scratch, 780k steps with the cosine cool-down from 600k)."""
    c = cli.compose_exp("b2_sbm", [])
    L, O, T = c.loss, c.optim, c.train
    assert (L.label_dropout, L.anchor_alpha, L.time.mix_frac, L.time.mix_lo, L.balance.mode) == (0.1, 0.65, 0.5, 0.3, "ema")
    assert (L.time.p_mean, L.time.p_mean_end, L.time.anneal_start, L.time.anneal_steps) == (0.8, 0.0, 25000, 1)   # from b2_btm
    assert L.endpoint.late_weight is None and L.transport.weight == 1.0 and L.endpoint.weight == 1.0
    assert (L.guide.enabled, L.guide.w, L.guide.kappa, L.guide.endpoint, L.guide.diag_frac, L.guide.start_step) == \
        (True, 1.0, 0.8, True, 0.5, 35000)
    assert (L.guide.teacher_decay, L.guide.teacher_refresh) == (1.0, 10000) and 1.0 in T.ema.decays
    assert (O.lr, O.schedule, O.stage_step, O.decay_start_step, O.decay_end_step, O.min_factor) == \
        (1e-3, "staged", 1000000000, 600000, 780000, 0.0)
    assert (T.max_steps, T.global_batch, T.per_device_batch, T.checkpoint_every, T.val_every) == (780000, 256, 64, 20000, 0)
    assert T.sync_free and T.ema.foreach and list(T.ema.decays) == [0.999, 0.9999, 1.0] and T.ema.active == 0.999
    assert (T.monitor.every, T.monitor.n_per_rank, list(T.monitor.label_scales), T.monitor.omega) == (5000, 2500, [1.0, 1.5], 0.8)
    assert c.model.name == "DiT-B/2" and c.init["from"] is None and c.seed == 42


def test_xl_sbm_recipe():
    """exp/xl_sbm.yaml: the SBM-XL/2 recipe (the BTM base continued with self-guidance to 355k steps)."""
    c = cli.compose_exp("xl_sbm", [])
    L, O, T = c.loss, c.optim, c.train
    assert c.model.name == "DiT-XL/2" and c.model.tangent_only_jvp and not c.init.resume_state and c.init.weights == "ema"
    assert OmegaConf.to_container(c, resolve=True)["init"]["from"] == f"{R.store()}/refs/bases/dit_xl2_1p55M_ema9999.pt"
    assert (L.time.p_mean, L.time.mix_frac, L.time.mix_lo, L.anchor_alpha, L.label_dropout) == (-1.2, 0.5, 0.3, 0.65, 0.1)
    assert (L.guide.enabled, L.guide.w, L.guide.kappa, L.guide.endpoint, L.guide.diag_frac, L.guide.start_step) == \
        (True, 1.0, 0.7, True, 0.5, 2000)
    assert (L.guide.teacher_decay, L.guide.teacher_refresh) == (1.0, 10000)
    assert dict(L.endpoint.late_weight) == {"t0": 0.6, "power": 1.0, "cap": 8.0}
    assert (L.balance.mode, L.transport.weight, L.endpoint.weight) == ("none", 2000, 1)
    assert (L.transport.p, L.endpoint.p) == (1.0, 0.0)
    assert (O.name, O.lr, O.schedule) == ("muon", 5e-4, "constant")
    assert (T.max_steps, T.global_batch, T.per_device_batch, T.checkpoint_every, T.val_every) == (355000, 256, 32, 5000, 0)
    assert list(T.ema.decays) == [0.999, 0.9999, 1.0] and T.ema.foreach and T.sync_free
    assert (T.monitor.omega, list(T.monitor.label_scales), T.monitor.n_per_rank) == (0.95, [1.0, 1.1, 1.2], 2500)


def test_override_and_frozen_yaml_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SBM_STORE", str(tmp_path))
    cfg = cli.compose_exp("b2_sbm", ["name=x", "loss.guide.kappa=0.6", "train.max_steps=10"])
    assert cfg.name == "x" and cfg.loss.guide.kappa == 0.6 and cfg.train.max_steps == 10
    frozen = tmp_path / "run.yaml"
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), frozen)
    again = cli.compose_exp(str(frozen), ["train.max_steps=20"])
    assert again.train.max_steps == 20 and again.data.latent_pack_dir == f"{tmp_path}/data/imagenet_latents_packed_adm"


def test_config_command_prints_resolved(capsys):
    cli.main(["config", "b2_sbm"])
    out = capsys.readouterr().out
    assert "kappa: 0.8" in out and "${" not in out and f"{R.store()}/data/imagenet_latents_packed_adm" in out


def test_xl_finetune_recipe():
    """exp/xl_finetune.yaml: the XL recipe on the released export's weights, a new run with the guide on from step 0."""
    c = cli.compose_exp("xl_finetune", [])
    xl = cli.compose_exp("xl_sbm", [])
    assert OmegaConf.to_container(c, resolve=True)["init"]["from"] == f"{R.store()}/ckpt/imagenet/sbm_sdvae_xl_step355000_ema0_9999.pt"
    assert not c.init.resume_state and c.loss.guide.start_step == 0
    for key in ("model", "optim", "data"):
        assert c[key] == xl[key]
    assert {k: v for k, v in c.loss.items() if k != "guide"} == {k: v for k, v in xl.loss.items() if k != "guide"}


@pytest.mark.parametrize("name", ["sbm_sdvae_b_step780000_ema0_999.pt", "sbm_sdvae_xl_step355000_ema0_9999.pt"])
def test_exports_load(name):
    from sbm.infer import load_export
    f = R.store() / "ckpt" / "imagenet" / name
    if not f.exists():
        pytest.skip(f"{f} not present")
    model, cfg, meta = load_export(f, device="cpu")
    assert meta["read"]["protocol"] in ("b2_adm50k", "xl_val50k") and cfg.model.name == model_name(name)
    assert torch.isfinite(next(model.parameters())).all()
    # the export also loads where a training checkpoint does (init.from / sbm eval)
    m = load_weights(build_model(**OmegaConf.to_container(cfg.model)), f)
    sd = torch.load(f, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    assert all(torch.equal(v, sd["model." + k]) for k, v in m.state_dict().items())


def model_name(file_name):
    return "DiT-XL/2" if "_xl_" in file_name else "DiT-B/2"
