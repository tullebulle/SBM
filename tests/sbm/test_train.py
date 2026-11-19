"""`sbm train` end to end on the CPU with a tiny DiT: the run folder (run.yaml with the commit, ckpt/, logs/, evals/,
wandb/), checkpoints on schedule, the EMA copies in the checkpoint, and resuming a run folder where it stopped."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sbm import cli
from sbm import run as R

REPO = Path(__file__).resolve().parents[2]
TINY = {"model.name": "DiT-tiny", "model.input_size": 8, "model.label_dim": 10, "wandb.mode": "disabled",
        "train.stage_checkpoints_locally": False, "data.num_workers": 0}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SBM_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def frozen(tmp_path, **over):
    cfg = OmegaConf.load(REPO / "configs" / "sbm" / "train.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in {**TINY, **over}.items()]))
    path = tmp_path / "exp.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def only_run(store):
    runs = list((store / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


def steps_in(run_dir):
    return sorted(int(p.stem[4:]) for p in (run_dir / "ckpt").glob("step*.ckpt"))


def fake_pack(tmp_path, n=32, classes=10):
    pack = tmp_path / "packed" / "train"
    pack.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(pack / "part_000.npy", rng.standard_normal((n, 2, 8, 8, 8)).astype(np.float32))
    (pack / "part_000.json").write_text(json.dumps({"labels": [i % classes for i in range(n)], "files": [f"{i}.pt" for i in range(n)]}))
    return tmp_path / "packed"


def test_pretrain_run_folder_ema_and_resume(tmp_path, store):
    exp = frozen(tmp_path, **{"name": "tiny-pre", "data.latent_pack_dir": str(fake_pack(tmp_path)), "train.precision": "32-true",
                              "train.global_batch": 8, "train.per_device_batch": 8, "train.max_steps": 4,
                              "train.checkpoint_every": 2, "train.ema.min_step": 0})
    cli.main(["train", exp])
    run_dir = only_run(store)
    assert run_dir.name.endswith("_tiny-pre")
    for sub in ("ckpt", "logs", "evals", "wandb"):
        assert (run_dir / sub).is_dir()
    cfg = OmegaConf.load(run_dir / "run.yaml")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert cfg.run.git.commit == head and cfg.run.env.torch == torch.__version__
    assert steps_in(run_dir) == [2, 4]
    ck = torch.load(run_dir / "ckpt" / "step0000004.ckpt", map_location="cpu", weights_only=False)
    assert ck["global_step"] == 4
    assert ck["ema_active_decay"] == 0.999 and set(ck["ema_state_dicts"]) == {"0_99", "0_9999"}
    assert not torch.equal(ck["state_dict"]["model.final_layer.linear.weight"],
                           ck["current_model_state"]["model.final_layer.linear.weight"])   # EMA differs from raw
    assert "cbm_loss_state" not in ck   # the per-step balancer carries no state
    from sbm.models import build_model, load_weights
    load_weights(build_model("DiT-tiny", input_size=8, label_dim=10), run_dir / "ckpt" / "step0000004.ckpt", "raw")
    # resume: extend the frozen config and train the same folder again
    cfg.train.max_steps = 6
    OmegaConf.save(cfg, run_dir / "run.yaml")
    cli.main(["train", str(run_dir)])
    assert steps_in(run_dir) == [2, 4, 6]


def test_sbm_recipe_trains_and_forks(tmp_path, store):
    """The b2_sbm recipe's switches on a tiny model (guide from step 2 with the frozen teacher, EMA balancer, time mixture),
    then a fork with init.resume_state: the loss state (the balancer's EMA norms) rides in the checkpoint and is restored."""
    pack = fake_pack(tmp_path)
    over = {"name": "tiny-sbm", "data.latent_pack_dir": str(pack), "train.precision": "32-true", "train.global_batch": 8,
            "train.per_device_batch": 8, "train.max_steps": 4, "train.checkpoint_every": 2, "train.ema.min_step": 0,
            "train.ema.decays": "[0.999,1.0]", "train.ema.foreach": True, "train.sync_free": True,
            "loss.label_dropout": 0.1, "loss.anchor_alpha": 0.65, "loss.time.mix_frac": 0.5, "loss.balance.mode": "ema",
            "loss.guide.enabled": True, "loss.guide.kappa": 0.8, "loss.guide.endpoint": True, "loss.guide.diag_frac": 0.5,
            "loss.guide.start_step": 2, "loss.guide.teacher_decay": 1.0, "loss.guide.teacher_refresh": 2}
    cli.main(["train", frozen(tmp_path, **over)])
    parent = only_run(store)
    ck = torch.load(parent / "ckpt" / "step0000004.ckpt", map_location="cpu", weights_only=False)
    assert set(ck["cbm_loss_state"]) == {"balance_norms"} and set(ck["ema_state_dicts"]) == {"1"}
    assert all(torch.isfinite(v).all() for v in ck["state_dict"].values())
    child = frozen(tmp_path, **{**over, "name": "tiny-fork", "train.max_steps": 6, "init.from": str(parent / "ckpt" / "step0000004.ckpt"),
                                 "init.resume_state": True})
    cli.main(["train", child])
    fork = next(p for p in (store / "runs").iterdir() if p.name.endswith("_tiny-fork"))
    assert steps_in(fork) == [6]
    c6 = torch.load(fork / "ckpt" / "step0000006.ckpt", map_location="cpu", weights_only=False)
    assert c6["global_step"] == 6 and "balance_norms" in c6["cbm_loss_state"]


def test_weights_only_init(tmp_path, store):
    """init.from with resume_state false: the weights are taken, the step counter and optimizer start fresh."""
    pack = fake_pack(tmp_path)
    base = {"data.latent_pack_dir": str(pack), "train.precision": "32-true", "train.global_batch": 8, "train.per_device_batch": 8,
            "train.checkpoint_every": 2, "train.ema.min_step": 0}
    cli.main(["train", frozen(tmp_path, **{**base, "name": "tiny-base", "train.max_steps": 2})])
    parent = only_run(store)
    cli.main(["train", frozen(tmp_path, **{**base, "name": "tiny-init", "train.max_steps": 2, "init.from": str(parent)})])
    child = next(p for p in (store / "runs").iterdir() if p.name.endswith("_tiny-init"))
    c = torch.load(child / "ckpt" / "step0000002.ckpt", map_location="cpu", weights_only=False)
    p = torch.load(parent / "ckpt" / "step0000002.ckpt", map_location="cpu", weights_only=False)
    assert c["global_step"] == 2 and not torch.equal(c["state_dict"]["model.final_layer.linear.weight"],
                                                     p["state_dict"]["model.final_layer.linear.weight"])


def test_missing_data_is_a_clear_error(tmp_path, store):
    exp = frozen(tmp_path, **{"name": "tiny-nodata", "data.latent_pack_dir": str(tmp_path / "nowhere"), "train.precision": "32-true",
                              "train.global_batch": 8, "train.per_device_batch": 8, "train.max_steps": 1})
    with pytest.raises(FileNotFoundError, match="latent_pack_dir"):
        cli.main(["train", exp])
