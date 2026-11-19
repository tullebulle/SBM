"""`python -m sbm_t2i.cli train` end to end on the CPU with a tiny DiT: the run folder (run.yaml with the commit, ckpt/, logs/, evals/,
wandb/), checkpoints on schedule, and resuming a run folder where it stopped. Both modes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sbm_t2i import cli
from sbm_t2i import run as R

REPO = Path(__file__).resolve().parents[2]
TINY = {"model.name": "DiT-tiny", "model.input_size": 8, "model.label_dim": 10, "wandb.mode": "disabled",
        "train.stage_checkpoints_locally": False, "data.num_workers": 0}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SBM_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def frozen(tmp_path, **over):
    cfg = OmegaConf.load(REPO / "configs" / "t2i" / "train.yaml")
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


def test_finetune_run_folder_and_resume(tmp_path, store):
    g = torch.Generator().manual_seed(0)
    bank = tmp_path / "bank.pt"
    torch.save({"targets": torch.randn(20, 2, 4, 8, 8, generator=g).to(torch.bfloat16),
                "labels": torch.randint(10, (20, 2), generator=g)}, bank)
    exp = frozen(tmp_path, **{"mode": "finetune", "name": "tiny-ft", "data.bank": str(bank), "model.train_pos_embed": True,
                              "optim.name": "adamw", "optim.lr": 1e-4, "optim.clip": 2.0, "loss.endpoint.p": 1.0,
                              "loss.signed.alpha": 1.0, "train.precision": "32-true", "train.tf32": False,
                              "train.max_steps": 4, "train.checkpoint_every": 2, "train.val_every": "null",
                              "finetune.micro_pairs": 8})
    cli.main(["train", exp])
    run_dir = only_run(store)
    assert run_dir.name.endswith("_tiny-ft")
    for sub in ("ckpt", "logs", "evals", "wandb"):
        assert (run_dir / sub).is_dir()
    cfg = OmegaConf.load(run_dir / "run.yaml")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert cfg.run.git.commit == head and cfg.run.env.torch == torch.__version__
    assert steps_in(run_dir) == [2, 4]
    # resume: extend the frozen config and train the same folder again
    cfg.train.max_steps = 6
    OmegaConf.save(cfg, run_dir / "run.yaml")
    cli.main(["train", str(run_dir)])
    assert steps_in(run_dir) == [2, 4, 6]
    ck = torch.load(run_dir / "ckpt" / "step0000006.ckpt", map_location="cpu", weights_only=False)
    assert ck["global_step"] == 6 and "model.pos_embed" in ck["state_dict"]


def test_pretrain_run_with_ema(tmp_path, store):
    pack = tmp_path / "packed" / "train"
    pack.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(pack / "part_000.npy", rng.standard_normal((32, 2, 8, 8, 8)).astype(np.float32))
    (pack / "part_000.json").write_text(json.dumps({"labels": [i % 10 for i in range(32)], "files": [f"{i}.pt" for i in range(32)]}))
    exp = frozen(tmp_path, **{"name": "tiny-pre", "data.latent_pack_dir": str(tmp_path / "packed"), "train.precision": "32-true",
                              "train.global_batch": 8, "train.per_device_batch": 8, "train.max_steps": 3,
                              "train.checkpoint_every": 3, "train.val_every": "null", "train.ema.min_step": 0,
                              "loss.cfg_distill.enabled": True})
    cli.main(["train", exp])
    run_dir = only_run(store)
    assert steps_in(run_dir) == [3]
    ck = torch.load(run_dir / "ckpt" / "step0000003.ckpt", map_location="cpu", weights_only=False)
    assert ck["ema_active_decay"] == 0.999 and set(ck["ema_state_dicts"]) == {"0_99", "0_9999"}
    assert not torch.equal(ck["state_dict"]["model.final_layer.linear.weight"],
                           ck["current_model_state"]["model.final_layer.linear.weight"])   # EMA differs from raw
    from sbm_t2i.models import build_model, load_weights
    load_weights(build_model("DiT-tiny", input_size=8, label_dim=10), run_dir / "ckpt" / "step0000003.ckpt", "raw")
