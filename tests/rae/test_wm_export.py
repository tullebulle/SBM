"""World-model exports: a Bridge_UNet checkpoint -> slim export -> the evaluator's loader (scripts/rae/eval_bridge.py),
which must give the network of the checkpoint's active EMA."""
import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.infer import slim_export_from_checkpoint
from sbm_rae.models import build_model

REPO = Path(__file__).resolve().parents[2]


def load_script(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_checkpoint(tmp_path, old_store="/data/store/"):
    model_cfg = {"name": "Bridge_UNet", "in_channels": 4, "num_steps_conditioning": "${dataset.n_history}", "num_actions": 7,
                 "cond_channels": 32, "depths": [1, 1], "channels": [16, 32], "attn_depths": [0, 1], "action_token_dim": 32,
                 "action_token_mode": "per_frame_component", "film_actions": True}
    cfg = {"model": model_cfg, "dataset": {"name": "bridge", "n_history": 4, "action_dim": 7,
                                           "processed_dir": old_store + "data/bridge/processed/bridge_v2",
                                           "latent_cache_dir": old_store + "data/bridge/latents_sd3"},
           "data_dir": old_store + "data/bridge/processed/bridge_v2", "work_dir": "/somewhere/run",
           "vae": {"enabled": True, "latents_scale": 1.5305, "latents_bias": 0.0}, "logging": {"entity": "someone-org"}}
    torch.manual_seed(0)
    model = build_model(OmegaConf.create(cfg).model)
    active = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    other = {k: v + 1.0 for k, v in active.items()}
    for sd in (active, other):
        sd["latents_scale"], sd["latents_bias"] = torch.full((4, 1, 1), 1.5305), torch.zeros(4, 1, 1)
    ck = {"state_dict": active, "ema_state_dicts": {"0_9999": other}, "ema_active_decay": 0.999, "global_step": 660000,
          "epoch": 3, "hyper_parameters": {"cfg": cfg}, "guided_cbm_loss": {"ratio_ema": None, "frozen_weight": None}}
    return ck, model


def test_world_model_export_loads_in_the_evaluator(tmp_path):
    ck, model = tiny_checkpoint(tmp_path)
    path = tmp_path / "export.pt"
    torch.save(slim_export_from_checkpoint(ck, 0.999), path)
    eval_bridge = load_script("eval_bridge", "scripts/rae/eval_bridge.py")
    net, cfg, step, which = eval_bridge.load_checkpoint(str(path), ema=None, device="cpu")
    assert step == 660000 and "0.999" in which and cfg.dataset.action_dim == 7
    for k, v in model.state_dict().items():
        assert torch.equal(net.state_dict()[k], v), k
    with pytest.raises(KeyError, match="EMA 0.999 only"):
        eval_bridge.load_checkpoint(str(path), ema=0.9999, device="cpu")
