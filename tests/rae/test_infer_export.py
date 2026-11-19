"""The EMA-only slim export (sbm_rae.infer, scripts/rae/export_record.py) and the FID scorer's handling of it."""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.ema import ema_decay_label
from sbm_rae.infer import (SLIM_FORMAT, build_model_from_cfg, is_slim_export, load_model_from_export, load_slim_export,
                           module_from_export, slim_export_from_checkpoint)
from sbm_rae.models.dit import DiT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABELS = 10


def tiny_cfg():
    """A training config whose model block builds a (small) registered network; the rest is what TrainingModule reads."""
    return {
        "seed": 1,
        "model": {"name": "DiT-S/2", "input_size": 8, "in_channels": 4, "label_dim": LABELS, "attn_func": "default",
                  "mlp_ratio": 4.0, "learn_sigma": False, "qk_norm": False, "is_zero_data": True, "dropout": 0.0},
        "dataset": {"name": "imagenet", "num_classes": LABELS, "img_resolution": 32, "use_precomputed_latents": False},
        "vae": {"enabled": False, "latents_scale": 0.5, "latents_bias": 0.25},
        "freeze": {"trainable": None},
        "trainer": {"overfitting": {"enabled": False}, "ema": {"enabled": True, "decays": [0.999, 0.9999], "active_decay": 0.999}},
        "sampling": {"enabled": False},
        "loss": {"name": "guided_cbm"},
        "anchor": {"enabled": True, "alpha": 0.4},
    }


def fake_lightning_checkpoint(cfg):
    torch.manual_seed(0)
    model = build_model_from_cfg(OmegaConf.create(cfg))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    active = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(1)
    other = {k: v + 0.01 * torch.randn(v.shape, generator=g) for k, v in active.items()}   # a different, finite EMA copy
    scale = torch.full((1, 4, 1, 1), 0.5)
    bias = torch.full((1, 4, 1, 1), 0.25)
    for sd in (active, other):
        sd["latents_scale"], sd["latents_bias"] = scale.clone(), bias.clone()
    return {
        "state_dict": active, "current_model_state": {k: v - 1.0 for k, v in active.items()},
        "ema_state_dicts": {ema_decay_label(0.9999): other}, "ema_active_decay": 0.999,
        "ema_decays": {ema_decay_label(d): d for d in (0.999, 0.9999)},
        "hyper_parameters": {"cfg": cfg}, "global_step": 7, "epoch": 1,
        "guided_cbm_loss": {"ratio_ema": 1.5, "frozen_weight": 2.0},
        "optimizer_states": [{"should": "not be exported"}],
    }, model


def test_export_picks_the_requested_ema_copy():
    cfg = tiny_cfg()
    ck, model = fake_lightning_checkpoint(cfg)
    exp = slim_export_from_checkpoint(ck, 0.9999, meta={"note": "test"})
    assert is_slim_export(exp) and exp["format"] == SLIM_FORMAT
    assert set(exp["model_state_dict"]) == set(model.state_dict())
    assert not any(k.startswith("model.") for k in exp["model_state_dict"])
    ref = ck["ema_state_dicts"]["0_9999"]
    for k, v in exp["model_state_dict"].items():
        assert torch.equal(v, ref["model." + k])
    assert torch.equal(exp["latents_scale"], ref["latents_scale"]) and torch.equal(exp["latents_bias"], ref["latents_bias"])
    assert exp["ema_decay"] == 0.9999 and exp["ema_label"] == "0_9999" and exp["global_step"] == 7
    assert exp["meta"] == {"note": "test"} and exp["guided_cbm_loss"] == ck["guided_cbm_loss"]
    assert "optimizer_states" not in exp and "ema_state_dicts" not in exp
    # the active decay comes from state_dict (the swapped-in active EMA), not from ema_state_dicts
    active = slim_export_from_checkpoint(ck, 0.999)
    assert torch.equal(next(iter(active["model_state_dict"].values())), ck["state_dict"]["model." + next(iter(active["model_state_dict"]))])
    with pytest.raises(KeyError, match="no EMA"):
        slim_export_from_checkpoint(ck, 0.99)


def test_export_roundtrips_and_rebuilds_the_network(tmp_path):
    cfg = tiny_cfg()
    ck, _ = fake_lightning_checkpoint(cfg)
    path = tmp_path / "export.pt"
    torch.save(slim_export_from_checkpoint(ck, 0.9999), path)
    assert is_slim_export(path)
    lightning_path = tmp_path / "plain.ckpt"
    torch.save(ck, lightning_path)
    assert not is_slim_export(lightning_path)

    exp = load_slim_export(path)
    model, info = load_model_from_export(exp, strict=True)
    assert info["missing"] == [] and info["unexpected"] == [] and info["global_step"] == 7
    assert info["n_params"] == sum(p.numel() for p in model.parameters())
    reference = build_model_from_cfg(OmegaConf.create(cfg))
    reference.load_state_dict({k[len("model."):]: v for k, v in ck["ema_state_dicts"]["0_9999"].items() if k.startswith("model.")})
    x, y = torch.randn(2, 4, 8, 8), torch.tensor([1, 2])
    with torch.no_grad():
        torch.testing.assert_close(model(x, y, omega=torch.tensor([0.8, 0.8])), reference(x, y, omega=torch.tensor([0.8, 0.8])))


def test_module_from_export_sets_weights_and_latent_buffers():
    cfg = OmegaConf.create(tiny_cfg())
    ck, _ = fake_lightning_checkpoint(OmegaConf.to_container(cfg))
    exp = slim_export_from_checkpoint(ck, 0.9999)
    model = build_model_from_cfg(cfg)
    module = module_from_export(cfg, exp, model, loss_fn=None)
    assert module.model is model
    torch.testing.assert_close(module.latents_scale, torch.full((1, 4, 1, 1), 0.5))
    torch.testing.assert_close(module.latents_bias, torch.full((1, 4, 1, 1), 0.25))
    k = next(iter(exp["model_state_dict"]))
    assert torch.equal(dict(module.model.named_parameters()).get(k, module.model.state_dict()[k]), exp["model_state_dict"][k])


def test_eval_fid_treats_the_export_as_one_active_variant(tmp_path):
    import scripts.rae.eval_fid as eval_fid

    cfg = tiny_cfg()
    ck, _ = fake_lightning_checkpoint(cfg)
    path = tmp_path / "export.pt"
    torch.save(slim_export_from_checkpoint(ck, 0.9999), path)
    distributed = eval_fid.DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1)
    fid_cfg = OmegaConf.create({"fid": {"ema_decays": [0.9999], "strict_ema_decays": True, "score_raw": False}})
    assert eval_fid._weight_variants_for_checkpoint(fid_cfg, path, distributed) == [
        eval_fid.WeightVariant(source="active", label="0_9999", decay=0.9999)]
    # another decay is requested: lenient -> the exported copy with a note, strict -> an error
    lenient = OmegaConf.create({"fid": {"ema_decays": [0.999], "strict_ema_decays": False}})
    assert eval_fid._weight_variants_for_checkpoint(lenient, path, distributed)[0].label == "0_9999"
    strict = OmegaConf.create({"fid": {"ema_decays": [0.999], "strict_ema_decays": True}})
    with pytest.raises(ValueError, match="slim export"):
        eval_fid._weight_variants_for_checkpoint(strict, path, distributed)
    # the training config and the step come out of the export like out of a checkpoint
    assert eval_fid._training_cfg_from_checkpoint(path).model.name == "DiT-S/2"
    assert eval_fid._training_step_of(path) == 7


def test_train_init_weights_from_accepts_the_export(tmp_path, capsys):
    """init_weights_from=<EMA-only export> starts a fine-tune from the exported EMA copy (scripts/rae/train.py)."""
    import types

    import scripts.rae.train as train

    cfg = tiny_cfg()
    ck, _ = fake_lightning_checkpoint(cfg)
    path = tmp_path / "export.pt"
    exp = slim_export_from_checkpoint(ck, 0.9999)
    torch.save(exp, path)
    module = types.SimpleNamespace(model=build_model_from_cfg(OmegaConf.create(cfg)),
                                   latents_scale=torch.full((1, 4, 1, 1), 0.5), latents_bias=torch.full((1, 4, 1, 1), 0.25))
    train._init_weights_from_checkpoint(OmegaConf.create({"init_weights_from": str(path)}), module)
    for k, v in module.model.state_dict().items():
        assert torch.equal(v, exp["model_state_dict"][k])
    out = capsys.readouterr().out
    assert "EMA 0.9999" in out and "WARNING" not in out
    # with a guided_cbm loss on the module, the export's loss state (the frozen transport weight) comes along
    from sbm_rae.guided_cbm import GuidedCBMLoss
    loss = GuidedCBMLoss.__new__(GuidedCBMLoss)
    loss._ratio_ema = loss._frozen_weight = None
    module.loss_fn = loss
    train._init_weights_from_checkpoint(OmegaConf.create({"init_weights_from": str(path)}), module)
    assert loss._frozen_weight == 2.0 and float(loss._ratio_ema) == 1.5
    assert "restored the guided_cbm loss state" in capsys.readouterr().out
    # a different latent affine in the fine-tune's config is flagged; the raw-weights source does not exist in an export
    module.latents_scale = torch.ones(1, 4, 1, 1)
    train._init_weights_from_checkpoint(OmegaConf.create({"init_weights_from": str(path)}), module)
    assert "WARNING: the export's latents_scale" in capsys.readouterr().out
    with pytest.raises(ValueError, match="EMA-only export"):
        train._init_weights_from_checkpoint(
            OmegaConf.create({"init_weights_from": str(path), "init_weights_source": "current_model_state"}), module)
