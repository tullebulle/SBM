"""Slim inference entry point: the EMA-only export of a training checkpoint, without optimizer / teacher / EMA-callback state.

scripts/rae/export_record.py writes such a file from a Lightning checkpoint; eval_fid.py reads it in place of the checkpoint
(the original .ckpt path keeps working). Layout of the export (a plain torch.save dict):

    format            "sbm_rae_slim_export_v1"
    model_state_dict  the chosen EMA copy, keyed like the network's own state_dict (DiT.load_state_dict(..., strict=True))
    latents_scale / latents_bias   the TrainingModule's latent affine buffers, (1, C, 1, 1)
    hyper_parameters  {"cfg": the full Hydra training config of the run} (what TrainingModule.save_hyperparameters stored)
    global_step / epoch            the checkpoint's counters
    ema_decay / ema_label          which EMA copy this is (e.g. 0.9999 / "0_9999")
    meta              provenance and the read settings that gave the published numbers
"""
import torch

SLIM_FORMAT = "sbm_rae_slim_export_v1"


def is_slim_export(obj):
    """True for a loaded export dict, or for a path whose file is one (read with mmap, so cheap)."""
    if isinstance(obj, dict):
        return str(obj.get("format", "")).startswith("sbm_rae_slim_export")
    try:
        return is_slim_export(torch.load(str(obj), map_location="meta", weights_only=False))
    except Exception:
        return False


def load_slim_export(path, map_location="cpu"):
    export = torch.load(str(path), map_location=map_location, mmap=True, weights_only=False)
    if not is_slim_export(export):
        raise ValueError(f"{path} is not an sbm_rae slim export (no 'format' key)")
    return export


def build_model_from_cfg(cfg, model_overrides=None):
    """The network of a training config (cfg.model), with optional overrides (e.g. {"attn_func": "default"})."""
    from omegaconf import OmegaConf

    from sbm_rae.models import build_model

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if model_overrides:
        model_cfg = OmegaConf.merge(model_cfg, OmegaConf.create(dict(model_overrides)))
    return build_model(model_cfg)


def load_model_from_export(export, model=None, model_overrides=None, strict=True):
    """Build (or take) the network and load the export's weights; returns (model, info dict)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(export["hyper_parameters"]["cfg"])
    if model is None:
        model = build_model_from_cfg(cfg, model_overrides)
    missing, unexpected = model.load_state_dict(export["model_state_dict"], strict=strict)
    info = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_tensors": len(export["model_state_dict"]),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "global_step": export.get("global_step"),
        "ema_decay": export.get("ema_decay"),
    }
    return model, info


def module_from_export(cfg, export, model, loss_fn, weighting_model=None):
    """A TrainingModule around `model` (already built from cfg.model, the scorer's way) with the export's weights and
    latent affine buffers; the equivalent of TrainingModule.load_from_checkpoint + the EMA copy, without the rest."""
    from sbm_rae.trainer import TrainingModule

    module = TrainingModule(cfg, model, loss_fn, weighting_model=weighting_model)
    module.model.load_state_dict(export["model_state_dict"], strict=True)
    with torch.no_grad():
        module.latents_scale.copy_(export["latents_scale"].to(module.latents_scale.dtype))
        module.latents_bias.copy_(export["latents_bias"].to(module.latents_bias.dtype))
    return module


def slim_export_from_checkpoint(checkpoint, ema_decay, meta=None, keep_loss_state=True):
    """Build the export dict from a loaded Lightning checkpoint of sbm_rae.trainer.TrainingModule.

    `ema_decay` picks the copy: the checkpoint's active EMA lives in `state_dict` (sbm_rae.ema swaps it in at save time),
    every other decay in `ema_state_dicts[<label>]`; both are keyed like the module's state_dict ("model.*" plus the
    latents_scale / latents_bias buffers). The tensors are cloned (a checkpoint opened with mmap=True stays untouched).
    """
    from sbm_rae.ema import ema_decay_label

    label = ema_decay_label(ema_decay)
    active = checkpoint.get("ema_active_decay", None)
    if active is not None and abs(float(ema_decay) - float(active)) < 1e-12:
        source = checkpoint["state_dict"]
    else:
        stored = checkpoint.get("ema_state_dicts", None) or {}
        if label not in stored:
            raise KeyError(f"no EMA {ema_decay} (label {label!r}) in the checkpoint: active {active}, stored {sorted(stored)}")
        source = stored[label]
    model_sd = {k[len("model."):]: v.detach().clone().contiguous() for k, v in source.items() if k.startswith("model.")}
    if not model_sd:
        raise ValueError("the chosen weight copy has no 'model.*' tensors")
    export = {
        "format": SLIM_FORMAT,
        "model_state_dict": model_sd,
        "latents_scale": source["latents_scale"].detach().clone(),
        "latents_bias": source["latents_bias"].detach().clone(),
        "hyper_parameters": {"cfg": checkpoint["hyper_parameters"]["cfg"]},
        "global_step": checkpoint.get("global_step"),
        "epoch": checkpoint.get("epoch"),
        "ema_decay": float(ema_decay),
        "ema_label": label,
        "meta": dict(meta or {}),
    }
    if keep_loss_state and checkpoint.get("guided_cbm_loss") is not None:
        # the loss's tiny state (the frozen transport weight of balance.mode=freeze), for a fine-tune that starts here
        export["guided_cbm_loss"] = checkpoint["guided_cbm_loss"]
    return export
