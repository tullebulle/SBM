import os
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
import lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from sbm_rae.lr_override import ForceLearningRate
from sbm_rae.checkpoint_io import LocalThenSyncCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from sbm_rae.data import get_data_module
from sbm_rae.ema import EMAWeightAveraging, resolve_active_ema_decay, resolve_ema_decays
from sbm_rae.infer import is_slim_export
from sbm_rae.loss import get_loss_fn, resolve_guidance_config, resolve_signed_config
from sbm_rae.models import build_model
from sbm_rae.models.weight import LossWeightingNetwork
from sbm_rae.trainer import (
    IterativeSamplingCallback,
    TrainingModule,
    overfitting_invariant_plot_enabled,
    train_interpolation_plot_enabled,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


def _resolve_num_devices(accelerator, devices):
    if isinstance(devices, int):
        if devices < 1:
            raise ValueError("trainer.devices must be >= 1")
        return devices

    if isinstance(devices, (list, tuple)):
        if len(devices) < 1:
            raise ValueError("trainer.devices must select at least one device")
        return len(devices)

    if isinstance(devices, str):
        stripped = devices.strip()
        if stripped == "auto":
            if accelerator in {"cpu", "mps"}:
                return 1
            return max(torch.cuda.device_count(), 1)
        if stripped.isdigit():
            return int(stripped)
        if "," in stripped:
            return len([part for part in stripped.split(",") if part.strip()])

    if devices is None:
        return 1

    raise ValueError(f"Unsupported trainer.devices value: {devices}")


def _configure_batching(cfg):
    trainer_cfg = cfg.trainer
    per_device_batch_size = int(
        trainer_cfg.get("per_device_batch_size", trainer_cfg.get("batch_size", 1))
    )
    global_batch_size = int(
        trainer_cfg.get("global_batch_size", per_device_batch_size)
    )
    num_devices = _resolve_num_devices(trainer_cfg.accelerator, trainer_cfg.devices)
    num_nodes = int(trainer_cfg.get("num_nodes", 1))
    world_size = num_devices * num_nodes
    if world_size < 1:
        raise ValueError("Resolved world size must be >= 1")

    per_step_batch_size = per_device_batch_size * world_size
    if global_batch_size < per_step_batch_size:
        raise ValueError(
            "trainer.global_batch_size must be >= per_device_batch_size * world_size "
            f"({global_batch_size} < {per_step_batch_size})"
        )
    if global_batch_size % per_step_batch_size != 0:
        raise ValueError(
            "trainer.global_batch_size must be divisible by per_device_batch_size * world_size "
            f"({global_batch_size} is not divisible by {per_step_batch_size})"
        )

    trainer_cfg.per_device_batch_size = per_device_batch_size
    trainer_cfg.global_batch_size = global_batch_size
    trainer_cfg.accumulate_grad_batches = global_batch_size // per_step_batch_size
    trainer_cfg.resolved_num_devices = num_devices
    trainer_cfg.resolved_world_size = world_size
    trainer_cfg.effective_global_batch_size = (
        per_device_batch_size * world_size * trainer_cfg.accumulate_grad_batches
    )


def _resolve_training_limits(cfg):
    trainer_cfg = cfg.trainer
    max_epochs = trainer_cfg.get("max_epochs", trainer_cfg.get("num_epochs", None))
    max_steps = trainer_cfg.get("max_steps", trainer_cfg.get("num_train_steps", None))

    if max_epochs is None:
        max_epochs = -1
    else:
        max_epochs = int(max_epochs)

    if max_steps is None:
        max_steps = -1
    else:
        max_steps = int(max_steps)

    if max_epochs < 0 and max_steps < 0:
        raise ValueError("Set either trainer.num_epochs/max_epochs or trainer.num_train_steps/max_steps")

    trainer_cfg.resolved_max_epochs = max_epochs
    trainer_cfg.resolved_max_steps = max_steps


def _unwrap_targets(dataset):
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = _unwrap_targets(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]
    raise ValueError("Unable to read class targets from validation dataset")


def _build_sampling_labels(cfg, datamodule):
    if not cfg.sampling.enabled:
        return None

    if bool(cfg.trainer.get("overfitting", {}).get("enabled", False)):
        return None

    if cfg.sampling.class_labels:
        labels = torch.tensor(cfg.sampling.class_labels, dtype=torch.long)
    elif cfg.sampling.use_val_labels:
        targets = _unwrap_targets(datamodule.val_dataset)
        generator = torch.Generator().manual_seed(cfg.seed + cfg.sampling.seed_offset)
        indices = torch.randperm(len(targets), generator=generator)
        labels = torch.tensor([targets[index] for index in indices[: cfg.sampling.n_samples]], dtype=torch.long)
    else:
        labels = torch.arange(cfg.sampling.n_samples, dtype=torch.long) % cfg.dataset.num_classes

    if labels.numel() < cfg.sampling.n_samples:
        repeat = (cfg.sampling.n_samples + labels.numel() - 1) // labels.numel()
        labels = labels.repeat(repeat)

    return labels[: cfg.sampling.n_samples]


def _build_logger(cfg):
    if cfg.logging.use_wandb:
        return WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            entity=cfg.logging.get("entity", None),
            save_dir=cfg.work_dir,
            offline=cfg.logging.get("offline", False),
            save_code=cfg.logging.get("save_code", True),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    return CSVLogger(save_dir=cfg.work_dir, name="csv")


def _init_weights_from_checkpoint(cfg, lightning_module):
    """Start a FRESH run from another checkpoint's weights (`init_weights_from`).

    Unlike `resume_from_checkpoint`, this restores the network weights and nothing
    else: the optimizer, schedulers, global step and data position all start at zero,
    and the EMA shadows (which live in a callback) start as copies of these weights
    rather than as the source run's shadows. It exists because a run whose model has parameters the source checkpoint's
    optimizer never saw (the amortised signed rule adds a guidance embedding) cannot
    restore that optimizer state at all: its parameter groups no longer match, and
    dropping them makes Lightning refuse the resume outright.

    The arm's step axis therefore starts at 0 while its weights are the source's, so
    read it against the source's step plus its own, and expect a short transient while
    the optimizer moments rebuild.
    """
    path = cfg.get("init_weights_from", None)
    if not path:
        return
    if cfg.get("resume_from_checkpoint", None):
        # A run started from init_weights_from that is later resumed (same arguments plus resume_from_checkpoint): the
        # run's own checkpoint holds everything, so the init weights are not reloaded on top of it.
        print(f"resume_from_checkpoint is set: init_weights_from ({path}) is ignored for this resumed run")
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # `state_dict` is the checkpoint's active EMA copy; `current_model_state` the raw
    # optimiser weights (init_weights_source=current_model_state continues those).
    source_key = str(cfg.get("init_weights_source", "state_dict"))
    if is_slim_export(checkpoint):
        # An EMA-only export (sbm_rae.infer, e.g. a released checkpoint): one EMA copy, keyed like the network
        # itself. It has no raw optimiser weights, so only the default source applies.
        if source_key != "state_dict":
            raise ValueError(f"{path} is an EMA-only export: init_weights_source={source_key!r} is not available "
                             "(it holds one EMA copy; use the default init_weights_source=state_dict)")
        weights = dict(checkpoint["model_state_dict"])
        source_key = f"model_state_dict, EMA {checkpoint.get('ema_decay')}"
        for name in ("latents_scale", "latents_bias"):
            exported, own = checkpoint.get(name), getattr(lightning_module, name, None)
            if exported is not None and own is not None and not torch.allclose(exported.to(own), own):
                print(f"WARNING: the export's {name} differs from this run's (set by its vae config); the weights were "
                      f"trained with the export's, keep the vae config of the source run")
    else:
        if source_key not in checkpoint:
            raise KeyError(f"init_weights_source={source_key!r} is not in {path}: {sorted(checkpoint)}")
        weights = {
            key[len("model."):]: value
            for key, value in checkpoint[source_key].items()
            if key.startswith("model.")
        }
    missing, unexpected = lightning_module.model.load_state_dict(weights, strict=False)
    # Tensors a fresh head adds (e.g. the auxiliary velocity head) may be absent from
    # the source; anything else missing is an error.
    allowed = tuple(cfg.get("init_weights_allow_missing", None) or [])
    missing = [key for key in missing if not key.startswith(allowed)] if allowed else missing
    if missing:
        raise ValueError(f"init_weights_from is missing {len(missing)} tensors, e.g. {missing[:3]}")
    print(f"initialised weights from {path}[{source_key}] ({len(weights)} tensors, {len(unexpected)} unused)")
    # An export also carries the guided-CBM loss state (the balancer's ratio EMA and the frozen transport weight): a
    # fine-tune that starts there with balance.mode=freeze keeps the source's weight instead of failing at the guide start
    # for lack of balancer history. A plain checkpoint keeps the old behaviour (the balancer restarts).
    loss_fn = getattr(lightning_module, "loss_fn", None)
    loss_state = checkpoint.get("guided_cbm_loss") if is_slim_export(checkpoint) else None
    if loss_state is not None and hasattr(loss_fn, "teacher_schedule") and hasattr(loss_fn, "load_state_dict"):
        loss_fn.load_state_dict(loss_state)
        print(f"restored the guided_cbm loss state from the export: {loss_state}")


def _resolve_resume_checkpoint_path(cfg):
    checkpoint_path = cfg.get("resume_from_checkpoint", None)
    if checkpoint_path in {None, "", "null"}:
        return None

    resolved_path = Path(checkpoint_path).expanduser()
    if not resolved_path.is_absolute():
        raise ValueError(
            "resume_from_checkpoint must be an absolute path. "
            f"Got: {checkpoint_path}"
        )

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resolved_path}")

    return str(resolved_path)


def _configure_repa_model(cfg):
    repa_cfg = cfg.get("repa", {}) or {}
    enabled = bool(repa_cfg.get("enabled", False))
    # loss.guided_cbm.repa aligns to the clean latent itself: it needs only the model's projector (model.repa_*, set by
    # the launcher), not the top-level repa block, which also turns on the data pipeline's representation targets
    guided_repa = (cfg.loss.get("guided_cbm", None) or {}).get("repa", None) or {}
    if not enabled and cfg.loss.get("name") == "guided_cbm" and bool(guided_repa.get("enabled", False)):
        cfg.model.repa_enabled = True
        return
    cfg.model.repa_enabled = enabled
    if not enabled:
        return

    cfg.model.repa_encoder_depth = int(repa_cfg.get("encoder_depth", 8))
    cfg.model.repa_z_dim = int(repa_cfg.get("z_dim", 768))
    cfg.model.repa_projector_dim = int(repa_cfg.get("projector_dim", 2048))


def _guided_cbm_scale_cond(cfg):
    """loss.guided_cbm.guide.scale_cond: the guided-CBM network takes the guidance scale w as an input."""
    if cfg.loss.get("name") != "guided_cbm" or cfg.loss.get("guided_cbm", None) is None:
        return False
    from sbm_rae.guided_cbm import guidance_scale_cond

    return guidance_scale_cond(cfg.loss.guided_cbm)[0]


def _configure_guidance_model(cfg):
    # The amortised signed rule hands alpha to the network through the SAME w
    # embedding classifier-free guidance uses, and that embedding only exists when
    # the model is built with it enabled. Without this the strength is accepted and
    # silently ignored, which makes an amortised run a slower native run.
    cfg.model.guidance_enabled = (
        resolve_guidance_config(cfg)["enabled"]
        or resolve_signed_config(cfg)["amortised"]["enabled"]
        or _guided_cbm_scale_cond(cfg)
    )


def _build_single_weighting_network(weighting_cfg):
    return LossWeightingNetwork(
        channels=weighting_cfg.get("channels", 128),
        clamp_min=weighting_cfg.get("clamp_min", -10.0),
        clamp_max=weighting_cfg.get("clamp_max", 10.0),
    )


def _weighting_head_enabled(weighting_cfg, name):
    if name in weighting_cfg:
        return bool(weighting_cfg.get(name))
    return True


def _build_weighting_model(cfg):
    weighting_cfg = cfg.get("weighting_model", None)
    if not weighting_cfg or not weighting_cfg.get("enabled", False):
        return None

    enabled_heads = ["transport", "endpoint"]
    if cfg.loss.get("self_consistency", False):
        enabled_heads.append("self_consistency")

    weighting_heads = {
        name: _build_single_weighting_network(weighting_cfg)
        for name in enabled_heads
        if _weighting_head_enabled(weighting_cfg, name)
    }
    if not weighting_heads:
        return None

    return torch.nn.ModuleDict(weighting_heads)


def _resolve_strategy(cfg):
    """Build the strategy, widening the collective timeout for plain DDP.

    Checkpoints here are ~2 GB and land on a shared filesystem. When rank 0
    blocks in that write for longer than NCCL's default 30-minute watchdog, the
    other ranks abort a scalar metric allreduce and the run dies with
    "Watchdog caught collective operation timeout" -- a filesystem stall
    presenting as a distributed failure. A longer timeout rides it out.
    """
    strategy = cfg.trainer.get("strategy", "auto")
    if strategy != "ddp":
        return strategy
    minutes = int(cfg.trainer.get("ddp_timeout_minutes", 120))
    return DDPStrategy(timeout=timedelta(minutes=minutes))


def _resolve_trainer_loop_settings(cfg):
    overfitting_enabled = bool(cfg.trainer.get("overfitting", {}).get("enabled", False))
    if overfitting_enabled:
        return {
            "max_epochs": 100000,
            "max_steps": 100000,
            "limit_train_batches": 1,
            "limit_val_batches": 0,
            "val_check_interval": 1,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
        }

    return {
        "max_epochs": cfg.trainer.resolved_max_epochs,
        "max_steps": cfg.trainer.resolved_max_steps,
        "limit_train_batches": cfg.trainer.get("limit_train_batches", 1.0),
        "limit_val_batches": cfg.trainer.limit_val_batches,
        "val_check_interval": cfg.trainer.val_check_interval,
        # CIFAR is ~97 optimizer steps per epoch at a global batch of 512, far
        # fewer than ImageNet, and Lightning rejects a val_check_interval
        # larger than one epoch's batches unless validation is counted over
        # total steps instead. null here means "count global steps".
        "check_val_every_n_epoch": cfg.trainer.get("check_val_every_n_epoch", 1),
        "num_sanity_val_steps": cfg.trainer.get("num_sanity_val_steps", 2),
    }


@hydra.main(config_path="../../configs/rae", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg.work_dir = HydraConfig.get().runtime.output_dir
    _configure_batching(cfg)
    _resolve_training_limits(cfg)
    resume_checkpoint_path = _resolve_resume_checkpoint_path(cfg)

    seed_everything(cfg.seed, workers=True)

    print(
        "Batching config: "
        f"per_device_batch_size={cfg.trainer.per_device_batch_size}, "
        f"devices={cfg.trainer.resolved_num_devices}, "
        f"num_nodes={cfg.trainer.num_nodes}, "
        f"accumulate_grad_batches={cfg.trainer.accumulate_grad_batches}, "
        f"effective_global_batch_size={cfg.trainer.effective_global_batch_size}, "
        f"max_epochs={cfg.trainer.resolved_max_epochs}, "
        f"max_steps={cfg.trainer.resolved_max_steps}"
    )

    datamodule = get_data_module(cfg)
    # Only the latent-pack path needs the cache; images encoded on the fly (vae.kind=rae) do not.
    if bool(cfg.dataset.get("use_precomputed_latents", False)):
        datamodule.require_latent_cache()
    datamodule.require_representation_cache()
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    _configure_repa_model(cfg)
    _configure_guidance_model(cfg)
    model = build_model(cfg.model)
    weighting_model = _build_weighting_model(cfg)
    loss_fn = get_loss_fn(cfg)
    if str(cfg.dataset.get("name", "imagenet")) == "bridge":
        from sbm_rae.bridge_trainer import BridgeTrainingModule   # frames encoded on the GPU, (x, cond) split per clip

        lightning_module = BridgeTrainingModule(cfg, model, loss_fn, weighting_model=weighting_model)
    else:
        lightning_module = TrainingModule(cfg, model, loss_fn, weighting_model=weighting_model)
    _init_weights_from_checkpoint(cfg, lightning_module)

    # Checkpoint cadence is counted in steps, not epochs. A CIFAR epoch is ~97
    # optimizer steps (~30 s), and each checkpoint carries the model, two EMA
    # copies, and Muon/AdamW state - about 2 GB for DiT-B/2. Saving per epoch
    # meant rank 0 wrote and re-copied 2 GB roughly twice a minute while the
    # other ranks sat in a collective waiting for it: measured 95 GB written
    # and 49 GB read back in 40 minutes, for ~2200 steps of actual progress.
    # save_last="link" symlinks last.ckpt at the newest checkpoint instead of
    # writing a second full copy of it.
    every_n_train_steps = cfg.trainer.get("checkpoint_every_n_steps", None)
    every_n_epochs = cfg.trainer.get("checkpoint_every_n_epochs", None)
    if every_n_train_steps and every_n_epochs:
        raise ValueError(
            "Set only one of trainer.checkpoint_every_n_steps and "
            "trainer.checkpoint_every_n_epochs; Lightning rejects both at once"
        )
    if not every_n_train_steps and not every_n_epochs:
        raise ValueError(
            "Set trainer.checkpoint_every_n_steps or trainer.checkpoint_every_n_epochs"
        )

    shared_checkpoints = Path(cfg.work_dir) / "checkpoints"
    checkpoint_kwargs = dict(
        # No "=" in the name: eval_fid.py takes the path as a Hydra
        # override, and Hydra parses "=" as a key/value separator.
        filename="step{step:07d}-epoch{epoch:04d}",
        auto_insert_metric_name=False,
        every_n_train_steps=every_n_train_steps or None,
        every_n_epochs=every_n_epochs or None,
        save_on_train_epoch_end=bool(every_n_epochs),
        save_top_k=-1,
        save_last="link",
    )

    # Writing a multi-GB checkpoint straight to a shared filesystem puts that write
    # inside the barrier that follows each save, which can stall or time out a
    # multi-node run. trainer.checkpoint_local_dir stages it on node-local disk and
    # mirrors it in the background instead.
    local_scratch = cfg.trainer.get("checkpoint_local_dir", None)
    if local_scratch:
        local_scratch = Path(str(local_scratch))
        checkpoint_callback = LocalThenSyncCheckpoint(
            dirpath=shared_checkpoints,
            shared_dirpath=shared_checkpoints,
            local_dirpath=local_scratch,
            sparse_after=cfg.trainer.get("checkpoint_sparse_after", None),
            sparse_every=cfg.trainer.get("checkpoint_sparse_every", None),
            **checkpoint_kwargs,
        )
    else:
        if cfg.trainer.get("checkpoint_sparse_every", None):
            raise ValueError("trainer.checkpoint_sparse_every needs trainer.checkpoint_local_dir (LocalThenSyncCheckpoint)")
        checkpoint_callback = ModelCheckpoint(dirpath=shared_checkpoints, **checkpoint_kwargs)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
    ]

    # A changed lr.val is discarded on resume unless it is reasserted after
    # Lightning restores the optimizer and scheduler state. See sbm_rae.lr_override.
    if bool(cfg.lr.get("force_on_resume", False)):
        callbacks.append(ForceLearningRate(cfg.lr.val))

    if cfg.trainer.ema.enabled:
        ema_decays = resolve_ema_decays(cfg.trainer.ema)
        callbacks.append(
            EMAWeightAveraging(
                decays=ema_decays,
                active_decay=resolve_active_ema_decay(cfg.trainer.ema, ema_decays),
                min_step=cfg.trainer.ema.min_step,
                use_buffers=cfg.trainer.ema.use_buffers,
                foreach=bool(cfg.trainer.ema.get("foreach", False)),   # speed: one multi-tensor update per EMA
            )
        )

    sampling_labels = _build_sampling_labels(cfg, datamodule)
    if (
        cfg.sampling.enabled
        or overfitting_invariant_plot_enabled(cfg)
        or train_interpolation_plot_enabled(cfg)
    ):
        callbacks.append(
            IterativeSamplingCallback(cfg, sampling_labels, datamodule.inverse_scaler)
        )

    loop_settings = _resolve_trainer_loop_settings(cfg)
    trainer = pl.Trainer(
        default_root_dir=cfg.work_dir,
        logger=_build_logger(cfg),
        max_epochs=loop_settings["max_epochs"],
        max_steps=loop_settings["max_steps"],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=_resolve_strategy(cfg),
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        val_check_interval=loop_settings["val_check_interval"],
        check_val_every_n_epoch=loop_settings["check_val_every_n_epoch"],
        limit_train_batches=loop_settings["limit_train_batches"],
        limit_val_batches=loop_settings["limit_val_batches"],
        num_sanity_val_steps=loop_settings["num_sanity_val_steps"],
        use_distributed_sampler=False,
        profiler=None,
    )

    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=resume_checkpoint_path,
    )


if __name__ == "__main__":
    main()