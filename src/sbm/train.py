"""The trainer: `train(cfg, run_dir)`.

ImageNet SD-VAE latents from a DataLoader, one CBMLoss call per micro-batch, Lightning's automatic optimization
(bf16-mixed, DDP, gradient accumulation to the global batch), Muon + AdamW, EMA copies (train.ema.decays; a decay of 1.0
is the guide's frozen teacher, refreshed every loss.guide.teacher_refresh steps).

Everything the run writes goes to its run folder: ckpt/ (Lightning checkpoints; resumed automatically), evals/, wandb/
(see sbm.run), and monitor/ when the in-training FID monitor or the step timer is on (sbm.monitor). `init.from` starts
from another checkpoint or export: its weights only (fine-tuning, a new run from step 0), or (init.resume_state) its
whole training state (optimizer, scheduler, EMA copies, global step, the loss state), a continuation of that run.

Distributed: on one node, `python -m sbm.cli train` uses every GPU (Lightning starts one process per GPU); under torchrun
the processes come from torchrun (train.devices = its --nproc_per_node, nodes = WORLD_SIZE / LOCAL_WORLD_SIZE).
"""
from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from sbm import run as R
from sbm.data import LatentData
from sbm.ema import EMA
from sbm.loss import CBMLoss
from sbm.models import build_model, fill_gw_keys, load_weights
from sbm.optim import (ForceLearningRate, GlobalStepLR, StagedCheckpoint, build_optimizer, extend_optimizer_state, lr_factor,
                       split_params)

LOSS_STATE_KEY = "cbm_loss_state"   # the checkpoint key of the loss state (kept: every existing checkpoint carries it)


class Module(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: torch.nn.Module):
        super().__init__()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)}, logger=False)
        self.cfg, self.model = cfg, model
        # the latent affine (scale 0.18215, bias 0), kept in the checkpoint for the evaluator
        self.register_buffer("latents_scale", torch.full((1, cfg.model.in_channels, 1, 1), float(cfg.data.latents_scale)))
        self.register_buffer("latents_bias", torch.zeros(1, cfg.model.in_channels, 1, 1))
        self.sync_free = bool(cfg.train.get("sync_free", False))
        self.loss = CBMLoss(cfg.loss, label_dim=cfg.model.label_dim, sync_free=self.sync_free)
        self._guide_weak_box = []   # loss.guide.autoguide's frozen weak model, in a list so it is no submodule (_guide_weak)

    def forward(self, x, labels, omega=None):
        return self.model(x, labels, omega=omega)

    def on_train_start(self):
        if self.cfg.train.per_rank_seed:
            # every rank its own stream (seed_everything seeds all ranks alike); the global step is folded in so a
            # resumed run does not replay the same draws
            seed = int(self.cfg.seed) + 7919 * (self.global_rank + 1) + int(self.global_step)
            torch.manual_seed(seed)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        losses, aux = self.loss(self.model, x, labels, self.global_step, teacher=self._guide_teacher(), weak=self._guide_weak())
        for k, v in {**losses, **aux}.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=k in losses)
        total = sum(losses.values())
        self.log("train/total_loss", total, on_step=True, prog_bar=True)
        return total

    def _guide_teacher(self):
        """loss.guide.teacher_decay = d: the guide reads from the EMA-d copy (the guided target's label component is the
        map's own estimate; the online map's SGD noise in it is fed back and amplified by 1 / (1 - gamma kappa)).
        None: the online map (iMF's rule as published)."""
        d = (self.cfg.loss.get("guide") or {}).get("teacher_decay")
        if d is None:
            return None
        from sbm.ema import EMA, ema_label
        ema = next((c for c in self.trainer.callbacks if isinstance(c, EMA)), None)
        if ema is None or ema_label(d) not in ema._models:
            raise ValueError(f"loss.guide.teacher_decay {d} needs train.ema.decays to contain it")
        teacher = ema._models[ema_label(d)]
        refresh = (self.cfg.loss.get("guide") or {}).get("teacher_refresh")
        if refresh:
            # a frozen teacher (decay 1.0 copy) refreshed from the active EMA every `teacher_refresh` steps: the guided
            # target's self-term then moves in jumps the online map cannot re-lock onto (the SBM recipes: 1.0 / 10000).
            # The copy is also filled at its first use (a decay-1.0 copy new to a resumed run would otherwise be the
            # random init until the first EMA update, or the raw weights after it).
            import itertools
            step = int(self.global_step)
            if int(teacher.n_averaged) == 0 or (step % int(refresh) == 0 and getattr(self, "_teacher_refresh_step", -1) != step):
                src = ema._models[ema.active_label]
                with torch.no_grad():
                    torch._foreach_copy_([t for t in itertools.chain(teacher.module.parameters(), teacher.module.buffers())],
                                         [t for t in itertools.chain(src.module.parameters(), src.module.buffers())])
                teacher.n_averaged.fill_(1)   # AveragedModel would otherwise overwrite it with the raw weights at its first update
                self._teacher_refresh_step = step
                self.log("train/teacher_refresh_step", float(step), on_step=True, rank_zero_only=True)
        return teacher.module.model

    def _guide_weak(self):
        """loss.guide.autoguide (Karras et al. 2024): the frozen WEAK model whose reads the autoguidance term subtracts.
        The run's architecture (sbm.models.build_model) with the weights of an earlier checkpoint of the line
        (autoguide.ckpt; copy autoguide.which = ema | raw | ema_<label>, resolved as init.weights is), loaded once at its first
        use, no gradient, eval mode, on the device, never updated or refreshed; a resume rebuilds it from the same file. Held
        in a list, so it is no submodule (not in the state dict, the optimizer, DDP or the EMA copies: a resume is unchanged).
        None when autoguidance is off."""
        if not self.loss.ag:
            return None
        if not self._guide_weak_box:
            src = resolve_source(self.loss.ag_ckpt)
            weak = load_weights(build_model(**self.cfg.model), src, self.loss.ag_which)
            weak.requires_grad_(False)
            weak.eval()
            self._guide_weak_box.append(weak.to(self.device))
            print(f"[autoguide] weak model from {src} [{self.loss.ag_which}], weight {self.loss.ag_weight} "
                  f"(kappa {self.loss.g_kappa}); frozen, never refreshed", flush=True)
        return self._guide_weak_box[0]

    def on_before_optimizer_step(self, optimizer):
        if self.sync_free:   # two foreach kernels instead of three per parameter (same norm up to summation order)
            norms = torch._foreach_norm([p.grad.detach().float() for p in self.parameters() if p.grad is not None])
            self.log("train/grad_norm", torch.linalg.vector_norm(torch.stack(norms)), on_step=True)
            return
        sq = sum(p.grad.detach().float().pow(2).sum() for p in self.parameters() if p.grad is not None)
        self.log("train/grad_norm", torch.sqrt(sq), on_step=True)

    def on_save_checkpoint(self, checkpoint):
        state = self.loss.state_dict()
        if state:                       # only the EMA balancer carries loss state (loss.balance.mode=ema)
            checkpoint[LOSS_STATE_KEY] = state

    def on_load_checkpoint(self, checkpoint):
        if self.model.gw_embedder is not None:
            self._add_gw_embedder(checkpoint)
        if LOSS_STATE_KEY in checkpoint:
            self.loss.load_state_dict(checkpoint[LOSS_STATE_KEY])

    def _add_gw_embedder(self, checkpoint):
        """A checkpoint written before the guidance input: give its weights (active EMA, raw, every other EMA copy) this
        module's fresh embedder (zero output, so every copy computes what it did) and re-index its optimizer state for
        the new parameters (fresh state for them). Lightning calls this before any strict load reads the dict."""
        own = self.state_dict()
        added = fill_gw_keys(own, checkpoint["state_dict"], prefix="model.")
        if not added:
            return
        for sd in [checkpoint.get("current_model_state"), *checkpoint.get("ema_state_dicts", {}).values()]:
            if sd is not None:
                fill_gw_keys(own, sd, prefix="model.")
        states = checkpoint.get("optimizer_states") or []
        if states:
            extend_optimizer_state(states[0], split_params(self.model, self.cfg.optim), [k[len("model."):] for k in added])
        print(f"added the guidance embedder to a checkpoint without it ({len(added)} tensors, output zero)")

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        losses, _ = self.loss(self.model, x, labels, self.global_step, weak=self._guide_weak())
        for k, v in losses.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = build_optimizer(self.model, self.cfg.optim)
        self._peak = [g["lr"] for g in opt.param_groups]
        sched = GlobalStepLR(opt, lambda _: lr_factor(self.cfg.optim, self._step_or_zero()))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    def _step_or_zero(self):
        try:
            return int(self.trainer.global_step)
        except Exception:
            return 0


class EvalAtSteps(Callback):
    """Evaluate the in-memory model at given global steps with a named protocol (single process), into evals/."""

    def __init__(self, run_dir, protocol, steps):
        self.run_dir, self.protocol, self.steps = Path(run_dir), protocol, set(int(s) for s in steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps and trainer.is_global_zero:
            from sbm.eval import evaluate_module
            was = pl_module.training
            evaluate_module(pl_module, self.protocol, run_dir=self.run_dir, label=f"step{trainer.global_step:07d}")
            pl_module.train(was)


# ---------------------------------------------------------------------- entry point
def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    found = sorted(Path(ckpt_dir).glob("step*.ckpt"), key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    return found[-1] if found else None


def resolve_source(src: str) -> Path:
    """A checkpoint file, or a run folder (its latest checkpoint)."""
    p = Path(str(src))
    if p.is_dir():
        c = latest_checkpoint(p / "ckpt")
        if c is None:
            raise FileNotFoundError(f"no checkpoint in {p}/ckpt")
        return c
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def world(cfg):
    """(GPUs per node, nodes): torchrun's LOCAL_WORLD_SIZE / WORLD_SIZE when set, else train.devices on one node."""
    if os.environ.get("LOCAL_WORLD_SIZE"):
        local = int(os.environ["LOCAL_WORLD_SIZE"])
        return local, max(int(os.environ.get("WORLD_SIZE", local)) // local, 1)
    devices = max(torch.cuda.device_count(), 1) if cfg.train.devices == "auto" else int(cfg.train.devices)
    return devices, 1


def train(cfg: DictConfig, run_dir: str | Path):
    run_dir = Path(run_dir)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.tf32)
    torch.backends.cudnn.allow_tf32 = True
    if cfg.train.tf32:
        torch.set_float32_matmul_precision("high")
    pl.seed_everything(int(cfg.seed), workers=True)
    devices, nodes = world(cfg)

    model = build_model(**cfg.model)
    module = Module(cfg, model)
    ckpt_path = latest_checkpoint(run_dir / "ckpt")
    from_parent = False
    if ckpt_path is None and cfg.init.get("from"):
        src = resolve_source(cfg.init["from"])
        if cfg.init.resume_state:
            ckpt_path, from_parent = src, True
        else:
            load_weights(model, src, cfg.init.weights)
            print(f"initialised weights from {src} [{cfg.init.weights}]")
    max_steps = int(cfg.train.max_steps)
    if from_parent and cfg.train.get("steps_after_init"):
        max_steps = int(torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)["global_step"]) + int(cfg.train.steps_after_init)

    every = int(cfg.train.checkpoint_every)
    local = cfg.train.get("stage_checkpoints_locally")
    callbacks = [StagedCheckpoint(dirpath=run_dir / "ckpt", filename="step{step:07d}", auto_insert_metric_name=False,
                                  every_n_train_steps=every, save_top_k=-1, save_last=False,
                                  local_dir=str(Path(str(local)) / run_dir.name) if local else None),
                 LearningRateMonitor(logging_interval="step")]
    if cfg.train.ema.decays:
        callbacks.append(EMA(cfg.train.ema.decays, cfg.train.ema.active, cfg.train.ema.min_step,
                             foreach=bool(cfg.train.ema.get("foreach", False))))
    monitor = cfg.train.get("monitor") or {}
    if int(monitor.get("every", 0) or 0) > 0:
        from sbm.monitor import FIDMonitor
        callbacks.append(FIDMonitor(run_dir, **OmegaConf.to_container(monitor, resolve=True)))
    if int(cfg.train.get("step_timer", 0) or 0) > 0:
        from sbm.monitor import StepTimer
        callbacks.append(StepTimer(run_dir, int(cfg.train.step_timer), images_per_step=int(cfg.train.global_batch)))
    if ckpt_path is not None and from_parent and cfg.optim.get("force_on_resume", True):
        callbacks.append(ForceLearningRate(_peak_lrs(model, cfg)))
    if cfg.train.get("eval") and cfg.train.eval.get("protocol"):
        callbacks.append(EvalAtSteps(run_dir, cfg.train.eval.protocol, cfg.train.eval.steps))

    logger = WandbLogger(**R.wandb_kwargs(cfg, run_dir))
    per_step = int(cfg.train.per_device_batch) * devices * nodes
    if int(cfg.train.global_batch) % per_step:
        raise ValueError(f"train.global_batch {cfg.train.global_batch} is not a multiple of {per_step}")
    accumulate = int(cfg.train.global_batch) // per_step
    strategy = DDPStrategy(timeout=timedelta(minutes=120)) if devices * nodes > 1 else "auto"
    # the processes come from torchrun, or else Lightning starts one per GPU (also inside a Slurm allocation, whose
    # environment Lightning would otherwise take as the process layout)
    from lightning.fabric.plugins.environments import LightningEnvironment, TorchElasticEnvironment
    env = TorchElasticEnvironment() if os.environ.get("LOCAL_WORLD_SIZE") else LightningEnvironment()
    trainer = pl.Trainer(default_root_dir=str(run_dir), logger=logger, callbacks=callbacks, max_steps=max_steps, max_epochs=-1,
                         plugins=[env],
                         accelerator="auto", devices=devices, num_nodes=nodes, strategy=strategy, precision=cfg.train.precision,
                         accumulate_grad_batches=accumulate, gradient_clip_val=cfg.optim.get("clip"),
                         log_every_n_steps=int(cfg.train.log_every),
                         val_check_interval=None if not cfg.train.get("val_every") else int(cfg.train.val_every),
                         check_val_every_n_epoch=None, limit_val_batches=int(cfg.train.get("val_batches", 8)),
                         num_sanity_val_steps=0, use_distributed_sampler=False, enable_progress_bar=False)
    if trainer.is_global_zero:
        (run_dir / "wandb" / "url.txt").write_text(str(getattr(logger.experiment, "url", "") or ""))
    trainer.fit(module, ckpt_path=str(ckpt_path) if ckpt_path else None, datamodule=LatentData(cfg))


def _peak_lrs(model, cfg):
    from sbm.optim import build_optimizer
    return [g["lr"] for g in build_optimizer(model, cfg.optim).param_groups]
