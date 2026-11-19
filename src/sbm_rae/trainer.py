import math
import os
import re
import time
from contextlib import nullcontext

import lightning as pl
import torch
import torch.nn as nn
import torchvision
import wandb
from diffusers import AutoencoderKL
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from sbm_rae.ema import ema_decay_label, resolve_active_ema_decay
from sbm_rae.loss import resolve_anchor_alpha, resolve_guidance_config
from sbm_rae.guided_cbm import scale_cond_lr_mult
from sbm_rae.optim import ChainedOptimizer
from sbm_rae.sampling import build_overfitting_class_labels


def overfitting_invariant_plot_enabled(cfg):
    overfitting_cfg = cfg.trainer.get("overfitting", {})
    if not bool(overfitting_cfg.get("enabled", False)):
        return False

    plot_cfg = overfitting_cfg.get("invariant_plot", {}) or {}
    if isinstance(plot_cfg, bool):
        return plot_cfg
    return bool(plot_cfg.get("enabled", True))


def train_interpolation_plot_enabled(cfg):
    sampling_cfg = cfg.get("sampling", {})
    plot_cfg = sampling_cfg.get("train_interpolation_plot", {}) or {}
    if isinstance(plot_cfg, bool):
        return plot_cfg
    return bool(plot_cfg.get("enabled", False))


def _ema_log_prefix(base, cfg):
    ema_cfg = cfg.trainer.get("ema", {})
    if not bool(ema_cfg.get("enabled", False)):
        return base
    active_decay = resolve_active_ema_decay(ema_cfg)
    return f"{base}_ema_{ema_decay_label(active_decay)}"


def _channel_buffer(value, channels):
    if isinstance(value, (int, float)):
        values = [float(value)] * channels
    else:
        values = list(value)
        if len(values) != channels:
            raise ValueError(
                f"Expected {channels} latent affine values, got {len(values)}"
            )
    return torch.tensor(values, dtype=torch.float32).view(1, channels, 1, 1)


def _clone_batch_to_cpu(batch):
    if torch.is_tensor(batch):
        return batch.detach().cpu().clone()
    if isinstance(batch, tuple):
        return tuple(_clone_batch_to_cpu(item) for item in batch)
    if isinstance(batch, list):
        return [_clone_batch_to_cpu(item) for item in batch]
    if isinstance(batch, dict):
        return {key: _clone_batch_to_cpu(value) for key, value in batch.items()}
    return batch


def _unpack_batch(batch):
    if isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("x", batch.get("latents")))
        labels = batch.get("labels", batch.get("y"))
        repa_targets = batch.get("repa_targets", batch.get("representations"))
        if inputs is None or labels is None:
            raise ValueError("Batch dict must contain inputs/x/latents and labels/y")
        return inputs, labels, repa_targets

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Expected batch with at least 2 items, got {len(batch)}")
        repa_targets = batch[2] if len(batch) > 2 else None
        return batch[0], batch[1], repa_targets

    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def cosine_factor(step, start, end, floor=0.0):
    """Multiplier on the peak rate: 1 before `start`, cosine to `floor` at `end`, `floor` after."""
    if step < start:
        return 1.0
    if step >= end:
        return floor
    progress = (step - start) / float(end - start)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class GlobalStepLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    """A LambdaLR whose lambda reads trainer.global_step, so a checkpoint's saved
    scheduler state is irrelevant to it. Every checkpoint written under the
    constant schedule carries a ConstantLR state, and stock LambdaLR raises
    KeyError('lr_lambdas') when Lightning loads that on resume. This accepts any
    state: it keeps its own lambda and takes only the scalar counters."""

    def load_state_dict(self, state_dict):
        incoming = dict(state_dict)
        incoming.pop("lr_lambdas", None)
        for key in ("last_epoch", "_step_count", "_last_lr"):
            if key in incoming:
                setattr(self, key, incoming[key])


# freeze.trainable: train a part of the network, the rest frozen (requires_grad False: out of the optimizer,
# DDP and the backward). Presets name parameter-name regexes on model.<name>; any other value is a regex itself. A resume
# from a checkpoint of the full model keeps the trainable parameters' optimizer state (_drop_frozen_optimizer_state).
FREEZE_PRESETS = {"head": r"^(head_blocks\.|head_x_embedder\.|s_projector\.|final_layer\.)"}   # the DDT head
FREEZE_PRESETS["head_last4"] = r"^(head_blocks\.|head_x_embedder\.|s_projector\.|final_layer\.|blocks\.(1[7-9]|20)\.)"   # the head + the
# last four of the 21 trunk blocks of DiT-XL/1-DH (model.blocks.0-20): a partial unfreeze between head-only, which re-adapts,
# and the full model, which degrades.


def resolve_freeze_regex(cfg):
    spec = (cfg.get("freeze", None) or {}).get("trainable", None)
    if spec is None or str(spec).lower() in ("", "null", "none"):
        return None
    return FREEZE_PRESETS.get(str(spec), str(spec))


def apply_freeze(model, regex):
    """requires_grad False on every parameter of `model` whose name does not match `regex`; returns the ids of the
    parameters this switched off (those frozen by design, e.g. the fixed position embeddings, were never in the optimizer)."""
    pattern = re.compile(regex)
    kept, frozen_ids = [], set()
    for name, param in model.named_parameters():
        if pattern.search(name):
            kept.append(name)
        elif param.requires_grad:
            frozen_ids.add(id(param))
            param.requires_grad_(False)
    if not kept:
        raise ValueError(f"freeze.trainable: no parameter matches {regex!r}")
    n_kept = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print(f"[freeze] trainable: {len(kept)} tensors, {n_kept / 1e6:.1f} M parameters (e.g. {kept[0]}); "
          f"frozen: {len(frozen_ids)} tensors ({regex})", flush=True)
    return frozen_ids


class TrainingModule(pl.LightningModule):
    """Minimal Lightning Module for training a latent generative model."""
    def __init__(self, cfg, model, loss_fn, weighting_model=None):
        super().__init__()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=False)}, logger=False)
        self.cfg = cfg
        self.model = model
        self._freeze_regex = resolve_freeze_regex(cfg)
        self._freeze_frozen_ids = set()
        if self._freeze_regex is not None:
            self._freeze_frozen_ids = apply_freeze(model, self._freeze_regex)
        self.loss_fn = loss_fn
        self.weighting_model = weighting_model
        self.use_precomputed_latents = bool(cfg.dataset.get("use_precomputed_latents", False))
        self._vae_container = []
        self._vae_cfg = cfg.get("vae", {}) or {}
        # Pixel-space data (vae.enabled=false) trains directly on RGB, so there is no
        # encoder/decoder in the loop at all.
        self.vae_enabled = bool(self._vae_cfg.get("enabled", True))
        self._overfitting_enabled = bool(cfg.trainer.get("overfitting", {}).get("enabled", False))
        self._overfitting_class_label = None
        # loss.name=guided_cbm (sbm_rae.guided_cbm): the frozen teacher lives in a plain list, so it is no submodule (not
        # in the state dict, the optimizer, DDP or the EMA copies); it rides in the checkpoint under its own key
        self._guide_teacher_box = []
        self._guide_weak_box = []     # autoguidance's frozen weak model (loss.guided_cbm.guide.autoguide), not a submodule
        self._guide_teacher_step = None
        self._guide_teacher_pending = None
        if getattr(loss_fn, "tangent_only_jvp", False):
            from sbm_rae.models.tangent_only import use_tangent_only_jvp

            use_tangent_only_jvp(self.model)
        
        # Scaling parameters specific to SD VAEs
        in_channels = cfg.model.in_channels
        self.register_buffer(
            'latents_scale',
            _channel_buffer(self._vae_cfg.get("latents_scale", 0.18215), in_channels),
        )
        self.register_buffer(
            'latents_bias',
            _channel_buffer(self._vae_cfg.get("latents_bias", 0.0), in_channels),
        )
        self._last_train_batch_end_time = None
        self._current_train_batch_start_time = None
        self._current_train_data_time = None

    @property
    def vae(self):
        if not self.vae_enabled:
            return None
        if not self._vae_container and str(self._vae_cfg.get("kind", "sdvae")) == "rae":
            # DINOv2 representation latent with the released RAE decoder (sbm_rae.rae_adapter): same interface as the SD-VAE
            from sbm_rae.rae_adapter import RAEAutoencoder
            print("Initializing RAE (DINOv2-B-reg encoder, ViT-XL decoder) for latent space operations...")
            self._vae_container.append(RAEAutoencoder(decoder=str(self._vae_cfg.get("rae_decoder", "ViTXL_n08")),
                                                      size=str(self._vae_cfg.get("rae_size", "base"))))
        if not self._vae_container:
            print("Initializing VAE for latent space operations...")
            kw = {"local_files_only": self._vae_cfg.get("local_files_only", False)}
            if self._vae_cfg.get("subfolder", None):   # e.g. the SD3 VAE lives under vae/ of its diffusers repo
                kw["subfolder"] = str(self._vae_cfg["subfolder"])
            vae = AutoencoderKL.from_pretrained(
                self._vae_cfg.get("pretrained_model_name_or_path", "stabilityai/sd-vae-ft-mse"), **kw
            )
            vae.eval()
            for parameter in vae.parameters():
                parameter.requires_grad = False
            self._vae_container.append(vae)
        return self._vae_container[0] if self._vae_container else None

    def _should_preload_vae(self):
        if not self.vae_enabled:
            return False
        return (
            not self.use_precomputed_latents
            or bool(self.cfg.get("sampling", {}).get("enabled", False))
            or train_interpolation_plot_enabled(self.cfg)
            or overfitting_invariant_plot_enabled(self.cfg)
            or self.cfg.get("fid", None) is not None
        )

    def setup(self, stage: str):
        del stage
        if self._should_preload_vae():
            self.vae.to(self.device)

    def forward(self, x, labels=None, omega=None, w=None):
        output = self.model(x, labels, omega=omega, w=w)
        if isinstance(output, tuple):
            return output[0]
        return output

    def on_train_start(self):
        if not bool(self.cfg.get("per_rank_seed", False)):
            return
        # Also fold in the global step so a requeue does not replay the same stream.
        seed = int(self.cfg.seed) + 7919 * (int(self.global_rank) + 1) + int(self.global_step)
        torch.manual_seed(seed)
        print(f"per_rank_seed: rank {self.global_rank} torch seed {seed}")

    def on_train_epoch_start(self):
        self._last_train_batch_end_time = None
        self._current_train_batch_start_time = None
        self._current_train_data_time = None

    def on_train_batch_start(self, batch, batch_idx):
        del batch, batch_idx
        batch_start_time = time.perf_counter()
        self._current_train_batch_start_time = batch_start_time
        if self._last_train_batch_end_time is None:
            self._current_train_data_time = None
            return
        self._current_train_data_time = batch_start_time - self._last_train_batch_end_time

    def _compute_grad_norm(self):
        grad_norm_sq = torch.zeros((), device=self.device, dtype=torch.float32)

        for param in self.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach().float()
            grad_norm_sq += grad.pow(2).sum()

        return torch.sqrt(grad_norm_sq)

    def _trainable_parameters(self):
        return tuple(param for param in self.parameters() if param.requires_grad)

    def _split_muon_parameters(self, include_frozen=False):
        muon_params = []
        adamw_params = []

        for _, module in self.named_modules():
            for _, param in module.named_parameters(recurse=False):
                if not param.requires_grad and not (include_frozen and id(param) in self._freeze_frozen_ids):
                    continue
                if param.ndim == 2 and not isinstance(module, nn.Embedding):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        return muon_params, adamw_params

    def _compute_loss_grad_norm(self, loss, parameters):
        grad_norm_sq = torch.zeros((), device=loss.device, dtype=torch.float32)
        if not loss.requires_grad or not parameters:
            return grad_norm_sq

        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for gradient in gradients:
            if gradient is None:
                continue
            grad_norm_sq += gradient.detach().float().pow(2).sum()
        return torch.sqrt(grad_norm_sq)

    def _should_log_loss_grad_norms(self):
        logging_cfg = self.cfg.get("logging", {})
        loss_grad_norm_cfg = logging_cfg.get("loss_grad_norms", {})
        if isinstance(loss_grad_norm_cfg, bool):
            enabled = loss_grad_norm_cfg
            every_n_steps = int(self.cfg.trainer.get("log_every_n_steps", 1))
        else:
            enabled = bool(loss_grad_norm_cfg.get("enabled", False))
            every_n_steps = int(
                loss_grad_norm_cfg.get(
                    "every_n_steps",
                    self.cfg.trainer.get("log_every_n_steps", 1),
                )
            )
            
        should_log = enabled and every_n_steps > 0 and (self.global_step + 1) % every_n_steps == 0
        
        return should_log

    def _log_loss_grad_norms(self, losses):
        if not self._should_log_loss_grad_norms():
            return

        parameters = self._trainable_parameters()
        for name, loss in losses.items():
            self.log(
                f"train/grad_norm/{name}",
                self._compute_loss_grad_norm(loss, parameters),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def _encode_to_latents(self, x):
        if self.use_precomputed_latents or not self.vae_enabled:
            return x

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x_vae = x.to(dtype=self.vae.dtype)
                latents = self.vae.encode(x_vae).latent_dist.sample()
                latents = (latents - self.latents_bias) * self.latents_scale
                return latents.to(dtype=x.dtype)

    def _guide_weak(self):
        """loss.guided_cbm.guide.autoguide's frozen weak model: the model's architecture with the weights of an earlier
        checkpoint (its active EMA unless autoguide.ema), loaded once; held in a list, so it is not a submodule (not
        trained, averaged or saved). None when autoguidance is off."""
        if not getattr(self.loss_fn, "ag", False):
            return None
        if not self._guide_weak_box:
            import copy

            from sbm_rae.guided_cbm import load_weak_state

            weak = copy.deepcopy(self.model)
            state = load_weak_state(self.loss_fn.ag_ckpt, self.loss_fn.ag_ema)
            if getattr(weak, "guidance_enabled", False) and not any(k.startswith("w_embedder.") for k in state):
                # a weak checkpoint from before guide.scale_cond: its network had no scale input, so the copy's scale
                # embedding is zeroed (its output layer), which makes the weak model ignore w exactly
                missing, unexpected = weak.load_state_dict(state, strict=False)
                if unexpected or any(not k.startswith("w_embedder.") for k in missing):
                    raise RuntimeError(f"autoguide weak model: unexpected {unexpected}, missing {missing}")
                torch.nn.init.zeros_(weak.w_embedder.mlp[2].weight)
                torch.nn.init.zeros_(weak.w_embedder.mlp[2].bias)
            else:
                weak.load_state_dict(state, strict=True)
            weak.requires_grad_(False)
            weak.eval()
            self._guide_weak_box.append(weak.to(self.device))
            print(f"[guided_cbm] autoguide: weak model from {self.loss_fn.ag_ckpt} (EMA "
                  f"{'active' if self.loss_fn.ag_ema is None else self.loss_fn.ag_ema}), weight {self.loss_fn.ag_weight}",
                  flush=True)
        return self._guide_weak_box[0]

    def _guide_teacher(self):
        """loss.guided_cbm's frozen teacher: a copy of the active EMA made at guide.start_step and refreshed from it every
        teacher_refresh steps after; None before the start."""
        schedule = self.loss_fn.teacher_schedule()
        if schedule is None:
            return None
        start, refresh = schedule
        step = int(self.global_step)
        if step < start or not refresh:
            return None
        if not self.training:   # validation (the EMA callback has swapped weights): read the teacher, never copy it
            return self._guide_teacher_box[0] if self._guide_teacher_box else None
        from sbm_rae.ema import EMAWeightAveraging

        ema = next((c for c in self.trainer.callbacks if isinstance(c, EMAWeightAveraging)), None)
        if ema is None:
            raise ValueError("loss.guided_cbm's teacher is a copy of the active EMA: enable trainer.ema")
        source_decay = getattr(self.loss_fn, "teacher_source", None)
        source_model = ema.get_average_model(decay=source_decay)
        if source_model is None:
            raise ValueError(f"loss.guided_cbm.guide.teacher_source {source_decay} is not one of trainer.ema.decays")
        source = source_model.module.model
        src_label = "active EMA" if source_decay is None else f"EMA-{source_decay:g}"
        box = self._guide_teacher_box
        if not box:
            import copy

            teacher = copy.deepcopy(source)
            teacher.requires_grad_(False)
            teacher.eval()
            box.append(teacher)
            if self._guide_teacher_pending is not None:   # a resume: the teacher the run had, not a fresh copy
                state, saved_step = self._guide_teacher_pending
                teacher.load_state_dict(state)
                self._guide_teacher_step, self._guide_teacher_pending = saved_step, None
                print(f"[guided_cbm] step {step}: teacher restored from the checkpoint (copied at step {saved_step})")
            else:
                self._guide_teacher_step = step
                print(f"[guided_cbm] step {step}: teacher copied from the {src_label}")
        if getattr(self.loss_fn, "teacher_refresh_at_start", False) and not getattr(self, "_teacher_start_refreshed", False):
            self._teacher_start_refreshed = True
            if self._guide_teacher_step != step:
                self._guide_teacher_step = None   # forces the refresh below at this step
        teacher = box[0]
        if self._guide_teacher_step is None or ((step - start) % int(refresh) == 0 and self._guide_teacher_step != step):
            with torch.no_grad():
                for dst, src in zip(teacher.state_dict().values(), source.state_dict().values()):
                    dst.copy_(src)
            self._guide_teacher_step = step
            print(f"[guided_cbm] step {step}: teacher refreshed from the {src_label}")
        self.log("train/teacher_copy_step", float(self._guide_teacher_step), on_step=True, rank_zero_only=True)
        return teacher

    def on_save_checkpoint(self, checkpoint):
        if self._guide_teacher_box:
            teacher = self._guide_teacher_box[0]
            checkpoint["guided_cbm_teacher"] = {
                "state": {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()},
                "step": self._guide_teacher_step,
            }
        if hasattr(self.loss_fn, "teacher_schedule"):
            checkpoint["guided_cbm_loss"] = self.loss_fn.state_dict()

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("guided_cbm_teacher")
        if saved is not None:
            self._guide_teacher_pending = (saved["state"], saved.get("step"))
        if "guided_cbm_loss" in checkpoint and hasattr(self.loss_fn, "teacher_schedule"):
            self.loss_fn.load_state_dict(checkpoint["guided_cbm_loss"])
        if self._freeze_regex is not None and checkpoint.get("optimizer_states"):
            self._drop_frozen_optimizer_state(checkpoint)
        if scale_cond_lr_mult(self.cfg) != 1.0 and checkpoint.get("optimizer_states"):
            self._split_scale_lr_groups(checkpoint)

    def _drop_frozen_optimizer_state(self, checkpoint):
        """freeze.trainable on a resume from a checkpoint of the FULL model: the saved (chained Muon + AdamW) optimizer
        state indexes every parameter in _split_muon_parameters order (one group per optimizer); keep the entries of the
        still-trainable parameters, renumbered to the groups _configure_muon_optimizer builds now. A state that already
        has the reduced size (a requeue or time-limit leg of the frozen run itself) is left alone."""
        if scale_cond_lr_mult(self.cfg) != 1.0:
            raise ValueError("freeze.trainable does not combine with scale_cond.lr_mult on a resume")
        states = checkpoint["optimizer_states"]
        if len(states) != 1 or "optimizers" not in states[0] or len(states[0]["optimizers"]) != 2:
            raise ValueError("freeze.trainable resumes only the chained Muon + AdamW optimizer state")
        chained = states[0]["optimizers"]
        kept_sets = self._split_muon_parameters()
        for i, all_params in enumerate(self._split_muon_parameters(include_frozen=True)):
            kept_ids = {id(p) for p in kept_sets[i]}
            sd = chained[i]
            groups = sd["param_groups"]
            if len(groups) != 1:
                raise ValueError(f"optimizer {i}: freeze.trainable expects one param group, got {len(groups)}")
            saved = list(groups[0]["params"])
            if saved == list(range(len(kept_ids))):
                continue                                              # already reduced (a leg of the frozen run)
            if saved != list(range(len(all_params))):
                raise ValueError(f"optimizer {i}: expected param ids 0..{len(all_params) - 1} or 0..{len(kept_ids) - 1}, "
                                 f"got {len(saved)}")
            old_idx = [j for j, p in enumerate(all_params) if id(p) in kept_ids]
            g0 = dict(groups[0])
            g0["params"] = list(range(len(old_idx)))
            chained[i] = {"state": {new: sd["state"][old] for new, old in enumerate(old_idx) if old in sd["state"]},
                          "param_groups": [g0]}
            print(f"[freeze] optimizer {i}: state reduced from {len(all_params)} to {len(old_idx)} parameters", flush=True)

    def _scale_embedding_param_ids(self):
        """ids of the guidance-scale embedding's trainable parameters (model.w_embedder, guide.scale_cond)."""
        ids = {id(p) for n, p in self.model.named_parameters() if p.requires_grad and ("." + n).count(".w_embedder.")}
        if not ids:
            raise ValueError("loss.guided_cbm.guide.scale_cond.lr_mult needs the model's w_embedder (guidance_enabled)")
        return ids

    def _split_scale_lr_groups(self, checkpoint):
        """scale_cond.lr_mult != 1: bring a saved (chained Muon + AdamW) optimizer state to the param groups
        _configure_muon_optimizer builds. A state written without the knob (one group per optimizer, e.g. a fork of a
        run trained at 1x) is split into the other parameters, then the guidance-scale embedding at lr_mult x the
        group's rate, every parameter keeping its state; the scheduler's per-group lists follow. A state already split
        gets its embedding groups' rates set to lr_mult x the main groups' (the knob, not the checkpoint, decides)."""
        mult = scale_cond_lr_mult(self.cfg)
        states = checkpoint["optimizer_states"]
        if len(states) != 1 or "optimizers" not in states[0] or len(states[0]["optimizers"]) != 2:
            raise ValueError("scale_cond.lr_mult resumes only the chained Muon + AdamW optimizer state")
        w_ids = self._scale_embedding_param_ids()
        chained = states[0]["optimizers"]
        split = False
        for i, params in enumerate(self._split_muon_parameters()):
            sd = chained[i]
            groups = sd["param_groups"]
            emb = [j for j, p in enumerate(params) if id(p) in w_ids]
            if not emb:
                continue
            if len(groups) == 2:
                groups[1]["lr"] = groups[0]["lr"] * mult
                if "initial_lr" in groups[0]:
                    groups[1]["initial_lr"] = groups[0]["initial_lr"] * mult
                continue
            if len(groups) != 1 or list(groups[0]["params"]) != list(range(len(params))):
                raise ValueError(f"optimizer {i}: expected one param group with ids 0..{len(params) - 1}")
            rest = [j for j in range(len(params)) if id(params[j]) not in w_ids]
            new_id = {old: new for new, old in enumerate(rest + emb)}
            g0, g1 = dict(groups[0]), dict(groups[0])
            g0["params"], g1["params"] = list(range(len(rest))), list(range(len(rest), len(params)))
            g1["lr"] = groups[0]["lr"] * mult
            if "initial_lr" in g1:
                g1["initial_lr"] = groups[0]["initial_lr"] * mult
            chained[i] = {"state": {new_id[k]: v for k, v in sd["state"].items()}, "param_groups": [g0, g1]}
            split = True
        if split:
            # per-group scheduler lists, in ChainedOptimizer order (Muon groups, then AdamW groups): [m, a] -> [m, m', a, a']
            for sch in checkpoint.get("lr_schedulers") or []:
                for s in [sch] + list(sch.get("_schedulers", None) or []):
                    for key in ("base_lrs", "_last_lr"):
                        v = s.get(key)
                        if isinstance(v, list) and len(v) == 2:
                            s[key] = [v[0], v[0] * mult, v[1], v[1] * mult]
            print(f"[scale_cond] optimizer state split: the guidance-scale embedding gets its own groups at {mult:g}x",
                  flush=True)

    def _compute_losses(self, x, labels, repa_targets=None):
        if hasattr(self.loss_fn, "teacher_schedule"):
            teacher = self._guide_teacher()
            # speed (sbm_rae.speed): optional torch.compile of the guided-CBM calls (trainer.compile.*; off by default)
            c = self.cfg.trainer.get("compile", None) or {}
            if self.training and any(bool(c.get(k, False)) for k in ("teacher", "student_fwd", "student_jvp")):
                if getattr(self, "_compiled_calls", None) is None:
                    from sbm_rae.speed import CompiledCalls

                    self._compiled_calls = CompiledCalls(self.cfg.trainer.compile, self.model)
                    self._compiled_calls.install(self.loss_fn)
                    print(f"[speed] torch.compile on: teacher={self._compiled_calls.do_teacher} "
                          f"student_fwd={self._compiled_calls.do_fwd} student_jvp={self._compiled_calls.do_jvp} "
                          f"mode={self._compiled_calls.mode}", flush=True)
                teacher = self._compiled_calls.teacher(teacher)
            self.loss_fn.teacher = teacher
            if getattr(self.loss_fn, "ag", False):
                weak = self._guide_weak()
                if getattr(self, "_compiled_calls", None) is not None and self.training:
                    weak = self._compiled_calls.teacher(weak)   # the same compiled no-grad read as the teacher
                self.loss_fn.weak = weak
        if repa_targets is None:
            return self.loss_fn(
                self.model,
                self.weighting_model,
                x,
                labels,
                self.global_step,
            )

        return self.loss_fn(
            self.model,
            self.weighting_model,
            x,
            labels,
            self.global_step,
            repa_targets=repa_targets,
        )

    def _capture_overfitting_label(self, labels):
        if not self._overfitting_enabled or self._overfitting_class_label is not None:
            return

        if labels is None:
            return

        label_tensor = labels.detach().reshape(-1)
        if label_tensor.numel() == 0:
            return

        self._overfitting_class_label = int(label_tensor[0].item())

    def training_step(self, batch, batch_idx):
        del batch_idx
        x, labels, repa_targets = _unpack_batch(batch)
        self._capture_overfitting_label(labels)
        if not self.use_precomputed_latents:
            x = self._encode_to_latents(x)

        # loss_fn handles the core training objective
        losses, aux_losses = self._compute_losses(x, labels, repa_targets=repa_targets)
        
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Raw (unweighted) MSEs also on the progress bar: with WANDB_MODE=disabled the progress-bar line in the log is the
        # only loss trace a run leaves.
        for name, value in aux_losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False, prog_bar=name.endswith("_mse"), logger=True)

        self._log_loss_grad_norms(losses)
            
        total_loss = sum(losses.values())
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)
        
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # The four timing metrics below are logged rank-locally (sync_dist=False): they are per-rank wall-clock
        # measurements, and reducing them across ranks would add a collective on the logging path (a source of
        # multi-node deadlocks at the end of fit). Real losses keep sync_dist=True in validation_step.
        del outputs, batch_idx
        batch_end_time = time.perf_counter()

        if self._current_train_batch_start_time is not None:
            compute_time = batch_end_time - self._current_train_batch_start_time
            self.log(
                "train/compute_time",
                compute_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

            data_time = self._current_train_data_time
            if data_time is not None:
                step_time = data_time + compute_time
                self.log(
                    "train/data_time",
                    data_time,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "train/step_time",
                    step_time,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "train/data_fraction",
                    data_time / step_time if step_time > 0 else 0.0,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

        self._last_train_batch_end_time = batch_end_time

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        self.log(
            "train/grad_norm",
            self._compute_grad_norm(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, labels, repa_targets = _unpack_batch(batch)
        if not self.use_precomputed_latents:
            x = self._encode_to_latents(x)

        val_loss, _ = self._compute_losses(x, labels, repa_targets=repa_targets)

        val_prefix = _ema_log_prefix("val", self.cfg)
        
        for name, loss in val_loss.items():
            self.log(f"{val_prefix}/{name}", loss, on_step=False, on_epoch=True, sync_dist=True)
            
        return sum(val_loss.values())
            
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = str(self.cfg.get("optimizer", "Adam")).lower()
        if optimizer_name != "muon" and scale_cond_lr_mult(self.cfg) != 1.0:
            raise ValueError("loss.guided_cbm.guide.scale_cond.lr_mult is implemented for optimizer=Muon only")
        if optimizer_name == "muon":
            optimizer = self._configure_muon_optimizer()
        elif optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0)
            )
        elif optimizer_name == "adamw":
            # betas from the muon block so plain AdamW matches the AdamW group of the Muon recipe
            # ((0.9, 0.95), as in iMF) rather than PyTorch's (0.9, 0.999)
            optimizer = torch.optim.AdamW(
                params,
                lr=self.cfg.lr.val,
                betas=tuple(self.cfg.get("muon", {}).get("adamw_betas", (0.9, 0.999))),
                weight_decay=self.cfg.get("weight_decay", 0.0),
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        # Each group's LR relative to cfg.lr.val. Muon runs at lr.val and the
        # AdamW group at 0.1x, and ForceLearningRate needs the ratios to
        # reapply a changed lr.val on resume without re-deriving them.
        self._lr_group_scales = [
            float(group["lr"]) / float(self.cfg.lr.val) if float(self.cfg.lr.val) else 1.0
            for group in optimizer.param_groups
        ]

        schedule = str(self.cfg.lr.get("schedule", "constant")).lower()
        if schedule == "cosine":
            # A cosine tail in GLOBAL steps, read off trainer.global_step rather
            # than the scheduler's own counter, so a resume at any step lands on
            # the right point of the curve and lr.force_on_resume keeps working.
            start = int(self.cfg.lr.get("decay_start_step", 0))
            end = int(self.cfg.lr.get("decay_end_step", 0))
            floor = float(self.cfg.lr.get("min_factor", 0.0))
            if end <= start:
                raise ValueError("lr.schedule=cosine needs lr.decay_end_step > lr.decay_start_step")
            module = self

            def _factor(_):
                # Before a Trainer is attached the property raises rather than
                # returning None; treat any failure as step 0, i.e. the peak rate.
                try:
                    step = int(module.trainer.global_step)
                except Exception:
                    step = 0
                return cosine_factor(step, start, end, floor)

            scheduler = GlobalStepLambdaLR(optimizer, _factor)
        elif schedule == "staged":
            # Piecewise: factor 1 until stage_step, then stage_factor, then a cosine to min_factor
            # between decay_start_step and decay_end_step (e.g. 1e-3 for 15 epochs, 3e-4 after, a tail
            # to zero at the end). Read off the global step,
            # so a resume lands on the right point and lr.force_on_resume keeps working.
            stage_step = int(self.cfg.lr.get("stage_step", 0))
            stage_factor = float(self.cfg.lr.get("stage_factor", 1.0))
            start = int(self.cfg.lr.get("decay_start_step", 0))
            end = int(self.cfg.lr.get("decay_end_step", 0))
            floor = float(self.cfg.lr.get("min_factor", 0.0))
            module = self

            def _factor(_):
                try:
                    step = int(module.trainer.global_step)
                except Exception:
                    step = 0
                base = 1.0 if step < stage_step else stage_factor
                if end > start:
                    return base * cosine_factor(step, start, end, floor / base if base else 0.0)
                return base

            scheduler = GlobalStepLambdaLR(optimizer, _factor)
        elif schedule == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=float('inf'))
        else:
            raise ValueError("lr.schedule must be one of: constant, cosine, staged")

        if self.cfg.lr.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.cfg.lr.warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[self.cfg.lr.warmup_steps]
            )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

    def _configure_muon_optimizer(self):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer=Muon requires torch.optim.Muon. Update PyTorch to a "
                "version that includes the built-in Muon optimizer."
            )

        muon_params, adamw_params = self._split_muon_parameters()
        if not muon_params:
            raise ValueError("optimizer=Muon requires at least one trainable 2D parameter")

        muon_cfg = self.cfg.get("muon", {})
        muon_lr = self.cfg.lr.val
        weight_decay = self.cfg.get("weight_decay", 0.0)
        # scale_cond.lr_mult != 1: the guidance-scale embedding gets its own group in each optimizer at lr_mult x that
        # optimizer's rate (resumes of single-group states: _split_scale_lr_groups); at 1 the groups are as always
        lr_mult = scale_cond_lr_mult(self.cfg)
        w_ids = self._scale_embedding_param_ids() if lr_mult != 1.0 else set()

        def _groups(ps, lr):
            emb = [p for p in ps if id(p) in w_ids]
            if not emb:
                return ps
            return [{"params": [p for p in ps if id(p) not in w_ids]}, {"params": emb, "lr": lr * lr_mult}]

        muon_cls = torch.optim.Muon
        if bool(muon_cfg.get("batched", False)):   # speed (sbm_rae.speed): same math, Newton-Schulz batched per shape
            from sbm_rae.speed import BatchedMuon

            muon_cls = BatchedMuon
        muon_optimizer = muon_cls(
            _groups(muon_params, muon_lr),
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=muon_cfg.get("momentum", 0.95),
            nesterov=muon_cfg.get("nesterov", True),
            ns_coefficients=tuple(
                muon_cfg.get("ns_coefficients", (3.4445, -4.7750, 2.0315))
            ),
            eps=muon_cfg.get("eps", 1.0e-7),
            ns_steps =muon_cfg.get("ns_steps", 5),
            adjust_lr_fn=muon_cfg.get("adjust_lr_fn", None),
        )

        if not adamw_params:
            return muon_optimizer

        adam_lr = muon_cfg.get("adam_lr", muon_cfg.get("adamw_lr", 0.1 * muon_lr))
        adamw_optimizer = torch.optim.AdamW(
            _groups(adamw_params, adam_lr),
            lr=adam_lr,
            weight_decay=muon_cfg.get("adamw_weight_decay", weight_decay),
            betas=tuple(muon_cfg.get("adamw_betas", (0.9, 0.95))),
            eps=muon_cfg.get("adamw_eps", 1.0e-8),
        )
        return ChainedOptimizer([muon_optimizer, adamw_optimizer])


class IterativeSamplingCallback(Callback):
    """Callback for multi-step iterative generation: T(x), T^2(x), ..., T^n(x)"""
    def __init__(self, cfg, test_labels, inverse_scaler):
        super().__init__()
        self.cfg = cfg
        self.test_labels = test_labels 
        self.inverse_scaler = inverse_scaler
        self.image_shape = (cfg.model.in_channels, cfg.model.input_size, cfg.model.input_size)
        self._last_sampled_step = -1 
        self._last_overfitting_invariant_step = -1
        self._last_train_interpolation_step = -1
        self._train_interpolation_batch = None
        self._train_interpolation_source_noise = None
        self._current_ema_label = None
        
        # Define the specific iterative steps to track and plot
        self.steps_to_plot = [1, 2]
        self.max_steps = max(self.steps_to_plot)
        
    def setup(self, trainer, pl_module, stage=None):
        self.world_size = trainer.world_size
        self.rank = trainer.global_rank
        device = pl_module.device

        # Ensure sample count is divisible by world size for clean distributed gathering
        self.n_samples = (self.cfg.sampling.n_samples // self.world_size) * self.world_size
        g = torch.Generator(device='cpu').manual_seed(self.cfg.seed + 12345)
        
        self.shared_noise = torch.randn(
            (self.n_samples, *self.image_shape),
            generator=g,
            device=torch.device('cpu'),
        ).to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch_idx
        self._capture_train_interpolation_batch(batch)

        should_sample = self._should_run_iterative_sampling(trainer.global_step)
        should_plot_invariant = self._should_run_overfitting_invariant(trainer.global_step)
        should_plot_train_interpolation = self._should_run_train_interpolation_plot(trainer.global_step)
        if not should_sample and not should_plot_invariant and not should_plot_train_interpolation:
            return

        was_training = pl_module.training
        pl_module.eval()

        for ema_label, ema_context in self._ema_weight_contexts_for_sampling(trainer, pl_module):
            self._current_ema_label = ema_label
            with ema_context:
                if should_sample:
                    self._last_sampled_step = trainer.global_step
                    self._run_distributed_sampling(pl_module, trainer)

                if should_plot_invariant:
                    self._last_overfitting_invariant_step = trainer.global_step
                    self._run_overfitting_invariant_plot(pl_module, trainer, batch)

                if should_plot_train_interpolation:
                    self._last_train_interpolation_step = trainer.global_step
                    self._run_train_interpolation_plot(pl_module, trainer)
        self._current_ema_label = None

        if was_training:
            pl_module.train()

    def _should_run_iterative_sampling(self, global_step):
        if not bool(self.cfg.sampling.get("enabled", False)):
            return False
        every_n_steps = int(self.cfg.sampling.get("every_n_steps", 0))
        return every_n_steps > 0 and global_step % every_n_steps == 0 and global_step != self._last_sampled_step

    def _overfitting_invariant_plot_cfg(self):
        overfitting_cfg = self.cfg.trainer.get("overfitting", {})
        plot_cfg = overfitting_cfg.get("invariant_plot", {}) or {}
        return {} if isinstance(plot_cfg, bool) else plot_cfg

    def _overfitting_invariant_every_n_steps(self):
        plot_cfg = self._overfitting_invariant_plot_cfg()
        return int(plot_cfg.get("every_n_steps", self.cfg.sampling.get("every_n_steps", 0)))

    def _overfitting_invariant_max_images(self):
        plot_cfg = self._overfitting_invariant_plot_cfg()
        return int(plot_cfg.get("max_images", self.cfg.sampling.get("n_samples", 16)))

    def _should_run_overfitting_invariant(self, global_step):
        if not overfitting_invariant_plot_enabled(self.cfg):
            return False
        every_n_steps = self._overfitting_invariant_every_n_steps()
        return (
            every_n_steps > 0
            and global_step % every_n_steps == 0
            and global_step != self._last_overfitting_invariant_step
        )

    def _train_interpolation_plot_cfg(self):
        plot_cfg = self.cfg.sampling.get("train_interpolation_plot", {}) or {}
        return {} if isinstance(plot_cfg, bool) else plot_cfg

    def _train_interpolation_every_n_steps(self):
        plot_cfg = self._train_interpolation_plot_cfg()
        return int(plot_cfg.get("every_n_steps", self.cfg.sampling.get("every_n_steps", 0)))

    def _train_interpolation_max_images(self):
        plot_cfg = self._train_interpolation_plot_cfg()
        return int(
            plot_cfg.get(
                "max_images",
                self.cfg.trainer.get(
                    "per_device_batch_size",
                    self.cfg.sampling.get("n_samples", 16),
                ),
            )
        )

    def _train_interpolation_t_values(self):
        plot_cfg = self._train_interpolation_plot_cfg()
        return [float(t_value) for t_value in plot_cfg.get("t_values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]

    def _train_interpolation_seed_offset(self):
        plot_cfg = self._train_interpolation_plot_cfg()
        return int(plot_cfg.get("seed_offset", 23456))

    def _should_run_train_interpolation_plot(self, global_step):
        if not train_interpolation_plot_enabled(self.cfg):
            return False
        every_n_steps = self._train_interpolation_every_n_steps()
        return (
            every_n_steps > 0
            and global_step % every_n_steps == 0
            and global_step != self._last_train_interpolation_step
        )

    def _capture_train_interpolation_batch(self, batch):
        if self._train_interpolation_batch is not None or not train_interpolation_plot_enabled(self.cfg):
            return
        if batch is None:
            return
        self._train_interpolation_batch = _clone_batch_to_cpu(batch)

    def _model_autocast_context(self, device):
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _get_ema_callback(self, trainer):
        ema_cfg = self.cfg.trainer.get("ema", {})
        if not bool(ema_cfg.get("enabled", False)):
            return None

        for callback in trainer.callbacks:
            if hasattr(callback, "average_parameters") and hasattr(callback, "decays"):
                return callback
        return None

    def _ema_decays_for_sampling(self, ema_callback):
        ema_cfg = self.cfg.trainer.get("ema", {})
        sample_decays = ema_cfg.get("sample_decays", ema_cfg.get("sampling_decays", None))
        if sample_decays is None:
            return tuple(ema_callback.decays)

        if isinstance(sample_decays, (int, float, str)):
            sample_decays = [sample_decays]
        decays = tuple(float(decay) for decay in sample_decays)
        missing = [decay for decay in decays if all(abs(decay - candidate) > 1.0e-12 for candidate in ema_callback.decays)]
        if missing:
            raise ValueError(
                "trainer.ema.sample_decays must be a subset of trainer.ema.decays; "
                f"missing={missing}"
            )
        return decays

    def _ema_weight_contexts_for_sampling(self, trainer, pl_module):
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is None:
            return [(None, nullcontext())]

        return [
            (ema_decay_label(decay), ema_callback.average_parameters(pl_module, decay=decay))
            for decay in self._ema_decays_for_sampling(ema_callback)
        ]

    def _artifact_suffix(self):
        if self._current_ema_label is None:
            return ""
        return f"_ema_{self._current_ema_label}"

    def _artifact_log_prefix(self, base):
        if self._current_ema_label is None:
            return base
        return f"{base}_ema_{self._current_ema_label}"

    def _decode(self, pl_module, latents, vae_batch_size):
        latents = latents / pl_module.latents_scale + pl_module.latents_bias
        if not pl_module.vae_enabled:
            return self.inverse_scaler(latents).clamp(0.0, 1.0)

        images = []

        for start in range(0, latents.shape[0], vae_batch_size):
            end = min(start + vae_batch_size, latents.shape[0])
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    latents_batch = latents[start:end].to(dtype=pl_module.vae.dtype)
                    images_batch = pl_module.vae.decode(latents_batch).sample
            images.append(images_batch.to(dtype=latents.dtype))
        
        images = torch.cat(images, dim=0) 
        images = self.inverse_scaler(images)
        return images.clamp(0.0, 1.0)

    def _gather(self, tensor, indices, trainer):
        if trainer.world_size == 1:
            return tensor

        gathered = trainer.strategy.all_gather(tensor).reshape(-1, *tensor.shape[1:])
        gathered_indices = trainer.strategy.all_gather(indices).reshape(-1)
        order = torch.argsort(gathered_indices)
        
        return gathered[order]

    def _guidance_temperatures(self, device, dtype):
        guidance_cfg = resolve_guidance_config(self.cfg)
        if not guidance_cfg["enabled"]:
            return torch.ones(1, device=device, dtype=dtype)

        if guidance_cfg["mode"] == "negative":
            alpha_min, alpha_max = guidance_cfg["alpha_range"]
            return torch.linspace(
                alpha_min,
                alpha_max,
                steps=3,
                device=device,
                dtype=dtype,
            )

        return torch.linspace(guidance_cfg["w_min"], 1.0, steps=3, device=device, dtype=dtype)

    def _empty_class_labels(self, labels, pl_module):
        if labels is None:
            return None

        label_dim = getattr(pl_module.model, "label_dim", None)
        if label_dim is None:
            model_cfg = self.cfg.get("model", {}) or {}
            label_dim = model_cfg.get("label_dim", None)
        if label_dim is None:
            dataset_cfg = self.cfg.get("dataset", {}) or {}
            label_dim = dataset_cfg.get("num_classes", None)
        if label_dim is None:
            raise ValueError("Unable to resolve empty class label for CFG sampling")

        return torch.full_like(labels, int(label_dim))

    def _cfg_sampling_output(self, pl_module, model_input, labels, omega=None, w=None):
        output = pl_module.model(model_input, labels, omega=omega, w=w)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        guidance_cfg = resolve_guidance_config(self.cfg)
        if (
            not guidance_cfg["enabled"]
            or guidance_cfg["mode"] == "negative"
            or labels is None
            or w is None
        ):
            return pred

        uncond_output = pl_module.model(
            model_input,
            self._empty_class_labels(labels, pl_module),
            omega=omega,
            w=torch.ones_like(w),
        )
        if isinstance(uncond_output, tuple):
            uncond_pred = uncond_output[0]
        else:
            uncond_pred = uncond_output

        w_broadcast = w.view(-1, *((1,) * (pred.ndim - 1)))
        return w_broadcast * pred + (1.0 - w_broadcast) * uncond_pred

    def _anchor_temperatures(self, device, dtype):
        alpha = resolve_anchor_alpha(self.cfg)
        if alpha == 0.0:
            return torch.ones(1, device=device, dtype=dtype)
        return torch.linspace(1.0 - alpha, 1.0, steps=3, device=device, dtype=dtype)

    def _conditioning_grid(self, device, dtype):
        anchor_temperatures = self._anchor_temperatures(device, dtype)
        guidance_temperatures = self._guidance_temperatures(device, dtype)
        return [
            (anchor_temperature, guidance_temperature)
            for anchor_temperature in anchor_temperatures
            for guidance_temperature in guidance_temperatures
        ]

    def _run_overfitting_invariant_plot(self, pl_module, trainer, batch):
        if batch is None:
            return

        print(f"Running Overfitting Invariant Plot at step {trainer.global_step}...")
        device = pl_module.device
        inputs, labels, _ = _unpack_batch(batch)
        max_images = self._overfitting_invariant_max_images()
        if max_images <= 0:
            return

        local_limit = min(inputs.shape[0], max(1, math.ceil(max_images / trainer.world_size)))
        inputs = inputs[:local_limit].to(device, non_blocking=True)
        labels_device = labels[:local_limit].to(device, non_blocking=True).long() if labels is not None else None

        if pl_module.use_precomputed_latents:
            target_latents = inputs
            target_images = self._decode(pl_module, target_latents, self.cfg.sampling.vae_batch_size)
        else:
            target_latents = pl_module._encode_to_latents(inputs)
            target_images = self.inverse_scaler(inputs.detach()).clamp(0.0, 1.0)

        with torch.no_grad():
            with self._model_autocast_context(device):
                model_output = pl_module.model(target_latents, labels_device)
                if isinstance(model_output, tuple):
                    pred_latents = model_output[0]
                else:
                    pred_latents = model_output

        pred_images = self._decode(pl_module, pred_latents, self.cfg.sampling.vae_batch_size)
        latent_mse = (pred_latents.detach().float() - target_latents.detach().float()).square().mean()
        image_mse = (pred_images.detach().float() - target_images.detach().float()).square().mean()
        pl_module.log(
            "train/overfit_invariant_latent_mse",
            latent_mse,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        pl_module.log(
            "train/overfit_invariant_image_mse",
            image_mse,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        device_indices = torch.arange(local_limit, device=device) + self.rank * local_limit
        all_targets = self._gather(target_images, device_indices, trainer)[:max_images]
        all_predictions = self._gather(pred_images, device_indices, trainer)[:max_images]

        if trainer.is_global_zero:
            self._save_overfitting_invariant_plot(pl_module, all_targets, all_predictions)

    def _train_interpolation_source(self, target_latents):
        expected_shape = tuple(target_latents.shape)
        if (
            self._train_interpolation_source_noise is None
            or tuple(self._train_interpolation_source_noise.shape) != expected_shape
        ):
            generator = torch.Generator(device="cpu").manual_seed(
                int(self.cfg.seed) + self._train_interpolation_seed_offset() + int(self.rank)
            )
            self._train_interpolation_source_noise = torch.randn(
                expected_shape,
                generator=generator,
                device=torch.device("cpu"),
            )

        return self._train_interpolation_source_noise.to(
            device=target_latents.device,
            dtype=target_latents.dtype,
            non_blocking=True,
        )

    def _run_train_interpolation_plot(self, pl_module, trainer):
        if self._train_interpolation_batch is None:
            return

        print(f"Running Train Interpolation Plot at step {trainer.global_step}...")
        device = pl_module.device
        inputs, labels, _ = _unpack_batch(self._train_interpolation_batch)
        max_images = self._train_interpolation_max_images()
        if max_images <= 0:
            return

        local_limit = min(inputs.shape[0], max(1, math.ceil(max_images / trainer.world_size)))
        inputs = inputs[:local_limit].to(device, non_blocking=True)
        labels_device = labels[:local_limit].to(device, non_blocking=True).long() if labels is not None else None

        if pl_module.use_precomputed_latents:
            target_latents = inputs
        else:
            target_latents = pl_module._encode_to_latents(inputs)

        source_noise = self._train_interpolation_source(target_latents)
        device_indices = torch.arange(local_limit, device=device) + self.rank * local_limit
        prediction_rows = []
        t_values = self._train_interpolation_t_values()

        for t_value in t_values:
            interpolants = source_noise + float(t_value) * (target_latents - source_noise)
            with torch.no_grad():
                with self._model_autocast_context(device):
                    model_output = pl_module.model(interpolants, labels_device)
                    if isinstance(model_output, tuple):
                        pred_latents = model_output[0]
                    else:
                        pred_latents = model_output

            pred_images = self._decode(pl_module, pred_latents, self.cfg.sampling.vae_batch_size)
            all_predictions = self._gather(pred_images, device_indices, trainer)[:max_images]
            if trainer.is_global_zero:
                prediction_rows.append(all_predictions.detach().cpu())

            del interpolants, pred_latents, pred_images, all_predictions
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if trainer.is_global_zero and prediction_rows:
            self._save_train_interpolation_plot(pl_module, prediction_rows, t_values)
    
    def _run_distributed_sampling(self, pl_module, trainer):
        print(f"Running Iterative Distributed Sampling at step {trainer.global_step}...")
        device = pl_module.device
        
        all_indices = torch.arange(self.n_samples, device=device)
        device_indices = all_indices[self.rank::self.world_size]
        
        if self.cfg.trainer.get("overfitting", {}).get("enabled", False) and pl_module._overfitting_class_label is not None:
            labels_device = build_overfitting_class_labels(
                device_indices.numel(),
                pl_module._overfitting_class_label,
            ).to(device)
        else:
            labels_device = self.test_labels[device_indices.cpu()].to(device) if self.test_labels is not None else None
        
        print("sampling labels", labels_device)
        conditioning_grid = self._conditioning_grid(
            device,
            self.shared_noise.dtype,
        )
        image_rows_by_step = {step: [] for step in self.steps_to_plot}

        for anchor_temperature, guidance_temperature in conditioning_grid:
            current_input = self.shared_noise[device_indices]
            anchor_values = anchor_temperature.expand(device_indices.numel())
            guidance_values = guidance_temperature.expand(device_indices.numel())
            saved_latents = {}

            # Iterative Loop: T(x), T^2(x), ... T^max_steps(x)
            for step in range(1, self.max_steps + 1):
                with self._model_autocast_context(device):
                    with torch.no_grad():
                        pred_latents = self._cfg_sampling_output(
                            pl_module,
                            current_input,
                            labels_device,
                            omega=anchor_values,
                            w=guidance_values,
                        )

                # Save the latents if this is one of the designated target steps
                if step in self.steps_to_plot:
                    saved_latents[step] = pred_latents

                # Feed the output back in as the input for the next iteration
                current_input = pred_latents

            # Decode, gather, and save images for each target step
            for step, latents in saved_latents.items():
                images = self._decode(pl_module, latents, self.cfg.sampling.vae_batch_size)
                all_images = self._gather(images, device_indices, trainer)

                if trainer.is_global_zero:
                    image_rows_by_step[step].append(all_images.detach().cpu())

        if trainer.is_global_zero:
            conditioning_values = [
                (float(anchor_temperature), float(guidance_temperature))
                for anchor_temperature, guidance_temperature in conditioning_grid
            ]
            for step, image_rows in image_rows_by_step.items():
                if image_rows:
                    self._save_samples(
                        pl_module,
                        torch.cat(image_rows, dim=0),
                        step,
                        conditioning_values=conditioning_values,
                    )

    def _save_samples(self, pl_module, samples, iteration_step, conditioning_values=None):
        suffix = self._artifact_suffix()
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}{suffix}")
        os.makedirs(save_dir, exist_ok=True)

        caption = None
        if conditioning_values is None:
            nrow = math.ceil(math.sqrt(samples.shape[0]))
        else:
            nrow = samples.shape[0] // len(conditioning_values)
            caption = "Rows: " + ", ".join(
                f"omega={anchor_temperature:.3f}, w={guidance_temperature:.3f}"
                for anchor_temperature, guidance_temperature in conditioning_values
            )
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        log_prefix = self._artifact_log_prefix("val")
        
        # Log to Weights & Biases with the specific iteration number
        experiment = getattr(pl_module.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            image = wandb.Image(grid, caption=caption) if caption is not None else wandb.Image(grid)
            experiment.log({
                f"{log_prefix}/iterative_step_{iteration_step}_samples": [image],
                "global_step": pl_module.global_step
            })

        # Save locally with the iteration number in the filename
        torchvision.utils.save_image(
            grid,
            os.path.join(save_dir, f"iterative_step_{iteration_step}_samples{suffix}.png"),
        )

    def _save_overfitting_invariant_plot(self, pl_module, target_images, pred_images):
        suffix = self._artifact_suffix()
        save_dir = os.path.join(self.cfg.work_dir, f"overfit_invariant_{pl_module.global_step}{suffix}")
        os.makedirs(save_dir, exist_ok=True)

        paired_images = torch.stack((target_images, pred_images), dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(paired_images, nrow=2, padding=2)
        log_prefix = self._artifact_log_prefix("train")

        experiment = getattr(pl_module.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            experiment.log({
                f"{log_prefix}/overfit_target_vs_T_target": [
                    wandb.Image(grid, caption="Each row: target, T(target)")
                ],
                "global_step": pl_module.global_step,
            })

        torchvision.utils.save_image(
            grid,
            os.path.join(save_dir, f"target_vs_T_target{suffix}.png"),
        )

    def _save_train_interpolation_plot(self, pl_module, prediction_rows, t_values):
        suffix = self._artifact_suffix()
        save_dir = os.path.join(self.cfg.work_dir, f"train_interpolation_{pl_module.global_step}{suffix}")
        os.makedirs(save_dir, exist_ok=True)

        row_width = prediction_rows[0].shape[0]
        grid_images = torch.cat(prediction_rows, dim=0)
        grid = torchvision.utils.make_grid(grid_images, nrow=row_width, padding=2)
        log_prefix = self._artifact_log_prefix("train")
        caption = "Rows: " + ", ".join(f"t={t_value:.1f}" for t_value in t_values)

        experiment = getattr(pl_module.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            experiment.log({
                f"{log_prefix}/train_batch_T_I_t": [wandb.Image(grid, caption=caption)],
                "global_step": pl_module.global_step,
            })

        torchvision.utils.save_image(
            grid,
            os.path.join(save_dir, f"train_batch_T_I_t{suffix}.png"),
        )