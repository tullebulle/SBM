"""The trainer: `train(cfg, run_dir)`.

mode pretrain   the default: data from a DataLoader (text-image shards, or class-conditional latents), one loss call per
                micro-batch, Lightning's automatic optimization (bf16-mixed, DDP, gradient accumulation to the global
                batch), Muon + AdamW, EMA copies (train.ema). The text-to-image recipes (configs/t2i/exp) use this mode
                with loss.name guided_cbm.
mode finetune   a single-GPU class-conditional fine-tuning loop: fp32 masters with bf16 autocast, TF32 matmul off; each
                update draws `pairs` rows of a fixed latent bank, accumulates the transport and the endpoint gradient over
                micro-batches in separate fp32 buffers, mixes them as g_t + lambda g_e with lambda = share * |g_t| / |g_e|
                over all parameters (share 0.25: the endpoint's applied norm share is 0.2) or a fixed lambda, clips the
                global norm, and steps AdamW (betas 0.9/0.95, wd 0, constant lr).

Everything the run writes goes to its run folder: ckpt/ (Lightning checkpoints; resumed automatically), logs/, evals/,
wandb/ (see sbm_t2i.run), and monitor/ when the in-training monitor or the step timer is on (sbm_t2i.monitor). `init.from`
starts from another checkpoint or an inference export: its weights only, or (init.resume_state) its whole training state
(optimizer, scheduler, EMA, global step), which is how a training stage continues the previous one.
"""
from __future__ import annotations

import math
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

from sbm_t2i import run as R
from sbm_t2i.data import Bank, LatentData, Steps, TextImageData
from sbm_t2i.ema import EMA
from sbm_t2i.loss import CBMLoss, draw_pairs, pair_terms
from sbm_t2i.models import build_model, fill_gw_keys, load_weights
from sbm_t2i.optim import (ForceLearningRate, GlobalStepLR, StagedCheckpoint, build_optimizer, extend_optimizer_state, lr_factor,
                       split_params)


class Module(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: torch.nn.Module):
        super().__init__()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)}, logger=False)
        self.cfg, self.model = cfg, model
        # latent affine buffers kept in every checkpoint (the class-conditional evaluator decodes with them)
        self.register_buffer("latents_scale", torch.full((1, cfg.model.in_channels, 1, 1), float(cfg.data.latents_scale)))
        self.register_buffer("latents_bias", torch.zeros(1, cfg.model.in_channels, 1, 1))
        self.finetune = cfg.mode == "finetune"
        if (cfg.loss.get("gcond") or {}).get("enabled"):
            if self.finetune or model.gw_embedder is None:
                raise ValueError("loss.gcond needs mode pretrain and model.gw_embedder=true")
        self.sync_free = bool(cfg.train.get("sync_free", False))
        # loss.name guided_cbm: the guided objective of the RAE-latent ImageNet maps (sbm_t2i.guided_cbm); its frozen teacher
        # and the autoguidance weak model are held in lists (not submodules), so they are neither trained nor EMA-copied
        self.guided = str(cfg.loss.get("name", "cbm")) == "guided_cbm"
        if self.guided:
            from sbm_t2i.guided_cbm import GuidedCBMLoss
            if self.finetune or cfg.data.get("kind", "latent") != "text_image":
                raise ValueError("loss.name guided_cbm is wired for mode pretrain on text_image data")
            self.loss = GuidedCBMLoss(cfg.loss.guided_cbm, label_dim=cfg.model.label_dim)
        else:
            self.loss = None if self.finetune else CBMLoss(cfg.loss, label_dim=cfg.model.label_dim, sync_free=self.sync_free)
        self._g_teacher_box, self._g_weak_box, self._g_teacher_step, self._g_teacher_pending = [], [], None, None
        # codec.kind: a frozen representation autoencoder; the map trains on its latent of the batch's images
        # (sbm_t2i.codec), held by reference like the text encoder (not trained, checkpointed or EMA-copied)
        self._codec = None
        if (cfg.get("codec") or {}).get("kind"):
            from sbm_t2i.codec import build_codec
            self._codec = build_codec(cfg.codec)
        if self.finetune:
            self.automatic_optimization = False
            self.bank = Bank(cfg.data.bank)
        # text-to-image: batches are (image uint8, token ids, token mask); the frozen text encoder and an external
        # guidance teacher sit outside the module tree (sbm_t2i.text.Frozen)
        self.text = cfg.data.get("kind", "latent") == "text_image"
        self._text_encoder = self._ext_teacher = None
        if self.text:
            from sbm_t2i.text import TextEncoder
            self._text_encoder = TextEncoder(cfg.data.text_encoder, cfg.data.get("text_len", 256))
        teacher_from = (cfg.loss.get("guide") or {}).get("teacher_from")
        if teacher_from and not self.finetune:
            from sbm_t2i.text import Frozen
            kw = {**OmegaConf.to_container(cfg.model, resolve=True), "tangent_only_jvp": False, "omega_tokens": 0}
            self._ext_teacher = Frozen(load_weights(build_model(**kw), resolve_source(teacher_from),
                                                    cfg.loss.guide.get("teacher_weights", "ema")))
        # gradient switches, off by default: optim.freeze drops the gradients of the matching parameters before the
        # step (neither optimizer moves them or their state; the head-only training stage freezes everything but the
        # head); optim.skip_spike skips a step whose gradient norm exceeds factor x the running geometric mean of the
        # accepted steps' norms
        self._freeze = [str(p) for p in (cfg.optim.get("freeze") or [])]
        sk = cfg.optim.get("skip_spike") or {}
        self._skip_factor = float(sk.get("factor", 0.0) or 0.0)
        self._skip_cfg = (float(sk.get("ema", 0.99)), int(sk.get("warmup", 50)))
        self._skip_state = {"ema": None, "n": 0, "skipped": 0}
        # optim.zero_key_bias: the key bias of every attention without qk-norm (TextDiT's trunk and DDT head) set to zero in
        # the network, its EMA copies and the teacher, and held there (its gradient and AdamW first moment zeroed). It
        # has no gradient in exact arithmetic (softmax is invariant to q.b_k, constant along each query's row), so the
        # function is unchanged; what it removes is a numerically driven drift that inflates the logits in bf16
        self._zero_kb = bool(cfg.optim.get("zero_key_bias", False))
        self._grad_knobs_on = bool(self._freeze) or self._skip_factor > 0 or self._zero_kb

    def forward(self, x, labels, omega=None):
        return self.model(x, labels, omega=omega)

    def on_train_start(self):
        if self.cfg.train.per_rank_seed and not self.finetune:
            # every rank its own stream (seed_everything seeds all ranks alike); the global step is folded in so a
            # requeue does not replay the same draws
            seed = int(self.cfg.seed) + 7919 * (self.global_rank + 1) + int(self.global_step)
            torch.manual_seed(seed)
        if self._zero_kb:
            from sbm_t2i.ema import EMA
            nets = [self.model] + [m.module.model for c in self.trainer.callbacks if isinstance(c, EMA) for m in c._models.values()]
            norms = [zero_key_bias(n) for n in nets]
            print(f"[optim.zero_key_bias] key biases zeroed in the network and {len(nets) - 1} EMA copies "
                  f"(network's key-bias norm was {norms[0]:.4g})", flush=True)

    # ------------------------------------------------------------------ pretrain
    def training_step(self, batch, batch_idx):
        if self.finetune:
            return self._finetune_step()
        model, teacher = self.model, (None if self.guided else self._guide_teacher())
        if self.text:   # the batch's prompts become rows the loss indexes; the null prompt is the index model.label_dim
            from sbm_t2i.text import TextBound
            image, ids, mask = batch
            # codec: the frozen encoder's normalised latent of the images; else MiniT2I's pixels in [-1, 1]
            x = self._codec.encode(image) if self._codec is not None else image.float().div_(127.5).sub_(1.0)
            context, mask = self._text_encoder(ids, mask)
            labels = torch.arange(x.shape[0], device=x.device)
            from sbm_t2i.dit_t2i import PromptBound, TextDiT
            bound = PromptBound if isinstance(self.model, TextDiT) else TextBound   # TextDiT: the prompt path once a batch
            bind = lambda m: bound(m, context, mask, self.cfg.model.label_dim)   # noqa: E731
            model = bind(model)
            teacher = None if teacher is None else bind(teacher)
        else:
            x, labels = batch
        if self.guided:
            teacher, weak = self._g_teacher(), self._g_weak()
            if self._zero_kb and teacher is not None:   # a teacher restored from a checkpoint still holds its key biases
                zero_key_bias(teacher)
            calls = self._g_compiled()
            if calls is not None:   # train.compile: the frozen readers compiled, the student's calls installed on the loss
                teacher, weak = calls.reader(teacher), calls.reader(weak)
            self.loss.teacher = None if teacher is None else bind(teacher)
            self.loss.weak = None if weak is None else bind(weak)
            if hasattr(model, "prompts"):   # each network's prompt path now, outside the JVP transform
                model.prompts()
                with torch.no_grad():
                    for reader in (self.loss.teacher, self.loss.weak):
                        if reader is not None:
                            reader.prompts()
            losses, aux = self.loss(model, None, x, labels, self.global_step)
        else:
            losses, aux = self.loss(model, x, labels, self.global_step, teacher=teacher)
        for k, v in {**losses, **aux}.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=k in losses)
        total = sum(losses.values())
        self.log("train/total_loss", total, on_step=True, prog_bar=True)
        return total

    def _guide_teacher(self):
        """loss.guide.teacher_decay = d: the guide reads from the EMA-d copy (the guided target's label component is the
        map's own estimate, so the online map's SGD noise in it is fed back and amplified by 1 / (1 - gamma kappa); an
        averaged copy damps it). None: the online map (iMF's rule as published)."""
        if self._ext_teacher is not None:   # loss.guide.teacher_from: a frozen pretrained network, never refreshed
            return self._ext_teacher.on(self.device)
        d = (self.cfg.loss.get("guide") or {}).get("teacher_decay")
        if d is None:
            return None
        from sbm_t2i.ema import EMA, ema_label
        ema = next((c for c in self.trainer.callbacks if isinstance(c, EMA)), None)
        if ema is None or ema_label(d) not in ema._models:
            raise ValueError(f"loss.guide.teacher_decay {d} needs train.ema.decays to contain it")
        teacher = ema._models[ema_label(d)]
        refresh = (self.cfg.loss.get("guide") or {}).get("teacher_refresh")
        if refresh:
            # a frozen teacher (decay 1.0 copy) refreshed from the active EMA every `teacher_refresh` steps: the guided
            # target's self-term then moves in jumps the online map cannot lock onto. The copy is also filled at its first
            # use (a decay-1.0 copy new to a resumed run would otherwise be the random init until the first EMA update, or
            # the raw weights after it).
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

    # ------------------------------------------------------------------ loss.name guided_cbm
    def _g_teacher(self):
        """loss.guided_cbm's frozen teacher: a copy of the active EMA (or guide.teacher_source's copy) made at
        guide.start_step and refreshed from it every teacher_refresh steps; restored from the checkpoint on a resume.
        None before the start."""
        schedule = self.loss.teacher_schedule()
        if schedule is None:
            return None
        start, refresh = schedule
        step = int(self.global_step)
        if step < start or not refresh:
            return None
        if not self.training:   # a read with the EMA weights swapped in: use the teacher, never copy it
            return self._g_teacher_box[0] if self._g_teacher_box else None
        from sbm_t2i.ema import EMA, ema_label
        ema = next((c for c in self.trainer.callbacks if isinstance(c, EMA)), None)
        if ema is None:
            raise ValueError("loss.guided_cbm's teacher is a copy of the active EMA: set train.ema.decays")
        decay = self.loss.teacher_source
        label = ema.active_label if decay is None else ema_label(decay)
        if label not in ema._models:
            raise ValueError(f"loss.guided_cbm.guide.teacher_source {decay} is not one of train.ema.decays")
        source = ema._models[label].module.model
        what = "active EMA" if decay is None else f"EMA-{decay:g}"
        box = self._g_teacher_box
        if not box:
            import copy
            teacher = copy.deepcopy(source).requires_grad_(False).eval()
            box.append(teacher)
            if self._g_teacher_pending is not None:   # a resume: the teacher the run had, not a fresh copy
                state, saved = self._g_teacher_pending
                teacher.load_state_dict(state)
                self._g_teacher_step, self._g_teacher_pending = saved, None
                print(f"[guided_cbm] step {step}: teacher restored from the checkpoint (copied at step {saved})", flush=True)
            else:
                self._g_teacher_step = step
                print(f"[guided_cbm] step {step}: teacher copied from the {what}", flush=True)
        if self.loss.teacher_refresh_at_start and not getattr(self, "_g_start_refreshed", False):
            self._g_start_refreshed = True
            if self._g_teacher_step != step:
                self._g_teacher_step = None
        teacher = box[0]
        if self._g_teacher_step is None or ((step - start) % int(refresh) == 0 and self._g_teacher_step != step):
            with torch.no_grad():
                for dst, src in zip(teacher.state_dict().values(), source.state_dict().values()):
                    dst.copy_(src)
            self._g_teacher_step = step
            print(f"[guided_cbm] step {step}: teacher refreshed from the {what}", flush=True)
        self.log("train/teacher_copy_step", float(self._g_teacher_step), on_step=True, rank_zero_only=True)
        return teacher

    def _g_compiled(self):
        """train.compile (sbm_t2i.speed.CompiledCalls), built at the first step; None when every switch is off."""
        c = self.cfg.train.get("compile") or {}
        if not any(bool(c.get(k, False)) for k in ("teacher", "student_fwd", "student_jvp")):
            return None
        if getattr(self, "_g_calls", None) is None:
            from sbm_t2i.speed import CompiledCalls
            # the balancer retains the student graph for its gradient norms until the guide start
            balancing = self.loss.balance != "none" and int(self.global_step) < self.loss.g_start
            self._g_calls = CompiledCalls(c, self.model, keep_backward_buffers=balancing)
            self._g_calls.install(self.loss)
            print(f"[speed] torch.compile on: teacher={self._g_calls.do_teacher} student_fwd={self._g_calls.do_fwd} "
                  f"student_jvp={self._g_calls.do_jvp} mode={self._g_calls.mode}", flush=True)
        return self._g_calls

    def _g_weak(self):
        """loss.guided_cbm.guide.autoguide's frozen weak model: this architecture with an earlier checkpoint's weights (its
        active EMA unless autoguide.ema), loaded once. None when autoguidance is off."""
        if not self.loss.ag:
            return None
        if not self._g_weak_box:
            import copy
            from sbm_t2i.guided_cbm import load_weak_state
            weak = copy.deepcopy(self.model)
            weak.load_state_dict(load_weak_state(self.loss.ag_ckpt, self.loss.ag_ema), strict=True)
            self._g_weak_box.append(weak.requires_grad_(False).eval().to(self.device))
            print(f"[guided_cbm] autoguide: weak model from {self.loss.ag_ckpt} (EMA "
                  f"{'active' if self.loss.ag_ema is None else self.loss.ag_ema}), weight {self.loss.ag_weight}", flush=True)
        return self._g_weak_box[0]

    def on_before_optimizer_step(self, optimizer):
        if not self.finetune:
            if self.sync_free:   # two foreach kernels instead of three per parameter (same norm up to summation order)
                norms = torch._foreach_norm([p.grad.detach().float() for p in self.parameters() if p.grad is not None])
                self.log("train/grad_norm", torch.linalg.vector_norm(torch.stack(norms)), on_step=True)
            else:
                sq = sum(p.grad.detach().float().pow(2).sum() for p in self.parameters() if p.grad is not None)
                self.log("train/grad_norm", torch.sqrt(sq), on_step=True)
            if self._grad_knobs_on:
                self._grad_knobs()

    def _grad_knobs(self):
        """optim.freeze and optim.skip_spike (see __init__): a parameter whose gradient is None is skipped by both
        optimizers (Muon's momentum and AdamW's moments and step count stay as they are)."""
        import fnmatch
        import math
        if not hasattr(self, "_knob_params"):
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            self._knob_params = [p for _, p in named]
            self._frozen_params = [p for n, p in named if any(fnmatch.fnmatchcase(n, pat) for pat in self._freeze)]
            if self._freeze:
                print(f"[optim.freeze] {len(self._frozen_params)} tensors "
                      f"({sum(p.numel() for p in self._frozen_params) / 1e6:.2f}M parameters) get no update", flush=True)
        for p in self._frozen_params:
            p.grad = None
        if self._zero_kb:
            if not hasattr(self, "_kb_moments_zeroed"):   # AdamW's first moment of the slices (restored from a checkpoint)
                adam = optimizer_of(self.trainer.optimizers[0], "adam")
                for bias, view in key_bias_views(self.model, with_param=True):
                    st = adam.state.get(bias, {}) if adam is not None else {}
                    if "exp_avg" in st:
                        st["exp_avg"][view].zero_()
                self._kb_moments_zeroed = True
            for bias, view in key_bias_views(self.model, with_param=True):
                if bias.grad is not None:
                    bias.grad[view] = 0
        if self._skip_factor <= 0:
            return
        grads = [p.grad.detach().float() for p in self._knob_params if p.grad is not None]
        g = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))) if grads else 0.0
        st, (decay, warmup) = self._skip_state, self._skip_cfg
        threshold = self._skip_factor * math.exp(st["ema"]) if st["ema"] is not None else math.inf
        spike = not math.isfinite(g) or (st["n"] >= warmup and g > threshold)
        if spike:
            for p in self._knob_params:
                p.grad = None
            st["skipped"] += 1
        elif g > 0:
            st["ema"] = math.log(g) if st["ema"] is None else decay * st["ema"] + (1.0 - decay) * math.log(g)
            st["n"] += 1
        self._skip_info = {"grad_norm": g, "threshold": threshold, "skipped": int(spike), "total_skipped": st["skipped"]}
        self.log("train/skip_spike", float(spike), on_step=True)
        self.log("train/skip_spike_total", float(st["skipped"]), on_step=True)

    def on_save_checkpoint(self, checkpoint):
        if self.guided:   # the teacher and the frozen balance weight
            if self._g_teacher_box:
                checkpoint["guided_cbm_teacher"] = {
                    "state": {k: v.detach().cpu().clone() for k, v in self._g_teacher_box[0].state_dict().items()},
                    "step": self._g_teacher_step}
            checkpoint["guided_cbm_loss"] = self.loss.state_dict()
            return
        state = self.loss.state_dict() if self.loss is not None else {}
        if state:                       # only some optional terms carry loss state (loss.transport.p_within, frechet.save_state)
            checkpoint["cbm_loss_state"] = state

    def on_load_checkpoint(self, checkpoint):
        if self.guided:
            saved = checkpoint.get("guided_cbm_teacher")
            if saved is not None:
                self._g_teacher_pending = (saved["state"], saved.get("step"))
            if "guided_cbm_loss" in checkpoint:
                self.loss.load_state_dict(checkpoint["guided_cbm_loss"])
            return
        if getattr(self.model, "gw_embedder", None) is not None:
            self._add_gw_embedder(checkpoint)
        if self.loss is not None and "cbm_loss_state" in checkpoint:
            self.loss.load_state_dict(checkpoint["cbm_loss_state"])

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
        losses, _ = self.loss(self.model, x, labels, self.global_step)
        for k, v in losses.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = build_optimizer(self.model, self.cfg.optim)
        self._peak = [g["lr"] for g in opt.param_groups]
        if self.finetune:
            return opt
        sched = GlobalStepLR(opt, lambda _: lr_factor(self.cfg.optim, self._step_or_zero()))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    def _step_or_zero(self):
        try:
            return int(self.trainer.global_step)
        except Exception:
            return 0

    # ------------------------------------------------------------------ finetune
    def _finetune_step(self):
        step = self.global_step + 1        # updates are counted from 1 (the draw streams are keyed by it)
        opt = self.optimizers()
        for g, peak in zip(opt.param_groups, self._peak):
            g["lr"] = peak * lr_factor(self.cfg.optim, step)
        params, grads, log = finetune_gradients(self.model, self.bank, self.cfg, step, self.device)
        for p, g in zip(params, grads):
            p.grad = g
        clip = float(self.cfg.optim.clip) if self.cfg.optim.clip else float("inf")
        raw = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.update(grad_norm=float(raw), clipped=float(float(raw) > clip), lr=opt.param_groups[0]["lr"])
        for k, v in log.items():
            self.log(f"train/{k}", float(v), on_step=True, on_epoch=False)


def key_bias_views(net, with_param=False):
    """The key-bias slices of every self-attention without qk-norm in `net` (sbm_t2i.dit_t2i.SelfAttention): views into the
    qkv bias, or (bias parameter, slice) pairs."""
    from sbm_t2i.dit_t2i import SelfAttention
    out = []
    for m in net.modules():
        if isinstance(m, SelfAttention) and isinstance(m.k_norm, torch.nn.Identity) and m.qkv.bias is not None:
            C = m.qkv.bias.numel() // 3
            out.append((m.qkv.bias, slice(C, 2 * C)) if with_param else m.qkv.bias[C:2 * C])
    return out


@torch.no_grad()
def zero_key_bias(net):
    """optim.zero_key_bias: zero `net`'s key biases in place; returns their norm before."""
    views = key_bias_views(net)
    norm = float(torch.sqrt(sum(v.float().pow(2).sum() for v in views))) if views else 0.0
    for v in views:
        v.zero_()
    return norm


def optimizer_of(opt, kind):
    """The AdamW (kind adam) or Muon (kind muon) inside sbm_t2i.optim.ChainedOptimizer (or the optimizer itself)."""
    subs = getattr(opt, "optimizers", (opt,))
    for o in subs:
        if (kind == "adam") == isinstance(o, torch.optim.AdamW):
            return o
    return None


def finetune_gradients(model, bank, cfg, step, device):
    """One update of the fine-tuning recipe, before the clip: (parameters, their mixed gradients, scalars to log).

    `pairs` bank rows (seed + 100003 step), split into micro-batches of `micro_pairs`; per micro-batch the per-pair
    draws, the two terms (loss.pair_terms), and each term's gradient of `mean * (micro / pairs)` accumulated into its
    own fp32 buffer; then g = g_transport + lambda g_endpoint (see endpoint_weight), cast to the parameter dtype."""
    f, lc = cfg.finetune, cfg.loss
    params = [p for p in model.parameters() if p.requires_grad]   # all of them in the recipe (pos_embed too)
    pairs, micro = int(f.pairs), min(int(f.micro_pairs), int(f.pairs))
    ids, gen = bank.rows(f.seed, step, pairs)
    gd = lc.get("guide") or {}
    guide = None
    if gd.get("enabled", False) and step >= int(gd.get("start_step", 0)):
        guide = {"kappa": float(gd.get("kappa", 0.0)), "t_lo": float(gd.get("t_lo", 0.35)), "t_hi": float(gd.get("t_hi", 0.6)),
                 "endpoint": bool(gd.get("endpoint", True)), "null": int(model.label_dim)}
    drop = float(lc.get("label_dropout", 0.0))
    totals = {t: [torch.zeros_like(p, dtype=torch.float32) for p in params] for t in ("transport", "endpoint")}
    values, aux = {"transport": 0.0, "endpoint": 0.0}, {}
    for offset in range(0, pairs, micro):
        chunk = ids[offset:offset + micro]
        weight = len(chunk) / pairs
        targets = bank.targets[chunk].flatten(0, 1).to(device)
        labels = bank.labels[chunk].flatten().to(device)
        if drop > 0:   # keep the null class trained (the guide reads it); drawn from the bank stream's CPU generator
            keep = torch.rand(labels.shape, generator=gen) >= drop
            labels = torch.where(keep.to(device), labels, torch.full_like(labels, int(model.label_dim)))
        draws = draw_pairs(targets, int(f.seed), step, pairs=pairs, offset=offset, time_mean=float(lc.time.p_mean),
                           time_std=float(lc.time.p_std), omega_range=tuple(f.omega_range))
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            terms, aux = pair_terms(model, targets, labels, draws, alpha=float(lc.signed.alpha), paired=bool(lc.signed.paired),
                                    generator=gen, transport_pc=(float(lc.transport.p), float(lc.transport.c)),
                                    endpoint_pc=(float(lc.endpoint.p), float(lc.endpoint.c)), guide=guide)
        for i, term in enumerate(("transport", "endpoint")):
            grads = torch.autograd.grad(terms[term].mean() * weight, params, retain_graph=i == 0, allow_unused=True)
            for acc, g in zip(totals[term], grads):
                if g is not None:
                    acc.add_(g.detach().float())
            values[term] += float(terms[term].detach().mean()) * weight
        del terms, grads
    norms = {t: float(torch.sqrt(sum(v.square().sum(dtype=torch.float64) for v in totals[t]))) for t in totals}
    lam = endpoint_weight(f.endpoint_weight, norms)
    mixed = [(t + (e * lam if lam != 1.0 else e)).to(p.dtype) for p, t, e in zip(params, totals["transport"], totals["endpoint"])]
    log = {**values, "transport_grad_norm": norms["transport"], "endpoint_grad_norm": norms["endpoint"], "endpoint_weight": lam,
           "endpoint_share_applied": lam * norms["endpoint"] / max(norms["transport"] + lam * norms["endpoint"], 1e-30),
           **{k: v for k, v in aux.items() if k != "signed_shift"}}
    return params, mixed, log


def endpoint_weight(setting, norms, lo=1e-2, hi=1e6):
    """'balanceK': lambda = K |g_t| / |g_e| clamped to [1e-2, 1e6] (applied endpoint share K / (1 + K)); a number: fixed."""
    if isinstance(setting, str) and setting.startswith("balance"):
        k = float(setting[len("balance"):] or 1.0)
        t, e = norms["transport"], norms["endpoint"]
        if not (e > 0 and math.isfinite(t) and math.isfinite(e)):
            return 1.0
        return float(min(max(k * t / e, lo), hi))
    return float(setting)


class EvalAtSteps(Callback):
    """Evaluate the in-memory model at given global steps with a named protocol (single process), into evals/."""

    def __init__(self, run_dir, protocol, steps):
        self.run_dir, self.protocol, self.steps = Path(run_dir), protocol, set(int(s) for s in steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps and trainer.is_global_zero:
            from sbm_t2i.eval import evaluate_module
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


def torchrun() -> bool:
    """True when the process was started by torchrun (torch.distributed.run)."""
    return all(k in os.environ for k in ("TORCHELASTIC_RUN_ID", "LOCAL_WORLD_SIZE", "WORLD_SIZE"))


def world(cfg):
    """(devices per node, nodes): torchrun's LOCAL_WORLD_SIZE and WORLD_SIZE when it started the process, else
    train.devices ("auto": every visible GPU) on train.num_nodes nodes."""
    if torchrun():
        local = int(os.environ["LOCAL_WORLD_SIZE"])
        return local, int(os.environ["WORLD_SIZE"]) // local
    devices = max(torch.cuda.device_count(), 1) if cfg.train.devices == "auto" else int(cfg.train.devices)
    return devices, int(cfg.train.get("num_nodes", 1) or 1)


def cluster_plugins(devices, nodes):
    """The process environment Lightning should trust: torchrun's when it started us; Lightning's own launcher on a
    single node (it starts one process per GPU), whatever scheduler variables the shell carries; otherwise Lightning's
    auto-detection (e.g. srun across nodes)."""
    from lightning.pytorch.plugins.environments import LightningEnvironment, TorchElasticEnvironment
    if torchrun():
        return [TorchElasticEnvironment()]
    if nodes == 1:
        return [LightningEnvironment()]
    return []


def train(cfg: DictConfig, run_dir: str | Path):
    run_dir = Path(run_dir)
    finetune = cfg.mode == "finetune"
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.tf32)
    torch.backends.cudnn.allow_tf32 = True
    if cfg.train.tf32:
        torch.set_float32_matmul_precision("high")
    pl.seed_everything(int(cfg.seed), workers=True)
    devices, nodes = world(cfg)
    if finetune and devices * nodes != 1:
        raise ValueError("mode finetune is single-GPU (its gradients are accumulated by hand, not all-reduced)")

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
            # an inference export (sbm_t2i.infer) also carries the guided-CBM loss state: a fine-tune from it keeps the
            # frozen balance weight instead of re-learning it (a plain checkpoint keeps the old behaviour)
            from sbm_t2i.infer import is_export
            src_ck = torch.load(str(src), map_location="cpu", mmap=True, weights_only=False)
            if module.guided and is_export(src_ck) and src_ck.get("guided_cbm_loss") is not None:
                module.loss.load_state_dict(src_ck["guided_cbm_loss"])
                print(f"restored the guided_cbm loss state from the export: {src_ck['guided_cbm_loss']}")
    max_steps = int(cfg.train.max_steps)
    if from_parent and cfg.train.get("steps_after_init"):
        max_steps = int(torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)["global_step"]) + int(cfg.train.steps_after_init)

    every = int(cfg.train.checkpoint_every)
    # train.stage_checkpoints_locally: a node-local folder each checkpoint is written to first, then copied into the run
    # folder in the background (true: <tmp>/sbm-ckpt-<run name>); off: written to the run folder directly
    local = cfg.train.get("stage_checkpoints_locally")
    if local is True:
        import tempfile
        local = str(Path(tempfile.gettempdir()) / f"sbm-ckpt-{run_dir.name}")
    callbacks = [StagedCheckpoint(dirpath=run_dir / "ckpt", filename="step{step:07d}", auto_insert_metric_name=False,
                                  every_n_train_steps=every, save_top_k=-1, save_last=False, local_dir=local or None),
                 LearningRateMonitor(logging_interval="step")]
    if not finetune and cfg.train.ema.decays:
        callbacks.append(EMA(cfg.train.ema.decays, cfg.train.ema.active, cfg.train.ema.min_step,
                             foreach=bool(cfg.train.ema.get("foreach", False))))
    monitor = cfg.train.get("monitor") or {}
    if not finetune and int(monitor.get("every", 0) or 0) > 0:
        from sbm_t2i.monitor import FIDMonitor, TextMonitor
        callbacks.append((TextMonitor if module.text else FIDMonitor)(run_dir, **OmegaConf.to_container(monitor, resolve=True)))
    if int(cfg.train.get("step_timer", 0) or 0) > 0:
        from sbm_t2i.monitor import StepTimer
        images = 2 * int(cfg.finetune.pairs) if finetune else int(cfg.train.global_batch)
        callbacks.append(StepTimer(run_dir, int(cfg.train.step_timer), images_per_step=images))
    if ckpt_path is not None and from_parent and cfg.optim.get("force_on_resume", True):
        callbacks.append(ForceLearningRate(_peak_lrs(model, cfg)))
    if cfg.train.get("eval") and cfg.train.eval.get("protocol"):
        callbacks.append(EvalAtSteps(run_dir, cfg.train.eval.protocol, cfg.train.eval.steps))

    logger = WandbLogger(**R.wandb_kwargs(cfg, run_dir))
    if finetune:
        strategy, precision, accumulate, data = "auto", "32-true", 1, None
        loaders = dict(train_dataloaders=torch.utils.data.DataLoader(Steps(max_steps), batch_size=None))
    else:
        per_step = int(cfg.train.per_device_batch) * devices * nodes
        if int(cfg.train.global_batch) % per_step:
            raise ValueError(f"train.global_batch {cfg.train.global_batch} is not a multiple of {per_step}")
        accumulate = int(cfg.train.global_batch) // per_step
        strategy = DDPStrategy(timeout=timedelta(minutes=120)) if devices * nodes > 1 else "auto"
        precision = cfg.train.precision
        loaders = dict(datamodule=TextImageData(cfg) if module.text else LatentData(cfg))
    trainer = pl.Trainer(default_root_dir=str(run_dir), logger=logger, callbacks=callbacks, max_steps=max_steps, max_epochs=-1,
                         accelerator="auto", devices=devices, num_nodes=nodes, strategy=strategy, precision=precision,
                         plugins=cluster_plugins(devices, nodes),
                         accumulate_grad_batches=accumulate,
                         gradient_clip_val=None if finetune else cfg.optim.get("clip"),
                         log_every_n_steps=int(cfg.train.log_every),
                         val_check_interval=None if finetune or not cfg.train.get("val_every") else int(cfg.train.val_every),
                         check_val_every_n_epoch=None, limit_val_batches=0 if finetune else int(cfg.train.get("val_batches", 8)),
                         num_sanity_val_steps=0, use_distributed_sampler=False, enable_progress_bar=False)
    if trainer.is_global_zero:
        (run_dir / "wandb" / "url.txt").write_text(str(getattr(logger.experiment, "url", "") or ""))
    trainer.fit(module, ckpt_path=str(ckpt_path) if ckpt_path else None, **loaders)


def _peak_lrs(model, cfg):
    from sbm_t2i.optim import build_optimizer
    return [g["lr"] for g in build_optimizer(model, cfg.optim).param_groups]
