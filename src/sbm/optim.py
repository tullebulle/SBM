"""Optimizers, learning-rate schedules and the checkpoint writer.

muon   torch.optim.Muon on every trainable 2-D weight outside embeddings, AdamW on the rest (embeddings, biases,
       vectors) at `adam_lr` (default lr / 10): the B/2 and XL recipes.
adamw  one AdamW over every trainable parameter.
Schedules are read off the GLOBAL step (a resume lands on the right point): constant | cosine (decay_start_step ->
decay_end_step to min_factor) | staged (x stage_factor from stage_step, then the cosine tail), after an optional
linear warm-up from 0.1.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class ChainedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = tuple(optimizers)
        groups = [g for o in self.optimizers for g in o.param_groups]
        super().__init__(groups, defaults={})
        self.param_groups, self.state = groups, {}

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict):
        for o, s in zip(self.optimizers, state_dict["optimizers"], strict=True):
            o.load_state_dict(s)
        self.param_groups, self.state = [g for o in self.optimizers for g in o.param_groups], {}


def split_params(module: nn.Module, o):
    """The trainable (name, parameter) pairs of each optimizer build_optimizer makes, in its order:
    [all] for adamw, [Muon's, AdamW's] for muon."""
    if o.name == "adamw":
        return [[(n, p) for n, p in module.named_parameters() if p.requires_grad]]
    if o.name != "muon":
        raise ValueError(f"optim.name must be muon or adamw, got {o.name}")
    muon, adam = [], []
    for mname, m in module.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            if p.requires_grad:
                (muon if p.ndim == 2 and not isinstance(m, nn.Embedding) else adam).append((f"{mname}.{pname}" if mname else pname, p))
    return [muon, adam]


def extend_optimizer_state(state: dict, groups, new_names) -> dict:
    """An optimizer state saved before the parameters `new_names` existed, re-indexed for the optimizers built now
    (`groups` = split_params' lists): every old parameter keeps its state (Muon momentum, Adam moments), the new ones
    get none (a fresh start). One parameter group per optimizer, as build_optimizer makes them. In place; returned."""
    new_names = set(new_names)
    subs = state["optimizers"] if "optimizers" in state else [state]
    if len(subs) != len(groups):
        raise ValueError(f"the saved state has {len(subs)} optimizers, the run builds {len(groups)}")
    for sd, named in zip(subs, groups):
        (group,) = sd["param_groups"]
        names = [n for n, _ in named]
        old = [n for n in names if n not in new_names]
        if len(group["params"]) != len(old):
            raise ValueError(f"saved optimizer group has {len(group['params'])} parameters, expected {len(old)} "
                             f"({len(names)} now, minus {len(names) - len(old)} new)")
        pos = {n: i for i, n in enumerate(names)}
        remap = {saved: pos[n] for saved, n in zip(group["params"], old)}
        sd["state"] = {remap[k]: v for k, v in sd["state"].items()}
        group["params"] = list(range(len(names)))
    return state


def build_optimizer(module: nn.Module, o):
    """o: the `optim` config block."""
    lr, wd = float(o.lr), float(o.get("weight_decay", 0.0))
    betas = tuple(o.get("betas", (0.9, 0.95)))
    if o.name == "adamw":
        return torch.optim.AdamW([p for _, p in split_params(module, o)[0]], lr=lr, betas=betas, weight_decay=wd,
                                 eps=float(o.get("eps", 1e-8)))
    muon, adam = ([p for _, p in g] for g in split_params(module, o))
    opt = torch.optim.Muon(muon, lr=lr, weight_decay=wd, momentum=0.95, nesterov=True,
                           ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7, ns_steps=5, adjust_lr_fn="original")
    adam_lr = float(o.adam_lr) if o.get("adam_lr") is not None else 0.1 * lr
    return ChainedOptimizer([opt, torch.optim.AdamW(adam, lr=adam_lr, weight_decay=wd, betas=betas, eps=1e-8)])


def cosine_factor(step, start, end, floor=0.0):
    if step < start:
        return 1.0
    if step >= end:
        return floor
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * (step - start) / float(end - start)))


def lr_factor(o, step):
    """The multiplier on the peak rate at a global step."""
    s = o.get("schedule", "constant")
    if s == "constant":
        f = 1.0
    elif s == "cosine":
        f = cosine_factor(step, int(o.decay_start_step), int(o.decay_end_step), float(o.min_factor))
    elif s == "staged":
        base = 1.0 if step < int(o.stage_step) else float(o.stage_factor)
        start, end = int(o.decay_start_step), int(o.decay_end_step)
        f = base * cosine_factor(step, start, end, float(o.min_factor) / base if base else 0.0) if end > start else base
    else:
        raise ValueError("optim.schedule must be constant, cosine or staged")
    w = int(o.get("warmup_steps", 0))
    if w > 0 and step < w:
        f *= 0.1 + 0.9 * step / w
    return f


class GlobalStepLR(torch.optim.lr_scheduler.LambdaLR):
    """A LambdaLR over the global step whose saved state is irrelevant (any scheduler state loads)."""

    def load_state_dict(self, state_dict):
        for key in ("last_epoch", "_step_count", "_last_lr"):
            if key in state_dict:
                setattr(self, key, state_dict[key])


class ForceLearningRate(Callback):
    """Reassert the configured peak rate after Lightning restored a checkpoint's optimizer and scheduler (which carry
    the OLD rate): needed whenever a run continues another with a different lr."""

    def __init__(self, lrs):
        self.lrs = list(lrs)

    def on_train_start(self, trainer, pl_module):
        for o in trainer.optimizers:
            for g, lr in zip(o.param_groups, self.lrs):
                g["lr"] = g["initial_lr"] = lr
        for c in trainer.lr_scheduler_configs:
            c.scheduler.base_lrs = list(self.lrs)
        print(f"ForceLearningRate: {self.lrs}")


class StagedCheckpoint(ModelCheckpoint):
    """Write the checkpoint to node-local scratch, then copy it to the run folder on a background thread, so a multi-GB
    write to a slow shared filesystem does not stall every rank inside the post-save barrier."""

    def __init__(self, *args, local_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_dir = Path(local_dir) if local_dir else None
        self._threads = []

    @staticmethod
    def _mirror(src, dst):
        try:
            tmp = dst.with_suffix(dst.suffix + ".partial")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            os.remove(src)
        except Exception as exc:  # never let the copy kill the run
            print(f"checkpoint copy to {dst} failed: {exc}")

    def _save_checkpoint(self, trainer, filepath):
        if self.local_dir is None:
            return super()._save_checkpoint(trainer, filepath)
        local = self.local_dir / Path(filepath).name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        super()._save_checkpoint(trainer, str(local))
        if trainer.is_global_zero:
            th = threading.Thread(target=self._mirror, args=(local, Path(filepath)), daemon=True)
            th.start()
            self._threads = [t for t in self._threads if t.is_alive()] + [th]

    def teardown(self, trainer, pl_module, stage):
        for th in self._threads:
            th.join(timeout=1800)
        return super().teardown(trainer, pl_module, stage)
