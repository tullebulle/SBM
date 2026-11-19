"""EMA copies of the Lightning module.

Several decays are tracked at once, from `min_step` on, with a constant decay (torch AveragedModel + get_ema_avg_fn).
On save the checkpoint's `state_dict` becomes the ACTIVE EMA copy (what every B/2 number scores), the raw weights move
to `current_model_state`, the other decays to `ema_state_dicts` (keyed by label, e.g. `0_9999`). Resume restores all
three. Validation runs on the active copy.

`foreach` (off by default): the same update, decay * ema + (1 - decay) * current, as three foreach kernels per copy
instead of AveragedModel's per-tensor loop, and without its two bool(n_averaged) host syncs per copy and step (the
counter is read from the device once per run). The arithmetic is the same op for op, so the copies are bit-identical.
"""
from __future__ import annotations

import itertools
from contextlib import contextmanager
from copy import deepcopy

import torch
from lightning.pytorch.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


def ema_label(decay):
    return f"{float(decay):.10g}".replace("-", "m").replace(".", "_")


class EMA(Callback):
    def __init__(self, decays=(0.99, 0.999, 0.9999), active=0.999, min_step=100, foreach=False):
        super().__init__()
        self.foreach, self._averaged = bool(foreach), {}
        self.decays = tuple(float(d) for d in decays)
        self._decay_of = {ema_label(d): d for d in self.decays}
        self.active = float(active)
        if all(abs(self.active - d) > 1e-12 for d in self.decays):
            raise ValueError(f"the active EMA decay {self.active} is not one of {self.decays}")
        self.min_step = int(min_step)
        self._models, self._latest = {}, 0

    @property
    def active_label(self):
        return ema_label(self.active)

    def setup(self, trainer, pl_module, stage=None):
        if stage == "fit":
            self._models = {ema_label(d): AveragedModel(model=pl_module, device=pl_module.device,
                                                        avg_fn=get_ema_avg_fn(decay=d), use_buffers=True)
                            for d in self.decays}
            self._averaged = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step <= self._latest or trainer.global_step - 1 < self.min_step:
            return
        if self.foreach:
            self._update_foreach(pl_module)
        else:
            for m in self._models.values():
                m.update_parameters(pl_module)
        self._latest = trainer.global_step

    @torch.no_grad()
    def _update_foreach(self, pl_module):
        """AveragedModel.update_parameters(use_buffers=True) with get_ema_avg_fn, op for op, as foreach kernels."""
        current = [t.detach() for t in itertools.chain(pl_module.parameters(), pl_module.buffers())]
        for label, m in self._models.items():
            averaged = [t.detach() for t in itertools.chain(m.module.parameters(), m.module.buffers())]
            if label not in self._averaged:            # one host read per copy per run (after any resume)
                self._averaged[label] = bool(m.n_averaged != 0)
            if not self._averaged[label]:
                torch._foreach_copy_(averaged, current)
                self._averaged[label] = True
            else:
                decay = self._decay_of[label]
                torch._foreach_mul_(averaged, decay)
                torch._foreach_add_(averaged, torch._foreach_mul(current, 1 - decay))
            m.n_averaged += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._models:
            self._swap(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._models:
            self._swap(pl_module)

    def state_dict(self):
        return {"latest_update_step": self._latest}

    def load_state_dict(self, state_dict):
        self._latest = state_dict["latest_update_step"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self._models:
            return
        weights, counters = {}, {}
        for label, m in self._models.items():
            state = m.state_dict()
            weights[label] = {k[7:]: v for k, v in state.items() if k.startswith("module.")}
            counters[label] = {k: v for k, v in state.items() if not k.startswith("module.")}
        a = self.active_label
        checkpoint["current_model_state"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = weights[a]
        checkpoint["averaging_state"] = counters[a]
        checkpoint["ema_state_dicts"] = {k: v for k, v in weights.items() if k != a}
        checkpoint["ema_averaging_states"] = {k: v for k, v in counters.items() if k != a}
        checkpoint["ema_decays"] = {ema_label(d): d for d in self.decays}
        checkpoint["ema_active_decay"] = self.active

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if not self._models:
            return
        if "current_model_state" in checkpoint and "averaging_state" in checkpoint:
            states = {self.active_label: {**{"module." + k: v for k, v in checkpoint["state_dict"].items()},
                                          **checkpoint["averaging_state"]}}
            for label, state in checkpoint.get("ema_state_dicts", {}).items():
                states[label] = {**{"module." + k: v for k, v in state.items()},
                                 **checkpoint.get("ema_averaging_states", {}).get(label, {})}
            for label, m in self._models.items():
                if label in states:
                    m.load_state_dict(states[label])
                else:
                    m.module.load_state_dict(deepcopy(checkpoint["current_model_state"]), strict=False)
            pl_module.load_state_dict(checkpoint["current_model_state"])
            return
        for m in self._models.values():
            m.module.load_state_dict(deepcopy(checkpoint["state_dict"]), strict=False)

    def _swap(self, pl_module, decay=None):
        m = self._models[self.active_label if decay is None else ema_label(decay)]
        for a, c in zip(list(m.module.parameters()) + list(m.module.buffers()),
                        list(pl_module.parameters()) + list(pl_module.buffers())):
            tmp = a.data.clone()
            a.data.copy_(c.data)
            c.data.copy_(tmp)

    @contextmanager
    def average_parameters(self, pl_module, decay=None):
        self._swap(pl_module, decay)
        try:
            yield
        finally:
            self._swap(pl_module, decay)
