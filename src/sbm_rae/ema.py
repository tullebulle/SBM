from contextlib import contextmanager, nullcontext
from copy import deepcopy

from lightning.pytorch.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn, get_ema_multi_avg_fn


def ema_decay_label(decay):
    return f"{float(decay):.10g}".replace("-", "m").replace(".", "_")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def resolve_ema_decays(ema_cfg):
    decays = ema_cfg.get("decays", None)
    if decays is None:
        decays = ema_cfg.get("decay", 0.9999)

    resolved_decays = tuple(float(decay) for decay in _as_list(decays))
    if not resolved_decays:
        raise ValueError("trainer.ema.decays must contain at least one decay")
    for decay in resolved_decays:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
    return resolved_decays


def resolve_active_ema_decay(ema_cfg, decays=None):
    if decays is None:
        decays = resolve_ema_decays(ema_cfg)

    active_decay = ema_cfg.get("active_decay", None)
    if active_decay is None:
        active_decay = decays[-1]
    active_decay = float(active_decay)

    if all(abs(active_decay - decay) > 1.0e-12 for decay in decays):
        raise ValueError(
            "trainer.ema.active_decay must be one of trainer.ema.decays; "
            f"got {active_decay} for decays={decays}"
        )
    return active_decay


class EMAWeightAveraging(Callback):
    def __init__(
        self,
        decay=None,
        decays=None,
        active_decay=None,
        min_step=100,
        use_buffers=True,
        device=None,
        foreach=False,
    ):
        super().__init__()
        self.foreach = bool(foreach)   # speed (trainer.ema.foreach): one multi-tensor lerp per update instead of a per-tensor loop
        if decays is None:
            decays = decay if decay is not None else 0.9999

        self.decays = tuple(float(decay) for decay in _as_list(decays))
        if not self.decays:
            raise ValueError("decays must contain at least one EMA decay")
        for decay in self.decays:
            if not 0.0 <= decay < 1.0:
                raise ValueError(f"EMA decay must be in [0, 1), got {decay}")

        if active_decay is None:
            active_decay = self.decays[-1]
        self.active_decay = float(active_decay)
        if all(abs(self.active_decay - decay) > 1.0e-12 for decay in self.decays):
            raise ValueError(
                "active_decay must be one of decays; "
                f"got {self.active_decay} for decays={self.decays}"
            )

        self.min_step = int(min_step)
        self.use_buffers = bool(use_buffers)
        self.device = device
        self._average_models = {}
        self._latest_update_step = 0

    @property
    def active_label(self):
        return ema_decay_label(self.active_decay)

    @property
    def decay_labels(self):
        return {decay: ema_decay_label(decay) for decay in self.decays}

    @property
    def _average_model(self):
        return self.get_average_model()

    def get_average_model(self, decay=None):
        label = self.active_label if decay is None else ema_decay_label(decay)
        return self._average_models.get(label)

    def setup(self, trainer, pl_module, stage=None):
        del trainer
        if stage != "fit":
            return

        device = self.device or pl_module.device
        self._average_models = {
            ema_decay_label(decay): AveragedModel(
                model=pl_module,
                device=device,
                avg_fn=None if self.foreach else get_ema_avg_fn(decay=decay),
                multi_avg_fn=get_ema_multi_avg_fn(decay=decay) if self.foreach else None,
                use_buffers=self.use_buffers,
            )
            for decay in self.decays
        }

    def should_update(self, step_idx=None, epoch_idx=None):
        del epoch_idx
        return (step_idx is not None) and (step_idx >= self.min_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        step_idx = trainer.global_step - 1
        if trainer.global_step <= self._latest_update_step:
            return
        if not self.should_update(step_idx=step_idx):
            return

        for average_model in self._average_models.values():
            average_model.update_parameters(pl_module)
        self._latest_update_step = trainer.global_step

    def on_validation_epoch_start(self, trainer, pl_module):
        del trainer
        if self._average_model is not None:
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        del trainer
        if self._average_model is not None:
            self._swap_models(pl_module)

    def on_train_end(self, trainer, pl_module):
        del trainer
        if self._average_model is not None:
            self._copy_average_to_current(pl_module)

    def state_dict(self):
        return {"latest_update_step": self._latest_update_step}

    def load_state_dict(self, state_dict):
        self._latest_update_step = state_dict["latest_update_step"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        del trainer, pl_module
        if not self._average_models:
            return

        average_states = {}
        averaging_states = {}
        for label, average_model in self._average_models.items():
            state = average_model.state_dict()
            average_states[label] = {
                name[7:]: value for name, value in state.items() if name.startswith("module.")
            }
            averaging_states[label] = {
                name: value for name, value in state.items() if not name.startswith("module.")
            }

        active_label = self.active_label
        checkpoint["current_model_state"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = average_states[active_label]
        checkpoint["averaging_state"] = averaging_states[active_label]
        checkpoint["ema_state_dicts"] = {
            label: state for label, state in average_states.items() if label != active_label
        }
        checkpoint["ema_averaging_states"] = {
            label: state for label, state in averaging_states.items() if label != active_label
        }
        checkpoint["ema_decays"] = {ema_decay_label(decay): decay for decay in self.decays}
        checkpoint["ema_active_decay"] = self.active_decay

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        del trainer
        if not self._average_models:
            return

        average_states = {}
        if "current_model_state" in checkpoint and "averaging_state" in checkpoint:
            active_state = {
                "module." + name: value for name, value in checkpoint["state_dict"].items()
            }
            active_state.update(checkpoint["averaging_state"])
            average_states[self.active_label] = active_state

            for label, state in checkpoint.get("ema_state_dicts", {}).items():
                average_state = {"module." + name: value for name, value in state.items()}
                average_state.update(checkpoint.get("ema_averaging_states", {}).get(label, {}))
                average_states[label] = average_state

            for label, average_model in self._average_models.items():
                if label in average_states:
                    average_model.load_state_dict(average_states[label])
                else:
                    average_model.module.load_state_dict(
                        deepcopy(checkpoint["current_model_state"]),
                        strict=False,
                    )
            pl_module.load_state_dict(checkpoint["current_model_state"])
            return

        for average_model in self._average_models.values():
            average_model.module.load_state_dict(deepcopy(checkpoint["state_dict"]), strict=False)

    def _swap_models(self, pl_module, decay=None):
        average_model = self.get_average_model(decay=decay)
        assert average_model is not None
        average_params = list(average_model.module.parameters()) + list(average_model.module.buffers())
        current_params = list(pl_module.parameters()) + list(pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module, decay=None):
        average_model = self.get_average_model(decay=decay)
        assert average_model is not None
        average_params = list(average_model.module.parameters()) + list(average_model.module.buffers())
        current_params = list(pl_module.parameters()) + list(pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)

    @contextmanager
    def average_parameters(self, pl_module, decay=None):
        average_model = self.get_average_model(decay=decay)
        if average_model is None:
            with nullcontext():
                yield
            return

        self._swap_models(pl_module, decay=decay)
        try:
            yield
        finally:
            self._swap_models(pl_module, decay=decay)