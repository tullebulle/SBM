"""Make a changed `lr.val` actually take effect when resuming a checkpoint.

Lightning's resume path restores the optimizer state *and* the LR scheduler
state after `configure_optimizers` has run. Both carry a learning rate:
`optimizer.param_groups[i]["lr"]`, and the scheduler's `base_lrs`, which
`_LRScheduler.load_state_dict` overwrites because it is a plain attribute.
Every scheduler step then recomputes the LR from the restored `base_lrs`, so a
new `lr.val` on the command line is silently discarded and training continues
at the old rate.

`train/lr` is logged, so the fix is verifiable: after enabling this the logged
LR must equal the requested one.

Enable with `lr.force_on_resume=true`. It is off by default because silently
overriding a restored schedule is the wrong default for an ordinary requeue --
this is for the deliberate case where the schedule is meant to change.
"""

from lightning.pytorch.callbacks import Callback


def _iter_schedulers(scheduler):
    """Yield `scheduler` and any nested children (SequentialLR, ChainedScheduler)."""
    yield scheduler
    for attribute in ("_schedulers", "_scheduler_list"):
        for child in getattr(scheduler, attribute, None) or []:
            yield from _iter_schedulers(child)


class ForceLearningRate(Callback):
    """Reassert the configured LR after Lightning has restored checkpoint state.

    `on_train_start` is the first hook that runs after optimizer and scheduler
    restore, which is why the override lives here rather than in
    `configure_optimizers`.
    """

    def __init__(self, learning_rate, enabled=True):
        self.learning_rate = float(learning_rate)
        self.enabled = bool(enabled)

    def on_train_start(self, trainer, pl_module):
        if not self.enabled:
            return

        scales = getattr(pl_module, "_lr_group_scales", None)

        for optimizer in trainer.optimizers:
            for index, group in enumerate(optimizer.param_groups):
                # Keep the Muon/AdamW ratio; fall back to the group's own ratio
                # against the restored LR if the scales were never recorded.
                scale = scales[index] if scales and index < len(scales) else 1.0
                group["lr"] = self.learning_rate * scale
                # Some schedulers read initial_lr rather than base_lrs.
                if "initial_lr" in group:
                    group["initial_lr"] = self.learning_rate * scale

        for config in trainer.lr_scheduler_configs:
            for scheduler in _iter_schedulers(config.scheduler):
                base = getattr(scheduler, "base_lrs", None)
                if not base:
                    continue
                scheduler.base_lrs = [
                    self.learning_rate * (scales[i] if scales and i < len(scales) else 1.0)
                    for i in range(len(base))
                ]
                scheduler._last_lr = list(scheduler.base_lrs)

        resolved = [g["lr"] for o in trainer.optimizers for g in o.param_groups]
        print(f"ForceLearningRate: param group LRs set to {resolved}")
