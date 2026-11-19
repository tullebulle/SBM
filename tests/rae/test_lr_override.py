"""The LR-on-resume regression: a changed lr.val must survive checkpoint restore.

An intended drift anneal ran 40k steps at the old LR because Lightning restores
`base_lrs` from the checkpoint, and the run's loss kept falling anyway, which
made the no-op look like a successful anneal.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sbm_rae.lr_override import ForceLearningRate


def _optimizer_and_scheduler(lr, warmup_steps=5000):
    """Mirror trainer.configure_optimizers: two groups (Muon 1x, AdamW 0.1x)."""
    parameters = [torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(2))]
    optimizer = torch.optim.AdamW(
        [{"params": [parameters[0]], "lr": lr}, {"params": [parameters[1]], "lr": 0.1 * lr}]
    )
    constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=10**9)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, constant], milestones=[warmup_steps]
        )
    else:
        scheduler = constant
    return optimizer, scheduler


def _fake_trainer(optimizer, scheduler):
    config = types.SimpleNamespace(scheduler=scheduler)
    return types.SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[config])


def _resumed_at(new_lr, warmup_steps=5000, steps_before=6000):
    """Train at 1e-3, checkpoint, then rebuild at `new_lr` and restore."""
    old_optimizer, old_scheduler = _optimizer_and_scheduler(1e-3, warmup_steps)
    for _ in range(steps_before):
        old_scheduler.step()
    saved = (old_optimizer.state_dict(), old_scheduler.state_dict())

    optimizer, scheduler = _optimizer_and_scheduler(new_lr, warmup_steps)
    optimizer.load_state_dict(saved[0])
    scheduler.load_state_dict(saved[1])
    return optimizer, scheduler


def test_resume_silently_reverts_lr_without_the_override():
    """Documents the bug, so the test fails loudly if Lightning ever changes."""
    optimizer, scheduler = _resumed_at(1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)


@pytest.mark.parametrize("warmup_steps", [0, 5000])
def test_force_learning_rate_survives_restore_and_stepping(warmup_steps):
    optimizer, scheduler = _resumed_at(1e-4, warmup_steps=warmup_steps)
    module = types.SimpleNamespace(_lr_group_scales=[1.0, 0.1])

    ForceLearningRate(1e-4).on_train_start(_fake_trainer(optimizer, scheduler), module)

    # Must hold immediately and stay held: the scheduler recomputes the LR from
    # base_lrs on every step, so overriding param_groups alone would not last.
    for _ in range(50):
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-6)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5, rel=1e-6)


def test_disabled_override_is_a_no_op():
    optimizer, scheduler = _resumed_at(1e-4)
    module = types.SimpleNamespace(_lr_group_scales=[1.0, 0.1])
    ForceLearningRate(1e-4, enabled=False).on_train_start(_fake_trainer(optimizer, scheduler), module)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)


def test_trainer_records_group_scales():
    """The Muon/AdamW ratio must come from the trainer, not be hard-coded."""
    source = (SRC / "sbm_rae" / "trainer.py").read_text()
    assert "_lr_group_scales" in source


def test_cosine_factor_is_flat_then_cosine_then_floor():
    from sbm_rae.trainer import cosine_factor
    assert cosine_factor(0, 100, 200) == 1.0
    assert cosine_factor(99, 100, 200) == 1.0
    assert abs(cosine_factor(150, 100, 200) - 0.5) < 1e-12
    assert cosine_factor(200, 100, 200) == 0.0
    assert cosine_factor(10_000, 100, 200) == 0.0
    assert abs(cosine_factor(150, 100, 200, floor=0.1) - 0.55) < 1e-12
    assert cosine_factor(300, 100, 200, floor=0.1) == 0.1


def test_shipped_config_defaults_to_the_constant_schedule():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs" / "rae"), version_base=None):
        cfg = compose(config_name="train")
        cos = compose(config_name="train", overrides=["lr.schedule=cosine", "lr.decay_start_step=120000", "lr.decay_end_step=160000"])
    assert cfg.lr.schedule == "constant"
    assert cos.lr.schedule == "cosine" and cos.lr.decay_end_step == 160000


def test_cosine_scheduler_accepts_a_constant_schedule_checkpoint_state():
    """Every checkpoint so far carries a ConstantLR state; resuming under cosine must not raise."""
    import torch
    from omegaconf import OmegaConf
    from sbm_rae.trainer import TrainingModule, cosine_factor

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.final_layer = torch.nn.Linear(4, 4)

    def module(schedule):
        cfg = OmegaConf.create({"optimizer": "adamw", "weight_decay": 0.0, "lr": {"val": 1e-3, "warmup_steps": 50, "force_on_resume": True,
               "schedule": schedule, "decay_start_step": 1000, "decay_end_step": 2000, "min_factor": 0.0},
               "loss": {}, "trainer": {}, "sampling": {"enabled": False}})
        # Bare construction: the real initialiser needs the dataset, data and
        # sampling blocks, none of which the scheduler touches. The two private
        # attributes are what LightningModule's `trainer` property reads.
        m = TrainingModule.__new__(TrainingModule); torch.nn.Module.__init__(m)
        m._trainer = None; m._jit_is_scripting = False; m._fabric = None; m.cfg = cfg; m.model = Tiny()
        return m

    mc = module("constant"); oc = mc.configure_optimizers(); sc = oc["lr_scheduler"]["scheduler"]
    for _ in range(80):
        oc["optimizer"].step(); sc.step()
    saved = sc.state_dict()

    mk = module("cosine"); ok = mk.configure_optimizers(); sk = ok["lr_scheduler"]["scheduler"]
    sk.load_state_dict(saved)

    class T:
        global_step = 0
    mk._trainer = T()   # what Lightning's setter stores; bypasses its type check
    for step in (1000, 1500, 2000, 3000):
        mk.trainer.global_step = step; ok["optimizer"].step(); sk.step()
        assert abs(ok["optimizer"].param_groups[0]["lr"] - 1e-3 * cosine_factor(step, 1000, 2000)) < 1e-12
