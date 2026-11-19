from contextlib import nullcontext

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.ema import EMAWeightAveraging
from sbm_rae.trainer import (
    IterativeSamplingCallback,
    TrainingModule,
    overfitting_invariant_plot_enabled,
    train_interpolation_plot_enabled,
)


def _training_module(loss_grad_norms=None):
    if loss_grad_norms is None:
        loss_grad_norms = {"enabled": True, "every_n_steps": 1}

    trainer_cfg = {
        "log_every_n_steps": 1,
        "overfitting": {"enabled": False},
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
    }

    cfg = OmegaConf.create(
        {
            "dataset": {"use_precomputed_latents": True},
            "model": {"in_channels": 1},
            "vae": {},
            "trainer": trainer_cfg,
            "logging": {"loss_grad_norms": loss_grad_norms},
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "lr": {"val": 1.0e-3, "warmup_steps": 0},
        }
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    return TrainingModule(cfg, model, loss_fn=None)


def test_compute_loss_grad_norm_uses_each_loss_term_independently():
    training_module = _training_module()
    parameters = training_module._trainable_parameters()

    linear_loss = 3.0 * training_module.model.weight.sum()
    quadratic_loss = training_module.model.weight.square().sum()

    torch.testing.assert_close(
        training_module._compute_loss_grad_norm(linear_loss, parameters),
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        training_module._compute_loss_grad_norm(quadratic_loss, parameters),
        torch.tensor(4.0),
    )
    assert training_module.model.weight.grad is None


def test_log_loss_grad_norms_skips_computation_when_disabled():
    training_module = _training_module(loss_grad_norms={"enabled": False})
    calls = []

    def compute_loss_grad_norm(loss, parameters):
        calls.append((loss, parameters))
        return torch.zeros(())

    training_module._compute_loss_grad_norm = compute_loss_grad_norm
    training_module._log_loss_grad_norms({"loss": training_module.model.weight.sum()})

    assert calls == []


def test_multi_ema_callback_updates_each_decay_and_restores_current_weights():
    training_module = _training_module()
    ema_callback = EMAWeightAveraging(
        decays=[0.5, 0.9],
        active_decay=0.9,
        min_step=0,
        use_buffers=False,
    )
    trainer = type("Trainer", (), {"global_step": 0})()
    ema_callback.setup(trainer, training_module, stage="fit")

    with torch.no_grad():
        training_module.model.weight.fill_(4.0)
    trainer.global_step = 1
    ema_callback.on_train_batch_end(trainer, training_module, outputs=None, batch=None, batch_idx=0)

    with torch.no_grad():
        training_module.model.weight.fill_(6.0)
    trainer.global_step = 2
    ema_callback.on_train_batch_end(trainer, training_module, outputs=None, batch=None, batch_idx=1)

    torch.testing.assert_close(
        ema_callback.get_average_model(0.5).module.model.weight,
        torch.tensor([[5.0]]),
    )
    torch.testing.assert_close(
        ema_callback.get_average_model(0.9).module.model.weight,
        torch.tensor([[4.2]]),
    )

    with ema_callback.average_parameters(training_module, decay=0.5):
        torch.testing.assert_close(training_module.model.weight, torch.tensor([[5.0]]))

    torch.testing.assert_close(training_module.model.weight, torch.tensor([[6.0]]))


def test_overfitting_invariant_plot_uses_its_own_periodic_gate():
    cfg = OmegaConf.create(
        {
            "seed": 123,
            "model": {"in_channels": 1, "input_size": 2},
            "trainer": {
                "overfitting": {
                    "enabled": True,
                    "invariant_plot": {"enabled": True, "every_n_steps": 5, "max_images": 3},
                },
                "ema": {"enabled": False},
            },
            "sampling": {"enabled": False, "every_n_steps": 100, "n_samples": 8, "vae_batch_size": 2},
        }
    )
    callback = IterativeSamplingCallback(cfg, test_labels=None, inverse_scaler=lambda images: images)

    assert overfitting_invariant_plot_enabled(cfg)
    assert callback._should_run_overfitting_invariant(10)
    assert not callback._should_run_iterative_sampling(100)

    callback._last_overfitting_invariant_step = 10
    assert not callback._should_run_overfitting_invariant(10)
    assert not callback._should_run_overfitting_invariant(11)
    assert callback._overfitting_invariant_max_images() == 3


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, labels=None):
        del labels
        self.inputs.append(x.detach().clone())
        return x + 1.0


class _GuidanceCapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.omegas = []
        self.ws = []

    def forward(self, x, labels=None, omega=None, w=None):
        del labels
        if omega is None:
            raise AssertionError("omega must be provided for validation anchor plots")
        if w is None:
            raise AssertionError("w must be provided for validation guidance plots")
        self.omegas.append(omega.detach().cpu().clone())
        self.ws.append(w.detach().cpu().clone())
        return x + omega.view(-1, 1, 1, 1).to(dtype=x.dtype) + w.view(-1, 1, 1, 1).to(dtype=x.dtype)


class _FakeSamplingModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = _CountingModel()
        self.use_precomputed_latents = True
        self.global_step = 2
        self.training = True
        self.logger = None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeGuidanceSamplingModule(_FakeSamplingModule):
    def __init__(self):
        super().__init__()
        self.model = _GuidanceCapturingModel()


class _FakeTrainer:
    def __init__(self, global_step):
        self.global_step = global_step
        self.world_size = 1
        self.global_rank = 0
        self.is_global_zero = True
        self.callbacks = []


class _FakeEMACallback:
    decays = (0.999, 0.9999)

    def average_parameters(self, pl_module, decay=None):
        del pl_module, decay
        return nullcontext()


def test_sampling_callback_uses_each_ema_decay_label_for_artifacts():
    cfg = OmegaConf.create(
        {
            "seed": 123,
            "work_dir": ".",
            "model": {"in_channels": 1, "input_size": 2},
            "trainer": {
                "ema": {"enabled": True, "decays": [0.999, 0.9999], "active_decay": 0.9999},
            },
            "sampling": {"enabled": False, "n_samples": 2, "vae_batch_size": 2},
        }
    )
    callback = IterativeSamplingCallback(cfg, test_labels=None, inverse_scaler=lambda images: images)
    trainer = _FakeTrainer(global_step=2)
    trainer.callbacks = [_FakeEMACallback()]

    contexts = callback._ema_weight_contexts_for_sampling(trainer, _FakeSamplingModule())

    assert [label for label, _ in contexts] == ["0_999", "0_9999"]
    callback._current_ema_label = "0_9999"
    assert callback._artifact_suffix() == "_ema_0_9999"
    assert callback._artifact_log_prefix("train") == "train_ema_0_9999"


class _CapturingInterpolationCallback(IterativeSamplingCallback):
    def __init__(self, cfg):
        super().__init__(cfg, test_labels=None, inverse_scaler=lambda images: images)
        self.saved_rows = None
        self.saved_t_values = None

    def _decode(self, pl_module, latents, vae_batch_size):
        del pl_module, vae_batch_size
        return latents

    def _save_train_interpolation_plot(self, pl_module, prediction_rows, t_values):
        del pl_module
        self.saved_rows = prediction_rows
        self.saved_t_values = t_values


class _CapturingSamplingCallback(IterativeSamplingCallback):
    def __init__(self, cfg):
        super().__init__(cfg, test_labels=None, inverse_scaler=lambda images: images)
        self.saved_samples = None
        self.saved_iteration_step = None
        self.saved_conditioning_values = None

    def _decode(self, pl_module, latents, vae_batch_size):
        del pl_module, vae_batch_size
        return latents

    def _save_samples(self, pl_module, samples, iteration_step, conditioning_values=None):
        del pl_module
        self.saved_samples = samples
        self.saved_iteration_step = iteration_step
        self.saved_conditioning_values = conditioning_values


@pytest.mark.xfail(
    reason="known failure of the validation-sampling grid plot (sampling.enabled is off in the released recipes)",
    strict=False,
)
def test_validation_sampling_plots_anchor_guidance_condition_grid():
    cfg = OmegaConf.create(
        {
            "seed": 123,
            "work_dir": ".",
            "anchor": {"enabled": True, "alpha": 0.4},
            "guidance": {"enabled": True, "w_min": 0.5, "class_drop_prob": 0.1, "beta": 1.0},
            "model": {"in_channels": 1, "input_size": 2},
            "trainer": {
                "overfitting": {"enabled": False},
                "ema": {"enabled": False},
            },
            "sampling": {
                "enabled": True,
                "every_n_steps": 1,
                "n_samples": 2,
                "vae_batch_size": 2,
            },
        }
    )
    callback = _CapturingSamplingCallback(cfg)
    callback.steps_to_plot = [1]
    callback.max_steps = 1
    module = _FakeGuidanceSamplingModule()
    trainer = _FakeTrainer(global_step=1)
    callback.setup(trainer, module)

    callback._run_distributed_sampling(module, trainer)

    expected_anchor_temperatures = torch.linspace(0.6, 1.0, steps=3)
    expected_guidance_temperatures = torch.tensor([0.5, 0.75, 1.0])
    expected_conditioning_values = [
        (float(anchor_temperature), float(guidance_temperature))
        for anchor_temperature in expected_anchor_temperatures
        for guidance_temperature in expected_guidance_temperatures
    ]
    assert callback.saved_conditioning_values == expected_conditioning_values
    assert callback.saved_iteration_step == 1
    assert callback.saved_samples.shape == (30, 1, 2, 2)
    assert len(module.model.omegas) == 15
    assert len(module.model.ws) == 15
    for omega, w, (expected_anchor, expected_guidance) in zip(
        module.model.omegas,
        module.model.ws,
        expected_conditioning_values,
    ):
        torch.testing.assert_close(
            omega,
            torch.full((2,), expected_anchor),
        )
        torch.testing.assert_close(
            w,
            torch.full((2,), expected_guidance),
        )


def test_train_interpolation_plot_uses_cached_first_train_batch_and_separate_t_values():
    cfg = OmegaConf.create(
        {
            "seed": 123,
            "work_dir": ".",
            "model": {"in_channels": 1, "input_size": 2},
            "trainer": {
                "per_device_batch_size": 2,
                "overfitting": {"enabled": False},
                "ema": {"enabled": False},
            },
            "sampling": {
                "enabled": False,
                "every_n_steps": 2,
                "n_samples": 2,
                "vae_batch_size": 2,
                "train_interpolation_plot": {
                    "enabled": True,
                    "every_n_steps": 2,
                    "max_images": 2,
                    "t_values": [0.0, 0.5, 1.0],
                    "seed_offset": 7,
                },
            },
        }
    )
    callback = _CapturingInterpolationCallback(cfg)
    module = _FakeSamplingModule()
    trainer = _FakeTrainer(global_step=1)
    callback.setup(trainer, module)

    first_batch = (
        torch.arange(12, dtype=torch.float32).view(3, 1, 2, 2),
        torch.tensor([1, 2, 3], dtype=torch.long),
    )
    second_batch = (first_batch[0] + 100.0, first_batch[1] + 10)

    callback.on_train_batch_end(trainer, module, outputs=None, batch=first_batch, batch_idx=0)
    trainer.global_step = 2
    module.global_step = 2
    callback.on_train_batch_end(trainer, module, outputs=None, batch=second_batch, batch_idx=1)

    assert train_interpolation_plot_enabled(cfg)
    assert callback.saved_t_values == [0.0, 0.5, 1.0]
    assert len(module.model.inputs) == 3
    assert all(model_input.shape == (2, 1, 2, 2) for model_input in module.model.inputs)
    torch.testing.assert_close(module.model.inputs[-1], first_batch[0][:2])
    assert callback.saved_rows is not None
    assert len(callback.saved_rows) == 3