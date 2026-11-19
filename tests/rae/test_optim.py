import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.optim import ChainedOptimizer
from sbm_rae.trainer import TrainingModule


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.embedding = torch.nn.Embedding(5, 3)

    def forward(self, x, labels=None):
        del labels
        return self.linear(x)


def _module_with_muon_optimizer():
    cfg = OmegaConf.create(
        {
            "dataset": {"use_precomputed_latents": True},
            "model": {"in_channels": 1},
            "vae": {},
            "trainer": {"overfitting": {"enabled": False}, "ema": {"enabled": False}},
            "logging": {"loss_grad_norms": {"enabled": False}},
            "optimizer": "Muon",
            "weight_decay": 0.0,
            "lr": {"val": 1.0e-3, "warmup_steps": 0},
            "muon": {"adamw_betas": [0.9, 0.95]},
        }
    )
    return TrainingModule(cfg, _ToyModel(), loss_fn=None)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon is unavailable")
def test_muon_optimizer_uses_adamw_for_non_muon_parameters():
    training_module = _module_with_muon_optimizer()
    optimizer = training_module.configure_optimizers()["optimizer"]

    assert isinstance(optimizer, ChainedOptimizer)
    assert isinstance(optimizer.optimizers[0], torch.optim.Muon)
    assert isinstance(optimizer.optimizers[1], torch.optim.AdamW)
    assert optimizer.optimizers[0].param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.optimizers[1].param_groups[0]["lr"] == pytest.approx(1.0e-4)

    muon_params = {
        id(param)
        for param_group in optimizer.optimizers[0].param_groups
        for param in param_group["params"]
    }
    adamw_params = {
        id(param)
        for param_group in optimizer.optimizers[1].param_groups
        for param in param_group["params"]
    }

    assert id(training_module.model.linear.weight) in muon_params
    assert id(training_module.model.linear.bias) in adamw_params
    assert id(training_module.model.embedding.weight) in adamw_params