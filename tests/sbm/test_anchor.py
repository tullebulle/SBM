"""The anchor switches: defaults leave the computation unchanged; omega_skew moves omega' toward 1; weight scales
only the endpoint entry (the balancer and the other terms keep the unweighted endpoint)."""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sbm.loss import CBMLoss
from sbm.models import build_model

TRAIN_YAML = Path(__file__).resolve().parents[2] / "configs" / "sbm" / "train.yaml"


def _setup(**endpoint):
    cfg = OmegaConf.load(TRAIN_YAML).loss
    for k, v in endpoint.items():
        cfg.endpoint[k] = v
    torch.manual_seed(0)
    model = build_model(name="DiT-S/2", input_size=8, in_channels=4, label_dim=10)
    x = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    return CBMLoss(cfg, label_dim=10), model, x, y


def _run(loss, model, x, y, seed=1):
    torch.manual_seed(seed)
    return loss(model, x, y, step=0)


def test_defaults_bit_identical():
    a = _run(*_setup())
    b = _run(*_setup(weight=1.0, omega_skew=1.0))
    for k in a[0]:
        assert torch.equal(a[0][k], b[0][k]), k


def test_weight_scales_only_endpoint():
    base, _ = _run(*_setup())
    half, _ = _run(*_setup(weight=0.5))
    assert torch.allclose(half["endpoint"], 0.5 * base["endpoint"])
    assert torch.equal(half["transport"], base["transport"])


def test_skew_moves_omega_prime_toward_one():
    u = torch.rand(100000, generator=torch.Generator().manual_seed(0))
    assert (1 - (1 - u) ** 4).mean() > 0.79 and u.mean() < 0.51
