"""The BTM loss against golden values from an independent implementation (tests/sbm/golden/b2.pt; CPU, fp32, tiny
DiT): loss terms, gradients (256 fixed projections + norm), and the RNG consumed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))
from helpers import VAEStub, project  # noqa: E402,F401  (VAEStub re-exported for the other tests)

from sbm.loss import CBMLoss  # noqa: E402
from sbm.models import build_model  # noqa: E402

REPO = HERE.parents[1]
DEFAULTS = OmegaConf.load(REPO / "configs" / "sbm" / "train.yaml")
RTOL = 1e-5


def tiny(train_pos_embed=False):
    return build_model("DiT-tiny", input_size=8, label_dim=10, train_pos_embed=train_pos_embed)


def loss_cfg(**over):
    return OmegaConf.merge(DEFAULTS.loss, OmegaConf.create(over))


B2 = {  # recipe -> (loss overrides, step)
    "native": ({}, 0),
    "native_restrict": ({"restrict_transport": True}, 0),
    "p_mean_anneal": ({"time": {"p_mean": 0.8, "p_mean_end": 0.0, "anneal_start": 25000, "anneal_steps": 1}}, 30000),
}


@pytest.fixture(scope="module")
def b2():
    return torch.load(HERE / "golden" / "b2.pt", weights_only=False)


def test_golden_file_recipes(b2):
    assert set(b2["recipes"]) == set(B2) and "frechet_stats" not in b2


def test_state_dict_layout(b2):
    """Same keys in the same order as the golden model: checkpoints load, and the parameter order (Muon groups) agrees."""
    assert list(tiny().state_dict()) == list(b2["model"])


@pytest.mark.parametrize("name", list(B2))
def test_b2_recipe(b2, name):
    over, step = B2[name]
    model = tiny()
    model.load_state_dict(b2["model"])
    loss = CBMLoss(loss_cfg(**over), label_dim=10)
    torch.manual_seed(1234)
    want = b2["recipes"][name]
    for call in want["calls"]:
        losses, _ = loss(model, b2["x"], b2["labels"], step)
        got = {k: float(v) for k, v in losses.items()}
        assert set(got) == set(call["losses"])
        for k, v in call["losses"].items():
            assert got[k] == pytest.approx(v, rel=RTOL, abs=1e-6), k
        params = [p for p in model.parameters() if p.requires_grad]
        g = project(torch.cat([x.flatten() for x in torch.autograd.grad(sum(losses.values()), params)]))
        assert g["norm"] == pytest.approx(call["grad"]["norm"], rel=RTOL)
        assert torch.allclose(g["proj"], call["grad"]["proj"], rtol=RTOL, atol=RTOL * call["grad"]["norm"])
    assert float(torch.rand(())) == want["rng_after"], "the loss draws from the RNG in a different order or amount"
