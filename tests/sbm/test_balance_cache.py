"""Optional amortized balancer VJPs (loss.balance.refresh_every); default and resume contracts."""
import copy

import pytest
import torch

import sbm.loss as loss_module
from test_golden import CBMLoss, loss_cfg


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = torch.nn.Linear(2, 1, bias=False)


def make(every=4, reset=False):
    return CBMLoss(loss_cfg(balance={"mode": "ema", "ema": 0.5,
                                    "refresh_every": every, "reset_on_resume": reset}), label_dim=10)


def terms(model, scale=1):
    value = model.final_layer.weight.sum()
    return value * scale, value * (3 * scale)


def test_skips_vjps_and_preserves_decay_horizon(monkeypatch):
    model, loss = Model(), make()
    original = loss_module.grad_norm
    calls = []

    def counted(value, params):
        calls.append(1)
        return original(value, params)

    monkeypatch.setattr(loss_module, "grad_norm", counted)
    first = loss._balance(model, *terms(model))
    previous = tuple(n.clone() for n in loss._balance_norms)
    for _ in range(3):
        current = loss._balance(model, *terms(model, 7))
        assert all(torch.equal(a, b) for a, b in zip(first[:2], current[:2]))
        assert current[2]["balance_refreshed"] == 0
    assert len(calls) == 2
    refreshed = loss._balance(model, *terms(model, 7))
    assert len(calls) == 4 and refreshed[2]["balance_refreshed"] == 1
    for before, after in zip(previous, loss._balance_norms):
        assert torch.allclose(after, (0.5 ** 4) * before + (1 - 0.5 ** 4) * before * 7)


def test_resume_preserves_phase_and_reset_rebootstraps():
    model, loss = Model(), make()
    for _ in range(2):
        loss._balance(model, *terms(model))
    state = copy.deepcopy(loss.state_dict())
    restored = make()
    restored.load_state_dict(state)
    for i in range(5):
        a = loss._balance(model, *terms(model, i + 2))
        b = restored._balance(model, *terms(model, i + 2))
        assert all(torch.equal(x, y) for x, y in zip(a[:2], b[:2]))
        assert a[2]["balance_refreshed"] == b[2]["balance_refreshed"]
    reset = make(reset=True)
    reset.load_state_dict(state)
    assert reset._balance_norms is None and reset._balance_cache is None
    assert reset._balance(model, *terms(model, 8))[2]["balance_refreshed"] == 1


def test_old_checkpoint_and_changed_interval_start_with_fresh_norms():
    model, old = Model(), make(every=1)
    old._balance(model, *terms(model))
    old_state = copy.deepcopy(old.state_dict())
    assert "balance_cache" not in old_state
    new = make()
    new.load_state_dict(old_state)
    before = tuple(n.clone() for n in new._balance_norms)
    new._balance(model, *terms(model, 3))
    for a, b in zip(before, new._balance_norms):
        assert torch.allclose(b, 0.5 * a + 0.5 * a * 3)  # not decay**4 on first resumed call
    changed = make(every=8)
    changed.load_state_dict(new.state_dict())
    assert changed._balance_cache is None
    assert changed._balance(model, *terms(model))[2]["balance_refreshed"] == 1


def test_default_has_no_cache_and_errors():
    model, loss = Model(), make(every=1)
    assert "balance_refreshed" not in loss._balance(model, *terms(model))[2]
    for overrides in ({"refresh_every": 0}, {"mode": "match_smaller_to_larger", "refresh_every": 4}):
        with pytest.raises(ValueError, match="refresh_every"):
            CBMLoss(loss_cfg(balance=overrides), label_dim=10)
