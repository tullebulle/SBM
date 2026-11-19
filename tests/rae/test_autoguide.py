"""loss.guided_cbm.guide.autoguide (autoguidance): the extra term + weight (b(y) - b_weak(y)) on the guided band.

    .venv/bin/python -m pytest tests/rae/test_autoguide.py -q
"""
import copy

import pytest
import torch

from sbm_rae.guided_cbm import GuidedCBMLoss, load_weak_state
from test_guided_cbm import LABELS, batch, grads, our_cfg, tiny_dit


def _loss(weight=None, ckpt="x.ckpt"):
    over = {} if weight is None else {"guide": {"autoguide": {"enabled": True, "ckpt": ckpt, "weight": weight}}}
    return GuidedCBMLoss(our_cfg(**over), label_dim=LABELS)


def _run(loss, model, x, y, seed=7, step=3):
    torch.manual_seed(seed)
    losses, aux = loss(model, None, x, y, step)
    return {k: v.detach() for k, v in losses.items()}, aux, grads(model, losses)


def test_weak_equal_to_teacher_changes_nothing():
    """A weak model identical to the teacher: b(y) - b_weak(y) = 0, so the losses and gradients equal the run without
    autoguidance exactly."""
    model, x, y = tiny_dit(head=True), *batch(n=8)
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.02 * torch.randn_like(p))
    off, on = _loss(), _loss(0.1)
    off.teacher = on.teacher = teacher
    on.weak = copy.deepcopy(teacher)
    l0, _, g0 = _run(off, model, x, y)
    l1, aux, g1 = _run(on, model, x, y)
    for k in l0:
        assert torch.equal(l0[k], l1[k]), k
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert float(aux["autoguide_shift_rms"]) == 0.0


def test_weak_model_moves_the_targets():
    """A weaker (perturbed) weak model: the targets move, everything stays finite, and the shift is reported."""
    model, x, y = tiny_dit(head=True), *batch(n=8)
    teacher = copy.deepcopy(model)
    weak = copy.deepcopy(teacher)
    with torch.no_grad():
        for p in weak.parameters():
            p.add_(0.05 * torch.randn_like(p))
    off, on = _loss(), _loss(0.1)
    off.teacher = on.teacher = teacher
    on.weak = weak
    l0, _, _ = _run(off, model, x, y)
    l1, aux, g1 = _run(on, model, x, y)
    assert any(not torch.equal(l0[k], l1[k]) for k in l0)
    assert all(torch.isfinite(v).all() for v in l1.values()) and all(torch.isfinite(g).all() for g in g1)
    assert float(aux["autoguide_shift_rms"]) > 0


def test_guided_step_without_weak_raises():
    model, x, y = tiny_dit(), *batch(n=8)
    on = _loss(0.1)
    on.teacher = copy.deepcopy(model)
    with pytest.raises(RuntimeError, match="weak model"):
        _run(on, model, x, y)


def test_config_is_checked():
    with pytest.raises(ValueError, match="autoguide"):
        _loss(0.0)                       # weight 0
    with pytest.raises(ValueError, match="autoguide"):
        _loss(0.1, ckpt=None)            # no checkpoint


def test_load_weak_state_picks_the_ema(tmp_path):
    """The active EMA (the checkpoint's state_dict, as sbm_rae.ema saves it) by default, the requested EMA copy otherwise;
    keys lose the "model." prefix, non-model entries are dropped."""
    a, b = {"model.w": torch.zeros(2), "latents_scale": torch.ones(1)}, {"model.w": torch.ones(2)}
    path = tmp_path / "c.ckpt"
    torch.save({"state_dict": a, "ema_state_dicts": {"0_9999": b, "0_99": {"model.w": torch.full((2,), 2.0)}},
                "ema_active_decay": 0.999}, path)
    assert torch.equal(load_weak_state(path)["w"], torch.zeros(2)) and set(load_weak_state(path)) == {"w"}
    assert torch.equal(load_weak_state(path, 0.999)["w"], torch.zeros(2))     # the active decay asked by name
    assert torch.equal(load_weak_state(path, 0.9999)["w"], torch.ones(2))
    with pytest.raises(KeyError):
        load_weak_state(path, 0.5)
