"""freeze.trainable: a head-only run trains only the DDT head, and a resume from a checkpoint of the full
model keeps the head's optimizer state (TrainingModule._drop_frozen_optimizer_state).

    .venv/bin/python -m pytest tests/rae/test_freeze.py -q
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.models.dit import DiT
from sbm_rae.trainer import TrainingModule

pytestmark = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon is unavailable")
TINY = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2, num_heads=4, label_dim=10, attn_func="default",
            head_depth=1, head_hidden_size=64, head_num_heads=4)


def module(trainable=None, seed=0):
    cfg = OmegaConf.create({
        "dataset": {"use_precomputed_latents": True}, "model": {"in_channels": 4}, "vae": {},
        "trainer": {"overfitting": {"enabled": False}, "ema": {"enabled": False}},
        "logging": {"loss_grad_norms": {"enabled": False}},
        "optimizer": "Muon", "weight_decay": 0.0, "lr": {"val": 1.0e-3, "warmup_steps": 0, "schedule": "constant"},
        "muon": {"adamw_betas": [0.9, 0.95], "adam_lr": 1.0e-4},
        "loss": {"name": "guided_cbm", "guided_cbm": {"guide": {"kappa": 0.7273}}},
        "freeze": {"trainable": trainable},
    })
    torch.manual_seed(seed)
    return TrainingModule(cfg, DiT(**TINY).float(), loss_fn=None)


def optimizer(m):
    return m.configure_optimizers()["optimizer"]


def step_with_fake_grads(m, opt, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g)
    opt.step()


def test_head_preset_freezes_the_trunk():
    m = module("head")
    names = {n for n, p in m.model.named_parameters() if p.requires_grad}
    assert names and all(n.startswith(("head_blocks.", "head_x_embedder.", "final_layer.")) for n in names)
    assert not any(n.startswith(("blocks.", "x_embedder.", "y_embedder.", "omega_embedder.")) for n in names)
    assert not m.model.pos_embed.requires_grad and not m.model.blocks[0].attn.qkv.weight.requires_grad
    muon, adamw = m._split_muon_parameters()
    assert muon and adamw
    full = module()
    assert [len(x) for x in full._split_muon_parameters()] == [len(x) for x in m._split_muon_parameters(include_frozen=True)]
    assert sum(len(x) for x in (muon, adamw)) < sum(len(x) for x in full._split_muon_parameters())


def test_off_by_default():
    m = module()
    assert m._freeze_regex is None and all(p.requires_grad for n, p in m.model.named_parameters() if not n.endswith("pos_embed"))


def test_resume_keeps_the_heads_optimizer_state():
    full = module()
    opt = optimizer(full)
    step_with_fake_grads(full, opt)
    ckpt = {"optimizer_states": [copy.deepcopy(opt.state_dict())]}
    names_full = {id(p): n for n, p in full.named_parameters()}
    saved = {}
    for i, params in enumerate(full._split_muon_parameters()):
        for j, p in enumerate(params):
            saved[names_full[id(p)]] = ckpt["optimizer_states"][0]["optimizers"][i]["state"][j]

    frozen = module("head")
    frozen.on_load_checkpoint(ckpt)
    opt2 = optimizer(frozen)
    opt2.load_state_dict(ckpt["optimizer_states"][0])          # fits the reduced groups
    names = {id(p): n for n, p in frozen.named_parameters()}
    checked = 0
    for i, params in enumerate(frozen._split_muon_parameters()):
        inner = opt2.optimizers[i]
        assert len(inner.param_groups) == 1 and len(inner.param_groups[0]["params"]) == len(params)
        for p in params:
            for k, v in saved[names[id(p)]].items():
                if torch.is_tensor(v):
                    assert torch.equal(inner.state[p][k], v), (names[id(p)], k)
                    checked += 1
    assert checked > 0
    # a reduced state resumes again unchanged (a requeue or time-limit leg of the frozen run)
    before = [list(o["param_groups"][0]["params"]) for o in ckpt["optimizer_states"][0]["optimizers"]]
    frozen.on_load_checkpoint(ckpt)
    assert [list(o["param_groups"][0]["params"]) for o in ckpt["optimizer_states"][0]["optimizers"]] == before
    step_with_fake_grads(frozen, opt2, seed=2)
