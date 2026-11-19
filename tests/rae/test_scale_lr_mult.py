"""loss.guided_cbm.guide.scale_cond.lr_mult: the guidance-scale embedding's own learning rate, and the resume that splits
an optimizer state written without it.

    .venv/bin/python -m pytest tests/rae/test_scale_lr_mult.py -q
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.models.dit import DiT
from sbm_rae.optim import ChainedOptimizer
from sbm_rae.trainer import TrainingModule
from test_scale_cond import TINY

pytestmark = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon is unavailable")


def module(lr_mult=None, scale_cond=True, batched=False, optimizer="Muon", seed=0):
    sc = {"enabled": scale_cond, "lo": 2.0, "hi": 8.0}
    if lr_mult is not None:
        sc["lr_mult"] = lr_mult
    cfg = OmegaConf.create({
        "dataset": {"use_precomputed_latents": True}, "model": {"in_channels": 4}, "vae": {},
        "trainer": {"overfitting": {"enabled": False}, "ema": {"enabled": False}},
        "logging": {"loss_grad_norms": {"enabled": False}},
        "optimizer": optimizer, "weight_decay": 0.0, "lr": {"val": 1.0e-3, "warmup_steps": 0, "schedule": "constant"},
        "muon": {"adamw_betas": [0.9, 0.95], "adam_lr": 1.0e-4, "batched": batched},
        "loss": {"name": "guided_cbm", "guided_cbm": {"guide": {"kappa": 0.7273, "scale_cond": sc}}},
    })
    torch.manual_seed(seed)
    return TrainingModule(cfg, DiT(guidance_enabled=True, **TINY).float(), loss_fn=None)


def opt_and_sched(m):
    out = m.configure_optimizers()
    return out["optimizer"], out["lr_scheduler"]["scheduler"]


def names(m):
    return {id(p): n for n, p in m.named_parameters()}


def fake_grads(m, seed):
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def test_off_by_default_one_group_per_optimizer():
    """lr_mult unset or 1 (or scale_cond off): the stock optimizer, one param group each for Muon and AdamW."""
    for m in (module(), module(lr_mult=1.0), module(lr_mult=10.0, scale_cond=False)):
        opt, _ = opt_and_sched(m)
        assert isinstance(opt, ChainedOptimizer)
        muon, adamw = m._split_muon_parameters()
        assert [len(o.param_groups) for o in opt.optimizers] == [1, 1]
        assert opt.optimizers[0].param_groups[0]["params"] == muon
        assert opt.optimizers[1].param_groups[0]["params"] == adamw


def test_lr_mult_gives_the_scale_embedding_its_own_groups():
    m = module(lr_mult=10.0)
    opt, _ = opt_and_sched(m)
    nm = names(m)
    muon, adamw = opt.optimizers
    assert [g["lr"] for g in muon.param_groups] == pytest.approx([1e-3, 1e-2])
    assert [g["lr"] for g in adamw.param_groups] == pytest.approx([1e-4, 1e-3])
    for o in (muon, adamw):
        assert all("w_embedder" not in nm[id(p)] for p in o.param_groups[0]["params"])
        assert o.param_groups[1]["params"] and all("w_embedder" in nm[id(p)] for p in o.param_groups[1]["params"])
    covered = [id(p) for o in opt.optimizers for g in o.param_groups for p in g["params"]]
    assert sorted(covered) == sorted(id(p) for p in m.parameters() if p.requires_grad)
    assert {nm[id(p)] for g in (muon.param_groups[1], adamw.param_groups[1]) for p in g["params"]} == \
        {"model.w_embedder.mlp.0.weight", "model.w_embedder.mlp.2.weight", "model.w_embedder.mlp.0.bias",
         "model.w_embedder.mlp.2.bias"}


@pytest.mark.parametrize("batched", [False, True])
def test_only_the_embedding_moves_faster(batched):
    """One step from the same weights, momentum and gradients: every other parameter moves exactly as at 1x, the
    embedding's Muon matrices and AdamW biases move lr_mult x as far (Muon's and Adam's updates are linear in lr)."""
    a, b = module(lr_mult=1.0, batched=batched), module(lr_mult=10.0, batched=batched)
    b.load_state_dict(a.state_dict())
    oa, _ = opt_and_sched(a)
    ob, _ = opt_and_sched(b)
    before = {n: p.detach().clone() for n, p in a.named_parameters()}
    fake_grads(a, 1)
    fake_grads(b, 1)
    oa.step()
    ob.step()
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for n, p0 in before.items():
        da, db = pa[n].detach() - p0, pb[n].detach() - p0
        if "w_embedder" in n:
            torch.testing.assert_close(db, 10 * da, rtol=1e-3, atol=1e-7)
            assert float(da.abs().max()) > 0
        else:
            torch.testing.assert_close(db, da, rtol=1e-5, atol=1e-8)


def test_resume_splits_a_single_group_state():
    """A checkpoint written at 1x (one group per optimizer) resumes under lr_mult: each parameter keeps its exact
    optimizer state, the embedding groups run at lr_mult x, the constant schedule keeps the rates, training steps on."""
    a = module(lr_mult=1.0)
    oa, sa = opt_and_sched(a)
    for s in range(2):
        fake_grads(a, s)
        oa.step()
        sa.step()
    ck = {"optimizer_states": [copy.deepcopy(oa.state_dict())], "lr_schedulers": [copy.deepcopy(sa.state_dict())]}

    b = module(lr_mult=10.0)
    b.load_state_dict(a.state_dict())
    b.on_load_checkpoint(ck)
    ob, sb = opt_and_sched(b)
    ob.load_state_dict(ck["optimizer_states"][0])
    sb.load_state_dict(ck["lr_schedulers"][0])

    na, nb = names(a), names(b)
    state_a = {na[id(p)]: st for o in oa.optimizers for p, st in o.state.items()}
    state_b = {nb[id(p)]: st for o in ob.optimizers for p, st in o.state.items()}
    assert set(state_a) == set(state_b) and state_a
    for n in state_a:
        assert set(state_a[n]) == set(state_b[n])
        for k, v in state_a[n].items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(state_b[n][k])), (n, k)
    lrs = [g["lr"] for g in ob.param_groups]
    assert lrs == pytest.approx([1e-3, 1e-2, 1e-4, 1e-3])
    fake_grads(b, 5)
    ob.step()
    sb.step()
    assert [g["lr"] for g in ob.param_groups] == pytest.approx(lrs)
    assert len(sb.get_last_lr()) == 4

    # a requeue of the split run: already two groups; the knob sets the embedding's rate
    ck2 = {"optimizer_states": [copy.deepcopy(ob.state_dict())], "lr_schedulers": [copy.deepcopy(sb.state_dict())]}
    c = module(lr_mult=30.0)
    c.load_state_dict(b.state_dict())
    c.on_load_checkpoint(ck2)
    oc, sc = opt_and_sched(c)
    oc.load_state_dict(ck2["optimizer_states"][0])
    sc.load_state_dict(ck2["lr_schedulers"][0])
    assert [g["lr"] for g in oc.param_groups] == pytest.approx([1e-3, 3e-2, 1e-4, 3e-3])


def test_checks():
    with pytest.raises(ValueError, match="lr_mult"):
        opt_and_sched(module(lr_mult=0.0))
    with pytest.raises(ValueError, match="Muon"):
        opt_and_sched(module(lr_mult=10.0, optimizer="AdamW"))
