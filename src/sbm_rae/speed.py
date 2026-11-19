"""Speed-ups for the guided-objective training step, each off unless its switch is set (unset, training runs the stock
eager code); together about 1.3-1.6x per step.

torch.compile (`trainer.compile.*`):

Three independent switches, each a torch.compile'd replacement for one kind of network call in sbm_rae.guided_cbm:

    teacher      the frozen teacher's 4 no-grad forwards per step
    student_fwd  the student's endpoint forward (with grad; with REPA it also returns the projection)
    student_jvp  the student's JVP (torch.func.jvp traced by dynamo; the tangent is returned detached, as the loss uses it)

The student calls go through a "twin": a copy of the module tree with plain nn.Linear / nn.Conv2d classes (the
tangent-only JVP layers of sbm_rae.models.tangent_only are an eager-mode trick) that shares the student's parameters and
buffers, so gradients, DDP hooks, the optimizer and the EMA all see the same tensors.

Optimizer / EMA (same math, fewer and larger kernels): `muon.batched` (BatchedMuon) and `trainer.ema.foreach` (one
multi-tensor update per EMA instead of a per-parameter loop).
"""
import copy

import torch
import torch.nn as nn

from sbm_rae.models.tangent_only import TangentOnlyConv2d, TangentOnlyLinear


def revert_tangent_only(module):
    """Plain nn.Linear / nn.Conv2d classes again (state-dict keys and parameters are unchanged)."""
    for m in module.modules():
        if isinstance(m, TangentOnlyLinear):
            m.__class__ = nn.Linear
        elif isinstance(m, TangentOnlyConv2d):
            m.__class__ = nn.Conv2d
    return module


def plain_twin(model):
    """Module tree with plain layer classes that SHARES `model`'s parameters and buffers."""
    twin = copy.deepcopy(model)
    revert_tangent_only(twin)
    named = dict(twin.named_modules())
    for name, p in model.named_parameters():
        mod, _, leaf = name.rpartition(".")
        setattr(named[mod], leaf, p)
    for name, b in model.named_buffers():
        mod, _, leaf = name.rpartition(".")
        named[mod]._buffers[leaf] = b
    return twin


def _jvp_detached(m, I, tangent, labels, omega, w=None):
    # w: the guidance-scale input (loss.guided_cbm.guide.scale_cond); None is the stock call
    wk = {} if w is None else {"w": w}
    pred, jv = torch.func.jvp(lambda i: m(i, labels, omega=omega, **wk), (I,), (tangent,))
    return pred, jv.detach()


def _fwd(m, I, labels, omega, return_repa, w=None):
    # the stock call when REPA is off (return_repa left at the model's default) and without the scale input
    wk = {} if w is None else {"w": w}
    return m(I, labels, omega=omega, return_repa=True, **wk) if return_repa else m(I, labels, omega=omega, **wk)


class CompiledCalls:
    """Holds the compiled callables and installs them on a GuidedCBMLoss."""

    def __init__(self, cfg_compile, model):
        c = cfg_compile or {}
        self.mode = str(c.get("mode", "default"))
        self.do_teacher = bool(c.get("teacher", False))
        self.do_fwd = bool(c.get("student_fwd", False))
        self.do_jvp = bool(c.get("student_jvp", False))
        self.twin = plain_twin(model) if (self.do_fwd or self.do_jvp) else None
        # Under DDP, dynamo's DDPOptimizer splits every compiled graph at gradient-bucket boundaries; the split cuts
        # through the torch.func.jvp transform ("Cannot access storage of TensorWrapper"). The compiled regions here
        # are inner calls, not the DDP module, so the split is turned off (the all-reduce then starts after each
        # compiled backward instead of overlapping inside it).
        if self.do_jvp:
            torch._dynamo.config.optimize_ddp = False
        self._readers = {}            # id -> (frozen module, its compiled wrapper): the teacher, and the autoguide weak model
        if self.do_fwd:
            fwd = torch.compile(_fwd, mode=self.mode)
            self.student_fwd = lambda m, I, y, om, return_repa=False, w=None: fwd(self.twin, I, y, om, return_repa, w)
        else:
            self.student_fwd = None
        if self.do_jvp:
            jvp = torch.compile(_jvp_detached, mode=self.mode)
            self.student_jvp = lambda m, I, tan, y, om, w=None: jvp(self.twin, I, tan, y, om, w)
        else:
            self.student_jvp = None

    @property
    def enabled(self):
        return self.do_teacher or self.do_fwd or self.do_jvp

    def teacher(self, teacher):
        """The compiled wrapper of a frozen no-grad reader: the teacher, or autoguidance's weak model (built once per
        object; the teacher's in-place refresh keeps the same object, so its compiled code is reused)."""
        if teacher is None or not self.do_teacher:
            return teacher
        key = id(teacher)
        if key not in self._readers:
            revert_tangent_only(teacher)
            self._readers[key] = (teacher, torch.compile(teacher, mode=self.mode))
        return self._readers[key][1]

    def install(self, loss_fn):
        loss_fn.student_fwd = self.student_fwd
        loss_fn.student_jvp = self.student_jvp


# ---------------------------------------------------------------------------------------------------------------------
# Optimizer / EMA (same math as the stock versions, fewer and larger kernels)

class BatchedMuon(torch.optim.Muon):
    """torch.optim.Muon with the Newton-Schulz iterations batched over same-shaped matrices (bmm / baddbmm) and the
    momentum / weight-decay / update steps as foreach ops. Per matrix the arithmetic is the stock one
    (torch/optim/_muon.py: _single_tensor_muon and _zeropower_via_newtonschulz)."""

    @torch.no_grad()
    def step(self, closure=None):
        from torch.optim._muon import _adjust_lr

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mom = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            a, b, c = group["ns_coefficients"]
            eps, ns_steps = float(group["eps"]), int(group["ns_steps"])
            params, grads, bufs = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                params.append(p)
                grads.append(p.grad)
                bufs.append(st["momentum_buffer"])
            if not params:
                continue
            torch._foreach_lerp_(bufs, grads, 1 - mom)
            updates = torch._foreach_lerp(grads, bufs, mom) if group["nesterov"] else bufs
            by_shape = {}
            for i, u in enumerate(updates):
                by_shape.setdefault(tuple(u.shape), []).append(i)
            if wd != 0.0:
                torch._foreach_mul_(params, 1 - lr * wd)
            for shape, idx in by_shape.items():
                tall = shape[0] > shape[1]
                x = torch.stack([updates[i] for i in idx]).bfloat16()
                if tall:
                    x = x.transpose(1, 2)
                x = x / x.norm(dim=(1, 2), keepdim=True).clamp(min=eps)
                for _ in range(ns_steps):
                    gram = x @ x.transpose(1, 2)
                    gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                    x = torch.baddbmm(x, gram_update, x, beta=a)
                if tall:
                    x = x.transpose(1, 2)
                adjusted = _adjust_lr(lr, group["adjust_lr_fn"], params[idx[0]].shape)
                torch._foreach_add_([params[i] for i in idx], list(x.unbind(0)), alpha=-adjusted)
        return loss


def ema_multi_avg_fn(decay):
    """The foreach EMA: ema <- ema + (1 - decay) (p - ema) in one multi-tensor kernel per dtype/device group."""
    from torch.optim.swa_utils import get_ema_multi_avg_fn

    return get_ema_multi_avg_fn(decay)
