"""Speed-ups for the guided-CBM step of the latent text-to-image map, each off unless its switch is set (together about
1.3-1.6x per step).

torch.compile (`train.compile.*`), three independent switches, each a compiled replacement for one kind of network call
in sbm_t2i.guided_cbm (installed as the loss's `student_fwd` / `student_jvp`, and wrapping the frozen readers):

    teacher      the frozen teacher's no-grad forwards per step (4; and autoguidance's weak model, 2), fused attention
    student_fwd  the student's endpoint forward (with grad), fused attention
    student_jvp  the student's JVP (torch.func.jvp traced by dynamo; the tangent returned detached, as the loss uses it),
                 plain-product attention (forward AD differentiates it)

Only the image path (TextDiT.image_path) is compiled. The prompt path runs eagerly once per batch and network
(sbm_t2i.dit_t2i.PromptBound), and its rows come in as tensors. The student calls go through "twins": copies of the module
tree with plain nn.Linear / nn.Conv2d classes (the tangent-only layers are an eager-mode trick) that SHARE the
student's parameters and buffers, so gradients, DDP, the optimizer and the EMA all see the same tensors.

Optimizer (`optim.batched`): BatchedMuon, torch.optim.Muon with the Newton-Schulz iterations batched per weight shape
(same arithmetic per matrix). The EMA's foreach update is sbm_t2i.ema's `train.ema.foreach`.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from sbm_t2i.models import TangentOnlyConv2d, TangentOnlyLinear


def revert_tangent_only(module):
    """Plain nn.Linear / nn.Conv2d classes again (state-dict keys and parameters are unchanged)."""
    for m in module.modules():
        if isinstance(m, TangentOnlyLinear):
            m.__class__ = nn.Linear
        elif isinstance(m, TangentOnlyConv2d):
            m.__class__ = nn.Conv2d
    return module


def plain_twin(model):
    """A module tree with plain layer classes that SHARES `model`'s parameters and buffers."""
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


def _jvp_detached(m, I, tangent, txt, pooled, omega, w=None):
    pred, jv = torch.func.jvp(lambda i: m.image_path(i, txt, pooled, omega=omega, w=w), (I,), (tangent,))
    return pred, jv.detach()


def _fwd(m, I, txt, pooled, omega, w=None):
    return m.image_path(I, txt, pooled, omega=omega, w=w)


class _Reader:
    """A frozen TextDiT whose image path is compiled; its prompt path stays eager (PromptBound runs it once a batch)."""

    def __init__(self, net, mode):
        self.net, self.prompt = net, net.prompt
        self.image_path = torch.compile(net.image_path, mode=mode)

    def __getattr__(self, name):
        return getattr(self.net, name)


class CompiledCalls:
    """The compiled callables; `install` puts the student's on a GuidedCBMLoss (whose `model` is a PromptBound), `reader`
    wraps a frozen network."""

    def __init__(self, cfg_compile, model, keep_backward_buffers=False):
        from sbm_t2i.dit_t2i import set_attention_mode
        c = cfg_compile or {}
        self.mode = str(c.get("mode", "default"))
        self.do_teacher = bool(c.get("teacher", False))
        self.do_fwd = bool(c.get("student_fwd", False))
        self.do_jvp = bool(c.get("student_jvp", False))
        # keep_backward_buffers: until the guide start the balancer takes gradients of the compiled student forward with
        # retain_graph=True, which a compiled backward that donates its buffers refuses (it fails at the first recompile,
        # e.g. for a smaller batch). No arithmetic changes, only buffer reuse. Process-wide.
        if keep_backward_buffers and self.do_fwd:
            torch._functorch.config.donated_buffer = False
        # DDP: dynamo's DDPOptimizer splits compiled graphs at bucket boundaries, through the torch.func.jvp transform
        # ("Cannot access storage of TensorWrapper"); the compiled regions are inner calls, so the split is turned off
        if self.do_jvp:
            torch._dynamo.config.optimize_ddp = False
        self._readers = {}
        self.student_fwd = self.student_jvp = None
        # fwd_attention (default fused): the student's endpoint forward with torch's fused SDPA (fused) or the plain
        # q k^T product (plain; the released text-to-image recipes train with it)
        self.fwd_attention = str(c.get("fwd_attention", "fused"))
        if self.fwd_attention not in ("fused", "plain"):
            raise ValueError(f"train.compile.fwd_attention must be fused or plain, got {self.fwd_attention}")
        if self.do_fwd:
            twin = set_attention_mode(plain_twin(model), self.fwd_attention)
            fwd = torch.compile(_fwd, mode=self.mode)

            def student_fwd(m, I, y, omega, return_repa=False, w=None):
                if return_repa:
                    raise ValueError("TextDiT has no REPA projector (loss.guided_cbm.repa)")
                return fwd(twin, I, *m.rows(y), omega, w)
            self.student_fwd = student_fwd
        if self.do_jvp:
            twin_j = set_attention_mode(plain_twin(model), "plain")
            if bool(c.get("jvp_attn_fp32", False)):   # fp32 attention in the JVP's network only
                from sbm_t2i.dit_t2i import set_attention_fp32
                set_attention_fp32(twin_j, True)
            jvp = torch.compile(_jvp_detached, mode=self.mode)
            self.student_jvp = lambda m, I, tan, y, omega, w=None: jvp(twin_j, I, tan, *m.rows(y), omega, w)

    @property
    def enabled(self):
        return self.do_teacher or self.do_fwd or self.do_jvp

    def reader(self, net):
        """The compiled wrapper of a frozen no-grad network (the teacher, the weak model), built once per object (the
        teacher's in-place refresh keeps the object, so its compiled code is reused)."""
        if net is None or not self.do_teacher:
            return net
        from sbm_t2i.dit_t2i import set_attention_mode
        key = id(net)
        if key not in self._readers:
            set_attention_mode(revert_tangent_only(net), "fused")
            self._readers[key] = (net, _Reader(net, self.mode))
        return self._readers[key][1]

    def install(self, loss_fn):
        loss_fn.student_fwd = self.student_fwd
        loss_fn.student_jvp = self.student_jvp


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
