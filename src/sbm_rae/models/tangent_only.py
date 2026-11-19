"""Tangent-only JVP for the DiT's token-path layers (as in the SD-VAE package, src/sbm/models.py).

Under torch.func.jvp the stock forward-mode formula of a Linear (and of the patch convolution) also multiplies the
primal by the weight's tangent. The weights are not dual here, so that tangent is materialised as zeros and costs a
full GEMM per layer, about one forward per JVP. Swapped in, the token-path Linears and the patch convolution run the
stock op on the primal (same value, same reverse-mode graph) and compute the tangent as the layer applied to the
input's tangent, without the bias and outside the reverse-mode graph (the loss detaches the tangent). Primal, tangent
and gradients are unchanged (tests/rae/test_guided_cbm.py).
"""

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F


class _TangentOnly:
    """Mixin for nn.Linear / nn.Conv2d: inside a forward-AD level the output's tangent is `_tangent(input tangent)`."""

    def forward(self, x):
        primal, tangent = fwAD.unpack_dual(x)
        if tangent is None:
            return super().forward(x)
        out = super().forward(primal)
        with torch.no_grad():
            out_tangent = self._tangent(tangent)
        return fwAD.make_dual(out, out_tangent)


class TangentOnlyLinear(_TangentOnly, nn.Linear):
    def _tangent(self, t):
        return F.linear(t, self.weight)


class TangentOnlyConv2d(_TangentOnly, nn.Conv2d):
    def _tangent(self, t):
        return self._conv_forward(t, self.weight, None)


def use_tangent_only_jvp(model):
    """Switch the token-path layers (patch conv; qkv, proj, fc1, fc2 per block; the final linear) to the tangent-only
    forward. Parameters and state-dict keys are unchanged; the adaLN Linears see no tangent and are left alone."""
    model.x_embedder.proj.__class__ = TangentOnlyConv2d
    layers = [model.final_layer.linear]
    for block in model.blocks:
        layers += [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2]
    if getattr(model, "head_depth", 0) > 0:
        # the DDT head is conditioned per token by the trunk output, so its adaLN Linears (and the final layer's) carry
        # the input's tangent as well
        model.head_x_embedder.proj.__class__ = TangentOnlyConv2d
        layers += [model.final_layer.adaLN_modulation[-1]]
        if isinstance(model.s_projector, nn.Linear):
            layers.append(model.s_projector)
        for block in model.head_blocks:
            layers += [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2, block.adaLN_modulation[-1]]
    for layer in layers:
        layer.__class__ = TangentOnlyLinear
    return model
