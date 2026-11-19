"""Helpers shared by the tests: the perturbation of the tiny DiT's zero-initialised layers, an SD-VAE stand-in, the
projection that summarises a gradient, and the data of the golden file b2.pt.

b2.pt holds golden values of the BTM loss on a tiny DiT (depth 2, width 64, 8x8 latents, 10 classes; CPU, fp32) computed
with an independent earlier implementation: per recipe the loss terms, the gradient of their sum (norm + 256 fixed Gaussian
projections) and one draw from the RNG afterwards, so the implementation here must also consume the global RNG in the
same order and amount (tests/sbm/test_golden.py).
"""
from __future__ import annotations

import torch


def perturb(model, seed=7, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            if p.requires_grad:
                p.add_(scale * torch.randn(p.shape, generator=g))
    return model


class VAEStub(torch.nn.Module):
    """decode(z).sample: (B, 4, 8, 8) latents -> (B, 3, 16, 16) images (stands in for the SD-VAE in the eval / monitor tests)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(11)
        self.w = torch.nn.Parameter(0.3 * torch.randn(3, 4, 3, 3, generator=g))

    def decode(self, z):
        img = torch.nn.functional.interpolate(torch.nn.functional.conv2d(z, self.w, padding=1), scale_factor=2)
        return type("Out", (), {"sample": torch.tanh(img)})()


def project(v, k=256, seed=99):
    """A gradient summarised by its norm and k fixed Gaussian projections (the full vector is ~150k numbers)."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(k, v.numel(), generator=g, dtype=torch.float64)
    return {"norm": float(v.double().norm()), "proj": P @ v.double()}


def data(seed=3, B=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 4, 8, 8, generator=g), torch.arange(B) % 10
