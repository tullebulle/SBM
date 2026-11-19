"""The frozen latent codec of the latent text-to-image map (config `codec`).

kind rae: a representation autoencoder (RAE, Zheng et al. 2025, arXiv 2510.11690) run by the RAE code
(github.com/bytetriper/RAE, `stage1.RAE`; vendored in third_party/RAE, `codec.rae_src`), exactly as the RAE-latent
ImageNet line uses it (sbm_rae.rae_adapter): the image in [0, 1], resized bicubically to `encoder_size`, through the frozen
encoder (DINOv2 with registers: its last layer's patch tokens, layer-normed), as a [C, h, w] grid normalised per element
by the statistics `stats` (mean, var). The decoder (a ViT, patch `decoder_patch` on the same token grid, so the output
side is decoder_patch x h) inverts it. The ImageNet DINOv2-S latent: encoder_size 224 -> 16 x 16 x 384; RAE's released
ViT-XL decoder outputs 256 px.

The codec is held by reference outside the module tree (sbm_t2i.text.Frozen): not trained, not checkpointed, shared by the
EMA copies. encode() runs at `dtype` with autocast off, so the latent the map trains on does not depend on the trainer's
precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from sbm_t2i.text import Frozen

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class RAECodec(Frozen):
    def __init__(self, rae_src, encoder, decoder, decoder_config, stats, encoder_size=224, decoder_patch=16,
                 dtype="fp32", encoder_cls="Dinov2withNorm"):
        if str(rae_src) not in sys.path:
            sys.path.insert(0, str(rae_src))
        from stage1 import RAE
        rae = RAE(encoder_cls=encoder_cls, encoder_config_path=str(encoder), encoder_input_size=int(encoder_size),
                  encoder_params={"dinov2_path": str(encoder), "normalize": True},
                  decoder_config_path=str(decoder_config), decoder_patch_size=int(decoder_patch),
                  pretrained_decoder_path=str(decoder) if decoder else None, noise_tau=0.0, reshape_to_2d=True,
                  normalization_stat_path=str(stats))
        super().__init__(rae)
        self.dtype = DTYPES[str(dtype)]
        self.module.to(self.dtype)
        grid = int(encoder_size) // int(rae.encoder_patch_size)
        self.latent_shape = (int(rae.latent_dim), grid, grid)
        self.out_size = grid * int(decoder_patch)

    @torch.no_grad()
    def encode(self, images):
        """uint8 [B, 3, S, S] (or floats in [0, 1]) -> the normalised latent [B, C, h, w], fp32."""
        x = images.float().div(255.0) if images.dtype == torch.uint8 else images.float()
        rae = self.on(x.device)
        with torch.autocast(x.device.type, enabled=False):
            return rae.encode(x.to(self.dtype)).float()

    @torch.no_grad()
    def decode(self, z):
        """normalised latent [B, C, h, w] -> images in [-1, 1] [B, 3, out_size, out_size], fp32."""
        rae = self.on(z.device)
        with torch.autocast(z.device.type, enabled=False):
            return rae.decode(z.to(self.dtype)).float().mul(2.0).sub(1.0)


def build_codec(cfg):
    """`cfg`: the `codec` config block."""
    kind = str(cfg.kind)
    if kind != "rae":
        raise ValueError(f"codec.kind must be rae (or null for none), got {kind!r}")
    for key in ("rae_src", "encoder", "decoder_config", "stats"):
        if not cfg.get(key) or not Path(str(cfg.get(key))).exists():
            raise FileNotFoundError(f"codec.{key}: {cfg.get(key)!r} does not exist")
    return RAECodec(cfg.rae_src, cfg.encoder, cfg.get("decoder"), cfg.decoder_config, cfg.stats,
                    encoder_size=int(cfg.get("encoder_size", 224)), decoder_patch=int(cfg.get("decoder_patch", 16)),
                    dtype=str(cfg.get("dtype", "fp32")), encoder_cls=str(cfg.get("encoder_cls", "Dinov2withNorm")))
