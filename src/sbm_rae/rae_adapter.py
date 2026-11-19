"""An RAE (Zheng et al. 2025, arXiv 2510.11690) behind the slice of the diffusers AutoencoderKL interface the trainer,
sampler and FID scorer use, so `vae.kind=rae` swaps the SD-VAE latent for a DINOv2 representation latent with no other
change: `encode(x).latent_dist.sample()` and `decode(z).sample`, images in [-1, 1].

Encoder: frozen DINOv2 with registers at 224 px (the 16x16 patch tokens; 384 channels for `size` small, 768 for base),
normalized per channel by the released ImageNet statistics. Decoder: the released ViT-XL decoder, 256 px. Both are frozen.
The RAE code is vendored in third_party/RAE/src (github.com/bytetriper/RAE; override with SBM_RAE_SRC); the
weights live in the store, $SBM_STORE/rae/assets/rae (decoder, stats) and $SBM_STORE/rae/assets/hf_home (the encoder,
offline: HF_HOME must point there with HF_HUB_OFFLINE=1). See sbm_rae.paths.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from sbm_rae.paths import rae_assets_dir, rae_src_dir

RAE_SRC = str(rae_src_dir())


class _Dist:
    def __init__(self, z):
        self._z = z

    def sample(self):
        return self._z


class _Encoded:
    def __init__(self, z):
        self.latent_dist = _Dist(z)


class _Decoded:
    def __init__(self, images):
        self.sample = images


class RAEAutoencoder(nn.Module):
    def __init__(self, decoder="ViTXL_n08", encoder=None, input_size=224, assets=None, decode_bf16=True, size="base"):
        """size: the DINOv2-with-registers model (small 384 / base 768 / large 1024 channels); it picks the encoder, the
        released decoder (decoders/dinov2/wReg_<size>/<decoder>) and the normalization statistics."""
        super().__init__()
        if RAE_SRC not in sys.path:
            sys.path.insert(0, RAE_SRC)
        from stage1 import RAE

        assets = Path(assets) if assets else rae_assets_dir()
        if size not in ("small", "base", "large"):
            raise ValueError(f"RAE size must be small, base or large, got {size!r}")
        encoder = encoder or f"facebook/dinov2-with-registers-{size}"
        self.size = size
        self.rae = RAE(
            encoder_cls="Dinov2withNorm", encoder_config_path=encoder, encoder_input_size=input_size,
            encoder_params={"dinov2_path": encoder, "normalize": True},
            decoder_config_path=str(assets / "decoder_config_ViTXL"),
            pretrained_decoder_path=str(assets / "decoders" / "dinov2" / f"wReg_{size}" / decoder / "model.pt"),
            noise_tau=0.0, reshape_to_2d=True,
            normalization_stat_path=str(assets / "stats" / "dinov2" / f"wReg_{size}" / "imagenet1k" / "stat.pt"),
        )
        self.rae.eval().requires_grad_(False)
        self.decode_bf16 = bool(decode_bf16)

    @property
    def dtype(self):
        return torch.float32

    def train(self, mode=True):  # always frozen and in eval mode (the encoder adds noise in training mode)
        return super().train(False)

    @torch.no_grad()
    def encode(self, x):
        """x: images in [-1, 1] -> normalized DINOv2 latent (B, 768, 16, 16), in fp32 (the training targets)."""
        return _Encoded(self.rae.encode(((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)))

    @torch.no_grad()
    def decode(self, z):
        """normalized latent -> images in [-1, 1] (callers map to [0, 1] with (x + 1) / 2)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.decode_bf16 and z.is_cuda):
            images = self.rae.decode(z.float())
        return _Decoded(images.float() * 2.0 - 1.0)
