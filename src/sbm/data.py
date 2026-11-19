"""Training data: ImageNet-256 SD-VAE latents.

Latents are sampled from each image's stored posterior (mean, logvar) with a random horizontal flip and scaled by
0.18215. The packed cache (scripts/sbm/encode_imagenet.py: part_*.npy of (n, 2, 8, 32, 32) float32 = [moments, moments of
the flipped image], plus part_*.json with the labels) is read memory-mapped; a per-file cache (one .pt per image,
data.latent_dir) is the optional fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class PackedLatents(Dataset):
    def __init__(self, pack_dir, latents_scale=0.18215):
        self.parts = sorted(Path(pack_dir).glob("part_*.npy"))
        if not self.parts:
            raise ValueError(f"no part_*.npy under {pack_dir}")
        self.targets, sizes = [], []
        for part in self.parts:
            meta = json.loads(part.with_suffix(".json").read_text())
            self.targets.extend(int(y) for y in meta["labels"])
            sizes.append(len(meta["labels"]))
        self._offsets = np.concatenate([[0], np.cumsum(sizes)])
        self._arrays = None
        self.scale = float(latents_scale)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        flip = bool(torch.rand(()) < 0.5)
        if self._arrays is None:      # opened lazily so each DataLoader worker maps the files itself
            self._arrays = [np.load(p, mmap_mode="r") for p in self.parts]
        part = int(np.searchsorted(self._offsets, index, side="right") - 1)
        row = self._arrays[part][index - int(self._offsets[part]), 1 if flip else 0]
        moments = torch.from_numpy(np.array(row, dtype=np.float32, copy=True)).unsqueeze(0)
        latent = DiagonalGaussianDistribution(moments).sample() * self.scale
        return latent[0].float(), int(self.targets[index])


class FileLatents(Dataset):
    """<root>/<class dir>/<image>.pt with `moments` / `moments_flip` (the per-file latent cache)."""

    def __init__(self, root, latents_scale=0.18215):
        from torchvision.datasets import DatasetFolder
        self.folder = DatasetFolder(str(root), loader=str, extensions=(".pt",))
        self.targets = self.folder.targets
        self.scale = float(latents_scale)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        path, target = self.folder.samples[index]
        item = torch.load(path, map_location="cpu")
        if "latents" in item:
            return item["latents"].float(), int(target)
        moments = item["moments_flip" if bool(torch.rand(()) < 0.5) else "moments"].float()
        moments = moments if moments.ndim == 4 else moments.unsqueeze(0)
        return (DiagonalGaussianDistribution(moments).sample() * self.scale)[0].float(), int(target)


class LatentData(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _dataset(self, split):
        d = self.cfg.data
        packed = Path(str(d.latent_pack_dir)) / split if d.get("latent_pack_dir") else None
        if packed is not None and packed.is_dir():
            return PackedLatents(packed, d.latents_scale)
        if not d.get("latent_dir"):
            raise FileNotFoundError(f"no packed latents at {packed} (data.latent_pack_dir) and data.latent_dir is not set")
        return FileLatents(Path(str(d.latent_dir)) / split, d.latents_scale)

    def setup(self, stage=None):
        self.train_set = self._dataset("train")
        self.val_set = self._dataset("val") if self.cfg.train.get("val_every") else None

    def _loader(self, dataset, shuffle):
        t = self.trainer
        sampler = (DistributedSampler(dataset, num_replicas=t.world_size, rank=t.global_rank, shuffle=shuffle,
                                      drop_last=False, seed=int(self.cfg.seed)) if t is not None and t.world_size > 1 else None)
        n = int(self.cfg.data.num_workers)
        return DataLoader(dataset, batch_size=int(self.cfg.train.per_device_batch), shuffle=shuffle and sampler is None,
                          sampler=sampler, num_workers=n, pin_memory=True, persistent_workers=n > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False) if self.val_set is not None else []
