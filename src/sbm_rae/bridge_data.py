"""Bridge-V2 / RT-1 clips for the world models.

DriftWorld's processed layout (their bridge_driftworld/datasets/download_data.py, run unchanged on the TFDS release):
<root>/<split>/<stem>.mp4 (camera image_0, 256x256, the TFDS 5 Hz frames) + <stem>.npz (frame-aligned 7-D actions, rescaled
by their script) + <stem>.txt (instruction). Their sampling: one random window of `n_frames` consecutive frames per episode
per epoch (episodes shorter than that are dropped); `full_video` returns every frame (their evaluation loader).

A batch is {"x": (B, T, 3, H, W) uint8 frames, "act": (B, T, A) float32, "idx": (B,) long}; the trainer encodes the frames
with the codec on the GPU (sbm_rae.bridge_trainer), so the loader moves bytes, not floats.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl


def read_frames(path, start=0, n=None):
    """Frames [start, start + n) of an mp4 as (n, 3, H, W) uint8 (every frame when n is None); decord, else PyAV."""
    try:
        import decord

        vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        total = len(vr)
        idx = list(range(start, total if n is None else min(start + n, total)))
        arr = vr.get_batch(idx).asnumpy()
    except ImportError:
        import av

        frames = []
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i < start:
                    continue
                if n is not None and i >= start + n:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
        arr = np.stack(frames)
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).contiguous()
    # RT-1: DriftWorld's conversion keeps the camera's 256x320 and their loader resizes to input_h x input_w = 256x256
    # (torchvision Resize, bilinear + antialias); BRIDGE_FRAME_SIZE=256 in the environment does the same here (training,
    # latent cache, evaluation); unset = frames as stored (Bridge-V2 is 256x256 already).
    S = os.environ.get("BRIDGE_FRAME_SIZE")
    if S and tuple(x.shape[-2:]) != (int(S), int(S)):
        x = torch.nn.functional.interpolate(x.float(), size=(int(S), int(S)), mode="bilinear", antialias=True, align_corners=False)
        x = x.round().clamp_(0, 255).to(torch.uint8)
    return x


class BridgeClips(Dataset):
    def __init__(self, root, split, n_frames=5, action_dim=7, full_video=False, max_episodes=None, seed=0):
        self.dir = Path(root) / split
        self.n_frames = int(n_frames)
        self.action_dim = int(action_dim)
        self.full_video = bool(full_video)
        index = self.dir / "_index.json"
        if index.exists():
            entries = json.loads(index.read_text())
        else:
            entries = []
            for mp4 in sorted(self.dir.glob("*.mp4")):
                npz = mp4.with_suffix(".npz")
                if not npz.exists():
                    continue
                entries.append([mp4.stem, int(np.load(npz)["arr_0"].shape[0])])
            tmp = index.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(entries))
            os.replace(tmp, index)
        entries = [e for e in entries if e[1] >= self.n_frames]
        if max_episodes is not None and int(max_episodes) > 0:
            rng = random.Random(int(seed))
            rng.shuffle(entries)
            entries = sorted(entries[: int(max_episodes)])
        self.stems = [e[0] for e in entries]
        self.lengths = [int(e[1]) for e in entries]
        if not self.stems:
            raise FileNotFoundError(f"no episodes with >= {self.n_frames} frames under {self.dir}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem, length = self.stems[i], self.lengths[i]
        actions = np.load(self.dir / f"{stem}.npz")["arr_0"].astype(np.float32)
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"{stem}: action dim {actions.shape[1]} != {self.action_dim}")
        if self.full_video:
            frames = read_frames(self.dir / f"{stem}.mp4")
            L = min(frames.shape[0], actions.shape[0])
            return {"x": frames[:L], "act": torch.from_numpy(actions[:L]), "idx": i}
        start = random.randint(0, length - self.n_frames)
        frames = read_frames(self.dir / f"{stem}.mp4", start, self.n_frames)
        if frames.shape[0] != self.n_frames:
            raise RuntimeError(f"{stem}: decoded {frames.shape[0]} frames from {start}, wanted {self.n_frames} (length {length})")
        return {"x": frames, "act": torch.from_numpy(actions[start : start + self.n_frames]), "idx": i}


class BridgeLatentClips(Dataset):
    """Windows of `n_frames` cached latents (scripts/rae/cache_bridge_latents.py): <cache>/<split>/latents.npy (fp16 memmap),
    actions.npy, index.json. Returns {"x": (T, C, h, w) float16 latents (already scaled), "act": (T, A), "idx"}; the
    trainer skips the codec for these."""

    def __init__(self, cache, split, n_frames=5, full_video=False, max_episodes=None, seed=0):
        self.dir = Path(cache) / split
        meta = json.loads((self.dir / "index.json").read_text())
        self.n_frames = int(n_frames)
        self.full_video = bool(full_video)
        self.codec, self.scale, self.shape = meta["codec"], float(meta["scale"]), tuple(meta["shape"])
        eps = [e for e in meta["episodes"] if e["length"] >= self.n_frames]
        if max_episodes is not None and int(max_episodes) > 0:
            rng = random.Random(int(seed))
            rng.shuffle(eps)
            eps = sorted(eps[: int(max_episodes)], key=lambda e: e["offset"])
        self.episodes = eps
        self.stems = [e["stem"] for e in eps]
        self.lengths = [int(e["length"]) for e in eps]
        self._lat = self._act = None

    def _open(self):
        if self._lat is None:   # one memmap per worker process
            self._lat = np.load(self.dir / "latents.npy", mmap_mode="r")
            self._act = np.load(self.dir / "actions.npy", mmap_mode="r")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        self._open()
        e = self.episodes[i]
        off, length = int(e["offset"]), int(e["length"])
        if self.full_video:
            sl = slice(off, off + length)
        else:
            start = random.randint(0, length - self.n_frames)
            sl = slice(off + start, off + start + self.n_frames)
        return {"x": torch.from_numpy(np.ascontiguousarray(self._lat[sl])), "act": torch.from_numpy(np.ascontiguousarray(self._act[sl])),
                "idx": i}


def collate_full_video(batch):
    """Pads full episodes to the longest in the batch; "len" holds the true lengths."""
    L = max(b["x"].shape[0] for b in batch)
    x = torch.zeros(len(batch), L, *batch[0]["x"].shape[1:], dtype=torch.uint8)
    act = torch.zeros(len(batch), L, batch[0]["act"].shape[1], dtype=torch.float32)
    lens = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n], act[i, :n], lens[i] = b["x"], b["act"], n
    return {"x": x, "act": act, "len": lens, "idx": torch.tensor([b["idx"] for b in batch])}


class BridgeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dataset
        self.root = str(d.processed_dir)
        self.latent_cache = d.get("latent_cache_dir", None)     # set: windows of cached latents instead of frames
        self.n_frames = int(d.get("n_frames", 5))
        self.action_dim = int(d.get("action_dim", 7))
        self.val_episodes = d.get("val_episodes", 256)
        self.batch_size = int(cfg.trainer.per_device_batch_size)
        self.num_workers = int(cfg.trainer.get("num_workers", 8))
        self.train_set = self.val_set = None

    # the ImageNet module's cache hooks (scripts/rae/train.py calls them): nothing to build here
    def require_latent_cache(self):
        pass

    def require_representation_cache(self):
        pass

    def prepare_data(self):
        pass

    @staticmethod
    def inverse_scaler(x):
        return x

    def setup(self, stage=None):
        if self.train_set is None:
            if self.latent_cache:
                self.train_set = BridgeLatentClips(self.latent_cache, "train", self.n_frames)
                self.val_set = BridgeLatentClips(self.latent_cache, "val", self.n_frames, max_episodes=self.val_episodes,
                                                 seed=int(self.cfg.seed))
                print(f"[bridge] latent cache {self.latent_cache}: codec {self.train_set.codec}, latent {self.train_set.shape}", flush=True)
            else:
                self.train_set = BridgeClips(self.root, "train", self.n_frames, self.action_dim)
                self.val_set = BridgeClips(self.root, "val", self.n_frames, self.action_dim, max_episodes=self.val_episodes,
                                           seed=int(self.cfg.seed))
            print(f"[bridge] train episodes {len(self.train_set)}, val episodes {len(self.val_set)} (clip {self.n_frames} frames)",
                  flush=True)

    def _loader(self, dataset, shuffle):
        t = self.trainer
        sampler = None
        if t is not None and t.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=t.world_size, rank=t.global_rank, shuffle=shuffle,
                                         drop_last=shuffle, seed=int(self.cfg.seed))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0,
                          drop_last=shuffle, prefetch_factor=4 if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)
