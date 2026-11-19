"""Training data: image-caption tar shards for the text-to-image maps (data.kind text_image), and, for class-conditional
maps, ImageNet-256 SD-VAE latents (data.kind latent) or fixed latent banks (mode finetune).

Latents are sampled from each image's stored posterior (mean, logvar) with a random horizontal flip and scaled by
0.18215. The packed cache (part_*.npy of (n, 2, 8, 32, 32) float32 plus part_*.json with labels; scripts/sbm/encode_imagenet.py)
is read memory-mapped; the per-file cache (one .pt per image) is the fallback.
"""
from __future__ import annotations

import glob
import io
import json
import random
import tarfile
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
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


def center_square(img, size):
    """MiniT2I's image transform: shorter side to `size` (bicubic), centre crop; uint8 [3, size, size]."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) != size:
        r = size / min(w, h)
        img = img.resize((max(size, round(w * r)), max(size, round(h * r))), Image.BICUBIC)
    left, top = round((img.width - size) / 2), round((img.height - size) / 2)
    img = img.crop((left, top, left + size, top + size))
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


class TextImageShards(IterableDataset):
    """An endless stream of (image uint8 [3, S, S], token ids [L], token mask [L]) from tar shards holding
    {key}.jpg|png|webp + {key}.txt (the WebDataset layout). The shards are dealt round-robin to (rank, worker), reshuffled
    every pass, and the samples pass through a shuffle buffer of raw bytes. With `caption_dir`, a shard's captions come
    from <caption_dir>/<shard stem>.json ({key: caption}) and keys it lacks are skipped (CC12M: the LLaVA-NeXT recaptions
    over a mirror's shards). The stream is seeded per (seed, rank, worker) but not replayed by a resumed run."""

    IMAGES = ("jpg", "jpeg", "png", "webp")

    def __init__(self, shards, image_size, tokenizer_path, text_len=256, seed=0, rank=0, world=1, buffer=1000, caption_dir=None):
        patterns = [shards] if isinstance(shards, str) else list(shards)
        self.shards = sorted(f for pat in patterns for f in glob.glob(str(pat)))
        if not self.shards:
            raise ValueError(f"no tar shards match {patterns}")
        self.size, self.tokenizer_path, self.text_len = int(image_size), str(tokenizer_path), int(text_len)
        self.seed, self.rank, self.world, self.buffer = int(seed), int(rank), int(world), int(buffer)
        self.caption_dir = None if caption_dir is None else Path(str(caption_dir))

    def _read(self, shard):
        """(image bytes, caption) per sample of one shard; a sample's members are adjacent in the tar."""
        captions = None
        if self.caption_dir is not None:
            captions = json.loads((self.caption_dir / (Path(shard).stem + ".json")).read_text())
        key, image, text = None, None, None
        with tarfile.open(shard, "r|*") as tar:
            for member in tar:
                if not member.isfile() or "." not in member.name:
                    continue
                stem, ext = member.name.rsplit(".", 1)
                if stem != key:
                    if image is not None and text:
                        yield image, text
                    key, image, text = stem, None, (captions.get(Path(stem).name) if captions is not None else None)
                ext = ext.lower()
                if ext in self.IMAGES:
                    image = tar.extractfile(member).read()
                elif ext == "txt" and captions is None:
                    text = tar.extractfile(member).read().decode("utf-8", "replace").strip()
        if image is not None and text:
            yield image, text

    def __iter__(self):
        from PIL import Image
        from sbm_t2i.text import tokenize, tokenizer
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        mine = self.shards[self.rank::self.world][wid::workers] or self.shards   # fewer shards than readers: all, own order
        rng = random.Random(f"{self.seed}-{self.rank}-{wid}")
        tok, buf = tokenizer(self.tokenizer_path), []
        while True:
            order = list(mine)
            rng.shuffle(order)
            for shard in order:
                for item in self._read(shard):
                    if len(buf) < self.buffer:
                        buf.append(item)
                        continue
                    j = rng.randrange(self.buffer)
                    (image, text), buf[j] = buf[j], item
                    try:
                        x = center_square(Image.open(io.BytesIO(image)), self.size)
                    except Exception:   # a truncated or unreadable image: skip the sample
                        continue
                    ids, mask = tokenize(tok, [text], self.text_len)
                    yield x, ids[0], mask[0]


class MixedTextImageShards(IterableDataset):
    """Several TextImageShards streams interleaved in a fixed block (data.mix): each block of
    sum(per_batch) samples takes per_batch[i] from source i, spread evenly. A DataLoader worker builds whole batches from
    its own stream, so with sum(per_batch) == train.per_device_batch every batch holds exactly per_batch[i] samples of
    source i. Each source keeps its own seeded shard order and shuffle buffer."""

    def __init__(self, sources, per_batch):
        self.sources, self.per_batch = list(sources), [int(n) for n in per_batch]
        if len(self.sources) != len(self.per_batch) or min(self.per_batch) < 0 or sum(self.per_batch) < 1:
            raise ValueError(f"data.mix: {len(self.sources)} sources, per_batch {self.per_batch}")
        # the block's order: source i at the positions closest to an even spread of its per_batch[i] slots
        slots = sorted((k / n + 0.5 / n, i) for i, n in enumerate(self.per_batch) for k in range(n))
        self.order = [i for _, i in slots]

    def __iter__(self):
        streams = [iter(s) for s in self.sources]
        while True:
            for i in self.order:
                yield next(streams[i])


class TextImageData(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        d, t = self.cfg.data, self.trainer
        # data.image_size: the side the images are read at when it is not the map's input (a codec's encoder resolution)
        size = d.get("image_size") or self.cfg.model.input_size
        rank, world = (t.global_rank, t.world_size) if t is not None else (0, 1)

        def shards(src_shards, caption_dir, seed):
            return TextImageShards(src_shards, size, d.text_encoder, d.get("text_len", 256), seed, rank=rank,
                                   world=world, buffer=int(d.get("shuffle_buffer", 1000)), caption_dir=caption_dir)

        mix = d.get("mix")
        if mix:   # data.mix (off by default): several sources in every batch, per_batch samples each
            if sum(int(m.per_batch) for m in mix) != int(self.cfg.train.per_device_batch):
                raise ValueError(f"data.mix per_batch {[int(m.per_batch) for m in mix]} must sum to train.per_device_batch "
                                 f"{self.cfg.train.per_device_batch}")
            dataset = MixedTextImageShards([shards(m.shards, m.get("caption_dir"), self.cfg.seed + 1000 * i)
                                            for i, m in enumerate(mix)], [m.per_batch for m in mix])
        else:
            dataset = shards(d.shards, d.get("caption_dir"), self.cfg.seed)
        n = int(d.num_workers)
        return DataLoader(dataset, batch_size=int(self.cfg.train.per_device_batch), num_workers=n, pin_memory=True,
                          persistent_workers=n > 0, drop_last=True)


class Bank:
    """A fixed bank of real latent pairs, {targets: (N, 2, 4, 32, 32) bf16 (already scaled), labels: (N, 2)}.
    Each update draws `pairs` rows with replacement from a CPU generator seeded seed + 100003 * step; the same generator
    then draws the signed rule's roll shifts, so a run's stream is fixed by (seed, step) alone."""

    def __init__(self, path):
        bank = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        self.targets, self.labels = bank["targets"], bank["labels"]

    def rows(self, seed, step, pairs):
        g = torch.Generator().manual_seed(int(seed) + int(step) * 100003)
        return torch.randint(len(self.labels), (pairs,), generator=g), g


class Steps(Dataset):
    """Mode finetune has no data loader of its own (the bank is sampled inside the step): one dummy item per update."""

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i
