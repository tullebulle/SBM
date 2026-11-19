
import functools
import gc
import os
import shutil
import time
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder



LATENT_CACHE_WAIT_SECONDS = 5
LATENT_CACHE_TIMEOUT_SECONDS = 12 * 60 * 60


def image_scaler(image):
    return image * 2.0 - 1.0


def inverse_image_scaler(image):
    return (image + 1.0) / 2.0


def _channel_buffer(value, channels):
    if isinstance(value, (int, float)):
        values = [float(value)] * channels
    else:
        values = list(value)
        if len(values) != channels:
            raise ValueError(
                f"Expected {channels} latent affine values, got {len(values)}"
            )
    return torch.tensor(values, dtype=torch.float32).view(1, channels, 1, 1)


def _clone_batch(batch):
    if torch.is_tensor(batch):
        return batch.detach().clone()
    if isinstance(batch, tuple):
        return tuple(_clone_batch(item) for item in batch)
    if isinstance(batch, list):
        return [_clone_batch(item) for item in batch]
    if isinstance(batch, dict):
        return {key: _clone_batch(value) for key, value in batch.items()}
    return batch


def _compact_cpu_tensor(tensor):
    return tensor.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)


class _SingleBatchDataset(Dataset):
    def __init__(self, batch):
        self.batch = _clone_batch(batch)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(f"Index {index} out of range for fixed overfitting batch")
        return _clone_batch(self.batch)


def _single_batch_collate(batches):
    if len(batches) != 1:
        raise ValueError(f"Expected one fixed batch, got {len(batches)}")
    return batches[0]


def center_crop_arr(pil_image, image_size):
    """ADM's (guided-diffusion) centre crop, as in DiT and RAE training and in the ADM reference statistics: halve with a
    box filter while the short side is at least 2 x image_size, bicubic to image_size on the short side, then crop the
    centre (offsets rounded down)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_imagenet_transform(resolution, crop="torchvision", hflip=False):
    """crop (dataset.crop): torchvision = Resize (bilinear) + CenterCrop, the original; adm = center_crop_arr.
    hflip (dataset.hflip): a random horizontal flip (p 0.5) after the crop."""
    if crop == "torchvision":
        steps = [transforms.Resize(resolution), transforms.CenterCrop(resolution)]
    elif crop == "adm":
        steps = [transforms.Lambda(functools.partial(center_crop_arr, image_size=resolution))]
    else:
        raise ValueError(f"dataset.crop must be torchvision or adm, got {crop!r}")
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class _ImageFolderWithRelativePath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        relative_path = Path(path).relative_to(self.root).with_suffix(".pt")
        return sample, target, str(relative_path)


def _sample_cached_latent(cache_item, latents_scale, latents_bias, source=None, flip=None):
    if "latents" in cache_item:
        return cache_item["latents"].to(dtype=torch.float32)

    # `flip` lets a caller that also holds per-patch targets make the same choice for both;
    # left as None the flip is drawn here, as before.
    if flip is None:
        flip = bool(torch.rand(()) < 0.5)
    moments_key = "moments_flip" if flip else "moments"
    moments = cache_item[moments_key].to(dtype=torch.float32)
    if moments.ndim == 3:
        moments = moments.unsqueeze(0)
    elif moments.ndim == 4 and moments.shape[0] == 1:
        pass
    else:
        location = f" in '{source}'" if source is not None else ""
        raise ValueError(
            f"Cached latent moments{location} must have shape (C, H, W) or "
            f"(1, C, H, W), got {tuple(moments.shape)}. Batched latent cache "
            "files are ambiguous and would otherwise always read sample 0."
        )
    posterior = DiagonalGaussianDistribution(moments)
    latent = posterior.sample()
    latent = (latent - latents_bias) * latents_scale
    return latent[0].to(dtype=torch.float32)


class _CacheBuildDataset(Dataset):
    def __init__(self, dataset, entries):
        self.dataset = dataset
        self.entries = list(entries)
        first_item = dataset[0]
        sample = first_item[0]
        self.has_relative_path = len(first_item) >= 3
        self._dummy_image = torch.zeros_like(sample)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sample_index, is_valid = self.entries[index]
        if is_valid:
            return (*self.dataset[sample_index], torch.tensor(True, dtype=torch.bool))
        if self.has_relative_path:
            return self._dummy_image.clone(), 0, "", torch.tensor(False, dtype=torch.bool)
        return self._dummy_image.clone(), 0, torch.tensor(False, dtype=torch.bool)


class LatentFileDataset(DatasetFolder):
    def __init__(self, root_dir, latents_scale=0.18215, latents_bias=0.0):
        self.root_dir = Path(root_dir)
        self.latents_scale = _channel_buffer(latents_scale, 4)
        self.latents_bias = _channel_buffer(latents_bias, 4)
        super().__init__(root=str(self.root_dir), loader=str, extensions=(".pt",))

    def __getitem__(self, index):
        path, target = self.samples[index]
        cache_item = torch.load(path, map_location="cpu")
        if "label" in cache_item and int(cache_item["label"]) != int(target):
            raise ValueError(
                f"Latent file '{path}' has label {int(cache_item['label'])}, "
                f"but its class directory maps to {int(target)}"
            )
        latent = _sample_cached_latent(cache_item, self.latents_scale, self.latents_bias, source=path)
        return latent, int(target)

    def get_with_flip(self, index, flip):
        path, target = self.samples[index]
        cache_item = torch.load(path, map_location="cpu")
        latent = _sample_cached_latent(
            cache_item, self.latents_scale, self.latents_bias, source=path, flip=flip
        )
        return latent, int(target)


class PackedLatentDataset(Dataset):
    """The packed SD-VAE latent cache written by scripts/sbm/encode_imagenet.py: a few float32 arrays of shape
    (n, 2, 8, 32, 32) (the posterior moments and those of the flipped image), memory-mapped, in DatasetFolder order.
    """

    def __init__(self, pack_dir, latents_scale=0.18215, latents_bias=0.0):
        import json
        import numpy as np

        self.pack_dir = Path(pack_dir)
        self.latents_scale = _channel_buffer(latents_scale, 4)
        self.latents_bias = _channel_buffer(latents_bias, 4)
        self.part_paths = sorted(self.pack_dir.glob("part_*.npy"))
        if not self.part_paths:
            raise ValueError(f"No part_*.npy under '{self.pack_dir}'; run scripts/sbm/encode_imagenet.py first")
        self.targets, self.files, sizes = [], [], []
        for part in self.part_paths:
            meta = json.loads(part.with_suffix(".json").read_text())
            self.targets.extend(int(label) for label in meta["labels"])
            self.files.extend(meta["files"])
            sizes.append(len(meta["labels"]))
        self._offsets = np.concatenate([[0], np.cumsum(sizes)])
        self._arrays = None

    def __len__(self):
        return len(self.targets)

    def _moments(self, index, flip):
        import numpy as np

        if self._arrays is None:
            # opened lazily so each DataLoader worker maps the files itself
            self._arrays = [np.load(part, mmap_mode="r") for part in self.part_paths]
        part = int(np.searchsorted(self._offsets, index, side="right") - 1)
        row = self._arrays[part][index - int(self._offsets[part]), 1 if flip else 0]
        return torch.from_numpy(np.array(row, dtype=np.float32, copy=True))

    def get_with_flip(self, index, flip):
        moments = self._moments(index, flip).unsqueeze(0)
        latent = DiagonalGaussianDistribution(moments).sample()
        latent = (latent - self.latents_bias) * self.latents_scale
        return latent[0].to(dtype=torch.float32), int(self.targets[index])

    def __getitem__(self, index):
        return self.get_with_flip(index, bool(torch.rand(()) < 0.5))


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        latent_data_dir = cfg.get("latent_data_dir", None)
        self.latent_data_dir = Path(latent_data_dir) if latent_data_dir else None
        self.repa_cfg = cfg.get("repa", {}) or {}
        self.repa_enabled = bool(self.repa_cfg.get("enabled", False))
        if self.repa_enabled:
            # the REPA representation-cache pipeline (DINOv2 patch targets) is not included
            raise ValueError("repa.enabled=true: the REPA data pipeline is not included in this code")
        self.use_precomputed_latents = bool(cfg.dataset.get("use_precomputed_latents", False))
        self.overfitting_enabled = bool(cfg.trainer.get("overfitting", {}).get("enabled", False))
        self.filter_for_classes = cfg.dataset.get("filter_for_classes", False)
        self.class_indices = cfg.dataset.get("filter_class_indices", [])
        self.batch_size = cfg.trainer.get(
            "per_device_batch_size",
            cfg.trainer.get("batch_size", 1),
        )
        self.num_workers = cfg.trainer.num_workers
        self.world_size = int(cfg.trainer.get("resolved_world_size", 1))
        self.resolution = cfg.dataset.img_resolution
        self.latent_cache_batch_size = int(
            cfg.dataset.get("latent_cache_batch_size", min(self.batch_size, 8))
        )
        self.latent_cache_num_workers = int(
            cfg.dataset.get("latent_cache_num_workers", min(self.num_workers, 4))
        )

        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        # dataset.crop / dataset.hflip: the training-image preparation of the on-the-fly encoding path.
        # Validation (and any latent cache built here) gets the crop without the flip; the cached path flips through its
        # mirrored copies, so hflip there would double it.
        self.crop = str(cfg.dataset.get("crop", "torchvision"))
        self.hflip = bool(cfg.dataset.get("hflip", False))
        if self.use_precomputed_latents and (self.crop != "torchvision" or self.hflip):
            raise ValueError("dataset.crop / dataset.hflip apply to on-the-fly encoding "
                             "(dataset.use_precomputed_latents=false); the latent caches were built with the torchvision crop")
        self.transform = build_imagenet_transform(self.resolution, crop=self.crop)
        self.train_transform = build_imagenet_transform(self.resolution, crop=self.crop, hflip=self.hflip)
        self.dims = (3, self.resolution, self.resolution)
        self._overfitting_train_batch = None

    def _cache_build_context(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _build_cache_dataset_view(self, dataset, rank, world_size, batch_size=None):
        if world_size <= 1:
            return dataset

        if batch_size is None:
            batch_size = self.latent_cache_batch_size

        global_batch_size = int(batch_size) * world_size
        dummy_index = 0
        entries = []
        for global_start in range(0, len(dataset), global_batch_size):
            batch_end = min(global_start + global_batch_size, len(dataset))
            valid_count = batch_end - global_start
            batch_indices = list(range(global_start, batch_end))
            batch_indices.extend([dummy_index] * (global_batch_size - valid_count))

            valid_flags = [True] * valid_count
            valid_flags.extend([False] * (global_batch_size - valid_count))

            local_start = rank * int(batch_size)
            local_end = local_start + int(batch_size)
            entries.extend(zip(batch_indices[local_start:local_end], valid_flags[local_start:local_end]))

        return _CacheBuildDataset(dataset, entries)

    def _distributed_context(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0, self.world_size

        rank = int(getattr(trainer, "global_rank", 0))
        world_size = int(getattr(trainer, "world_size", self.world_size))
        return rank, max(world_size, 1)

    def _distributed_sampler(self, dataset, shuffle):
        rank, world_size = self._distributed_context()
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
            seed=int(self.cfg.get("seed", 0)),
        )

    def _resolve_split_dir(self, split):
        split_dir = self.data_dir / split
        if split_dir.is_dir():
            return str(split_dir)

        expected_train = self.data_dir / "train"
        expected_val = self.data_dir / "val"
        raise FileNotFoundError(
            "ImageNet dataset directory is misconfigured. "
            f"Resolved data_dir='{self.data_dir}', but expected split directory '{split_dir}' does not exist. "
            f"Expected layout: '{expected_train}' and '{expected_val}'. "
            "Override 'data_dir=/path/to/imagenet' (configs/rae/train.yaml) or set the SBM_IMAGENET_DIR environment variable."
        )

    def _filter_by_classes(self, dataset):
        targets = dataset.targets  # list[int]
        indices = [i for i, y in enumerate(targets) if y in self.class_indices]
        return Subset(dataset, indices)

    def _set_precomputed_latents(self, enabled):
        self.use_precomputed_latents = bool(enabled)
        self.cfg.dataset.use_precomputed_latents = self.use_precomputed_latents

    def _latent_split_dir(self, split):
        if not self.latent_data_dir:
            raise ValueError(
                "latent_data_dir must be set to use or populate the latent cache"
            )
        return self.latent_data_dir / split

    def _latent_split_ready(self, split):
        if not self.latent_data_dir:
            return False

        try:
            self._build_latent_dataset(split)
        except (FileNotFoundError, RuntimeError, ValueError):
            return False
        return True

    def _latent_cache_ready(self):
        return self._latent_split_ready("train") and self._latent_split_ready("val")

    def _latent_pack_split_dir(self, split):
        pack_dir = self.cfg.get("latent_pack_dir", None)
        if pack_dir and (Path(str(pack_dir)) / split).is_dir():
            return Path(str(pack_dir)) / split
        return None

    def _build_latent_encoder(self):
        vae_cfg = self.cfg.get("vae", {})
        vae = AutoencoderKL.from_pretrained(
            vae_cfg.get("pretrained_model_name_or_path", "stabilityai/sd-vae-ft-mse"),
            local_files_only=vae_cfg.get("local_files_only", False),
        )
        vae.eval()
        for parameter in vae.parameters():
            parameter.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae.to(device)
        return vae, device

    def _write_latent_file(self, output_dir, relative_path, moments, moments_flip, label):
        output_path = output_dir / Path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
        torch.save(
            {
                "moments": _compact_cpu_tensor(moments),
                "moments_flip": _compact_cpu_tensor(moments_flip),
                "label": int(label),
            },
            temp_path,
        )
        os.replace(temp_path, output_path)

    def _write_latent_split_cache(self, split, vae, device):
        output_dir = self._latent_split_dir(split)
        temp_dir = self.latent_data_dir / f".{split}.tmp"

        rank, world_size = self._cache_build_context()
        if rank == 0:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()

        dataset = _ImageFolderWithRelativePath(self._resolve_split_dir(split), transform=self.transform)
        dataset = self._build_cache_dataset_view(
            dataset,
            rank,
            world_size,
            batch_size=self.latent_cache_batch_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.latent_cache_batch_size,
            shuffle=False,
            num_workers=self.latent_cache_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=self.latent_cache_num_workers > 0,
        )

        if rank == 0:
            print(f"Preparing latent cache for split '{split}' at '{output_dir}' using {world_size} rank(s)")
        with torch.no_grad():
            for batch in loader:
                if world_size > 1:
                    images, labels, relative_paths, valid_mask = batch
                else:
                    images, labels, relative_paths = batch
                    valid_mask = torch.ones(labels.shape[0], dtype=torch.bool)

                images = images.to(device, dtype=vae.dtype, non_blocking=device.type == "cuda")
                labels = labels.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
                valid_mask = valid_mask.to(device=device, dtype=torch.bool, non_blocking=device.type == "cuda")

                moments = vae.encode(images).latent_dist.parameters.to(dtype=torch.float32)
                moments_flip = vae.encode(images.flip(dims=[3])).latent_dist.parameters.to(dtype=torch.float32)

                moments = moments[valid_mask].cpu()
                moments_flip = moments_flip[valid_mask].cpu()
                labels = labels[valid_mask].to(device="cpu")
                relative_paths = [
                    path
                    for path, is_valid in zip(relative_paths, valid_mask.cpu().tolist())
                    if is_valid
                ]

                for index, relative_path in enumerate(relative_paths):
                    self._write_latent_file(
                        temp_dir,
                        relative_path,
                        moments[index],
                        moments_flip[index],
                        labels[index].item(),
                    )

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            temp_dir.rename(output_dir)
        if world_size > 1:
            dist.barrier()

    def _populate_latent_cache(self):
        if not self.latent_data_dir:
            return

        self.latent_data_dir.mkdir(parents=True, exist_ok=True)
        vae, device = self._build_latent_encoder()
        try:
            for split in ("train", "val"):
                if self._latent_split_ready(split):
                    continue
                self._write_latent_split_cache(
                    split,
                    vae,
                    device,
                )
        finally:
            vae.to("cpu")
            del vae
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def build_latent_cache(self):
        if not self.latent_data_dir:
            raise ValueError("latent_data_dir must be set to populate the latent cache")

        if self._latent_cache_ready():
            self._set_precomputed_latents(True)
            return

        self._populate_latent_cache()
        if not self._latent_cache_ready():
            raise RuntimeError(
                f"Latent cache build did not produce a valid cache at '{self.latent_data_dir}'"
            )
        self._set_precomputed_latents(True)

    def _ensure_latent_cache(self):
        if not self.latent_data_dir:
            if self.use_precomputed_latents:
                raise ValueError(
                    "dataset.use_precomputed_latents=true requires latent_data_dir to be set"
                )
            self._set_precomputed_latents(False)
            return

        if self._latent_cache_ready():
            self._set_precomputed_latents(True)
            return

        if self.world_size > 1:
            raise RuntimeError(
                "Latent cache is missing or incomplete for a distributed run. "
                "Auto-population is disabled when trainer.resolved_world_size > 1 because a single rank would build the cache while the others block. "
                "Run a single-process job once to populate it."
            )

        lock_dir = self.latent_data_dir / ".cache_lock"
        deadline = time.time() + LATENT_CACHE_TIMEOUT_SECONDS
        self.latent_data_dir.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                lock_dir.mkdir()
            except FileExistsError:
                if self._latent_cache_ready():
                    self._set_precomputed_latents(True)
                    return
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for latent cache lock at '{lock_dir}'"
                    )
                print(f"Waiting for latent cache to finish at '{self.latent_data_dir}'")
                time.sleep(LATENT_CACHE_WAIT_SECONDS)
                continue

            try:
                if not self._latent_cache_ready():
                    self._populate_latent_cache()
            finally:
                shutil.rmtree(lock_dir, ignore_errors=True)
            break

        if not self._latent_cache_ready():
            raise RuntimeError(
                f"Latent cache build did not produce a valid cache at '{self.latent_data_dir}'"
            )
        self._set_precomputed_latents(True)

    def require_latent_cache(self):
        if not self.latent_data_dir:
            raise ValueError(
                "Training requires latent_data_dir to be set when using precomputed latents. "
                "Populate the cache first with a single-process run."
            )

        if not self._latent_cache_ready():
            raise RuntimeError(
                "Latent cache is missing or incomplete. "
                f"Expected ready train/val caches under '{self.latent_data_dir}'. "
                "Populate the cache first with a single-process run."
            )

        self._set_precomputed_latents(True)

    def require_representation_cache(self):
        # kept for the shared entry point (scripts/rae/train.py calls it); the REPA data pipeline is not included
        if self.repa_enabled:
            raise ValueError("repa.enabled=true: the REPA data pipeline is not included in this code")

    def _resolve_latent_split_dir(self, split):
        if not self.latent_data_dir:
            raise ValueError(
                "latent_data_dir must be set to use the latent cache"
            )

        split_dir = self._latent_split_dir(split)
        if split_dir.is_dir():
            return str(split_dir)

        raise FileNotFoundError(
            "Precomputed latent dataset directory is misconfigured. "
            f"Resolved latent_data_dir='{self.latent_data_dir}', but expected split directory '{split_dir}' does not exist. "
            "Populate the cache from data_dir or override latent_data_dir=/path/to/latents."
        )

    def _build_latent_dataset(self, split):
        vae_cfg = self.cfg.get("vae", {})
        pack_dir = self.cfg.get("latent_pack_dir", None)
        if pack_dir and (Path(str(pack_dir)) / split).is_dir():
            return PackedLatentDataset(
                Path(str(pack_dir)) / split,
                latents_scale=vae_cfg.get("latents_scale", 0.18215),
                latents_bias=vae_cfg.get("latents_bias", 0.0),
            )
        return LatentFileDataset(
            self._resolve_latent_split_dir(split),
            latents_scale=vae_cfg.get("latents_scale", 0.18215),
            latents_bias=vae_cfg.get("latents_bias", 0.0),
        )

    def _build_dataset(self, split):
        if self.use_precomputed_latents:
            dataset = self._build_latent_dataset(split)
        else:
            transform = self.train_transform if split == "train" else self.transform
            dataset = ImageFolder(self._resolve_split_dir(split), transform=transform)

        if self.filter_for_classes and self.class_indices:
            dataset = self._filter_by_classes(dataset)
        return dataset

    def prepare_data(self):
        self._ensure_latent_cache()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.imagenet_train = self._build_dataset("train")
            self.imagenet_val = self._build_dataset("val")
            # Dataset-agnostic aliases used by the shared entrypoints.
            self.train_dataset = self.imagenet_train
            self.val_dataset = self.imagenet_val

        if stage == "test" or stage is None:
            self.imagenet_test = self._build_dataset("val")
            self.test_dataset = self.imagenet_test

    def _build_train_dataloader(self, *, persistent_workers=None):
        if persistent_workers is None:
            persistent_workers = self.num_workers > 0

        sampler = self._distributed_sampler(self.imagenet_train, shuffle=True)
        return DataLoader(
            self.imagenet_train,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0,
        )

    def _overfitting_train_dataloader(self):
        if self._overfitting_train_batch is None:
            loader = self._build_train_dataloader(persistent_workers=False)
            try:
                first_batch = next(iter(loader))
            except StopIteration as exc:
                raise RuntimeError("Cannot enable overfitting mode with an empty train dataset") from exc
            self._overfitting_train_batch = _clone_batch(first_batch)

        return DataLoader(
            _SingleBatchDataset(self._overfitting_train_batch),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_single_batch_collate,
        )

    def train_dataloader(self):
        if self.overfitting_enabled:
            return self._overfitting_train_dataloader()
        return self._build_train_dataloader()

    def val_dataloader(self):
        if self.overfitting_enabled:
            return self._overfitting_train_dataloader()

        sampler = self._distributed_sampler(self.imagenet_val, shuffle=False)
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        sampler = self._distributed_sampler(self.imagenet_test, shuffle=False)
        return DataLoader(
            self.imagenet_test,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def get_data_module(cfg):
    dataset_name = cfg.dataset.get("name", "imagenet")
    if dataset_name == "imagenet":
        return ImageNetDataModule(cfg)
    if dataset_name == "bridge":
        # Bridge-V2 / RT-1 clips for the world models (sbm_rae.bridge_data)
        from sbm_rae.bridge_data import BridgeDataModule

        return BridgeDataModule(cfg)

    raise ValueError(f"Unsupported dataset: {dataset_name}. Available: imagenet, bridge")
