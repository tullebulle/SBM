import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from sbm_rae.data import ImageNetDataModule, LatentFileDataset


def _storage_nbytes(tensor):
    return tensor.untyped_storage().nbytes()


def _write_latent_file_tree_split(root, split, class_labels):
    split_dir = root / split
    for class_label, labels in enumerate(class_labels):
        class_dir = split_dir / str(class_label)
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx, label in enumerate(labels):
            latent = torch.full((4, 1, 1), float(label * 10 + sample_idx))
            torch.save(
                {"latents": latent, "label": int(label)},
                class_dir / f"sample_{sample_idx:05d}.pt",
            )


def _latent_cfg(tmp_path, *, resolved_world_size=2):
    return OmegaConf.create(
        {
            "seed": 123,
            "data_dir": str(tmp_path / "images"),
            "latent_data_dir": str(tmp_path / "latents"),
            "dataset": {
                "use_precomputed_latents": True,
                "img_resolution": 8,
                "filter_for_classes": False,
                "filter_class_indices": [],
            },
            "vae": {"latents_scale": 1.0, "latents_bias": 0.0},
            "repa": {"enabled": False},
            "trainer": {
                "per_device_batch_size": 2,
                "num_workers": 0,
                "resolved_world_size": resolved_world_size,
                "overfitting": {"enabled": False},
            },
        }
    )


def _latent_datamodule(tmp_path, *, resolved_world_size=2):
    latent_root = tmp_path / "latents"
    class_labels = [[0, 0, 0, 0], [1, 1, 1, 1]]
    for split in ("train", "val"):
        _write_latent_file_tree_split(latent_root, split, class_labels)

    datamodule = ImageNetDataModule(_latent_cfg(tmp_path, resolved_world_size=resolved_world_size))
    datamodule.setup(stage="fit")
    return datamodule


def test_latent_file_dataset_reads_individual_cached_files(tmp_path):
    latent_root = tmp_path / "latents"
    _write_latent_file_tree_split(latent_root, "train", [[0, 0], [1]])

    dataset = LatentFileDataset(latent_root / "train", latents_scale=1.0, latents_bias=0.0)

    assert len(dataset) == 3
    assert dataset.targets == [0, 0, 1]
    latent, label = dataset[2]

    assert label == 1
    torch.testing.assert_close(latent, torch.full((4, 1, 1), 10.0))


def test_latent_cache_writer_saves_compact_sample_storage(tmp_path):
    datamodule = ImageNetDataModule(_latent_cfg(tmp_path, resolved_world_size=1))
    moments_batch = torch.arange(64 * 8 * 2 * 2, dtype=torch.float32).reshape(64, 8, 2, 2)
    moments_flip_batch = moments_batch + 1000.0

    datamodule._write_latent_file(
        tmp_path / "cache",
        "0/sample.pt",
        moments_batch[3],
        moments_flip_batch[3],
        0,
    )

    cache_item = torch.load(tmp_path / "cache" / "0" / "sample.pt", map_location="cpu")

    assert _storage_nbytes(cache_item["moments"]) == cache_item["moments"].numel() * cache_item["moments"].element_size()
    assert _storage_nbytes(cache_item["moments_flip"]) == cache_item["moments_flip"].numel() * cache_item["moments_flip"].element_size()


def test_latent_file_dataset_rejects_batched_moments(tmp_path):
    class_dir = tmp_path / "latents" / "train" / "0"
    class_dir.mkdir(parents=True)
    torch.save(
        {
            "moments": torch.zeros(2, 8, 1, 1),
            "moments_flip": torch.zeros(2, 8, 1, 1),
            "label": 0,
        },
        class_dir / "sample.pt",
    )
    dataset = LatentFileDataset(tmp_path / "latents" / "train", latents_scale=1.0, latents_bias=0.0)

    with pytest.raises(ValueError, match="Batched latent cache files"):
        dataset[0]


def test_latent_train_dataloader_uses_full_random_distributed_sampler(tmp_path):
    datamodule = _latent_datamodule(tmp_path)

    loader = datamodule.train_dataloader()

    sampler = loader.sampler

    assert isinstance(datamodule.imagenet_train, LatentFileDataset)
    assert isinstance(sampler, DistributedSampler)
    assert sampler.shuffle
    assert sampler.seed == 123


def test_latent_train_dataloader_shuffles_in_single_process(tmp_path):
    datamodule = _latent_datamodule(tmp_path, resolved_world_size=1)

    loader = datamodule.train_dataloader()

    assert loader.sampler is not None
    assert loader.batch_sampler.sampler is loader.sampler