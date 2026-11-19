import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from sbm_rae.data import ImageNetDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rae.train import _resolve_trainer_loop_settings


def _base_cfg(tmp_path, overfitting_enabled):
    return OmegaConf.create(
        {
            "seed": 123,
            "data_dir": str(tmp_path),
            "latent_data_dir": None,
            "dataset": {
                "use_precomputed_latents": False,
                "img_resolution": 8,
            },
            "trainer": {
                "per_device_batch_size": 2,
                "num_workers": 0,
                "resolved_world_size": 1,
                "resolved_max_epochs": 7,
                "resolved_max_steps": -1,
                "limit_val_batches": 8,
                "val_check_interval": 1000,
                "overfitting": {"enabled": overfitting_enabled},
            },
        }
    )


def test_overfitting_train_dataloader_reuses_frozen_first_batch(tmp_path):
    cfg = _base_cfg(tmp_path, overfitting_enabled=True)
    datamodule = ImageNetDataModule(cfg)
    datamodule.imagenet_train = TensorDataset(
        torch.arange(24, dtype=torch.float32).view(6, 4),
        torch.arange(6, dtype=torch.long),
    )

    loader = datamodule.train_dataloader()
    first_x, first_labels = next(iter(loader))
    first_x.zero_()

    second_x, second_labels = next(iter(datamodule.train_dataloader()))
    third_x, third_labels = next(iter(datamodule.train_dataloader()))

    assert len(loader) == 1
    torch.testing.assert_close(second_x, third_x)
    torch.testing.assert_close(second_labels, third_labels)
    assert not torch.equal(first_x, second_x)
    assert torch.equal(second_labels, third_labels)


def test_overfitting_val_dataloader_matches_frozen_train_batch(tmp_path):
    cfg = _base_cfg(tmp_path, overfitting_enabled=True)
    datamodule = ImageNetDataModule(cfg)
    datamodule.imagenet_train = TensorDataset(
        torch.arange(24, dtype=torch.float32).view(6, 4),
        torch.arange(6, dtype=torch.long),
    )
    datamodule.imagenet_val = TensorDataset(
        -torch.arange(24, dtype=torch.float32).view(6, 4),
        torch.arange(10, 16, dtype=torch.long),
    )

    train_x, train_labels = next(iter(datamodule.train_dataloader()))
    val_x, val_labels = next(iter(datamodule.val_dataloader()))

    torch.testing.assert_close(val_x, train_x)
    torch.testing.assert_close(val_labels, train_labels)


def test_overfitting_loop_settings_disable_validation(tmp_path):
    cfg = _base_cfg(tmp_path, overfitting_enabled=True)

    settings = _resolve_trainer_loop_settings(cfg)

    assert settings["limit_train_batches"] == 1
    assert settings["limit_val_batches"] == 0
    assert settings["num_sanity_val_steps"] == 0
    assert settings["val_check_interval"] == 1
    assert settings["max_steps"] == 100000


def test_standard_loop_settings_keep_configured_validation(tmp_path):
    cfg = _base_cfg(tmp_path, overfitting_enabled=False)

    settings = _resolve_trainer_loop_settings(cfg)

    assert settings["max_epochs"] == 7
    assert settings["max_steps"] == -1
    assert settings["limit_train_batches"] == 1.0
    assert settings["limit_val_batches"] == 8
    assert settings["val_check_interval"] == 1000