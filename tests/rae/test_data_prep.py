"""dataset.crop / dataset.hflip: the training-image preparation of the on-the-fly encoding path.

    .venv/bin/python -m pytest tests/rae/test_data_prep.py -q
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from sbm_rae.data import ImageNetDataModule, build_imagenet_transform, center_crop_arr


def photo(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def original_transform(resolution):
    """build_imagenet_transform as it was before the knobs."""
    return transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


@pytest.mark.parametrize("size", [(500, 375), (375, 500), (180, 216), (1234, 994), (256, 256)])
def test_default_is_the_original_transform(size):
    img = photo(*size)
    assert torch.equal(build_imagenet_transform(256)(img), original_transform(256)(img))


def test_adm_crop_is_center_crop_arr():
    """ADM's steps by hand on a 1100 x 700 image: one box halving (700 >= 512) to 550 x 350, bicubic to 402 x 256
    (round(550 * 256 / 350) = 402), crop offset (402 - 256) // 2 = 73."""
    img = photo(1100, 700)
    half = img.resize((550, 350), resample=Image.BOX)
    big = half.resize((402, 256), resample=Image.BICUBIC)
    want = np.asarray(big)[:, 73:73 + 256]
    got = center_crop_arr(img, 256)
    assert got.size == (256, 256)
    assert np.array_equal(np.asarray(got), want)
    t = build_imagenet_transform(256, crop="adm")(img)
    want_t = transforms.Normalize([0.5] * 3, [0.5] * 3)(transforms.ToTensor()(Image.fromarray(want)))
    assert torch.equal(t, want_t)


def test_adm_crop_differs_from_the_original():
    img = photo(500, 375)
    a, b = build_imagenet_transform(256)(img), build_imagenet_transform(256, crop="adm")(img)
    assert a.shape == b.shape == (3, 256, 256)
    assert not torch.equal(a, b)


def test_hflip_flips_about_half_the_draws():
    img = photo(500, 375)
    plain = build_imagenet_transform(256, crop="adm")(img)
    t = build_imagenet_transform(256, crop="adm", hflip=True)
    torch.manual_seed(0)
    outs = [t(img) for _ in range(64)]
    flipped = sum(torch.equal(o, plain.flip(-1)) for o in outs)
    same = sum(torch.equal(o, plain) for o in outs)
    assert flipped + same == 64 and 16 < flipped < 48


def test_unknown_crop_raises():
    with pytest.raises(ValueError, match="crop"):
        build_imagenet_transform(256, crop="bicubic")


def cfg(precomputed=False, **dataset):
    return OmegaConf.create({
        "seed": 0, "data_dir": "/nonexistent",
        "dataset": {"use_precomputed_latents": precomputed, "img_resolution": 256, "filter_for_classes": False,
                    "filter_class_indices": [], **dataset},
        "vae": {"latents_scale": 1.0, "latents_bias": 0.0}, "repa": {"enabled": False},
        "trainer": {"per_device_batch_size": 2, "num_workers": 0, "resolved_world_size": 1, "overfitting": {"enabled": False}},
    })


def test_datamodule_default_has_no_flip_and_the_original_crop():
    dm = ImageNetDataModule(cfg())
    img = photo(500, 375)
    for t in (dm.transform, dm.train_transform):
        assert not any(isinstance(s, transforms.RandomHorizontalFlip) for s in t.transforms)
        assert torch.equal(t(img), original_transform(256)(img))


def test_datamodule_flips_training_images_only(tmp_path):
    for split in ("train", "val"):
        (tmp_path / split / "n0").mkdir(parents=True)
        photo(500, 375).save(tmp_path / split / "n0" / "a.png")
    c = cfg(crop="adm", hflip=True)
    c.data_dir = str(tmp_path)
    dm = ImageNetDataModule(c)
    dm.setup(stage="fit")
    assert any(isinstance(s, transforms.RandomHorizontalFlip) for s in dm.imagenet_train.transform.transforms)
    assert not any(isinstance(s, transforms.RandomHorizontalFlip) for s in dm.imagenet_val.transform.transforms)
    want = build_imagenet_transform(256, crop="adm")(photo(500, 375))
    torch.manual_seed(0)
    train = [dm.imagenet_train[0][0] for _ in range(32)]
    assert all(torch.equal(x, want) or torch.equal(x, want.flip(-1)) for x in train)
    assert any(torch.equal(x, want.flip(-1)) for x in train)
    assert all(torch.equal(dm.imagenet_val[0][0], want) for _ in range(8))


def test_precomputed_latents_reject_the_knobs():
    with pytest.raises(ValueError, match="on-the-fly"):
        ImageNetDataModule(cfg(precomputed=True, hflip=True))
    with pytest.raises(ValueError, match="on-the-fly"):
        ImageNetDataModule(cfg(precomputed=True, crop="adm"))
