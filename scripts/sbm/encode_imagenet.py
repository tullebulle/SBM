"""Encode ImageNet-1k train into the SD-VAE latent pack the SD-VAE trainer reads (configs/sbm/train.yaml data.latent_pack_dir).

Every image is preprocessed as ADM / guided-diffusion does (`center_crop_arr`: halve with a BOX filter while the short side
is at least 512, BICUBIC resize of the short side to 256, center crop), mapped to [-1, 1] and encoded by the SD-VAE
(stabilityai/sd-vae-ft-mse); the posterior parameters (mean and log-variance, 8 x 32 x 32) of the image and of its
horizontal flip are stored, and the trainer samples a latent from one of the two at random.

Output: <out>/train/part_XXX.npy, float32 of shape (n, 2, 8, 32, 32), and part_XXX.json with the files, labels and classes,
40 parts of 25 classes each (classes in sorted WordNet-id order = the label ids, files sorted within a class).

    python scripts/sbm/encode_imagenet.py <imagenet>/train <store>/data/imagenet_latents_packed_adm [--shard i --num-shards n]

One GPU; parts already written are skipped (rerun to resume); --shard i --num-shards n encodes the parts p with
p % n == i, so n processes (one per GPU) split the work. About 5 minutes per part on one H100 with 14 loader workers.
--check <pack> encodes the first --limit images of each selected part and compares them with an existing pack instead of
writing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PARTS, CLASSES_PER_PART = 40, 25
EXTENSIONS = (".jpeg", ".jpg", ".png")


def center_crop_arr(pil_image, image_size=256):
    """ADM's preprocessing (guided-diffusion, image_datasets.py)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return torch.from_numpy(np.array(center_crop_arr(im))).permute(2, 0, 1)


def part_files(root: Path, classes: list[str], part: int):
    """(paths, files, labels) of part `part`: its 25 classes, files sorted by name."""
    paths, files, labels = [], [], []
    for label in range(part * CLASSES_PER_PART, (part + 1) * CLASSES_PER_PART):
        d = root / classes[label]
        if not d.is_dir():
            continue
        for name in sorted(f for f in os.listdir(d) if f.lower().endswith(EXTENSIONS)):
            paths.append(d / name)
            files.append(f"{classes[label]}/{Path(name).stem}.pt")
            labels.append(label)
    return paths, files, labels


@torch.no_grad()
def encode(vae, paths, device, batch, workers, sink):
    """Calls sink(start, moments) for consecutive batches; moments (b, 2, 8, 32, 32) float32 on the CPU."""
    loader = torch.utils.data.DataLoader(Images(paths), batch_size=batch, num_workers=workers, shuffle=False)
    i, t0 = 0, time.time()
    for b, u8 in enumerate(loader):
        x = u8.to(device).float() / 127.5 - 1.0
        m = vae.encode(x).latent_dist.parameters.float()
        mf = vae.encode(x.flip(dims=[3])).latent_dist.parameters.float()
        sink(i, torch.stack([m, mf], 1).cpu())
        i += len(u8)
        if b % 50 == 0:
            print(f"  {i}/{len(paths)} images, {i / max(time.time() - t0, 1e-6):.0f} img/s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_dir", help="ImageNet train folder: <train_dir>/<wnid>/*.JPEG")
    ap.add_argument("out", help="the pack folder (data.latent_pack_dir); parts go to <out>/train/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--parts", type=int, nargs="*", help="only these part indices (0-39)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--check", help="an existing pack folder to compare with (nothing is written)")
    ap.add_argument("--limit", type=int, default=512, help="images per part in --check mode")
    a = ap.parse_args(argv)
    from sbm.infer import load_vae

    root = Path(a.train_dir)
    classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
    if len(classes) != PARTS * CLASSES_PER_PART:
        sys.exit(f"{root} has {len(classes)} class folders, expected 1000")
    parts = a.parts if a.parts else list(range(PARTS))
    parts = [p for p in parts if p % a.num_shards == a.shard]
    # full fp32 convolutions: with cuDNN's TF32 the moments move by about 1e-2 rms (max ~2); without it the encoder
    # reproduces the paper's latent pack to about 2e-3 rms (3e-4 of the latents' scale)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    vae = load_vae(device)
    out = Path(a.out) / "train"
    out.mkdir(parents=True, exist_ok=True)
    for part in parts:
        paths, files, labels = part_files(root, classes, part)
        if a.check:
            ref_dir = Path(a.check) / "train"
            meta = json.loads((ref_dir / f"part_{part:03d}.json").read_text())
            row = {f: i for i, f in enumerate(meta["files"])}
            ref = np.load(ref_dir / f"part_{part:03d}.npy", mmap_mode="r")
            paths, files, labels = paths[:a.limit], files[:a.limit], labels[:a.limit]
            stats = {"n": 0, "max": 0.0, "mean_max": 0.0}

            def compare(start, m):
                idx = [row[f] for f in files[start:start + len(m)]]
                assert [meta["labels"][j] for j in idx] == labels[start:start + len(m)]
                d = (m - torch.from_numpy(np.stack([ref[j] for j in idx]))).abs()
                stats["n"] += len(m)
                stats["max"] = max(stats["max"], float(d.max()))
                stats["mean_max"] = max(stats["mean_max"], float(d[:, :, :4].max()))

            encode(vae, paths, device, a.batch, a.workers, compare)
            print(f"check part {part}: {stats['n']} images, max |diff| {stats['max']:.3g} (means {stats['mean_max']:.3g})", flush=True)
            continue
        final = out / f"part_{part:03d}.npy"
        if final.exists() and final.with_suffix(".json").exists():
            print(f"part {part}: done already", flush=True)
            continue
        print(f"part {part}: {len(paths)} images", flush=True)
        tmp = Path(f"{final}.tmp{os.getpid()}")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(paths), 2, 8, 32, 32))

        def write(start, m):
            arr[start:start + len(m)] = m.numpy()

        encode(vae, paths, device, a.batch, a.workers, write)
        arr.flush()
        del arr
        os.replace(tmp, final)
        final.with_suffix(".json").write_text(json.dumps({
            "files": files, "labels": labels, "classes": classes[part * CLASSES_PER_PART:(part + 1) * CLASSES_PER_PART],
            "preprocessing": "ADM center_crop_arr (BOX + BICUBIC), sd-vae-ft-mse moments"}))
        print(f"part {part}: written {final}", flush=True)


if __name__ == "__main__":
    main()
