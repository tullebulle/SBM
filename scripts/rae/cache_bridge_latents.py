"""Encode every world-model frame once: <processed>/<split>/*.mp4 -> <cache>/<split>/latents.npy (fp16 memmap,
(total frames, C, h, w), scaled as the trainer scales them), actions.npy (total frames, A) and index.json (per episode:
stem, offset, length), so training reads latent windows instead of decoding video and running the codec on every step
(the SD-VAE encoder costs ~10x the U-Net step at 256x256).

usage: python scripts/rae/cache_bridge_latents.py --processed <dir> --out <dir> --codec sdvae|sd3 [--split train val] [--workers 8]
                                              [--batch 64] [--shard K --shard-id i]   (shards write their own part files; --merge joins them)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbm_rae.bridge_data import BridgeClips  # noqa: E402


class _Full(Dataset):
    def __init__(self, clips, ids):
        self.clips, self.ids = clips, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.clips[self.ids[i]]


def build_codec(kind, device):
    if kind == "sdvae":
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).eval().to(device)
        return vae, 0.18215, 4
    if kind == "sd3":
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae", local_files_only=True).eval().to(device)
        return vae, float(vae.config.scaling_factor), 16
    raise ValueError(kind)


@torch.no_grad()
def encode(vae, scale, frames_u8, device, batch, bf16=False):
    out = []
    for i in range(0, frames_u8.shape[0], batch):
        x = frames_u8[i : i + batch].to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):   # --bf16: the encoder under autocast (MIG slices)
            z = vae.encode(x).latent_dist.sample().float() * scale
        out.append(z.to(torch.float16).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codec", default="sdvae", choices=["sdvae", "sd3"])
    ap.add_argument("--split", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--action-dim", type=int, default=7, help="7 = Bridge-V2, 10 = RT-1 (DriftWorld's conversion)")
    ap.add_argument("--merge", action="store_true", help="join the shard part files of every split into one memmap")
    ap.add_argument("--bf16", action="store_true", help="encode under bf16 autocast (4x faster on A100 MIG slices; the latents are stored as fp16 anyway)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        for split in a.split:
            d = out / split
            parts = sorted(d.glob("part*.index.json"))
            entries, lat_parts, act_parts = [], [], []
            for p in parts:
                meta = json.loads(p.read_text())
                lat_parts.append(np.load(str(p).replace(".index.json", ".latents.npy"), mmap_mode="r"))
                act_parts.append(np.load(str(p).replace(".index.json", ".actions.npy"), mmap_mode="r"))
                entries.append(meta)
            total = sum(m["total"] for m in entries)
            C, h, w = lat_parts[0].shape[1:]
            A = act_parts[0].shape[1]
            lat = np.lib.format.open_memmap(d / "latents.npy", mode="w+", dtype=np.float16, shape=(total, C, h, w))
            act = np.lib.format.open_memmap(d / "actions.npy", mode="w+", dtype=np.float32, shape=(total, A))
            index, off = [], 0
            for meta, lp, apart in zip(entries, lat_parts, act_parts):
                n = meta["total"]
                lat[off : off + n] = lp[:n]
                act[off : off + n] = apart[:n]
                for e in meta["episodes"]:
                    index.append({"stem": e["stem"], "offset": off + e["offset"], "length": e["length"]})
                off += n
            lat.flush(); act.flush()
            (d / "index.json").write_text(json.dumps({"codec": entries[0]["codec"], "scale": entries[0]["scale"], "total": total,
                                                      "shape": [C, h, w], "episodes": index}))
            print(f"{split}: merged {len(parts)} parts, {total} frames, {len(index)} episodes -> {d}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, scale, C = build_codec(a.codec, device)
    for split in a.split:
        clips = BridgeClips(a.processed, split, n_frames=1, action_dim=a.action_dim, full_video=True)
        ids = list(range(a.shard_id, len(clips), a.shard))
        total = sum(clips.lengths[i] for i in ids)
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        tag = f"part{a.shard_id:02d}of{a.shard:02d}" if a.shard > 1 else "all"
        lat = np.lib.format.open_memmap(d / f"{tag}.latents.npy", mode="w+", dtype=np.float16, shape=(total, C, 32, 32))
        act = np.lib.format.open_memmap(d / f"{tag}.actions.npy", mode="w+", dtype=np.float32, shape=(total, clips.action_dim))
        loader = DataLoader(_Full(clips, ids), batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None)
        episodes, off, t0 = [], 0, time.time()
        for k, item in enumerate(loader):
            frames, actions = item["x"], item["act"]
            n = min(frames.shape[0], actions.shape[0])
            z = encode(vae, scale, frames[:n], device, a.batch, bf16=a.bf16)
            lat[off : off + n] = z.numpy()
            act[off : off + n] = actions[:n].numpy()
            episodes.append({"stem": clips.stems[ids[k]], "offset": off, "length": int(n)})
            off += n
            if k % 500 == 0:
                print(f"{split} {tag}: episode {k}/{len(ids)} frames {off}/{total} {time.time() - t0:.0f}s", flush=True)
        lat.flush(); act.flush()
        if a.shard == 1:   # one shard: the final files directly (no merge step)
            os.replace(d / f"{tag}.latents.npy", d / "latents.npy")
            os.replace(d / f"{tag}.actions.npy", d / "actions.npy")
            (d / "index.json").write_text(json.dumps({"codec": a.codec, "scale": scale, "total": off, "shape": [C, 32, 32],
                                                      "episodes": episodes}))
        else:
            (d / f"{tag}.index.json").write_text(json.dumps({"codec": a.codec, "scale": scale, "total": off, "episodes": episodes}))
        print(f"{split} {tag}: done, {off} frames, {len(episodes)} episodes, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
