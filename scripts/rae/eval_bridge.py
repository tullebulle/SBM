"""World-model evaluation: DriftWorld's validation protocol (their eval_on_validation_shard.py + aggregate) on an SBM world model.

Every validation episode (full length L): the first n_history frames are given, the rest are generated autoregressively in
chunks of `--chunk` (8) frames, with the history reset to the ground truth at the start of every chunk; one network pass per
frame (T(z; history, actions) from noise z). Reference = the codec's reconstruction of the ground truth. Per-video SSIM / PSNR
(skimage, uint8) and LPIPS (AlexNet) averaged over the generated frames, then over videos; FID (pytorch-fid, Inception pool3)
over every generated frame against the reconstructed frames (features accumulated per shard). FVD (stylegan-v's fvd2048_16f
on the first 16 generated frames of each video, written to pred_first16/ and gt_first16/) is computed by scripts/rae/wm/eval.sh.
The same per-video metrics against the raw frames are reported with the suffix _raw (not part of DriftWorld's protocol).

usage (one shard per GPU, then the aggregate):
  python scripts/rae/eval_bridge.py --ckpt <export or training checkpoint> --out <dir> --rank r --world-size W
                                    [--omega 0.8] [--frame-size 256] [--chunk 8] [--max-videos N] [--ema 0.999]
                                    [--processed <dir>] [--action-dim 7|10]
  python scripts/rae/eval_bridge.py --out <dir> --aggregate [--skip-fid]   (adds <dir>/fvd.json if the FVD step wrote it)

--omega / --frame-size / --chunk default to the read an export records (meta.read), so an export reproduces its published
numbers without flags; for a training checkpoint they default to 0.8 / the dataset's frames / 8 (RT-1 reads use
--frame-size 256: its 256x320 frames resized as in training). --processed / --action-dim default to the checkpoint's
dataset config. The history frames are encoded with a sample of the VAE posterior (as in DriftWorld's evaluation), so
per-video values vary slightly from run to run; the map's noise is seeded per video.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbm_rae.bridge_data import BridgeClips  # noqa: E402
from sbm_rae.infer import is_slim_export  # noqa: E402
from sbm_rae.models import build_model  # noqa: E402
from sbm_rae.paths import store_root  # noqa: E402

# DriftWorld's processed Bridge-V2 clips in the store ($SBM_STORE/data/bridge); --processed overrides
PROCESSED = str(store_root() / "data" / "bridge" / "processed" / "bridge_v2")
N_FVD_FRAMES = 16
READ_DEFAULTS = {"omega": 0.8, "frame_size": None, "chunk": 8}


# ---------------------------------------------------------------- model + codec from a checkpoint
def load_checkpoint(path, ema=None, device="cuda"):
    """(model, cfg, step, which) from an export (sbm_rae.infer) or a training checkpoint (its active EMA, or `ema`)."""
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = OmegaConf.create(ck["hyper_parameters"]["cfg"])
    if is_slim_export(ck):   # one EMA copy, keys without the "model." prefix
        if ema is not None and abs(float(ema) - float(ck["ema_decay"])) > 1e-12:
            raise KeyError(f"{path}: an export of EMA {ck['ema_decay']} only (asked for {ema})")
        model = build_model(cfg.model)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval().requires_grad_(False).to(device)
        return model, cfg, int(ck.get("global_step") or 0), f"EMA {ck['ema_decay']} (export)"
    active = ck.get("ema_active_decay", None)
    if ema is None or (active is not None and abs(float(ema) - float(active)) < 1e-12):
        sd, which = ck["state_dict"], f"active EMA {active}"
    else:
        label = f"{float(ema):g}".replace(".", "_")
        stored = ck.get("ema_state_dicts", {})
        if label not in stored:
            raise KeyError(f"{path}: no EMA {ema} (stored {sorted(stored)}, active {active})")
        sd, which = stored[label], f"EMA {ema}"
    model = build_model(cfg.model)
    missing, unexpected = model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    if unexpected or missing:
        print(f"[eval] state dict: missing {list(missing)[:4]} unexpected {list(unexpected)[:4]}", flush=True)
    model.eval().requires_grad_(False).to(device)
    step = int(ck.get("global_step", 0))
    return model, cfg, step, which


def read_settings(path):
    """The read an export records under meta.read (omega, frame_size, chunk, ...); {} for a training checkpoint."""
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return {}
    meta = ck.get("meta") if isinstance(ck, dict) else None
    return dict((meta or {}).get("read") or {})


class Codec:
    """encode(frames uint8 (N, 3, H, W)) -> scaled latents; decode(latents) -> pixels in [-1, 1]; the trainer's scaling."""

    def __init__(self, cfg, device, chunk=32):
        v = cfg.get("vae", {}) or {}
        self.kind = str(v.get("kind", "sdvae"))
        self.device, self.chunk = device, chunk
        C = int(cfg.model.in_channels)
        self.scale = torch.full((1, C, 1, 1), float(v.get("latents_scale", 0.18215)), device=device)
        self.bias = torch.full((1, C, 1, 1), float(v.get("latents_bias", 0.0)), device=device)
        if self.kind == "rae":
            from sbm_rae.rae_adapter import RAEAutoencoder

            self.vae = RAEAutoencoder(decoder=str(v.get("rae_decoder", "ViTXL_n08")), size=str(v.get("rae_size", "base"))).to(device)
        else:
            from diffusers import AutoencoderKL

            kw = {"local_files_only": True}
            if v.get("subfolder"):
                kw["subfolder"] = v["subfolder"]
            self.vae = AutoencoderKL.from_pretrained(v.get("pretrained_model_name_or_path", "stabilityai/sd-vae-ft-mse"), **kw).eval().to(device)

    @torch.no_grad()
    def encode(self, frames_u8):
        out = []
        for i in range(0, frames_u8.shape[0], self.chunk):
            x = frames_u8[i : i + self.chunk].to(self.device).float().div_(127.5).sub_(1.0)
            z = self.vae.encode(x).latent_dist.sample()
            out.append((z - self.bias) * self.scale)
        return torch.cat(out)

    @torch.no_grad()
    def decode(self, z):
        out = []
        for i in range(0, z.shape[0], self.chunk):
            x = self.vae.decode(z[i : i + self.chunk] / self.scale + self.bias).sample
            out.append(x.float().clamp(-1, 1))
        return torch.cat(out)


# ---------------------------------------------------------------- DriftWorld's rollout
@torch.no_grad()
def rollout(model, obs, act, n, chunk, omega, w, seed):
    """obs (L, C, h, w) ground-truth latents, act (L, A): the ground-truth history [0:n] followed by generated frames, in
    chunks of `chunk` frames with the history reset to the ground truth at every chunk start. One fresh Gaussian draw per
    generated frame from a generator seeded with `seed` (the generator also draws one sample per video and one per chunk
    before the frames' draws; they are unused but keep the seeds of the published reads)."""
    L, C, h, w_ = obs.shape
    n_gen = L - n
    n_chunks = math.ceil(n_gen / chunk)
    padded = n + n_chunks * chunk
    obs_pad = torch.zeros(padded, C, h, w_, device=obs.device, dtype=obs.dtype)
    obs_pad[:L] = obs
    act_pad = torch.zeros(padded, act.shape[-1], device=obs.device, dtype=torch.float32)
    act_pad[:L] = act
    result = torch.zeros_like(obs_pad)
    result[:n] = obs[:n]
    g = torch.Generator(device=obs.device).manual_seed(seed)
    torch.randn(1, C, h, w_, device=obs.device, dtype=obs.dtype, generator=g)   # the per-video draw (unused)
    for ci in range(n_chunks):
        s = chunk * ci
        win = torch.zeros(n + chunk, C, h, w_, device=obs.device, dtype=obs.dtype)
        win[:n] = obs_pad[s : s + n]
        torch.randn(1, C, h, w_, device=obs.device, dtype=obs.dtype, generator=g)   # the per-chunk draw (unused)
        for i in range(chunk):
            cond = {"obs": win[i : n + i].reshape(1, n * C, h, w_), "act": act_pad[s + i : s + n + i].unsqueeze(0)}
            z = torch.randn(1, C, h, w_, device=obs.device, dtype=obs.dtype, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=obs.is_cuda):
                pred = model(z, cond, omega=omega, w=w)
            win[n + i] = pred.float()[0]
        result[s + n : s + n + chunk] = win[n:]
    return result[:L]


# ---------------------------------------------------------------- DriftWorld's metrics
def to_u8(x):
    """(N, 3, H, W) float in [-1, 1] -> uint8 numpy, their convert_to_uint8_np (round)."""
    return ((x.clamp(-1, 1) + 1.0) / 2.0 * 255).round().to(torch.uint8).cpu().numpy()


def ssim_psnr(gen_u8, gt_u8):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    s, p = [], []
    for a, b in zip(gen_u8, gt_u8):
        s.append(structural_similarity(a.transpose(1, 2, 0), b.transpose(1, 2, 0), channel_axis=2, data_range=255))
        p.append(peak_signal_noise_ratio(b, a, data_range=255))
    return float(np.mean(s)), float(np.mean(p))


_lpips = None
_inception = None


@torch.no_grad()
def inception_feats(x, device):
    """pytorch-fid's pool3 features (2048-d) of (N, 3, H, W) images in [-1, 1]: its InceptionV3 with resize + normalize."""
    global _inception
    from pytorch_fid.inception import InceptionV3

    if _inception is None:
        _inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True).to(device).eval()
    out = []
    for i in range(0, x.shape[0], 64):
        f = _inception((x[i : i + 64].to(device).float() + 1.0) / 2.0)[0]
        out.append(f.squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out)


def lpips_mean(gen, gt, device):
    global _lpips
    import lpips

    if _lpips is None:
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    vals = []
    with torch.no_grad():
        for i in range(0, gen.shape[0], 32):
            vals.append(_lpips(gen[i : i + 32].to(device), gt[i : i + 32].to(device)).flatten())
    return float(torch.cat(vals).mean())


def write_frames(frames_u8, out_dir, stem, per_video=False):
    import imageio

    if per_video:
        d = out_dir / stem
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames_u8):
            imageio.imwrite(str(d / f"frame_{i:02d}.png"), f.transpose(1, 2, 0))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames_u8):
            imageio.imwrite(str(out_dir / f"{stem}_f{i:04d}.png"), f.transpose(1, 2, 0))


# ---------------------------------------------------------------- shard
def run_shard(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, step, which = load_checkpoint(a.ckpt, a.ema, device)
    codec = Codec(cfg, device)
    n = int(cfg.dataset.get("n_history", 4))
    ds = cfg.get("dataset", None) or {}
    if a.processed is None:
        a.processed = str(ds.processed_dir) if "processed_dir" in ds else PROCESSED
    if a.action_dim is None:
        a.action_dim = int(ds.action_dim) if "action_dim" in ds else 7
    clips = BridgeClips(a.processed, "val", n_frames=1, action_dim=a.action_dim, full_video=True)
    ids = list(range(len(clips)))
    if a.max_videos:
        ids = ids[: a.max_videos]
    my = ids[a.rank :: a.world_size]
    print(f"[eval] {a.ckpt} step {step} ({which}); codec {codec.kind}; n_history {n}; chunk {a.chunk}; omega {a.omega}; frames {os.environ.get('BRIDGE_FRAME_SIZE') or 'as stored'}; "
          f"shard {a.rank}/{a.world_size}: {len(my)} of {len(ids)} videos", flush=True)
    dirs = {k: out / k for k in ("pred_first16", "gt_first16")}
    records_path = out / "shards" / f"records_{a.rank:02d}_of_{a.world_size:02d}.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records = {}   # (a rerun recomputes the shard: the Inception features are kept in memory until the end)
    feats = {"pred": [], "gt": [], "raw": []}
    t0 = time.time()
    for k, idx in enumerate(my):
        stem = clips.stems[idx]
        if stem in records:
            continue
        item = clips[idx]
        frames, act = item["x"], item["act"]
        L = min(frames.shape[0], act.shape[0])
        if L <= n:
            continue
        frames, act = frames[:L], act[:L]
        obs = codec.encode(frames)                                   # (L, C, h, w)
        gen_lat = rollout(model, obs, act.to(device), n, a.chunk, a.omega, None, seed=a.seed + idx)
        gen = codec.decode(gen_lat[n:])                              # generated frames only
        gt = codec.decode(obs[n:])                                   # their reference: the reconstruction
        raw = frames[n:].to(device).float().div(127.5).sub(1.0)
        gen_u8, gt_u8, raw_u8 = to_u8(gen), to_u8(gt), to_u8(raw)
        ssim, psnr = ssim_psnr(gen_u8, gt_u8)
        ssim_r, psnr_r = ssim_psnr(gen_u8, raw_u8)
        rec = {"stem": stem, "n_gen": int(L - n), "ssim": ssim, "psnr": psnr, "lpips": lpips_mean(gen, gt, device),
               "ssim_raw": ssim_r, "psnr_raw": psnr_r, "lpips_raw": lpips_mean(gen, raw, device)}
        feats["pred"].append(inception_feats(gen, device)); feats["gt"].append(inception_feats(gt, device)); feats["raw"].append(inception_feats(raw, device))
        if L - n >= N_FVD_FRAMES and idx < a.fvd_videos:   # the first-16 clips stylegan-v's fvd2048_16f samples from
            write_frames(gen_u8[:N_FVD_FRAMES], dirs["pred_first16"], stem, per_video=True)
            write_frames(gt_u8[:N_FVD_FRAMES], dirs["gt_first16"], stem, per_video=True)
            rec["fvd_frames"] = N_FVD_FRAMES
        records[stem] = rec
        if (k + 1) % 20 == 0 or k + 1 == len(my):
            tmp = records_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(records))
            os.replace(tmp, records_path)
            m = {key: float(np.mean([r[key] for r in records.values()])) for key in ("ssim", "psnr", "lpips")}
            print(f"[eval] shard {a.rank}: {k + 1}/{len(my)} videos, {time.time() - t0:.0f}s; running SSIM {m['ssim']:.4f} "
                  f"PSNR {m['psnr']:.3f} LPIPS {m['lpips']:.4f}", flush=True)
    tmp = records_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records))
    os.replace(tmp, records_path)
    for k, v in feats.items():
        if v:
            np.save(out / "shards" / f"feats_{k}_{a.rank:02d}_of_{a.world_size:02d}.npy", torch.cat(v).numpy().astype(np.float32))
    meta = {"ckpt": a.ckpt, "step": step, "weights": which, "codec": codec.kind, "omega": a.omega, "w": None, "chunk": a.chunk,
            "frame_size": a.frame_size,
            "n_history": n, "world_size": a.world_size}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[eval] shard {a.rank} done: {len(records)} videos in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- aggregate (per-video means, FID from the shards' features)
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid's calculate_frechet_distance without the scipy `disp` keyword (removed in scipy >= 1.18)."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_feats(out, a_key, b_key):
    """FID between two accumulated Inception feature sets (pytorch-fid's Frechet distance on their mean / covariance)."""

    def load(key):
        parts = sorted((out / "shards").glob(f"feats_{key}_*.npy"))
        return np.concatenate([np.load(p) for p in parts]) if parts else None
    fa, fb = load(a_key), load(b_key)
    if fa is None or fb is None:
        return None
    return frechet_distance(fa.mean(0).astype(np.float64), np.cov(fa.astype(np.float64), rowvar=False),
                            fb.mean(0).astype(np.float64), np.cov(fb.astype(np.float64), rowvar=False))


def aggregate(a):
    out = Path(a.out)
    recs = {}
    for p in sorted((out / "shards").glob("records_*.json")):
        recs.update(json.loads(p.read_text()))
    keys = ("ssim", "psnr", "lpips", "ssim_raw", "psnr_raw", "lpips_raw")
    summary = {"n_videos": len(recs), **{k: float(np.mean([r[k] for r in recs.values()])) for k in keys},
               "n_gen_frames": int(sum(r["n_gen"] for r in recs.values())),
               "n_fvd_videos": int(sum(1 for r in recs.values() if r.get("fvd_frames")))}
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    summary.update({k: meta.get(k) for k in ("ckpt", "step", "weights", "codec", "omega", "chunk", "frame_size")})
    print(f"[aggregate] {summary['n_videos']} videos, {summary['n_gen_frames']} generated frames: SSIM {summary['ssim']:.4f} "
          f"PSNR {summary['psnr']:.3f} LPIPS {summary['lpips']:.4f} (vs raw: {summary['ssim_raw']:.4f} / {summary['psnr_raw']:.3f} / "
          f"{summary['lpips_raw']:.4f})", flush=True)
    if not a.skip_fid:
        summary["fid"] = fid_from_feats(out, "gt", "pred")
        summary["fid_raw"] = fid_from_feats(out, "raw", "pred")
        summary["fid_rec_vs_raw"] = fid_from_feats(out, "raw", "gt")   # the codec's own reconstruction FID on these frames
        if summary["fid"] is None:   # a shard stopped before it saved its features (its records survive)
            print("[aggregate] FID: no feature files (shard incomplete) - per-video metrics only", flush=True)
        else:
            print(f"[aggregate] FID {summary['fid']:.3f} (vs raw {summary['fid_raw']:.3f}; reconstruction vs raw {summary['fid_rec_vs_raw']:.3f})", flush=True)
    fvd_file = out / "fvd.json"
    if fvd_file.exists():   # written by the stylegan-v step of scripts/rae/wm/eval.sh
        summary.update(json.loads(fvd_file.read_text()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="an export (sbm_rae.infer) or a training checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--processed", default=None, help="processed clips (default: the checkpoint's dataset.processed_dir, else Bridge-V2 in the store)")
    ap.add_argument("--action-dim", type=int, default=None, help="7 = Bridge-V2, 10 = RT-1 (default: the checkpoint's dataset.action_dim)")
    ap.add_argument("--frame-size", type=int, default=None, help="resize frames to this square size (default: the export's meta.read; RT-1: 256)")
    ap.add_argument("--ema", type=float, default=None, help="EMA decay to read from a training checkpoint (default: its active EMA)")
    ap.add_argument("--omega", type=float, default=None, help="the read's anchor time (default: the export's meta.read, else 0.8)")
    ap.add_argument("--chunk", type=int, default=None, help="frames generated per chunk (default: the export's meta.read, else 8)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=0, help="only the first N validation episodes (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fvd-videos", type=int, default=2048, help="the first-16 clips are written for the first N videos (stylegan-v samples 2048)")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--skip-fid", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a)
        return
    if not a.ckpt:
        ap.error("--ckpt is required for a shard")
    read = {**READ_DEFAULTS, **{k: v for k, v in read_settings(a.ckpt).items() if v is not None}}
    if a.omega is None:
        a.omega = float(read["omega"])
    if a.chunk is None:
        a.chunk = int(read["chunk"])
    if a.frame_size is None and read.get("frame_size"):
        a.frame_size = int(read["frame_size"])
    if a.frame_size:
        os.environ["BRIDGE_FRAME_SIZE"] = str(a.frame_size)
    run_shard(a)


if __name__ == "__main__":
    main()
