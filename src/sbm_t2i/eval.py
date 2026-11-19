"""Class-conditional FID (ImageNet, SD-VAE latent maps of this trainer) with a named protocol; every result carries its
full protocol and is written to <run>/evals/<name>.json. The text-to-image reads are sbm_t2i.text_eval and
scripts/t2i/geneval_sample.py.

    python -m sbm_t2i.cli eval <run_dir|checkpoint> [protocol=b2_adm50k] [ckpt=last|<file name>] [dest=monitor] [key=value ...]

dest=monitor writes to <run>/monitor/ instead of evals/ (quick reads at small n), named <protocol>_<ckpt>_<weights>_n<n>.

Protocols (any field can be overridden, and the override is recorded):

b2_adm50k   DiT-B/2. The active EMA-0.999 copy
            (`state_dict`), omega 0.8, 50k samples with labels arange(N) % 1000, noise from a CUDA generator seeded
            42 + 2024 + rank, batch 250, bf16 autocast, SD-VAE-ft-mse decode in fp32, torchmetrics Inception-2048
            (normalize=True). FID against ImageNet-val (refs/fid/imagenet_val_inception_stats.pt) and against the ADM/JiT
            train statistics (refs/fid/adm_jit_in256_stats.npz). Multistep: `steps` (a list),
            `renoise_t` (re-noise to the interpolant at that time between calls), `guide_scale` (rolled-class guidance,
            a guided call costs 2 NFE). A held-out noise stream: seed_offset 7777. `gw` (a map with a guidance input,
            model.gw_embedder): the guidance strength to sample at, one step (e.g. gw=1.5; default the model's
            gw_default, recorded in the settings; an error on a map without the input). `gw_lo`, `gw_hi` (a map with
            model.gw_interval): the guidance interval to read at (both or neither; default the map's [0, 1]).
xl_val50k   DiT-XL/2. Raw weights cast to bf16, omega 0.9,
            10 shards x 5000 images with labels arange(1000) repeated 5, bf16 noise from a CUDA generator seeded
            20261112 + shard, batch 8, decode in chunks of 8, Inception-2048 on uint8 images; FID against the moments
            of 50 ImageNet-val images per class (refs/fid/imagenet_val_real50k.pt).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from sbm_t2i import run as R

PROTOCOLS = {
    "b2_adm50k": dict(weights="ema", omega=0.8, n=50000, steps=[1], seed=42, seed_offset=2024, batch=250, vae_batch=64,
                      renoise_t=None, guide_scale=0.0, refs=["val", "adm"]),
    "xl_val50k": dict(weights="raw", omega=0.9, n=50000, shards=10, seed=20261112, batch=8, vae_batch=8, bf16_weights=True,
                      refs=["val50"]),
}
REF_FILES = {"val": "fid/imagenet_val_inception_stats.pt", "adm": "fid/adm_jit_in256_stats.npz",
             "val50": "fid/imagenet_val_real50k.pt"}
VAE = "stabilityai/sd-vae-ft-mse"
INCEPTION = "inception-2015-12-05-6726825d.pth"


def _ensure_inception_weights():
    """torchmetrics looks for the FID Inception weights in the torch hub cache; seed it from refs/ (no download)."""
    dst = Path(torch.hub.get_dir()) / "checkpoints" / f"weights-{INCEPTION}"
    src = R.store() / "refs" / "weights" / INCEPTION
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _vae(device):
    from diffusers import AutoencoderKL
    try:
        vae = AutoencoderKL.from_pretrained(VAE, local_files_only=True)
    except OSError:
        vae = AutoencoderKL.from_pretrained(VAE)
    return vae.to(device).eval().requires_grad_(False)


def _mean_cov(sum_, outer, n):
    mu = sum_.double() / n
    return mu, (outer.double() - n * torch.outer(mu, mu)) / (n - 1)


def _fid(mu1, s1, mu2, s2):
    from torchmetrics.image.fid import _compute_fid
    return float(_compute_fid(mu1.double(), s1.double(), mu2.double(), s2.double()).cpu())


def _ref(name, device):
    path = R.store() / "refs" / REF_FILES[name]
    if name == "adm":
        z = np.load(path)
        return torch.from_numpy(np.asarray(z["mu"], np.float64)).to(device), torch.from_numpy(np.asarray(z["sigma"], np.float64)).to(device)
    s = torch.load(path, map_location=device)
    if name == "val50":
        return _mean_cov(s["sum"], s["outer_sum"], int(s["count"]))
    return _mean_cov(s["sum"], s["cov_sum"], int(s["n"]))


@torch.no_grad()
def _sample_b2(model, labels, p, device, generator, omega):
    """One batch at every requested step count: {steps: latents}."""
    x = torch.randn((len(labels), model.in_channels, *model.x_embedder.img_size), generator=generator, device=device)
    out, cur = {}, x
    w = None if p.get("gw") is None else float(p["gw"])
    ikw = {} if p.get("gw_lo") is None else {"s_lo": float(p["gw_lo"]), "s_hi": float(p["gw_hi"])}
    for k in range(1, max(p["steps"]) + 1):
        if k > 1 and p.get("renoise_t") is not None:
            fresh = torch.randn(cur.shape, generator=generator, device=device, dtype=x.dtype)
            cur = (1.0 - float(p["renoise_t"])) * fresh + float(p["renoise_t"]) * cur.to(x.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            t = model(cur, labels, omega=omega, w=w, **ikw)
            if float(p.get("guide_scale") or 0.0):
                shift = 1 + int(torch.randint(int(labels.numel()) - 1, (1,)))
                t = t + float(p["guide_scale"]) * (t - model(cur, torch.roll(labels, shift), omega=omega, w=w, **ikw))
        cur = t
        if k in p["steps"]:
            out[k] = cur
    return out


@torch.no_grad()
def decoded_b2(model, p, device, scale, vae, labels, generator):
    """The b2 protocol's samples batch by batch: yields (samples so far, {steps: images in [0, 1]}). Shared by the
    evaluator and the in-training monitor (sbm_t2i.monitor)."""
    omega = torch.tensor(float(p["omega"]), device=device)
    done = 0
    for start in range(0, len(labels), int(p["batch"])):
        y = labels[start:start + int(p["batch"])].to(device)
        out = {}
        for k, lat in _sample_b2(model, y, p, device, generator, omega).items():
            # divide by a fp32 tensor (a Python scalar divides by its reciprocal, which is not bit-identical and moves
            # the decoded images by up to ~2/255)
            lat = lat / torch.full((1, lat.shape[1], 1, 1), scale, dtype=torch.float32, device=lat.device)
            imgs = []
            for s in range(0, len(lat), int(p["vae_batch"])):
                with torch.autocast(device.type, enabled=False):
                    imgs.append(vae.decode(lat[s:s + int(p["vae_batch"])]).sample.float())
            out[k] = ((torch.cat(imgs) + 1.0) / 2.0).clamp(0.0, 1.0)
        done += len(y)
        yield done, out


def _metric_fid(m, ref, device):
    n = m.fake_features_num_samples.double()
    mu = m.fake_features_sum.double() / n
    cov = (m.fake_features_cov_sum.double() - n * torch.outer(mu, mu)) / (n - 1)
    return _fid(*_ref(ref, device), mu, cov)


@torch.no_grad()
def _run_b2(model, p, device, scale):
    from torchmetrics.image.fid import FrechetInceptionDistance
    _ensure_inception_weights()
    vae = _vae(device)
    metrics = {k: FrechetInceptionDistance(feature=2048, normalize=True).to(device) for k in p["steps"]}
    if float(p.get("label_scale", 1.0)) != 1.0:   # label scale (default off): class rows extrapolated from the null row
        import copy
        model = copy.deepcopy(model)
        w = model.y_embedder.embedding_table.weight.data
        w[:-1] = w[-1:] + float(p["label_scale"]) * (w[:-1] - w[-1:])
    labels = torch.arange(int(p["n"])) % 1000
    g = torch.Generator(device=device.type).manual_seed(int(p["seed"]) + int(p["seed_offset"]))
    prefixes = sorted(int(k) for k in (p.get("prefixes") or []))
    if any(k % int(p["batch"]) or not 0 < k < int(p["n"]) for k in prefixes):
        raise ValueError(f"prefixes {prefixes} must be multiples of batch {p['batch']} below n {p['n']}")
    at_prefix = {}
    for done, images in decoded_b2(model, p, device, scale, vae, labels, g):
        for k, imgs in images.items():
            metrics[k].update(imgs, real=False)
        if done in prefixes:
            for k, m in metrics.items():
                for ref in p["refs"]:
                    at_prefix.setdefault(f"prefix_fid_{ref}", {}).setdefault(str(k), {})[str(done)] = _metric_fid(m, ref, device)
    out, moments = {}, {}
    for k, m in metrics.items():
        moments[k] = {"sum": m.fake_features_sum.cpu(), "cov_sum": m.fake_features_cov_sum.cpu(), "n": m.fake_features_num_samples.cpu()}
        for ref in p["refs"]:
            out.setdefault(f"fid_{ref}", {})[str(k)] = _metric_fid(m, ref, device)
    return {**out, **at_prefix}, moments


@torch.no_grad()
def _run_xl(model, p, device, scale):
    from torchmetrics.image.fid import FrechetInceptionDistance
    _ensure_inception_weights()
    vae = _vae(device).float()
    inception = FrechetInceptionDistance(feature=2048, normalize=True).to(device).inception.eval()
    if p.get("bf16_weights") or float(p.get("label_scale", 1.0)) != 1.0:
        import copy
        model = copy.deepcopy(model)   # a copy: the caller may be training these weights
        if p.get("bf16_weights"):
            model = model.to(torch.bfloat16)
        if float(p.get("label_scale", 1.0)) != 1.0:   # label scale (default off), as in _run_b2: class rows extrapolated from the null row
            w_tab = model.y_embedder.embedding_table.weight.data
            w_tab[:-1] = w_tab[-1:] + float(p["label_scale"]) * (w_tab[:-1] - w_tab[-1:])
    # a map with a guidance input (model.gw_embedder) is read at p['gw'] on the interval [gw_lo, gw_hi], as in _sample_b2
    gw = None if p.get("gw") is None else float(p["gw"])
    ikw = {} if p.get("gw_lo") is None else {"s_lo": float(p["gw_lo"]), "s_hi": float(p["gw_hi"])}
    count, s, o = 0, torch.zeros(2048, dtype=torch.float64, device=device), torch.zeros(2048, 2048, dtype=torch.float64, device=device)
    per_shard = int(p["n"]) // int(p["shards"])
    labels = torch.arange(1000).repeat(per_shard // 1000)
    with torch.inference_mode():
        for shard in range(int(p["shards"])):
            g = torch.Generator(device=device.type).manual_seed(int(p["seed"]) + shard)
            for start in range(0, per_shard, int(p["batch"])):
                c = labels[start:start + int(p["batch"])].to(device)
                z = torch.randn(len(c), 4, 32, 32, generator=g, device=device, dtype=torch.bfloat16)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    lat = model(z, c, omega=float(p["omega"]), w=gw, **ikw)
                imgs = torch.cat([vae.decode(lat[i:i + int(p["vae_batch"])].float() / scale).sample
                                  for i in range(0, len(lat), int(p["vae_batch"]))])
                f = inception(((imgs + 1.0) / 2.0).clamp(0.0, 1.0).float().mul(255).byte()).double()
                count += f.shape[0]; s += f.sum(0); o += f.T @ f
    mu, cov = _mean_cov(s, o, count)
    return {f"fid_{ref}": {"1": _fid(*_ref(ref, device), mu, cov)} for ref in p["refs"]}, {1: {"count": count, "sum": s.cpu(), "outer_sum": o.cpu()}}


def _resolve_gw(model, p):
    """A map with a guidance input is read at p['gw'], or at its default, which the record then carries."""
    if getattr(model, "gw_embedder", None) is not None and p.get("gw") is None:
        p["gw"] = model.gw_default
    elif getattr(model, "gw_embedder", None) is None and p.get("gw") is not None:
        raise ValueError(f"gw={p['gw']}: this map has no guidance-strength input (model.gw_embedder is off)")
    return p


def evaluate_module(module, protocol, *, run_dir, label, **overrides):
    """Score an in-memory Lightning module (its current weights) and record the result."""
    p = {**PROTOCOLS[protocol], **overrides}
    model, device = module.model.eval(), module.device
    p = _resolve_gw(model, p)
    return _record(run_dir, protocol, label, p, "in-memory", *_dispatch(protocol, model, p, device, float(module.latents_scale.flatten()[0])))


def _dispatch(protocol, model, p, device, scale):
    t0 = time.time()
    out, moments = (_run_xl if protocol.startswith("xl") else _run_b2)(model, p, device, scale)
    return out, moments, time.time() - t0


def _record(run_dir, protocol, label, p, checkpoint, out, moments, seconds, folder="evals"):
    name = f"{protocol}_{label}"
    result = {"protocol": protocol, "settings": p, "checkpoint": checkpoint, "seconds": seconds, **out,
              "fid": next(iter(out.values()))[str(max(int(k) for k in next(iter(out.values()))))]}
    if run_dir is not None and folder != "evals":
        d = Path(run_dir) / folder if not Path(folder).is_absolute() else Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(moments, d / f"{name}_moments.pt")
        (d / f"{name}.json").write_text(json.dumps(result, indent=1))
        print(d / f"{name}.json")
    elif run_dir is not None:
        torch.save(moments, Path(run_dir) / "evals" / f"{name}_moments.pt")
        print(R.record_eval(run_dir, name, result))
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}, indent=1))
    return result


def _parse(overrides):
    import yaml
    kv = {}
    for o in overrides:
        k, _, v = o.partition("=")
        kv[k] = yaml.safe_load(v)
    return kv


def evaluate(target, overrides=()):
    """python -m sbm_t2i.cli eval <run_dir|checkpoint> [protocol=...] [ckpt=...] [field=value ...]"""
    from sbm_t2i.models import build_model, load_weights
    from sbm_t2i.train import latest_checkpoint
    kv = _parse(overrides)
    protocol = kv.pop("protocol", "b2_adm50k")
    dest = kv.pop("dest", "evals")
    target = Path(target)
    if target.is_dir():
        run_dir = target
        name = kv.pop("ckpt", "last")
        ckpt = latest_checkpoint(run_dir / "ckpt") if name == "last" else run_dir / "ckpt" / name
        cfg = R.load_run(run_dir)
        model_cfg = dict(cfg.model)
    else:
        ckpt, run_dir = target, (target.parent.parent if target.parent.name == "ckpt" else None)
        head = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        hp = (head.get("hyper_parameters") or {}).get("cfg") or {}
        model_cfg = dict(hp.get("model") or {"name": head.get("model_name", "DiT-B/2")})
        model_cfg["name"] = model_cfg.get("name", head.get("model_name", "DiT-B/2"))
    p = {**PROTOCOLS[protocol], **kv}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(build_model(**model_cfg), ckpt, p["weights"]).to(device).eval()
    p = _resolve_gw(model, p)
    out, moments, secs = _dispatch(protocol, model, p, device, 0.18215)
    if run_dir is None:
        run_dir = R.store() / "evals_adhoc" / Path(ckpt).stem
        (run_dir / "evals").mkdir(parents=True, exist_ok=True)
    if dest != "evals":
        label = f"{Path(ckpt).stem}_{p['weights']}_n{p.get('n')}"
        return _record(run_dir, protocol, label, p, str(ckpt), out, moments, secs, folder=str(dest))
    label = Path(ckpt).stem + "".join(f"_{k}{v}" for k, v in sorted(kv.items()) if k not in ("weights",))
    return _record(run_dir, protocol, label, p, str(ckpt), out, moments, secs)
