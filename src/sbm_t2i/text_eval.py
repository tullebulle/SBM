"""Reads of the text-to-image maps: samplers and the scores that do not need a benchmark stack.

Samplers, both from noise 2 N(0, I):
  one_step     x = T(z; prompt, omega) (the map's read; `label_scale` moves the text states away from the mask token)
  euler_cfg    the base's sampler: Euler from t = 0 to 1 on v = (D - x) / max(1 - t, a_min), D the CFG mix of the
               prompt and null-prompt denoiser reads (MiniT2I: 100 steps, scale 6, a_min 0.05)
Scorer: Inception-2048 moments against MiniT2I's COCO statistics (mscoco_fid_stats_512.npz; FID on their 30K caption
list) and the CLIP score (100 x cosine of CLIP ViT-B/32 image and prompt embeddings). The Inception network is
torchmetrics' port, the statistics were made with pytorch-fid's: a consistent trend, not MiniT2I's protocol number.

    python -m sbm_t2i.text_eval ckpt=<checkpoint|run folder|release .safetensors> [which=ema] [mode=one_step|euler] [n=10000]
        [omega=0.95] [label_scale=1.0] [steps=100] [cfg=6.0] [batch=50] [seed=2066] [out=<json>] [grid=<jpg>]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sbm_t2i.run import store
from sbm_t2i.text import TextBound, tokenize, tokenizer

ASSETS = store() / "t2i"   # COCO captions + FID statistics (assets/coco), FLAN-T5 and CLIP (weights/)
DEFAULTS = dict(captions=str(ASSETS / "assets/coco/coco-30-val-2014-captions.json"),
                ref_stats=str(ASSETS / "assets/coco/mscoco_fid_stats_512.npz"),
                clip=str(ASSETS / "weights/clip-vit-base-patch32"), text_encoder=str(ASSETS / "weights/flan-t5-large"))
NOISE_SCALE = 2.0


def load_captions(path):
    data = json.loads(Path(path).read_text())
    return [data[k] for k in sorted(data, key=int)] if isinstance(data, dict) else list(data)


@torch.no_grad()
def one_step(model, context, mask, z, omega, null, label_scale=1.0):
    bound = TextBound(model, context, mask, null, label_scale)
    with torch.autocast(z.device.type, dtype=torch.bfloat16):
        return bound(z, torch.arange(z.shape[0], device=z.device), omega=omega).float()


@torch.no_grad()
def euler_cfg(model, context, mask, z, steps=100, cfg=6.0, a_min=0.05, autocast=True):
    x, ts = z, torch.linspace(0.0, 1.0, steps + 1, device=z.device)
    for a, b in zip(ts[:-1], ts[1:]):
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=autocast):
            d = model(torch.cat([x, x]), (torch.cat([context, context]), torch.cat([mask, torch.zeros_like(mask)]))).float()
        d_c, d_u = d.chunk(2)
        x = x + (b - a) * ((d_u + cfg * (d_c - d_u)) - x) / max(1.0 - float(a), a_min)
    return x


def to_unit(x):
    """[-1, 1] pixels -> [0, 1], as the 8-bit image the sample is saved as."""
    return (x.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).floor() / 255.0


def save_grid(images, path, columns=4):
    """images [N, 3, H, W] in [0, 1] -> a JPEG grid at half resolution."""
    from PIL import Image
    n, _, h, w = images.shape
    rows = n // columns
    g = (images[:rows * columns] * 255).byte().reshape(rows, columns, 3, h, w).permute(0, 3, 1, 4, 2).reshape(rows * h, columns * w, 3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(g.cpu().numpy()).resize((columns * w // 2, rows * h // 2), Image.LANCZOS).save(path, quality=92)


class Scorer:
    """Running sums over samples: Inception moments (s, o), the CLIP cosine sum, the count; all float64 tensors, so a
    multi-rank caller can all-reduce `sums` before `result`."""

    def __init__(self, device, ref_stats=DEFAULTS["ref_stats"], clip=DEFAULTS["clip"]):
        import numpy as np
        from torchmetrics.image.fid import FrechetInceptionDistance
        from transformers import CLIPModel, CLIPTokenizerFast

        from sbm_t2i.eval import _ensure_inception_weights
        _ensure_inception_weights()
        self.inception = FrechetInceptionDistance(feature=2048, normalize=True).to(device).inception.eval()
        z = np.load(ref_stats)
        self.ref = (torch.from_numpy(np.asarray(z["mu"], np.float64)).to(device), torch.from_numpy(np.asarray(z["sigma"], np.float64)).to(device))
        self.clip = CLIPModel.from_pretrained(clip, torch_dtype=torch.float32).to(device).eval()
        self.clip_tok = CLIPTokenizerFast.from_pretrained(clip)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.device = device
        self.reset()

    def reset(self):
        d = self.device
        self.sums = [torch.zeros(2048, dtype=torch.float64, device=d), torch.zeros(2048, 2048, dtype=torch.float64, device=d),
                     torch.zeros((), dtype=torch.float64, device=d), torch.zeros((), dtype=torch.float64, device=d)]

    @torch.no_grad()
    def add(self, images, prompts):
        """images [B, 3, H, W] in [0, 1] (square), their prompts."""
        f = self.inception((images * 255).byte()).double()
        px = (F.interpolate(images, size=224, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1) - self.mean) / self.std
        tok = self.clip_tok(list(prompts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        cos = F.cosine_similarity(self.clip.get_image_features(pixel_values=px), self.clip.get_text_features(**tok))
        self.sums[0] += f.sum(0)
        self.sums[1] += f.t().mm(f)
        self.sums[2] += cos.double().sum()
        self.sums[3] += images.shape[0]

    def result(self, sums=None):
        from sbm_t2i.eval import _fid, _mean_cov
        s, o, c, n = self.sums if sums is None else sums
        n = float(n)
        mu, cov = _mean_cov(s, o, n)
        return {"n": int(n), "fid_coco": _fid(*self.ref, mu, cov), "clip": 100.0 * float(c) / n,
                "mean_term": float((mu - self.ref[0]).pow(2).sum()), "trace_ratio": float(cov.diagonal().sum() / self.ref[1].diagonal().sum())}


def main(argv=None):
    from sbm_t2i.models import build_model, load_weights
    from sbm_t2i.text import TextEncoder
    from sbm_t2i.train import resolve_source
    a = dict(mode="one_step", which="ema", n=10000, omega=0.95, label_scale=1.0, steps=100, cfg=6.0, a_min=0.05, batch=50,
             seed=2066, model="MMJiT-B/16", out=None, grid=None, **DEFAULTS)
    a.update(kv.split("=", 1) for kv in (sys.argv[1:] if argv is None else argv))
    n, batch, dev = int(a["n"]), int(a["batch"]), torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    src = resolve_source(a["ckpt"])
    euler = a["mode"] == "euler"
    model = load_weights(build_model(a["model"], input_size=512, in_channels=3, omega_tokens=0 if euler else 4), src, a["which"])
    model = model.to(dev).eval()
    encoder, tok, scorer = TextEncoder(a["text_encoder"]), tokenizer(a["text_encoder"]), Scorer(dev, a["ref_stats"], a["clip"])
    captions = load_captions(a["captions"])[:n]
    g = torch.Generator(device="cuda").manual_seed(int(a["seed"]))
    t0, first = time.time(), None
    for i in range(0, n, batch):
        prompts = captions[i:i + batch]
        ids, mask = tokenize(tok, prompts)
        context, mask = encoder(ids.to(dev), mask.to(dev))
        z = NOISE_SCALE * torch.randn(len(prompts), 3, 512, 512, generator=g, device=dev)
        if euler:
            x = euler_cfg(model, context, mask, z, int(a["steps"]), float(a["cfg"]), float(a["a_min"]))
        else:
            x = one_step(model, context, mask, z, float(a["omega"]), 1000, float(a["label_scale"]))
        images = to_unit(x)
        scorer.add(images, prompts)
        first = images[:16] if first is None else first
        if (i // batch) % 20 == 0:
            print(f"{i + len(prompts)} / {n}  {time.time() - t0:.0f}s", flush=True)
    keys = ("mode", "which", "omega", "label_scale", "seed") if not euler else ("mode", "which", "steps", "cfg", "a_min", "seed")
    res = {"ckpt": str(src), **{k: a[k] for k in keys}, **scorer.result(), "seconds": time.time() - t0}
    print(json.dumps(res, indent=1))
    if a["out"]:
        Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(a["out"]).write_text(json.dumps(res, indent=1))
    if a["grid"]:
        save_grid(first, a["grid"])


if __name__ == "__main__":
    main()
