"""GenEval samples of a text-to-image checkpoint or export (sbm_t2i.infer), in GenEval's folder layout.

All 553 prompts of GenEval's evaluation_metadata.jsonl, 4 images each, from one noise stream seeded `seed`, in prompt
order (sbm_t2i.infer.generate, 16 prompts x 4 images per batch). One step: x = decode(T(z; prompt, omega)) with
z ~ N(0, I) in the codec's latent, the EMA copy `which` of the checkpoint, the prompt's FLAN-T5 states moved away from
the mask token by `label_scale` (1 = off); omega and label_scale default to the checkpoint's read settings (SBM-XL:
0.35 / 1.125, SBM-B: 0.3 / 1.25). Images are saved at the decoder's side (256 px for the RAE ViT-XL decoder).

    python scripts/t2i/geneval_sample.py ckpt=<export.pt | <run>/ckpt/step*.ckpt> out=<dir> [which=0.9999] [omega=...]
        [label_scale=...] [seed=0] [prompts_per_batch=16] [metadata=<evaluation_metadata.jsonl>]

Layout: <out>/<index:05d>/metadata.jsonl (the prompt's metadata line), <out>/<index:05d>/samples/<k:04d>.png, k = 0..3,
and <out>/protocol.json. A finished folder (protocol.json present) is left as it is. Score: scripts/t2i/geneval/score.sh.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sbm_t2i.infer import generate, load_model
from sbm_t2i.run import store

METADATA = str(store() / "t2i" / "geneval" / "evaluation_metadata.jsonl")   # GenEval's prompts/evaluation_metadata.jsonl
PER_PROMPT = 4


@torch.no_grad()
def main(argv):
    a = dict(x.split("=", 1) for x in argv)
    ckpt, out = Path(a["ckpt"]), Path(a["out"])
    which, seed, ppb = a.get("which", "0.9999"), int(a.get("seed", 0)), int(a.get("prompts_per_batch", 16))
    if (out / "protocol.json").exists():
        print(f"{out}: already sampled")
        return
    device = torch.device("cuda")
    model, cfg, step = load_model(ckpt, which)   # an export, or <run>/ckpt/<step>.ckpt + <run>/run.yaml
    omega = float(a.get("omega", cfg.read["omega"]))
    ls = float(a.get("label_scale", cfg.read["label_scale"]))
    lines = [ln for ln in Path(a.get("metadata", METADATA)).read_text().splitlines() if ln.strip()]
    g = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for i0 in range(0, len(lines), ppb):
        chunk = lines[i0:i0 + ppb]
        prompts = [json.loads(ln)["prompt"] for ln in chunk for _ in range(PER_PROMPT)]
        images = generate(model, cfg, prompts, omega=omega, label_scale=ls, device=device, batch_size=len(prompts),
                          generator=g).permute(0, 2, 3, 1).numpy()
        for j, line in enumerate(chunk):
            d = out / f"{i0 + j:05d}"
            (d / "samples").mkdir(parents=True, exist_ok=True)
            (d / "metadata.jsonl").write_text(line.strip() + "\n")
            for k in range(PER_PROMPT):
                Image.fromarray(np.ascontiguousarray(images[PER_PROMPT * j + k])).save(d / "samples" / f"{k:04d}.png")
    protocol = dict(ckpt=str(ckpt), step=step, which=which, omega=omega, label_scale=ls, seed=seed, n_prompts=len(lines),
                    per_prompt=PER_PROMPT, side=int(images.shape[1]), seconds=round(time.time() - t0, 1))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1))
    print(json.dumps(protocol))


if __name__ == "__main__":
    main(sys.argv[1:])
