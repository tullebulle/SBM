"""Weighted average ("soup") of checkpoints of a text-to-image run. Each EMA copy is averaged across the sources on its
own: the active EMA (`state_dict`) and every `ema_state_dicts` entry. The output keeps only those (no raw weights,
optimizer or teacher): scripts/t2i/geneval_sample.py reads it with which=<decay> when it sits in <dir>/ckpt/ next to a
copy of the run's run.yaml (<dir>/run.yaml), nohead.py takes it as the trunk, soup_to_resume.py makes it resumable.
    python scripts/t2i/ckpt/soup.py <out.ckpt> [--weights=w1,w2,...] <src.ckpt> <src.ckpt> [...]
--weights: one weight per source, normalised to sum 1; without it, uniform.
"""
import os
import sys
from pathlib import Path

import torch


def main(out, sources, weights=None):
    if len(sources) < 2:
        raise ValueError("need at least two sources")
    if weights is not None:
        if len(weights) != len(sources) or min(weights) < 0:
            raise ValueError(f"{len(weights)} weights for {len(sources)} sources")
        weights = [w / sum(weights) for w in weights]
    sums, meta = None, []
    for i, src in enumerate(sources):
        ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
        copies = {"state_dict": ck["state_dict"], **{f"ema_{k}": v for k, v in ck["ema_state_dicts"].items()}}
        w = 1.0 if weights is None else weights[i]
        if sums is None:
            sums = {c: {k: (v.double() * w if torch.is_floating_point(v) else v.clone()) for k, v in sd.items()}
                    for c, sd in copies.items()}
            active = ck.get("ema_active_decay")
        else:
            if set(copies) != set(sums) or any(set(copies[c]) != set(sums[c]) for c in sums):
                raise KeyError(f"{src}: EMA copies or keys differ from {sources[0]}")
            for c, sd in copies.items():
                for k, v in sd.items():
                    if torch.is_floating_point(v):
                        sums[c][k] += v.double() * w if weights is not None else v.double()
        meta.append((str(Path(src).resolve()), int(ck["global_step"])))
        print(f"[{i + 1}/{len(sources)}] {src}: step {ck['global_step']}", flush=True)
        del ck, copies
    n = len(sources) if weights is None else 1
    avg = {c: {k: (v / n).float() if v.dtype == torch.float64 else v for k, v in sd.items()} for c, sd in sums.items()}
    out_ck = {"global_step": max(s for _, s in meta), "ema_active_decay": active, "state_dict": avg.pop("state_dict"),
              "ema_state_dicts": {c[len("ema_"):]: sd for c, sd in avg.items()},
              "soup": {"sources": [p for p, _ in meta], "steps": [s for _, s in meta],
                       "weights": [1 / len(meta)] * len(meta) if weights is None else weights}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{out}.tmp{os.getpid()}"
    torch.save(out_ck, tmp)
    os.replace(tmp, out)
    print(f"-> {out} (steps {[s for _, s in meta]})", flush=True)


if __name__ == "__main__":
    args = sys.argv[2:]
    w = [float(x) for x in args.pop(0)[len("--weights="):].split(",")] if args[0].startswith("--weights=") else None
    main(sys.argv[1], args, w)
