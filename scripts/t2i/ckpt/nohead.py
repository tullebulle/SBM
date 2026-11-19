"""The no-head form of a checkpoint average: the average's trunk and embedders, with the DDT head taken from one
checkpoint (averaging the head over older members hurts wide averages). The head is the same tensor set as the head-only
stage's trainable set: head_blocks, head_x_embedder, s_projector, final_layer. Each EMA copy takes the head from the
same copy of the head checkpoint (state_dict <- its state_dict, ema 0_9999 <- its 0_9999). The output has soup.py's layout.
    python scripts/t2i/ckpt/nohead.py <out.ckpt> <average.ckpt> <head.ckpt>
"""
import os
import re
import sys
from pathlib import Path

import torch

HEAD = re.compile(r"^model\.(head_blocks\.|head_x_embedder\.|s_projector\.|final_layer\.)")


def main(out, avg_path, head_path):
    avg = torch.load(avg_path, map_location="cpu", mmap=True, weights_only=False)
    src = torch.load(head_path, map_location="cpu", mmap=True, weights_only=False)
    pairs = {"state_dict": (avg["state_dict"], src["state_dict"])}
    pairs.update({f"ema_{k}": (sd, src["ema_state_dicts"][k]) for k, sd in avg["ema_state_dicts"].items()})
    new, n_head = {}, None
    for c, (a, h) in pairs.items():
        keys = [k for k in a if HEAD.match(k)]
        if n_head is None:
            n_head = len(keys)
        if not keys or len(keys) != n_head or any(k not in h or h[k].shape != a[k].shape for k in keys):
            raise KeyError(f"{c}: head keys missing or mismatched ({len(keys)})")
        new[c] = {k: (h[k] if HEAD.match(k) else v) for k, v in a.items()}
        print(f"{c}: {len(keys)} head tensors ({sum(a[k].numel() for k in keys) / 1e6:.2f}M) from {head_path}, "
              f"{len(a) - len(keys)} from the average", flush=True)
    soup = dict(avg.get("soup", {}))
    soup.update({"trunk_from": str(Path(avg_path).resolve()), "head_from": str(Path(head_path).resolve()),
                 "head_step": int(src["global_step"]), "head_regex": HEAD.pattern})
    out_ck = {"global_step": avg["global_step"], "ema_active_decay": avg.get("ema_active_decay"),
              "state_dict": new.pop("state_dict"),
              "ema_state_dicts": {c[len("ema_"):]: sd for c, sd in new.items()}, "soup": soup}
    tmp = f"{out}.tmp{os.getpid()}"
    torch.save(out_ck, tmp)
    os.replace(tmp, out)
    # read back: every head tensor equals the head checkpoint's, every other tensor the average's
    chk = torch.load(out, map_location="cpu", mmap=True, weights_only=False)
    got = {"state_dict": chk["state_dict"], **{f"ema_{k}": v for k, v in chk["ema_state_dicts"].items()}}
    for c, (a, h) in pairs.items():
        bad = [k for k, v in got[c].items() if not torch.equal(v, h[k] if HEAD.match(k) else a[k])]
        if bad or set(got[c]) != set(a):
            raise RuntimeError(f"{out} {c}: read-back mismatch {bad[:3]}")
    print(f"-> {out}: read back ok (head from step {src['global_step']}, trunk from {Path(avg_path).name})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
