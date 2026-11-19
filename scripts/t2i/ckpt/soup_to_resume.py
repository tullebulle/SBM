"""Turn a checkpoint average (soup.py / nohead.py; reading only) into a checkpoint a run can resume from
(init.from=<out> init.resume_state=true). Every weight slot becomes one EMA copy of the average: the raw weights
`current_model_state`, the active EMA `state_dict`, every `ema_state_dicts` entry, and the guided-CBM teacher
`guided_cbm_teacher["state"]` (the model's keys without the "model." prefix). Everything else (optimizer, the guided-CBM
loss's frozen balance weight, EMA counters, step, loops) comes from a full checkpoint of the run. A head-only stage
trains its head against the trunk it is read with, so the raw weights and the teacher have to be the average too.

    python scripts/t2i/ckpt/soup_to_resume.py <out.ckpt> --soup <average.ckpt> --base <full checkpoint.ckpt> [--ema 0_9999]
(CPU, ~40 GB RAM)
"""
import argparse
import os
from pathlib import Path

import torch


def check(label, sd, avg):
    if set(sd) != set(avg):
        raise KeyError(f"{label}: {len(set(sd) ^ set(avg))} keys differ from the average's")
    bad = [k for k in sd if sd[k].shape != avg[k].shape or sd[k].dtype != avg[k].dtype]
    if bad:
        raise ValueError(f"{label}: shape / dtype differs from the average's in {bad[:3]}")
    moved = sum(1 for k in sd if torch.is_floating_point(sd[k]) and not torch.equal(sd[k], avg[k]))
    print(f"{label}: {moved} of {len(sd)} tensors replaced by the average", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--soup", help="the average; omitted: the base's own EMA copy --ema fills every slot")
    ap.add_argument("--base", required=True, help="a full (resumable) checkpoint of the run")
    ap.add_argument("--ema", default="0_9999", help="which EMA copy of the average fills every slot")
    args = ap.parse_args()
    base = torch.load(args.base, map_location="cpu", mmap=True, weights_only=False)
    if args.soup is None:
        soup = {"ema_state_dicts": base["ema_state_dicts"],
                "soup": {"steps": [base["global_step"]], "weights": [1.0]}}
        args.soup = args.base
    else:
        soup = torch.load(args.soup, map_location="cpu", mmap=True, weights_only=False)
    if "soup" not in soup:
        raise ValueError(f"{args.soup} has no 'soup' key: not a checkpoint average")
    avg = soup["ema_state_dicts"][args.ema]
    print("average:", args.soup, "EMA", args.ema, "\n  steps", soup["soup"]["steps"],
          "\n  weights", [round(w, 4) for w in soup["soup"]["weights"]], flush=True)
    print(f"base {args.base}: step {base['global_step']}, active EMA {base.get('ema_active_decay')}, "
          f"EMA entries {list(base.get('ema_state_dicts', {}))}, teacher copied at {base['guided_cbm_teacher']['step']}",
          flush=True)
    teacher_avg = {k[len("model."):]: v for k, v in avg.items() if k.startswith("model.")}
    check("current_model_state", base["current_model_state"], avg)
    check("state_dict", base["state_dict"], avg)
    for k, v in base.get("ema_state_dicts", {}).items():
        check(f"ema_state_dicts[{k}]", v, avg)
    check("guided_cbm_teacher", base["guided_cbm_teacher"]["state"], teacher_avg)

    # a separate copy per slot: shared storages would survive the save, and an in-place EMA update would then move them all
    def filled(sd):
        return {k: v.clone() for k, v in sd.items()}
    out = dict(base)
    out["current_model_state"] = filled(avg)
    out["state_dict"] = filled(avg)
    out["ema_state_dicts"] = {k: filled(avg) for k in base.get("ema_state_dicts", {})}
    out["guided_cbm_teacher"] = dict(base["guided_cbm_teacher"], state=filled(teacher_avg))
    out["soup_resume"] = {"soup": str(Path(args.soup).resolve()), "soup_steps": soup["soup"]["steps"],
                          "soup_weights": soup["soup"]["weights"], "ema": args.ema, "base": str(Path(args.base).resolve())}
    tmp = args.out + f".tmp{os.getpid()}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, tmp)
    # read back: every slot equals the average
    back = torch.load(tmp, map_location="cpu", mmap=True, weights_only=False)
    slots = [back["current_model_state"], back["state_dict"], *back["ema_state_dicts"].values()]
    assert all(set(s) == set(avg) and all(torch.equal(s[k], avg[k]) for k in avg) for s in slots)
    assert all(torch.equal(back["guided_cbm_teacher"]["state"][k], v) for k, v in teacher_avg.items())
    print("read back: every slot equals the average", flush=True)
    os.replace(tmp, args.out)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
