"""Export one EMA copy of a training checkpoint (RAE line and world models) as a slim inference file (sbm_rae.infer) and
verify that it loads.

    python scripts/rae/export_record.py --ckpt outputs/<run>/checkpoints/<step>.ckpt --out my_export.pt [--ema 0.9999]
        [--meta-json '{"read": {"omega": 0.8, "label_scale": 1.35}}'] [--expect-params-m 669.4] [--no-verify]

`meta.read` holds the read settings sample.py and the evaluators use by default. The checkpoint is opened with mmap, so a
14 GB DiT-XL checkpoint needs no memory beyond the exported copy (~2.7 GB).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from sbm_rae.infer import SLIM_FORMAT, load_model_from_export, load_slim_export, slim_export_from_checkpoint  # noqa: E402


def sha256_of(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--meta-json", default=None, help="a JSON file or an inline JSON object recorded under `meta`")
    ap.add_argument("--expect-params-m", type=float, default=None, help="fail unless the loaded model has this many M parameters (+-0.1)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    meta = {}
    if a.meta_json:
        meta = json.loads(Path(a.meta_json).read_text()) if os.path.isfile(a.meta_json) else json.loads(a.meta_json)
    meta.setdefault("source_checkpoint", str(Path(a.ckpt).resolve()))
    meta.setdefault("exported_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)
    print(f"loaded (mmap) {a.ckpt} in {time.time() - t0:.1f}s: global_step {ck.get('global_step')} epoch {ck.get('epoch')} "
          f"active EMA {ck.get('ema_active_decay')} stored {sorted((ck.get('ema_state_dicts') or {}).keys())}", flush=True)
    export = slim_export_from_checkpoint(ck, a.ema, meta=meta)
    n_tensors = len(export["model_state_dict"])
    n_params = sum(v.numel() for v in export["model_state_dict"].values())
    print(f"EMA {a.ema} ({export['ema_label']}): {n_tensors} tensors, {n_params / 1e6:.2f} M parameters", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    torch.save(export, tmp)
    os.replace(tmp, out)
    size = out.stat().st_size
    digest = sha256_of(out)
    print(f"wrote {out} ({size / 1e9:.3f} GB, sha256 {digest}) in {time.time() - t0:.1f}s", flush=True)
    del ck, export

    if not a.no_verify:
        t1 = time.time()
        loaded = load_slim_export(out)
        assert loaded["format"] == SLIM_FORMAT
        model, info = load_model_from_export(loaded, model_overrides={"attn_func": "default"}, strict=True)
        print(f"verify: built {loaded['hyper_parameters']['cfg']['model']['name']} from hparams.cfg.model, "
              f"load_state_dict(strict=True) ok, {info['n_params'] / 1e6:.2f} M parameters, {info['n_tensors']} tensors "
              f"({time.time() - t1:.1f}s)", flush=True)
        if a.expect_params_m is not None and abs(info["n_params"] / 1e6 - a.expect_params_m) > 0.1:
            raise SystemExit(f"parameter count {info['n_params'] / 1e6:.2f} M differs from the expected {a.expect_params_m} M")
        del model, loaded

    print("done")


if __name__ == "__main__":
    main()
