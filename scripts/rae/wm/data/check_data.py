"""Checks for the world models' data at each stage (scripts/rae/wm/data); exits non-zero on a failure.

    check_data.py raw <TFDS builder dir>
        every shard listed in dataset_info.json present, no aria2 control file left, and per split the bytes on disk equal
        dataset_info.json's numBytes + 16 bytes of TFRecord framing per example (catches truncated or partial downloads)
    check_data.py processed <processed dir> [--dataset bridge|rt1] [--scan --workers N]
        per split: episodes {i:09d}.mp4 / .npz / .txt / .meta.json for i = 0..n-1, no mp4 under 1 KB, manifest_<split>.jsonl
        with one row per episode; with --dataset the counts, frame totals and the manifests' sha256 against the paper's
        conversion (the manifest rows hold the sha1 of each episode's raw frames, so equal manifests mean the same episodes,
        ids, frames and instructions); --scan decodes every mp4 and matches its frame count to the npz
    check_data.py latents <latent cache dir> --processed <processed dir> [--dataset bridge|rt1]
        per split: index.json covers every processed episode once, contiguous offsets, latents.npy (frames, 16, 32, 32)
        float16 and actions.npy (frames, A); with --dataset the frame totals of the paper's cache

Stdlib only for `raw`; numpy for `processed` / `latents`, decord for --scan.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# the paper's data (DriftWorld's conversion of BridgeData V2 1.0.0 and RT-1 fractal20220817_data 0.1.0)
EXPECTED = {
    "bridge": {
        "action_dim": 7,
        "episodes": {"train": 53192, "val": 6872},
        "frames": {"train": 1999410, "val": 256181},
        "manifest_sha256": {"train": "c8fd405cc4fad49262ed684c4edd3692747f733aaadeeca7823c4ac8283421c1",
                            "val": "f4880eb8835fb024ccd9303e1bc5dbc0048c84d054aedbbfa489c77689d30eae"},
    },
    "rt1": {
        "action_dim": 10,
        "episodes": {"train": 85032, "val": 2180},
        "frames": {"train": 3691573, "val": 94827},
        "manifest_sha256": {"train": "2275af10cab53ddea6404d7a29dbe5474ebd915f73bce507b5d1e7fb9484006d",
                            "val": "03bccf60e241a1baee0fb507436620446823b85489ba935ecae45aca8d0f8f02"},
    },
}
SPLITS = ("train", "val")
STEM = re.compile(r"^\d{9}$")


class Report:
    def __init__(self):
        self.failures = 0

    def ok(self, msg):
        print(f"  ok    {msg}")

    def fail(self, msg):
        print(f"  FAIL  {msg}")
        self.failures += 1

    def check(self, cond, msg):
        (self.ok if cond else self.fail)(msg)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_raw(builder, rep):
    builder = Path(builder)
    info_path = builder / "dataset_info.json"
    if not info_path.is_file():
        rep.fail(f"{info_path} missing")
        return
    info = json.loads(info_path.read_text())
    rep.check((builder / "features.json").is_file(), "features.json present")
    names = os.listdir(builder)
    leftovers = [n for n in names if n.endswith(".aria2")]
    rep.check(not leftovers, f"no aria2 control files left ({len(leftovers)} found: {leftovers[:3]})")
    prefix = info["name"]
    for sp in info["splits"]:
        split, n_shards = sp["name"], len(sp["shardLengths"])
        examples = sum(int(x) for x in sp["shardLengths"])
        want = [f"{prefix}-{split}.tfrecord-{k:05d}-of-{n_shards:05d}" for k in range(n_shards)]
        missing = [w for w in want if not (builder / w).is_file()]
        rep.check(not missing, f"{split}: {n_shards - len(missing)} / {n_shards} shards ({examples} episodes)"
                               + (f", missing e.g. {missing[:3]}" if missing else ""))
        if missing:
            continue
        on_disk = sum((builder / w).stat().st_size for w in want)
        expect = int(sp["numBytes"]) + 16 * examples
        rep.check(on_disk == expect, f"{split}: {on_disk} bytes on disk, dataset_info.json says {expect}")


def _decode(mp4):
    import decord
    import numpy as np

    try:
        vr = decord.VideoReader(mp4, ctx=decord.cpu(0), num_threads=1)
        n = len(vr)
        last = vr[n - 1].asnumpy()
        a = np.load(mp4[:-4] + ".npz")["arr_0"].shape[0]
        return mp4, (n == a), f"{n} frames vs {a} actions", tuple(last.shape)
    except Exception as e:  # noqa: BLE001 - report every unreadable file
        return mp4, False, f"{type(e).__name__}: {e}", None


def check_processed(root, dataset, scan, workers, rep):
    import numpy as np

    exp = EXPECTED.get(dataset)
    for split in SPLITS:
        d = Path(root) / split
        if not d.is_dir():
            rep.fail(f"{d} missing")
            continue
        by_ext = {}
        for name in os.listdir(d):
            stem, _, ext = name.partition(".")
            if STEM.match(stem):
                by_ext.setdefault(ext, set()).add(stem)
        stems = by_ext.get("mp4", set())
        n = len(stems)
        rep.check(stems == {f"{i:09d}" for i in range(n)}, f"{split}: {n} episodes with ids 0..{n - 1}")
        for ext in ("npz", "txt", "meta.json"):
            rep.check(by_ext.get(ext, set()) == stems, f"{split}: every episode has its .{ext}")
        small = [s for s in sorted(stems) if (d / f"{s}.mp4").stat().st_size <= 1024]
        rep.check(not small, f"{split}: no mp4 under 1 KB" + (f" ({len(small)}: {small[:3]})" if small else ""))
        manifest = d / f"manifest_{split}.jsonl"
        if manifest.is_file():
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            rep.check(sorted(r["index"] for r in rows) == sorted(stems), f"{split}: manifest has one row per episode ({len(rows)})")
            frames = sum(int(r["n_frames"]) for r in rows)
        else:
            rep.fail(f"{split}: {manifest.name} missing (run the finalize step)")
            rows, frames = None, None
        if stems:
            a = np.load(d / f"{min(stems)}.npz")["arr_0"]
            rep.ok(f"{split}: actions {a.dtype} (T, {a.shape[1]}), {frames} frames in the manifest")
            if exp:
                rep.check(a.shape[1] == exp["action_dim"], f"{split}: action dim {a.shape[1]} == {exp['action_dim']}")
        if exp:
            rep.check(n == exp["episodes"][split], f"{split}: {n} episodes == the paper's {exp['episodes'][split]}")
            if rows is not None:
                rep.check(frames == exp["frames"][split], f"{split}: {frames} frames == the paper's {exp['frames'][split]}")
                digest = sha256(manifest)
                rep.check(digest == exp["manifest_sha256"][split],
                          f"{split}: manifest sha256 {digest[:16]}.. == the paper's {exp['manifest_sha256'][split][:16]}..")
        if scan and stems:
            files = [str(d / f"{s}.mp4") for s in sorted(stems)]
            bad, shapes = [], set()
            with ProcessPoolExecutor(workers) as ex:
                for mp4, good, msg, shape in ex.map(_decode, files, chunksize=64):
                    if shape:
                        shapes.add(shape)
                    if not good:
                        bad.append((mp4, msg))
                        print(f"  BAD   {mp4}: {msg}", flush=True)
            rep.check(not bad, f"{split}: {len(files) - len(bad)} / {len(files)} mp4s decode to their npz length; "
                               f"frame shapes {sorted(shapes)}")


def check_latents(cache, processed, dataset, rep):
    import numpy as np

    exp = EXPECTED.get(dataset)
    for split in SPLITS:
        d = Path(cache) / split
        if not (d / "index.json").is_file():
            rep.fail(f"{d}/index.json missing (run the merge step)")
            continue
        meta = json.loads((d / "index.json").read_text())
        eps = sorted(meta["episodes"], key=lambda e: e["offset"])
        stems = [e["stem"] for e in eps]
        want = sorted(n[:-4] for n in os.listdir(Path(processed) / split) if n.endswith(".mp4") and STEM.match(n[:-4]))
        rep.check(sorted(stems) == want, f"{split}: {len(stems)} episodes cached, once each, = the processed episodes ({len(want)})")
        off = 0
        contiguous = True
        for e in eps:
            contiguous &= e["offset"] == off and e["length"] > 0
            off += e["length"]
        rep.check(contiguous and off == meta["total"], f"{split}: contiguous offsets, {meta['total']} frames")
        rep.check(meta["codec"] == "sd3" and abs(float(meta["scale"]) - 1.5305) < 1e-6,
                  f"{split}: codec {meta['codec']}, scale {meta['scale']}")
        lat = np.load(d / "latents.npy", mmap_mode="r")
        act = np.load(d / "actions.npy", mmap_mode="r")
        rep.check(lat.shape == (meta["total"], 16, 32, 32) and lat.dtype == np.float16,
                  f"{split}: latents.npy {lat.shape} {lat.dtype}")
        rep.check(act.shape[0] == meta["total"] and (not exp or act.shape[1] == exp["action_dim"]),
                  f"{split}: actions.npy {act.shape} {act.dtype}")
        if exp:
            rep.check(meta["total"] == exp["frames"][split], f"{split}: {meta['total']} frames == the paper's {exp['frames'][split]}")
        parts = sorted(p.name for p in d.glob("part*.index.json"))
        if parts:
            print(f"  note  {split}: {len(parts)} part files remain (delete <split>/part* to free the space; or rerun the merge step with CLEAN_PARTS=1)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("raw")
    r.add_argument("builder")
    p = sub.add_parser("processed")
    p.add_argument("processed")
    p.add_argument("--dataset", choices=sorted(EXPECTED))
    p.add_argument("--scan", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    lt = sub.add_parser("latents")
    lt.add_argument("cache")
    lt.add_argument("--processed", required=True)
    lt.add_argument("--dataset", choices=sorted(EXPECTED))
    a = ap.parse_args()
    rep = Report()
    if a.stage == "raw":
        print(f"raw TFDS release: {a.builder}")
        check_raw(a.builder, rep)
    elif a.stage == "processed":
        print(f"processed episodes: {a.processed}")
        check_processed(a.processed, a.dataset, a.scan, a.workers, rep)
    else:
        print(f"latent cache: {a.cache}")
        check_latents(a.cache, a.processed, a.dataset, rep)
    print("CHECK PASSED" if rep.failures == 0 else f"CHECK FAILED ({rep.failures} failures)")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
