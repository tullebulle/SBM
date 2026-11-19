"""Re-deal the 120k fine-tuning mix (store/t2i/data/shards/mix120k: 175 shards, one source each, 288-1,000 samples) into
N equal shards of shuffled samples, for the alignment stage after CC12M (MiniT2I's second stage).

Why: TextImageShards deals whole shards to (rank, worker) and every worker yields at the same rate, so a sample's rate
goes as 1 / (samples its worker holds). On the source shards (175 over 16 ranks x 8 workers: 1 or 2 shards per worker,
288 to 1,000 samples each) DALL-E 3 samples would be drawn ~7x as often as BLIP3o ones. With N = 256 equal shards every
worker holds 2 of them (~931 samples), so every sample is drawn at the same rate.

Pass 1 lists the samples (tar headers only); a seeded permutation puts sample i at slot p, shard p % N. Pass 2 copies
each sample's members, bytes unchanged, into its shard. Samples keep their keys (source-prefixed); a key repeated in the
source (19 DALL-E 3 rows listed twice in one shard, both read by the loader) gets `_dup<n>` on its later copies, so no two
samples share a name. A sample without an image or a non-empty .txt is dropped, as the loader would drop it.

    python scripts/t2i/data/reshard_mix.py <src dir> <out dir> [--n 256] [--seed 0]
"""
import argparse
import io
import json
import os
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

IMAGES = ("jpg", "jpeg", "png", "webp")


def samples(path):
    """[(key, {ext: TarInfo})] of one tar, in order; members of a sample are adjacent."""
    out, key, members = [], None, None
    with tarfile.open(path, "r:") as tar:
        for m in tar:
            if not m.isfile() or "." not in m.name:
                continue
            stem, ext = m.name.rsplit(".", 1)
            if stem != key:
                if key is not None:
                    out.append((key, members))
                key, members = stem, {}
            members[ext.lower()] = m
    if key is not None:
        out.append((key, members))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    shards = sorted(src.glob("*.tar"))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.tar")):
        raise FileExistsError(f"{out} already holds shards")

    def named(s):
        """(output name, members) of one source shard's usable samples; a key seen before gets _dup<n>."""
        out = []
        for key, members in samples(s):
            if not any(e in members for e in IMAGES) or "txt" not in members or members["txt"].size == 0:
                continue
            seen[key] += 1
            out.append((key if seen[key] == 1 else f"{key}_dup{seen[key] - 1}", members))
        return out

    keep, dropped, seen = [], Counter(), Counter()
    for s in shards:
        n_all = len(samples(s))
        got = named(s)
        dropped[s.stem] += n_all - len(got)
        keep += [name for name, _ in got]
    dropped = {k: v for k, v in dropped.items() if v}
    if len(set(keep)) != len(keep):
        raise ValueError(f"{len(keep) - len(set(keep))} duplicate names after renaming")
    print(f"{len(keep)} samples in {len(shards)} shards; dropped {sum(dropped.values())}; "
          f"renamed {sum(1 for k in keep if '_dup' in k)} repeated keys", flush=True)

    perm = list(range(len(keep)))
    random.Random(args.seed).shuffle(perm)
    slot = {keep[i]: p for p, i in enumerate(perm)}   # name -> slot
    seen = Counter()   # pass 2 renames exactly as pass 1 did
    names = [out / f"mix120k-{j:04d}.tar.partial" for j in range(args.n)]
    writers = [tarfile.open(n, "w") for n in names]
    counts, sources = Counter(), defaultdict(Counter)
    try:
        for s in shards:
            with tarfile.open(s, "r:") as tar:
                for key, members in named(s):
                    j = slot[key] % args.n
                    for ext, m in members.items():
                        data = tar.extractfile(m).read()
                        info = tarfile.TarInfo(f"{key}.{m.name.rsplit('.', 1)[1]}")
                        info.size = len(data)
                        writers[j].addfile(info, io.BytesIO(data))
                    counts[j] += 1
                    sources[j][key.split("_", 1)[0].split("-")[0]] += 1
            print(f"  {s.name} done", flush=True)
    finally:
        for w in writers:
            w.close()

    # read back: every shard holds exactly its samples, each with an image and a non-empty caption
    total, names_out = 0, set()
    for j, n in enumerate(names):
        got = samples(n)
        names_out |= {k for k, _ in got}
        assert len(got) == counts[j], (n, len(got), counts[j])
        assert all(any(e in m for e in IMAGES) and m.get("txt") is not None and m["txt"].size > 0 for _, m in got), n
        total += len(got)
    assert total == len(keep) == len(names_out), (total, len(keep), len(names_out))
    for n in names:
        os.replace(n, str(n)[: -len(".partial")])
    summary = {"source": str(src.resolve()), "seed": args.seed, "shards": args.n, "images": total,
               "per_shard": [min(counts.values()), max(counts.values())],
               "by_source": dict(sum(sources.values(), Counter())), "dropped": dropped,
               "renamed_repeats": sorted(k for k in keep if "_dup" in k)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
