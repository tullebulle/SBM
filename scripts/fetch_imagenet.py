"""ImageNet-1k (ILSVRC 2012) from Hugging Face as the class folders the trainers read:

    <out>/train/<wnid>/<name>.JPEG    1,281,167 images
    <out>/val/<wnid>/<name>.JPEG      50,000 images

The source is the gated dataset ILSVRC/imagenet-1k: accept its terms on huggingface.co/datasets/ILSVRC/imagenet-1k (logged
in), then `hf auth login`. The JPEG bytes are written unchanged, under their original file names: the result equals the
image-net.org release file for file, except the 23 CMYK JPEGs (22 train, 1 val), which the Hugging Face copy holds
re-encoded as RGB. Each parquet file is
downloaded, unpacked and deleted in turn, so the disk peak is about the images themselves (147 + 7 GB); a re-run skips the
finished files.

    python scripts/fetch_imagenet.py [--out <store>/data/imagenet_1k] [--splits train val] [--workers 8]
"""
import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from sbm_rae.paths import store_root

REPO = "ILSVRC/imagenet-1k"
SPLITS = {"train": ("train", 1_281_167), "val": ("validation", 50_000)}   # our folder -> (HF split, images)


def class_ids(cache):
    """The 1000 WordNet ids in label order, read from the dataset's classes.py (parsed, not executed)."""
    text = Path(hf_hub_download(REPO, "classes.py", repo_type="dataset", local_dir=cache)).read_text()
    wnids = list(dict.fromkeys(re.findall(r'"(n\d{8})"', text)))
    assert len(wnids) == 1000, f"expected 1000 class ids in classes.py, found {len(wnids)}"
    return wnids


def file_name(path, wnid, fallback):
    """The original ImageNet name: the parquet keeps it, at most with the class id appended (`..._n01440764.JPEG`)."""
    name = PurePosixPath(path).name if path else fallback
    return re.sub(rf"_{wnid}(\.\w+)$", r"\1", name)


def unpack(parquet, dest, wnids):
    """Write every image of one parquet file to dest/<wnid>/<name>; returns the number written."""
    n = 0
    made = set()
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256, columns=["image", "label"]):
        image, labels = batch.column("image"), batch.column("label").to_pylist()
        data, paths = image.field("bytes").to_pylist(), image.field("path").to_pylist()
        for blob, path, label in zip(data, paths, labels):
            wnid = wnids[label]
            if wnid not in made:
                (dest / wnid).mkdir(parents=True, exist_ok=True)
                made.add(wnid)
            (dest / wnid / file_name(path, wnid, f"{Path(parquet).stem}_{n:06d}.JPEG")).write_bytes(blob)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(store_root() / "data" / "imagenet_1k"))
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=8, help="parquet files in flight at once")
    a = ap.parse_args()
    out = Path(a.out)
    cache, done_dir = out / ".parquet", out / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    wnids = class_ids(cache)
    files = HfApi().list_repo_files(REPO, repo_type="dataset")

    def job(split, name):
        marker = done_dir / (PurePosixPath(name).stem + ".json")
        if marker.exists():
            return json.loads(marker.read_text())["images"]
        local = hf_hub_download(REPO, name, repo_type="dataset", local_dir=cache)
        n = unpack(local, out / split, wnids)
        marker.write_text(json.dumps({"file": name, "images": n}))
        Path(local).unlink()
        print(f"{name}: {n} images -> {out / split}", flush=True)
        return n

    ok = True
    for split in a.splits:
        hf_split, expected = SPLITS[split]
        names = sorted(f for f in files if re.fullmatch(rf"data/{hf_split}-\d+-of-\d+\.parquet", f))
        with ThreadPoolExecutor(a.workers) as pool:
            total = sum(pool.map(lambda f: job(split, f), names))
        print(f"{split}: {total:,} images in {len(names)} files (expected {expected:,})")
        ok &= total == expected
    shutil.rmtree(cache, ignore_errors=True)
    if not ok:
        raise SystemExit("image counts differ from ImageNet-1k's: re-run to resume, or check the download")
    print(f"ImageNet-1k ready in {out}")


if __name__ == "__main__":
    main()
