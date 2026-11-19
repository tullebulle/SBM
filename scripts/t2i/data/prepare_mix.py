"""MiniT2I's 120K fine-tuning mix as uniform shards for sbm_t2i.data.TextImageShards: every image at 512 x 512 (shorter side
to 512 bicubic, centre crop: MiniT2I's load-time transform, applied once) as JPEG quality 97 / 4:4:4, with its caption.

  blip3o60k    BLIP3o/BLIP3o-60k            11 tars of {n}.jpg (1024 px) + {n}.txt
  dalle3       OpenDatasets/dalle-3-dataset 67 parquets, columns image / caption / image_hash
  sharegpt4o   FreedomIntelligence/ShareGPT-4o-Image, text_to_image.json (output_image, input_prompt) + 10 tars of
               image/<id>.png; prompts longer than 256 FLAN-T5 tokens are dropped, as the authors do

One task per source file, each writing <out>/<source>-<file>-<nnn>.tar of up to 1000 samples; a finished task leaves
<out>/done/<task>.json with its counts and is skipped on a re-run. usage: python scripts/t2i/data/prepare_mix.py [workers=16]   (paths under <store>/t2i)
"""
import io
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sbm_t2i.run import store  # noqa: E402

RAW, OUT = store() / "t2i/data/raw", store() / "t2i/data/shards/mix120k"
T5 = store() / "t2i/weights/flan-t5-large"
SIZE, PER_SHARD = 512, 1000


class Writer:
    def __init__(self, task):
        self.task, self.n, self.shard, self.tar = task, 0, 0, None

    def add(self, key, image, caption):
        """image: a PIL image; -> written as <task>_<key>.jpg + .txt"""
        from PIL import Image
        from sbm_t2i.data import center_square
        caption = " ".join(str(caption).split())
        if not caption:
            return False
        x = center_square(image, SIZE)
        buf = io.BytesIO()
        Image.fromarray(x.permute(1, 2, 0).numpy()).save(buf, "JPEG", quality=97, subsampling=0)
        if self.tar is None or self.n % PER_SHARD == 0:
            self.close()
            self.tar = tarfile.open(OUT / f"{self.task}-{self.shard:03d}.tar.tmp", "w")
            self.shard += 1
        for name, data in ((f"{self.task}_{key}.jpg", buf.getvalue()), (f"{self.task}_{key}.txt", caption.encode("utf-8"))):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            self.tar.addfile(info, io.BytesIO(data))
        self.n += 1
        return True

    def close(self):
        if self.tar is not None:
            self.tar.close()
            tmp = Path(self.tar.name)
            tmp.rename(tmp.with_suffix(""))   # .tar.tmp -> .tar
            self.tar = None


def run(task, source, path):
    from PIL import Image
    done = OUT / "done" / f"{task}.json"
    if done.exists():
        return json.loads(done.read_text())
    for stale in OUT.glob(f"{task}-*.tar*"):
        stale.unlink()
    w, skipped = Writer(task), 0
    if source == "blip3o60k":
        key, image, text = None, None, None

        def flush():
            nonlocal skipped
            if image is not None and text is not None:
                try:
                    skipped += not w.add(key, Image.open(io.BytesIO(image)), text)
                except Exception:
                    skipped += 1
        with tarfile.open(path, "r|*") as tar:
            for m in tar:
                if not m.isfile() or "." not in m.name:
                    continue
                stem, ext = m.name.rsplit(".", 1)
                if stem != key:
                    flush()
                    key, image, text = stem, None, None
                if ext.lower() in ("jpg", "jpeg", "png", "webp"):
                    image = tar.extractfile(m).read()
                elif ext.lower() == "txt":
                    text = tar.extractfile(m).read().decode("utf-8", "replace")
        flush()
    elif source == "dalle3":
        import pyarrow.parquet as pq
        row = 0
        for rows in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "caption", "image_hash"]):
            for rec in rows.to_pylist():
                row += 1
                raw = rec["image"]
                raw = raw.get("bytes") if isinstance(raw, dict) else raw
                try:
                    skipped += not w.add(rec.get("image_hash") or f"{row:06d}", Image.open(io.BytesIO(raw)), rec.get("caption") or "")
                except Exception:
                    skipped += 1
    elif source == "sharegpt4o":
        from sbm_t2i.text import tokenizer
        tok = tokenizer(T5)
        captions = {}
        for item in json.loads((RAW / "sharegpt4o/text_to_image.json").read_text()):
            name = item.get("output_image", "")
            if name.startswith("image/") and name.endswith(".png") and name[6:-4].isdigit():
                captions[int(name[6:-4])] = item.get("input_prompt", "")
        with tarfile.open(path, "r|*") as tar:
            for m in tar:
                if not m.isfile() or not (m.name.startswith("image/") and m.name.endswith(".png")) or not m.name[6:-4].isdigit():
                    continue
                caption = captions.get(int(m.name[6:-4]))
                if not caption or len(tok(caption, truncation=False).input_ids) > 256:
                    skipped += 1
                    continue
                try:
                    skipped += not w.add(m.name[6:-4], Image.open(io.BytesIO(tar.extractfile(m).read())), caption)
                except Exception:
                    skipped += 1
    w.close()
    out = {"task": task, "source": source, "written": w.n, "skipped": skipped, "shards": w.shard}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    workers = int(dict(a.split("=") for a in sys.argv[1:]).get("workers", 16))
    (OUT / "done").mkdir(parents=True, exist_ok=True)
    tasks = [(f"blip3o-{p.stem}", "blip3o60k", p) for p in sorted((RAW / "blip3o60k").glob("*.tar"))]
    tasks += [(f"dalle3-{i:02d}", "dalle3", p) for i, p in enumerate(sorted((RAW / "dalle3/data").glob("*.parquet")))]
    tasks += [(f"sharegpt4o-{p.stem.rsplit('_', 1)[1]}", "sharegpt4o", p) for p in sorted((RAW / "sharegpt4o").glob("text_to_image_part_*.tar"))]
    print(len(tasks), "source files", flush=True)
    total = {}
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(run, *t) for t in tasks]):
            r = f.result()
            total.setdefault(r["source"], [0, 0])
            total[r["source"]][0] += r["written"]
            total[r["source"]][1] += r["skipped"]
            print(json.dumps(r), flush=True)
    summary = {"written": {k: v[0] for k, v in total.items()}, "skipped": {k: v[1] for k, v in total.items()},
               "shards": len(list(OUT.glob("*.tar"))), "images": sum(v[0] for v in total.values())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
