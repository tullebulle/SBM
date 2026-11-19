"""CC12M for sbm_t2i.data.TextImageShards: the LLaVA-NeXT recaptions onto the mirror's shards.

The recaptions (CaptionEmporium/conceptual-captions-cc12m-llavanext, train.jsonl.gz) were made on pixparse/cc12m-wds
itself, so a row's `key` is the tar member's key. For every shard cc12m-train-NNNN.tar this writes
<out>/cc12m-train-NNNN.json = {key: caption_llava} (the caption MiniT2I pretrains and distils on); the loader then
skips the images without a recaption. Member names are read from the tar headers by seeking (no image is read); on
every 100th shard the row's url is checked against the shard's own {key}.json.
usage: python scripts/t2i/data/join_cc12m.py [threads=16]   (paths under <store>/t2i/data)
"""
import gzip
import json
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sbm_t2i.run import store  # noqa: E402

ROOT = store() / "t2i" / "data"
RAW, RECAP, OUT = ROOT / "raw/cc12m", ROOT / "raw/cc12m_recaption/train.jsonl.gz", ROOT / "shards/cc12m_captions"


def scan(item):
    index, shard = item
    keys, urls = [], {}
    with tarfile.open(shard, "r:") as tar:
        for m in tar:
            if m.name.endswith(".jpg"):
                keys.append(m.name[:-4])
            elif index % 100 == 0 and m.name.endswith(".json"):
                urls[m.name[:-5]] = json.loads(tar.extractfile(m).read()).get("url")
    return shard, keys, urls


if __name__ == "__main__":
    threads = int(dict(a.split("=") for a in sys.argv[1:]).get("threads", 16))
    OUT.mkdir(parents=True, exist_ok=True)
    captions, url_of, rows = {}, {}, 0
    with gzip.open(RECAP, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rows += 1
            text = " ".join((row.get("caption_llava") or "").split())
            if text:
                captions[row["key"]] = text
                if int(row["key"]) % 97 == 0:
                    url_of[row["key"]] = row.get("url")
            if rows % 2_000_000 == 0:
                print(rows, "recaption rows", flush=True)
    print(rows, "rows,", len(captions), "captions", flush=True)
    shards = sorted(RAW.glob("cc12m-train-*.tar"))
    images = matched = checked = agree = 0
    with ThreadPoolExecutor(threads) as pool:
        for n, (shard, keys, urls) in enumerate(pool.map(scan, enumerate(shards))):
            mine = {k: captions[k] for k in keys if k in captions}
            (OUT / (shard.stem + ".json")).write_text(json.dumps(mine))
            images, matched = images + len(keys), matched + len(mine)
            for k, u in urls.items():
                if k in url_of:
                    checked, agree = checked + 1, agree + (url_of[k] == u)
            if n % 200 == 0:
                print(n, "shards", images, "images", matched, "with a recaption", flush=True)
    summary = {"shards": len(shards), "images": images, "with_recaption": matched, "recaption_rows": rows,
               "url_checked": checked, "url_agree": agree}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
