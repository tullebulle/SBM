#!/bin/bash
# GenEval score of one image folder (Ghosh et al. 2023, github.com/djghosh13/geneval; install: build_env.sh here).
#
#   bash scripts/t2i/geneval/score.sh <image_dir> <out_dir>
#
# <image_dir> in GenEval's layout: <index:05d>/metadata.jsonl (that prompt's line of prompts/evaluation_metadata.jsonl)
# and <index:05d>/samples/<k>.png (any \d+.png; other files and folders are ignored). Runs GenEval's
# evaluation/evaluate_images.py (Mask2Former Swin-S COCO + CLIP ViT-L/14 colour check, defaults unchanged) and
# evaluation/summary_scores.py, and writes into <out_dir> (created; nothing else in it is touched):
#   results.jsonl   one line per image (correct, reason, detections)
#   summary.txt     summary_scores.py's printout
#   summary.json    {"overall", "single_object", "two_object", "counting", "colors", "position", "color_attr", "n_images"}
#                   overall = mean of the six task scores (GenEval's number); a task with no images reads null
# Rerunning overwrites (old outputs are removed first, so a failed run leaves no stale summary.json).
# Needs one GPU with compute capability 8.0 or 9.0 (A100, H100, H200): the env is torch 2.1.2+cu121 and mmcv-full 1.7.2,
# whose CUDA ops are built for sm_80 / sm_90. About 10 minutes for the 2,212 images.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${GENEVAL_ROOT:-${SBM_STORE:-$REPO/store}/t2i/geneval}
PY=$ROOT/env/bin/python
[[ $# -eq 2 ]] || { echo "usage: $0 <image_dir> <out_dir>" >&2; exit 2; }
IMG=$(realpath "$1")
[[ -d $IMG ]] || { echo "[geneval] no such folder: $IMG" >&2; exit 2; }
mkdir -p "$2"
OUT=$(realpath "$2")

n_img=$(find -L "$IMG" -mindepth 3 -maxdepth 3 -regextype posix-extended -regex '.*/[0-9]+/samples/[0-9]+\.png' | wc -l)
n_meta=$(find -L "$IMG" -mindepth 2 -maxdepth 2 -name metadata.jsonl | wc -l)
echo "[geneval] $(date +%FT%T) $(hostname) image_dir=$IMG ($n_meta prompts, $n_img images) out_dir=$OUT"
(( n_img > 0 )) || { echo "[geneval] no images in $IMG" >&2; exit 2; }
(( n_img == 2212 && n_meta == 553 )) || echo "[geneval] warning: the full benchmark is 553 prompts x 4 = 2212 images"

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$cc" in
  8.0|9.0) echo "[geneval] GPU: $gpu (sm ${cc/./})" ;;
  *) echo "[geneval] GPU '$gpu' (compute capability '${cc:-none}') unsupported: need A100 / H100 / H200" >&2; exit 3 ;;
esac

# Isolation from the caller's environment; open_clip reads its "openai" ViT-L/14 weights from ~/.cache/clip, kept in
# $ROOT/home; no network access is needed.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONSTARTUP
export HOME=$ROOT/home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
JOBTMP=$(mktemp -d "${TMPDIR:-/tmp}/geneval_XXXXXX")
trap 'rm -rf "$JOBTMP"' EXIT
export MPLCONFIGDIR=$JOBTMP/mpl OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

rm -f "$OUT/results.jsonl" "$OUT/summary.json" "$OUT/summary.txt"
cd "$ROOT/geneval"
t0=$(date +%s)
"$PY" evaluation/evaluate_images.py "$IMG" --outfile "$JOBTMP/results.jsonl" --model-path "$ROOT/models"
cp "$JOBTMP/results.jsonl" "$OUT/.results.jsonl.tmp" && mv "$OUT/.results.jsonl.tmp" "$OUT/results.jsonl"
"$PY" evaluation/summary_scores.py "$OUT/results.jsonl" > "$JOBTMP/summary.txt"
cat "$JOBTMP/summary.txt"
"$PY" - "$OUT/results.jsonl" "$JOBTMP/summary.json" <<'EOF'
import json, sys
import numpy as np
import pandas as pd
df = pd.read_json(sys.argv[1], orient="records", lines=True)
tags = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
scores = {t: (float(df.loc[df["tag"] == t, "correct"].mean()) if (df["tag"] == t).any() else None) for t in tags}
present = [float(df.loc[df["tag"] == t, "correct"].mean()) for t in df["tag"].unique()]   # summary_scores.py's mean
out = {"overall": float(np.mean(present)), **scores, "n_images": int(len(df))}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
EOF
cp "$JOBTMP/summary.txt" "$OUT/summary.txt"
cp "$JOBTMP/summary.json" "$OUT/.summary.json.tmp" && mv "$OUT/.summary.json.tmp" "$OUT/summary.json"
echo "[geneval] $(date +%FT%T) done in $(( $(date +%s) - t0 ))s: $OUT/summary.json"
