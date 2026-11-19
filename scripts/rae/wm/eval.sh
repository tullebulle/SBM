#!/bin/bash
# World-model validation read (DriftWorld's protocol, scripts/rae/eval_bridge.py): every validation episode generated in
# 8-frame chunks from the true history, one shard per GPU, then the FVD (stylegan-v's fvd2048_16f on the first 16 generated
# frames of each video) and the means -> <out>/summary.json (SSIM, PSNR, LPIPS, FID, FVD).
#
#   bash scripts/rae/wm/eval.sh <checkpoint> <out dir> [eval_bridge.py options, e.g. --max-videos 256 for a quick read]
#
# The read settings (anchor omega, frame size) come from the checkpoint. GPUS="0 1 2 3" picks devices (default: all visible).
# Needs `uv sync --extra wm`, `bash download.sh wm` and the processed validation split (scripts/rae/wm/data/prepare.sh).
# A full Bridge-V2 read takes about 50 min on 4 H200s, RT-1 about 25 min.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CKPT=${1:?usage: eval.sh <checkpoint> <out dir> [eval_bridge.py options]}; OUT=${2:?usage: eval.sh <checkpoint> <out dir>}; shift 2
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
STORE=$(cd "${SBM_STORE:-$REPO/store}" && pwd)
export SBM_STORE=$STORE HF_HOME=$STORE/rae/assets/hf_home HF_HUB_OFFLINE=1 TORCH_HOME=$STORE/rae/assets/torch_home
export DNNLIB_CACHE_DIR=$STORE/rae/assets/dnnlib OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
PY=$REPO/.venv/bin/python
GPUS=${GPUS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}
read -ra DEV <<< "$GPUS"; N=${#DEV[@]}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

echo "$N shards (logs: $OUT/shard_*.log)"
pids=()
for i in "${!DEV[@]}"; do
  CUDA_VISIBLE_DEVICES=${DEV[$i]} "$PY" "$REPO/scripts/rae/eval_bridge.py" --ckpt "$CKPT" --out "$OUT" \
    --world-size "$N" --rank "$i" "$@" > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "a shard failed: see $OUT/shard_*.log" >&2; exit 1; }

if [[ -d "$OUT/pred_first16" ]]; then
  echo "FVD (stylegan-v fvd2048_16f)"
  (cd "$STORE/refs/code/stylegan-v" && CUDA_VISIBLE_DEVICES=${DEV[0]} "$PY" src/scripts/calc_metrics_for_dataset.py \
     --real_data_path "$OUT/gt_first16" --fake_data_path "$OUT/pred_first16" --mirror 1 --gpus 1 --resolution 256 \
     --metrics fvd2048_16f --verbose 0 --use_cache 0) > "$OUT/fvd_stylegan_v.log" 2>&1
  "$PY" - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
m = re.search(r'\{"results": \{[^}]*\}', (out / "fvd_stylegan_v.log").read_text())
if not m:
    sys.exit(f"FVD: no result in {out / 'fvd_stylegan_v.log'}")
fvd = float(json.loads(m.group() + "}")["results"]["fvd2048_16f"])
(out / "fvd.json").write_text(json.dumps({"fvd": fvd, "fvd_metric": "fvd2048_16f (stylegan-v), first 16 frames of every video"}))
PY
fi
CUDA_VISIBLE_DEVICES=${DEV[0]} "$PY" "$REPO/scripts/rae/eval_bridge.py" --out "$OUT" --aggregate
