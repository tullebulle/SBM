#!/bin/bash
# Build a world-model dataset on this machine: download the TFDS release, convert it into per-episode files with
# DriftWorld's converter, check it, then cache the SD3-VAE latents of every frame (what the trainer reads).
#
#   bash scripts/rae/wm/data/prepare.sh bridge|rt1 [first step: download|convert|finalize|cache|merge]
#
# Once before: bash scripts/rae/wm/data/fetch_driftworld.sh; bash scripts/rae/wm/data/setup_convert_env.sh;
# `bash download.sh wm` (the SD3 VAE, a gated model: accept its license on huggingface.co first).
# Every step can be rerun: the download resumes, the conversion skips finished episodes.
#
# env: CONVERT_JOBS  parallel conversion processes, 4 cores each (default: cores / 4, at most 8)
#      GPUS          GPUs for the latent cache, e.g. "0 1 2 3" (default: every visible GPU)
#      CLEAN_PARTS=1 delete the per-GPU latent parts after the merge (they double the disk)
#      SCAN=0        skip decoding every mp4 in the check
#
# Disk (Bridge-V2 / RT-1): raw 133 / 119 GB, episodes 4 / 15 GB, latents 74 / 124 GB (twice that until merged).
# Time with 20 cores and 4 H200s: Bridge-V2 about 1.5 h download, 1 h conversion, 25 min cache; RT-1 40 min, 2 h, 40 min.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
source "$REPO/scripts/rae/wm/data/common.sh"
DS=${1:?usage: prepare.sh bridge|rt1 [download|convert|finalize|cache|merge]}; FIRST=${2:-download}
dataset_paths "$DS" || exit 1
LOGS=$STORE/data/$DS/logs; mkdir -p "$LOGS"
NPROC=$(nproc)

download() {
  mkdir -p "$BUILDER"
  local list=$LOGS/download_files.txt min
  if [[ $DS == bridge ]]; then
    local base=https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0
    curl -sf "$base/" | grep -o 'href="[^"]*"' | sed 's/href="//; s/"$//' | grep -v -E '^\?|^/|/$' | sort -u \
      | sed "s#^#$base/#" > "$list"
    min=1158
  else   # Open X-Embodiment's public bucket gs://gresearch/robotics, over HTTPS
    python3 -I - > "$list" <<'PY'
import json, urllib.request
api = ("https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/fractal20220817_data/0.1.0/"
       "&maxResults=1000&fields=items(name),nextPageToken")
names, tok = [], None
while True:
    d = json.load(urllib.request.urlopen(api + (f"&pageToken={tok}" if tok else ""), timeout=60))
    names += [it["name"] for it in d.get("items", []) if not it["name"].endswith("/")]
    tok = d.get("nextPageToken")
    if not tok:
        break
for n in sorted(names):
    print("https://storage.googleapis.com/gresearch/" + n)
PY
    min=1026
  fi
  local total; total=$(wc -l < "$list")
  (( total >= min )) || { echo "file listing too short ($total < $min): server unreachable or changed" >&2; return 1; }
  echo "$total files -> $BUILDER"
  local left=1
  for pass in 1 2 3; do
    if command -v aria2c >/dev/null; then
      (cd "$BUILDER" && aria2c -q -c -j 8 -x 4 -s 4 --file-allocation=none --auto-file-renaming=false --allow-overwrite=false -i "$list")
    else
      (cd "$BUILDER" && xargs -P 8 -n 1 wget -q -c < "$list")
    fi
    left=0
    while read -r url; do f=$BUILDER/$(basename "$url"); [[ -s $f && ! -e $f.aria2 ]] || left=$((left + 1)); done < "$list"
    echo "pass $pass: $left files incomplete"
    (( left == 0 )) && break
  done
  (( left == 0 )) || { echo "download incomplete: rerun to resume" >&2; return 1; }
}

convert() {
  check_driftworld || return 1
  [[ -x $PY_CONVERT ]] || { echo "no conversion env at $CONVERT_ENV: run scripts/rae/wm/data/setup_convert_env.sh" >&2; return 1; }
  "$PY_CONVERT" -I "$DATA_SCRIPTS/check_data.py" raw "$BUILDER" || { echo "raw release incomplete: rerun the download step" >&2; return 1; }
  mkdir -p "$(dirname "$PROCESSED")"
  local n=${CONVERT_JOBS:-$(( NPROC / 4 > 8 ? 8 : (NPROC / 4 > 0 ? NPROC / 4 : 1) ))} dir args
  if [[ $DS == bridge ]]; then
    dir=$DW/bridge_driftworld/datasets; args=(--dataset_name bridge_v2 --dataset_home "$BUILDER" --fps 10 --splits train val)
  else
    dir=$DW/rt1_driftworld/datasets; args=(--dataset_name rt_1 --dataset_home "$RAW" --fps 3 --val_holdout_frac 0.025 --val_seed 314)
  fi
  echo "converting in $n processes (logs: $LOGS/convert_*.log)"
  local pids=() i pin
  for ((i = 0; i < n; i++)); do
    # 4 cores per process, as in the paper's conversion (the mp4 bytes depend on the core count libx264 sees)
    pin=(); (( NPROC >= 4 * n )) && command -v taskset >/dev/null && pin=(taskset -c "$((4 * i))-$((4 * i + 3))")
    (cd "$dir" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TF_CPP_MIN_LOG_LEVEL=2 \
      "${pin[@]}" "$PY_CONVERT" download_data.py "${args[@]}" --output_dir "$(dirname "$PROCESSED")" --num_shards "$n" --shard_id "$i") \
      > "$LOGS/convert_$i.log" 2>&1 &
    pids+=($!)
  done
  local fail=0 p; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  (( fail == 0 )) || { echo "a conversion process failed: see $LOGS/convert_*.log (rerun to resume)" >&2; return 1; }
}

finalize() {
  check_driftworld || return 1
  local dir args
  if [[ $DS == bridge ]]; then
    dir=$DW/bridge_driftworld/datasets; args=(--dataset_name bridge_v2 --dataset_home "$BUILDER" --splits train val)
  else
    dir=$DW/rt1_driftworld/datasets; args=(--dataset_name rt_1 --dataset_home "$RAW" --val_holdout_frac 0.025 --val_seed 314)
  fi
  (cd "$dir" && CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 "$PY_CONVERT" download_data.py "${args[@]}" \
    --output_dir "$(dirname "$PROCESSED")" --finalize_manifest) > "$LOGS/finalize.log" 2>&1 \
    || { echo "manifest step failed: see $LOGS/finalize.log" >&2; return 1; }
  # episode ids, files, counts and the manifests' sha256 against the paper's conversion (+ decode every mp4 unless SCAN=0)
  local scan=(); [[ ${SCAN:-1} == 1 ]] && scan=(--scan --workers "$NPROC")
  "$PY_CONVERT" -I "$DATA_SCRIPTS/check_data.py" processed "$PROCESSED" --dataset "$DS" "${scan[@]}" || return 1
  # the loader's episode index, built once here so the cache processes don't race to write it
  rm -f "$PROCESSED/train/_index.json" "$PROCESSED/val/_index.json"
  "$PY_MAIN" - "$PROCESSED" "$ACTION_DIM" <<'PY'
import sys
from sbm_rae.bridge_data import BridgeClips
for split in ("train", "val"):
    c = BridgeClips(sys.argv[1], split, n_frames=1, action_dim=int(sys.argv[2]))
    print(f"{split}: {len(c)} episodes, {sum(c.lengths)} frames")
PY
}

cache() {
  local gpus=${GPUS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L | wc -l) - 1 )))} dev
  read -ra dev <<< "$gpus"
  local n=${#dev[@]} pids=() i
  (( n > 0 )) || { echo "no GPU visible" >&2; return 1; }
  # SD3-VAE posterior sample x 1.5305 of every frame (fp32 encode, fp16 storage); RT-1 frames resized to 256x256 first
  if [[ -n $FRAME_SIZE ]]; then export BRIDGE_FRAME_SIZE=$FRAME_SIZE; else unset BRIDGE_FRAME_SIZE; fi
  echo "caching latents on $n GPUs (logs: $LOGS/cache_*.log)"
  for i in "${!dev[@]}"; do
    CUDA_VISIBLE_DEVICES=${dev[$i]} SBM_STORE=$STORE HF_HOME=$STORE/rae/assets/hf_home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
      "$PY_MAIN" "$REPO/scripts/rae/cache_bridge_latents.py" --processed "$PROCESSED" --out "$LATENTS" --codec sd3 \
      --split val train --action-dim "$ACTION_DIM" --workers 6 --batch 64 --shard "$n" --shard-id "$i" > "$LOGS/cache_$i.log" 2>&1 &
    pids+=($!)
  done
  local fail=0 p; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  (( fail == 0 )) || { echo "a cache process failed: see $LOGS/cache_*.log" >&2; return 1; }
}

merge() {
  local s parts runs nn k t ext
  for s in val train; do   # the merge takes every part it finds: the parts must come from one run
    mapfile -t parts < <(cd "$LATENTS/$s" 2>/dev/null && ls part*.index.json 2>/dev/null)
    (( ${#parts[@]} > 0 )) || { echo "$s: no latent parts under $LATENTS/$s (run the cache step)" >&2; return 1; }
    mapfile -t runs < <(printf '%s\n' "${parts[@]}" | sed -E 's/^part[0-9]+of([0-9]+)\..*/\1/' | sort -u)
    (( ${#runs[@]} == 1 )) || { echo "$s: parts of runs with different GPU counts (${runs[*]}): delete the stale ones" >&2; return 1; }
    nn=${runs[0]}
    for ((k = 0; k < 10#$nn; k++)); do
      t=$(printf 'part%02dof%s' "$k" "$nn")
      for ext in index.json latents.npy actions.npy; do
        [[ -f $LATENTS/$s/$t.$ext ]] || { echo "$s: $t.$ext missing (cache process $k unfinished?)" >&2; return 1; }
      done
    done
  done
  CUDA_VISIBLE_DEVICES="" "$PY_MAIN" "$REPO/scripts/rae/cache_bridge_latents.py" --out "$LATENTS" --split val train --merge --processed unused || return 1
  "$PY_MAIN" -I "$DATA_SCRIPTS/check_data.py" latents "$LATENTS" --processed "$PROCESSED" --dataset "$DS" || return 1
  if [[ ${CLEAN_PARTS:-0} == 1 ]]; then rm -f "$LATENTS"/{train,val}/part*; echo "latent parts deleted"; fi
  du -sh "$LATENTS"
}

steps=(download convert finalize cache merge)
start=-1
for k in "${!steps[@]}"; do [[ ${steps[$k]} == "$FIRST" ]] && start=$k; done
(( start >= 0 )) || { echo "unknown step '$FIRST' (download|convert|finalize|cache|merge)" >&2; exit 1; }
for ((k = start; k < ${#steps[@]}; k++)); do
  echo "== $DS: ${steps[$k]} ($(date +%T))"
  "${steps[$k]}" || { echo "$DS: ${steps[$k]} failed; fix and rerun: bash $0 $DS ${steps[$k]}" >&2; exit 1; }
done
echo "== $DS ready: episodes $PROCESSED, latents $LATENTS"
