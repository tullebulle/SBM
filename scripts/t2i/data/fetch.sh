#!/bin/bash
# Download every row "size<TAB>url<TAB>destination" of a manifest (scripts/t2i/data/manifest_*.tsv) with curl, resuming
# partial files (huggingface_hub's file locks hang on some network filesystems), then check every size. Relative
# destinations land in the store ($SBM_STORE, default <repo>/store). Re-run to resume.
#   bash scripts/t2i/data/fetch.sh scripts/t2i/data/manifest_eval_assets.tsv [parallel=8]
# manifest_flan_t5 (the frozen text encoder, 3 GB) and manifest_eval_assets (COCO captions + FID statistics, CLIP B/32,
# GenEval's prompts) are what sampling and the training monitor read; manifest_cc12m (CC12M, 1.1 TB, + the LLaVA-NeXT
# recaptions) and manifest_mix120k (the 120k alignment mix) are the training data (then join_cc12m.py, prepare_mix.py,
# reshard_mix.py).
set -uo pipefail
umask 002
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export STORE=${SBM_STORE:-$REPO/store}
MANIFEST=${1:?usage: fetch.sh <manifest.tsv> [parallel]}; PAR=${2:-8}
get() {
  size=$1; url=$2; dst=$3; [[ $dst = /* ]] || dst=$STORE/$dst
  [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ] && return 0
  mkdir -p "$(dirname "$dst")"
  for try in 1 2 3 4 5; do
    curl -sSL --retry 5 --retry-delay 10 -C - -o "$dst" "$url" && [ "$(stat -c %s "$dst")" = "$size" ] && { echo "ok $dst"; return 0; }
    [ "$(stat -c %s "$dst" 2>/dev/null || echo 0)" -gt "$size" ] && rm -f "$dst"   # a stale append: start over
    sleep 20
  done
  echo "FAIL $dst"; return 1
}
export -f get
tr '\t' ' ' < "$MANIFEST" | xargs -P "$PAR" -L 1 bash -c 'get "$0" "$1" "$2"'
bad=0
while IFS=$'\t' read -r size url dst; do
  [[ $dst = /* ]] || dst=$STORE/$dst
  [ "$(stat -c %s "$dst" 2>/dev/null)" = "$size" ] || { echo "MISSING $dst"; bad=$((bad + 1)); }
done < "$MANIFEST"
echo "done: $(wc -l < "$MANIFEST") files, $bad missing"
[ "$bad" = 0 ]
