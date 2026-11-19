#!/bin/bash
# Clone DriftWorld's code (github.com/Susie-Lu/driftworld; its converters produce the processed layout the world models read)
# at the pinned commit into <store>/refs/code/driftworld and apply driftworld_rt1_sharding.patch to its RT-1 converter:
# --num_shards / --shard_id / --finalize_manifest as in their Bridge converter (shard k converts the global episode range
# [k*N/K, (k+1)*N/K) with the train / val counters set to their values at the block start, so the ids and the seeded
# validation hold-out equal a single-process run), and an mp4 under 1 KB (left by a job killed mid-write) counts as
# missing when a run resumes. With the default --num_shards 1 the patched script behaves as their original.
# Idempotent; a ~1 MB clone. DriftWorld's repository carries no license file, so its code is fetched, not included here.
#
#   bash scripts/rae/wm/data/fetch_driftworld.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
source "$REPO/scripts/rae/wm/data/common.sh"

if [[ ! -d $DW/.git ]]; then
  mkdir -p "$(dirname "$DW")"
  git clone -q "$DW_URL" "$DW"
fi
if [[ "$(git -C "$DW" rev-parse HEAD)" != "$DW_COMMIT" ]]; then
  git -C "$DW" apply -R "$DW_PATCH" 2>/dev/null || true          # an earlier patched checkout
  git -C "$DW" cat-file -e "$DW_COMMIT^{commit}" 2>/dev/null || git -C "$DW" fetch -q origin
  git -C "$DW" checkout -q "$DW_COMMIT"
fi
if ! git -C "$DW" apply -R --check "$DW_PATCH" 2>/dev/null; then  # not applied yet
  git -C "$DW" apply "$DW_PATCH"
fi
check_driftworld
echo "DriftWorld at $DW: $(git -C "$DW" rev-parse --short HEAD), RT-1 converter patched; converters match the pinned sha256"
