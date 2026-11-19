# Sourced by the world models' data scripts (scripts/rae/wm/data/*): the store layout and the pinned sources.
# The caller sets REPO (the repo root). Data lives under STORE=${SBM_STORE:-<repo>/store}, where configs/rae/dataset/
# {bridge,rt1}.yaml read it (${sbm_store:data/...}).

STORE=${SBM_STORE:-$REPO/store}
DATA_SCRIPTS=$REPO/scripts/rae/wm/data

# DriftWorld (github.com/Susie-Lu/driftworld): its TFDS -> mp4 / npz / txt converters. Cloned at the pinned commit, not vendored.
DW_URL=https://github.com/Susie-Lu/driftworld
DW_COMMIT=e24d385f738d0501a21156c5e62992e5e649e3e8
DW=${DRIFTWORLD_DIR:-$STORE/refs/code/driftworld}
DW_PATCH=$DATA_SCRIPTS/driftworld_rt1_sharding.patch
# sha256 of the two converters as used: Bridge unchanged, RT-1 with driftworld_rt1_sharding.patch applied
DW_BRIDGE_SHA256=6aace4a0aa5e26ebcc744fac352b5187dd6abc13575ea2f7974675bda560a10f
DW_RT1_SHA256=57cb5a503b9f40487534ecbc19d9ddc76730720edbef9641ee8db0bc3823b4a4

# the conversion environment (requirements_convert.txt: tensorflow / tensorflow-datasets; not in the repo's venv)
CONVERT_ENV=${CONVERT_ENV:-$STORE/envs/wm_convert}
PY_CONVERT=$CONVERT_ENV/bin/python
# the repo's venv (`uv sync --frozen --extra wm`): the latent cache and the loader's episode index
PY_MAIN=${SBM_PYTHON:-$REPO/.venv/bin/python}

# dataset_paths bridge|rt1 -> RAW (download root), BUILDER (the TFDS builder dir), PROCESSED, LATENTS, ACTION_DIM, FRAME_SIZE
dataset_paths() {
  case "${1:-}" in
    bridge)
      RAW=$STORE/data/bridge/raw
      BUILDER=$RAW/rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0
      PROCESSED=$STORE/data/bridge/processed/bridge_v2
      LATENTS=$STORE/data/bridge/latents_sd3
      ACTION_DIM=7 FRAME_SIZE= ;;
    rt1)
      RAW=$STORE/data/rt1/raw_tfds
      BUILDER=$RAW/fractal20220817_data/0.1.0
      PROCESSED=$STORE/data/rt1/processed/rt_1
      LATENTS=$STORE/data/rt1/latents_sd3
      ACTION_DIM=10 FRAME_SIZE=256 ;;   # RT-1 frames are stored at 256x320; the loader resizes them (BRIDGE_FRAME_SIZE)
    *) echo "dataset must be 'bridge' or 'rt1', got '${1:-}'" >&2; return 1 ;;
  esac
}

# check_driftworld: the clone is at the pinned commit with the RT-1 patch applied (fetch_driftworld.sh)
check_driftworld() {
  local ok=1
  echo "$DW_BRIDGE_SHA256  $DW/bridge_driftworld/datasets/download_data.py" | sha256sum -c --quiet - 2>/dev/null || ok=0
  echo "$DW_RT1_SHA256  $DW/rt1_driftworld/datasets/download_data.py" | sha256sum -c --quiet - 2>/dev/null || ok=0
  [[ $ok == 1 ]] || { echo "DriftWorld's converters missing or not as pinned under $DW: run scripts/rae/wm/data/fetch_driftworld.sh" >&2; return 1; }
}
