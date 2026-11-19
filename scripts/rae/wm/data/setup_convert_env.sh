#!/bin/bash
# Build the conversion environment (requirements_convert.txt: Python 3.12, tensorflow, tensorflow-datasets, decord2,
# imageio-ffmpeg) with uv at ${CONVERT_ENV:-<store>/envs/wm_convert}; CPU only, ~2 GB, a few minutes:
#   bash scripts/rae/wm/data/setup_convert_env.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
source "$REPO/scripts/rae/wm/data/common.sh"

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }
[[ -x $PY_CONVERT ]] || uv venv -q --python 3.12 "$CONVERT_ENV"
uv pip install -q --python "$PY_CONVERT" -r "$DATA_SCRIPTS/requirements_convert.txt"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 "$PY_CONVERT" -c '
import tensorflow as tf, tensorflow_datasets as tfds, decord, imageio, imageio_ffmpeg
print(f"conversion env OK: tensorflow {tf.__version__}, tfds {tfds.__version__}, imageio {imageio.__version__}, "
      f"ffmpeg {imageio_ffmpeg.get_ffmpeg_version()}, decord {decord.__version__}")'
echo "python: $PY_CONVERT"
