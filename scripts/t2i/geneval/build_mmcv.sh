#!/bin/bash
# mmcv-full 1.7.2 from source for env/ (torch 2.1.2+cu121) with CUDA ops for sm_80 and sm_90: OpenMMLab's cu121 wheel
# carries sm_50..sm_86 only (no PTX), so its ops (Mask2Former's MultiScaleDeformableAttention) cannot run on H100/H200.
# Writes wheels/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl and installs it into env/ in place of the OpenMMLab wheel.
# Needs nvcc (CUDA 12.x) on PATH and a C++ compiler; MAX_JOBS (default 32) parallel compile jobs, about an hour.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${GENEVAL_ROOT:-${SBM_STORE:-$REPO/store}/t2i/geneval}
cd "$ROOT"
command -v nvcc >/dev/null || { echo "[mmcv] nvcc not found: put CUDA 12.x on PATH" >&2; exit 2; }
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
TMP=${TMPDIR:-/tmp}/geneval_mmcv_$$
mkdir -p "$TMP" wheels
trap 'rm -rf "$TMP"' EXIT
echo "[mmcv] $(date) $(hostname) nvcc: $(nvcc --version | tail -2 | head -1) gcc: $(gcc --version | head -1)"
curl -sSL -o "$TMP/mmcv-full-1.7.2.tar.gz" https://files.pythonhosted.org/packages/source/m/mmcv-full/mmcv-full-1.7.2.tar.gz
tar -xzf "$TMP/mmcv-full-1.7.2.tar.gz" -C "$TMP"
cd "$TMP/mmcv-full-1.7.2"
UV_CACHE_DIR=$TMP/uvc uv pip install --python "$ROOT/env/bin/python" ninja   # parallel extension build
export MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=${MAX_JOBS:-32} PATH=$ROOT/env/bin:$PATH
"$ROOT/env/bin/python" -c "import torch, setuptools; print('torch', torch.__version__, 'setuptools', setuptools.__version__)"
"$ROOT/env/bin/python" setup.py bdist_wheel -d "$TMP/dist" 2>&1 | grep -vE "^(copying|creating|adding) " | tail -40
cp "$TMP"/dist/mmcv_full-1.7.2-*.whl "$ROOT/wheels/"
ls -la "$ROOT/wheels"
cd "$TMP" && mkdir -p x && cd x && unzip -q -o "$ROOT"/wheels/mmcv_full-1.7.2-*.whl 'mmcv/_ext*.so'
cuobjdump --list-elf mmcv/_ext*.so | grep -o 'sm_[0-9]*' | sort | uniq -c
UV_CACHE_DIR=$TMP/uvc uv pip install --python "$ROOT/env/bin/python" --reinstall --no-deps "$ROOT"/wheels/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl
UV_CACHE_DIR=$TMP/uvc uv pip freeze --python "$ROOT/env/bin/python" > "$ROOT/env_freeze.txt"
echo "[mmcv] installed: $(ls "$ROOT"/env/lib/python3.10/site-packages/mmcv/_ext*.so)"
cuobjdump --list-elf "$ROOT"/env/lib/python3.10/site-packages/mmcv/_ext*.so | grep -o 'sm_[0-9]*' | sort | uniq -c
echo "[mmcv] done $(date)"
