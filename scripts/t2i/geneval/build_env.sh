#!/bin/bash
# Builds the GenEval scoring stack under ROOT ($GENEVAL_ROOT, default <store>/t2i/geneval; store = $SBM_STORE or <repo>/store).
#   geneval/      djghosh13/geneval (prompts/evaluation_metadata.jsonl, evaluation/*.py)
#   mmdetection/  open-mmlab/mmdetection branch 2.x, editable install (evaluate_images.py reads its configs/)
#   models/       Mask2Former Swin-S COCO checkpoint (evaluation/download_models.sh)
#   home/.cache/clip/ViT-L-14.pt  open_clip "openai" ViT-L/14 (score.sh sets HOME=ROOT/home)
#   env/          uv venv: python 3.10, torch 2.1.2+cu121, mmcv-full 1.7.2 (CUDA ops sm_80/sm_90: build_mmcv.sh), mmdet 2.x,
#                 open-clip-torch 2.20.0, clip-benchmark 1.4.0 (as geneval's environment.yml, which pins torch 1.12 / cu113)
# Idempotent: every step skips what is already in place. About 20 minutes with 16 cores. Order:
#   bash scripts/t2i/geneval/build_env.sh && bash scripts/t2i/geneval/build_mmcv.sh   (the second needs nvcc, CUDA 12.x)
# env_freeze.txt here is the environment the released GenEval numbers were scored in.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${GENEVAL_ROOT:-${SBM_STORE:-$REPO/store}/t2i/geneval}
mkdir -p "$ROOT" && cd "$ROOT"
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv_cache_geneval_$$
export UV_PYTHON_INSTALL_DIR=$ROOT/python
export UV_LINK_MODE=copy
TMP=${TMPDIR:-/tmp}/geneval_build_$$
mkdir -p "$TMP"
trap 'rm -rf "$TMP" "$UV_CACHE_DIR"' EXIT

echo "[build] $(date) on $(hostname)"
[ -d geneval/.git ] || { git clone https://github.com/djghosh13/geneval.git && git -C geneval checkout af4902f24d3ca90ebbb446dd9891a59e0f82725f; }
[ -d mmdetection/.git ] || { git clone -b 2.x https://github.com/open-mmlab/mmdetection.git && git -C mmdetection checkout e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1; }
echo "[build] geneval $(git -C geneval rev-parse --short HEAD), mmdetection $(git -C mmdetection rev-parse --short HEAD) ($(git -C mmdetection describe --tags 2>/dev/null || true))"
wc -l geneval/prompts/evaluation_metadata.jsonl

DET=models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
if [ ! -s "$DET" ]; then
  bash geneval/evaluation/download_models.sh "$TMP/det"
  mkdir -p models && cp "$TMP/det/$(basename $DET)" "$DET"
fi
CLIPW=home/.cache/clip/ViT-L-14.pt
if [ ! -s "$CLIPW" ]; then
  curl -sSL -o "$TMP/ViT-L-14.pt" https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
  echo "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  $TMP/ViT-L-14.pt" | sha256sum -c
  mkdir -p "$(dirname $CLIPW)" && cp "$TMP/ViT-L-14.pt" "$CLIPW"
fi
ls -la models "$(dirname $CLIPW)"

if [ ! -x env/bin/python ]; then
  uv python install 3.10
  uv venv --python 3.10 env
fi
source env/bin/activate
cat > "$TMP/constraints.txt" <<EOF
torch==2.1.2+cu121
torchvision==0.16.2+cu121
numpy==1.23.5
EOF
uv pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match torch==2.1.2+cu121 torchvision==0.16.2+cu121 numpy==1.23.5
# mmcv-full: OpenMMLab's cu121/torch2.1 wheel has CUDA ops for sm_50..sm_86 only (A100, not H100/H200); build_mmcv.sh
# builds one for sm_80 + sm_90 into wheels/, used here once it exists
MMCV=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl
[ -f wheels/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl ] && MMCV=$ROOT/wheels/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl
uv pip install -c "$TMP/constraints.txt" --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match \
  "$MMCV" \
  addict yapf==0.33.0 opencv-python-headless==4.7.0.72 "setuptools<70" wheel pandas==1.5.3 "scipy<1.12" "matplotlib<3.8" \
  pycocotools terminaltables six tqdm scikit-learn==1.3.2 open-clip-torch==2.20.0 timm==0.9.16 "protobuf<4" ftfy regex \
  sentencepiece "huggingface-hub<1"
# the mmcv-full wheel pulls in opencv-python (GUI build) next to the headless one; keep only the headless one
uv pip uninstall opencv-python opencv-python-headless
uv pip install -c "$TMP/constraints.txt" opencv-python-headless==4.7.0.72
uv pip install -c "$TMP/constraints.txt" --no-deps clip-benchmark==1.4.0
uv pip install -c "$TMP/constraints.txt" --no-build-isolation -e mmdetection
uv pip freeze > env_freeze.txt

python - <<'EOF'
import torch, mmcv, mmdet, open_clip, numpy, pandas, cv2, clip_benchmark
from importlib.metadata import version
from mmcv.ops import MultiScaleDeformableAttention
import mmcv.ops, os, glob
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list())
print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "open_clip", version("open_clip_torch"), "cv2", cv2.__version__, "numpy", numpy.__version__, "pandas", pandas.__version__)
from mmcv.utils import collect_env
print({k: v for k, v in collect_env().items() if "CUDA" in k or "MMCV" in k or "Compiler" in k})
print("mmcv _ext:", glob.glob(os.path.join(os.path.dirname(mmcv.__file__), "_ext*.so")))
EOF
EXT=$(ls env/lib/python3.10/site-packages/mmcv/_ext*.so | head -1)
echo "[build] cuda archs in $EXT:"
cuobjdump --list-elf "$EXT" 2>/dev/null | grep -o 'sm_[0-9]*' | sort | uniq -c || true
cuobjdump --list-ptx "$EXT" 2>/dev/null | grep -o 'sm_[0-9]*' | sort | uniq -c || true
echo "[build] done $(date)"
