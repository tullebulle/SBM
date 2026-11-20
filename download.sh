#!/bin/bash
# Download the SBM checkpoints (Hugging Face: sbm-models/sbm), and the frozen networks they need, into ./store (or $SBM_STORE).
#
#   bash download.sh            # everything (~20 GB)
#   bash download.sh imagenet   # ImageNet 256: SBM-XL / SBM-B in the RAE and SD-VAE latents (+ FID reference statistics)
#   bash download.sh t2i        # text-to-image: SBM-XL, SBM-B (+ FLAN-T5-Large)
#   bash download.sh wm         # world models: Bridge-V2, RT-1 (+ example episodes, the SD3 VAE and the evaluation networks)
#
# The checkpoints are in a private repo: ask for access, then `hf auth login`. The world models need the SD3 VAE, a gated
# model: accept its license at https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python HF=.venv/bin/hf
[[ -x $PY ]] || { echo "no .venv: run 'uv sync' (and 'uv sync --extra wm' for the world models) first" >&2; exit 1; }
mkdir -p "${SBM_STORE:-store}"
export SBM_STORE=$(cd "${SBM_STORE:-store}" && pwd)

checkpoints() { "$HF" download sbm-models/sbm --include "$1/*" --local-dir "$SBM_STORE/ckpt"; }   # -> <store>/ckpt/<task>/

imagenet() {
  checkpoints imagenet
  "$PY" scripts/rae/fetch_assets.py --sdvae      # RAE decoder + DINOv2 encoder, SD-VAE, FID Inception
}
t2i() {
  checkpoints t2i
  "$PY" scripts/rae/fetch_assets.py              # RAE decoder (the text-to-image maps share the ImageNet latent)
  bash scripts/t2i/data/fetch.sh scripts/t2i/data/manifest_flan_t5.tsv
}
has_wm() { "$PY" -c "import decord, lpips" 2>/dev/null; }
wm() {
  has_wm || { echo "world models: run 'uv sync --extra wm' first" >&2; exit 1; }
  checkpoints wm                                 # + wm/examples/: four validation episodes for sample.py
  "$PY" scripts/rae/fetch_assets.py --wm         # SD3 VAE (gated), FID / LPIPS / FVD networks, stylegan-v's FVD code
}

case "${1:-all}" in
  imagenet) imagenet ;;
  t2i) t2i ;;
  wm) wm ;;
  all) imagenet; t2i
       if has_wm; then wm; else echo "world models skipped: run 'uv sync --extra wm', then 'bash download.sh wm'"; fi ;;
  *) echo "usage: bash download.sh [imagenet|t2i|wm|all]" >&2; exit 1 ;;
esac
echo "done: checkpoints in $SBM_STORE/ckpt/"
