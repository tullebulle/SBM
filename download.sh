#!/bin/bash
# Download the SBM checkpoints (Hugging Face: sbm-models/sbm) with the frozen networks they need, and the training data,
# into ./store (or $SBM_STORE).
#
#   bash download.sh              # every checkpoint (~20 GB)
#   bash download.sh imagenet     # ImageNet 256: SBM-XL / SBM-B in the RAE and SD-VAE latents (+ FID reference statistics)
#   bash download.sh t2i          # text-to-image: SBM-XL, SBM-B (+ FLAN-T5-Large)
#   bash download.sh wm           # world models: Bridge-V2, RT-1 (+ example episodes, the SD3 VAE and the evaluation networks)
#
# Training data (only needed to train; every step resumes when re-run):
#   bash download.sh imagenet-data     # ImageNet-1k, 154 GB -> store/data/imagenet_1k/{train,val}/<class>/*.JPEG
#   bash download.sh t2i-data          # CC12M + recaptions (1.2 TB) and the 120k alignment mix (0.25 TB), ready to train on
#   bash download.sh wm-data bridge    # or rt1: download (~130 GB), convert, encode with the SD3 VAE (GPUs); a few hours
#
# Access, on huggingface.co (then `hf auth login`): the checkpoints are in a private repo (ask for access); ImageNet-1k
# (datasets/ILSVRC/imagenet-1k) and the SD3 VAE of the world models (stabilityai/stable-diffusion-3-medium-diffusers) are
# gated: accept their terms on those pages first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python HF=.venv/bin/hf
[[ -x $PY ]] || { echo "no .venv: run 'uv sync' (and 'uv sync --extra wm' for the world models) first" >&2; exit 1; }
mkdir -p "${SBM_STORE:-store}"
export SBM_STORE=$(cd "${SBM_STORE:-store}" && pwd)

checkpoints() { "$HF" download sbm-models/sbm --include "$1/*" --local-dir "$SBM_STORE/ckpt"; }   # -> <store>/ckpt/<task>/
has_wm() { "$PY" -c "import decord, lpips" 2>/dev/null; }
need_wm() { has_wm || { echo "world models: run 'uv sync --extra wm' first" >&2; exit 1; }; }

imagenet() {
  checkpoints imagenet
  "$PY" scripts/rae/fetch_assets.py --sdvae      # RAE decoder + DINOv2 encoder, SD-VAE, FID Inception
}
t2i() {
  checkpoints t2i
  "$PY" scripts/rae/fetch_assets.py              # RAE decoder (the text-to-image maps share the ImageNet latent)
  bash scripts/t2i/data/fetch.sh scripts/t2i/data/manifest_flan_t5.tsv
}
wm() {
  need_wm
  checkpoints wm                                 # + wm/examples/: four validation episodes for sample.py
  "$PY" scripts/rae/fetch_assets.py --wm         # SD3 VAE (gated), FID / LPIPS / FVD networks, stylegan-v's FVD code
}

imagenet_data() {
  "$PY" scripts/rae/fetch_assets.py --sdvae      # the encoders the trainers use
  "$PY" scripts/fetch_imagenet.py                # -> <store>/data/imagenet_1k
}
t2i_data() {
  local d=scripts/t2i/data
  "$PY" scripts/rae/fetch_assets.py
  bash $d/fetch.sh $d/manifest_flan_t5.tsv
  bash $d/fetch.sh $d/manifest_eval_assets.tsv   # COCO captions + FID statistics, CLIP: the training monitor
  bash $d/fetch.sh $d/manifest_cc12m.tsv         # CC12M (webdataset tars) and the LLaVA-NeXT recaptions
  "$PY" $d/join_cc12m.py                         # -> <store>/t2i/data/shards/cc12m_captions
  bash $d/fetch.sh $d/manifest_mix120k.tsv       # BLIP3o-60k, the DALL-E 3 dataset, ShareGPT-4o-Image
  "$PY" $d/prepare_mix.py                        # -> <store>/t2i/data/shards/mix120k
  "$PY" $d/reshard_mix.py "$SBM_STORE/t2i/data/shards/mix120k" "$SBM_STORE/t2i/data/shards/mix120k_s256"
}
wm_data() {
  need_wm
  "$PY" scripts/rae/fetch_assets.py --wm         # the SD3 VAE encodes the frames
  bash scripts/rae/wm/data/fetch_driftworld.sh   # DriftWorld's converter
  bash scripts/rae/wm/data/setup_convert_env.sh  # its environment (tensorflow-datasets), CPU only
  bash scripts/rae/wm/data/prepare.sh "$1"       # download, convert, check, encode -> <store>/data/<dataset>
}

case "${1:-all}" in
  imagenet) imagenet ;;
  t2i) t2i ;;
  wm) wm ;;
  all) imagenet; t2i
       if has_wm; then wm; else echo "world models skipped: run 'uv sync --extra wm', then 'bash download.sh wm'"; fi ;;
  imagenet-data) imagenet_data; exit ;;
  t2i-data) t2i_data; echo "done: text-to-image training data in $SBM_STORE/t2i/data"; exit ;;
  wm-data) [[ ${2:-} == bridge || ${2:-} == rt1 ]] || { echo "usage: bash download.sh wm-data bridge|rt1" >&2; exit 1; }
           wm_data "$2"; exit ;;
  *) echo "usage: bash download.sh [imagenet|t2i|wm|all|imagenet-data|t2i-data|wm-data bridge|rt1]" >&2; exit 1 ;;
esac
echo "done: checkpoints in $SBM_STORE/ckpt/"
