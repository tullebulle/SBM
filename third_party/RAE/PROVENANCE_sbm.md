# third_party/RAE

The RAE code (Zheng et al. 2025, "Diffusion Transformers with Representation Autoencoders", github.com/bytetriper/RAE),
vendored for `sbm_rae.rae_adapter` (the DINOv2-with-registers encoder + the released ViT-XL decoder behind the SD-VAE
interface the trainer and scorer use). Copied on 9 Nov 2026 from a checkout of the upstream repository (upstream commit
unrecorded): `src/`, `configs/`, `LICENSE`, `README.md`, `requirements.txt`, `environment.yml`, `pack_images.py`.
Not copied: `assets/` (two demo PNGs) and `.git`. Nothing was modified.

The code is imported by path (`sbm_rae.paths.rae_src_dir()`: this directory's `src/`, or `SBM_RAE_SRC` / `BTM_RAE_SRC`);
the weights live in the store, `$SBM_STORE/rae/assets/rae` (decoder, stats) and `$SBM_STORE/rae/assets/hf_home` (the
encoder, offline).
