"""Where the sbm_rae code finds its data: one resolution point, the SBM_STORE environment variable.

SBM_STORE (default: <repo>/store) holds
    rae/assets/rae/         the RAE decoder (decoders/dinov2/wReg_<size>/<decoder>/model.pt), its config and statistics
    rae/assets/hf_home/     the HF cache with the DINOv2-with-registers encoder and the SD-VAE (offline, HF_HUB_OFFLINE=1)
    rae/assets/torch_home/  the torch hub cache (Inception weights for FID, DINOv2 hub code)
    ckpt/                   the released checkpoints (imagenet/, t2i/, wm/; download.sh)
    ckpt/imagenet/fid_cache/  the ADM reference and ImageNet-val Inception statistics (with the checkpoints)
    data/                   the datasets (bridge / rt1 processed clips, ImageNet latent packs)
The Hydra configs reach the same root through the ${sbm_store:<relative path>} resolver (registered below). Environment overrides for single pieces:
SBM_RAE_SRC (the RAE source tree), SBM_RAE_ASSETS, SBM_FID_CACHE.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STORE = REPO / "store"


def store_root() -> Path:
    return Path(os.environ.get("SBM_STORE") or str(DEFAULT_STORE))


def rae_assets_dir() -> Path:
    """The RAE codec assets (decoder_config_ViTXL/, decoders/, stats/)."""
    return Path(os.environ.get("SBM_RAE_ASSETS") or str(store_root() / "rae" / "assets" / "rae"))


def fid_cache_dir() -> Path:
    """Cached real-image Inception statistics (eval_fid.py) and the ADM reference statistics."""
    return Path(os.environ.get("SBM_FID_CACHE")
                or str(store_root() / "ckpt" / "imagenet" / "fid_cache"))


def rae_src_dir() -> Path:
    """The RAE source tree (github.com/bytetriper/RAE), vendored under third_party/RAE/src."""
    if os.environ.get("SBM_RAE_SRC"):
        return Path(os.environ["SBM_RAE_SRC"])
    return REPO / "third_party" / "RAE" / "src"


def register_config_resolver():
    """${sbm_store:rel/path} -> the absolute path under store_root(), read when the value is accessed (so SBM_STORE set
    after composition still applies). Registered on `import sbm_rae`."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("sbm_store"):
        OmegaConf.register_new_resolver("sbm_store", lambda rel="": str(store_root() / str(rel)))
