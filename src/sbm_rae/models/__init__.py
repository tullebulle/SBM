from omegaconf import OmegaConf

from sbm_rae.models.dit import DiT, DiT_B_2, DiT_XL_2, DiT_models as _DiT_models
from sbm_rae.models.bridge_unet import Bridge_models

DiT_models = {**_DiT_models, **Bridge_models}


def build_model(cfg):
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    name = model_cfg.pop("name")

    try:
        model_builder = DiT_models[name]
    except KeyError as exc:
        available = ", ".join(sorted(DiT_models))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc

    return model_builder(**model_cfg)


__all__ = [
    "DiT",
    "DiT_B_2",
    "DiT_XL_2",
    "DiT_models",
    "build_model",
]
