"""One-step class-conditional sampling from an exported map of the RAE line.

The map takes Gaussian noise and a class label to a DINOv2-S representation latent (16 x 16 x 384) in one network
evaluation; the RAE decoder turns it into a 256 x 256 image.

    from sbm_rae.generate import load_generator, generate
    gen = load_generator("store/ckpt/imagenet/sbm_rae_xl_step1042500_ema0_9999.pt")
    images = generate(gen, labels=[207, 360, 387, 974], seed=0)     # uint8, (N, 3, 256, 256)

The read settings (anchor omega, class-label scale) default to the ones stored in the export's `meta["read"]`, i.e. the
settings behind the published FID. scripts/rae/eval_fid.py uses the same model setup.
"""
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf, open_dict

from sbm_rae.infer import load_slim_export, module_from_export
from sbm_rae.loss import get_loss_fn, resolve_guidance_config, resolve_signed_config
from sbm_rae.models import build_model
from sbm_rae.models.weight import LossWeightingNetwork
from sbm_rae.sampling import generate_image_batches


def _guided_cbm_scale_cond(cfg):
    """loss.guided_cbm.guide.scale_cond: the network takes the guidance scale w as an input."""
    loss_cfg = cfg.get("loss", None) or {}
    if loss_cfg.get("name") != "guided_cbm" or loss_cfg.get("guided_cbm", None) is None:
        return False
    from sbm_rae.guided_cbm import guidance_scale_cond

    return guidance_scale_cond(cfg.loss.guided_cbm)[0]


def default_w(cfg):
    """The guidance-scale input a scale-conditioned network is read at (the guide's w0); None for other networks."""
    if not _guided_cbm_scale_cond(cfg):
        return None
    from sbm_rae.guided_cbm import guidance_scale_cond

    return float(guidance_scale_cond(cfg.loss.guided_cbm)[3])


def configure_model_cfg(cfg):
    """Set the architecture switches that the training config implies but does not store under `model`: the guidance
    (w) embedding and the REPA projector. Without them the network would not match the checkpoint's tensors."""
    with open_dict(cfg):
        cfg.model.guidance_enabled = (
            resolve_guidance_config(cfg)["enabled"]
            or resolve_signed_config(cfg)["amortised"]["enabled"]
            or _guided_cbm_scale_cond(cfg)
        )
        repa_cfg = cfg.get("repa", {}) or {}
        enabled = bool(repa_cfg.get("enabled", False))
        guided_repa = (cfg.loss.get("guided_cbm", None) or {}).get("repa", None) or {}
        if not enabled and cfg.loss.get("name") == "guided_cbm" and bool(guided_repa.get("enabled", False)):
            cfg.model.repa_enabled = True
        else:
            cfg.model.repa_enabled = enabled
            if enabled:
                cfg.model.repa_encoder_depth = int(repa_cfg.get("encoder_depth", 8))
                cfg.model.repa_z_dim = int(repa_cfg.get("z_dim", 768))
                cfg.model.repa_projector_dim = int(repa_cfg.get("projector_dim", 2048))
    return cfg


def build_weighting_model(cfg):
    """The learned loss-weighting heads, when the run trained with them (they hold tensors of the checkpoint)."""
    weighting_cfg = cfg.get("weighting_model", None)
    if not weighting_cfg or not weighting_cfg.get("enabled", False):
        return None
    heads = ["transport", "endpoint"] + (["self_consistency"] if cfg.loss.get("self_consistency", False) else [])
    nets = {
        name: LossWeightingNetwork(
            channels=weighting_cfg.get("channels", 128),
            clamp_min=weighting_cfg.get("clamp_min", -10.0),
            clamp_max=weighting_cfg.get("clamp_max", 10.0),
        )
        for name in heads
        if bool(weighting_cfg.get(name, True))
    }
    return torch.nn.ModuleDict(nets) if nets else None


def apply_label_scale(module, label_scale):
    """Read-time guidance on the conditioning input: every class row e_y of the label table becomes
    e_null + s (e_y - e_null), the null class being the table's last row. None or 1 leaves the map unchanged."""
    if label_scale is None or float(label_scale) == 1.0:
        return
    table = module.model.y_embedder.embedding_table.weight
    num_classes = int(module.cfg.dataset.num_classes)
    if table.shape[0] != num_classes + 1:
        raise ValueError(f"a label scale needs a null-class row: the label table has {table.shape[0]} rows for "
                         f"{num_classes} classes")
    with torch.no_grad():
        null = table[num_classes:].clone()
        table[:num_classes] = null + float(label_scale) * (table[:num_classes] - null)


@dataclass
class Generator:
    module: torch.nn.Module
    cfg: object
    read: dict = field(default_factory=dict)
    w: float | None = None

    @property
    def device(self):
        return next(self.module.model.parameters()).device


def load_generator(path, device="cuda", label_scale=None):
    """The export at `path` as a ready-to-sample module (plain attention, eval mode, decoder loaded)."""
    torch.set_float32_matmul_precision("high")   # TF32 matmuls with bf16 autocast, the setting of the published reads
    export = load_slim_export(path)
    cfg = OmegaConf.create(export["hyper_parameters"]["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.attn_func = "default"   # the flash-attention kernels are only needed for training's forward-mode derivative
    configure_model_cfg(cfg)
    module = module_from_export(cfg, export, build_model(cfg.model), get_loss_fn(cfg), weighting_model=build_weighting_model(cfg))
    read = dict((export.get("meta") or {}).get("read") or {})
    apply_label_scale(module, label_scale if label_scale is not None else read.get("label_scale"))
    module.eval()
    module.requires_grad_(False)
    module.to(device)
    module.setup("predict")
    return Generator(module=module, cfg=cfg, read=read, w=default_w(cfg))


@torch.no_grad()
def generate(gen, labels, seed=0, omega=None, batch_size=64):
    """uint8 images (N, 3, H, W) for the class `labels`, one network evaluation each; `omega` defaults to the export's
    read setting. The noise stream is fixed by `seed` and the batch size."""
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    omega = gen.read.get("omega") if omega is None else omega
    model_cfg = gen.cfg.model
    shape = (int(model_cfg.in_channels), int(model_cfg.input_size), int(model_cfg.input_size))
    out = []
    for images_by_steps, _ in generate_image_batches(
        gen.module, labels=labels, latent_shape=shape, sample_batch_size=batch_size, generation_steps=[1],
        vae_batch_size=25, seed=int(seed), inverse_scaler=lambda x: (x + 1.0) / 2.0, device=gen.device,
        omega=None if omega is None else float(omega), w=gen.w,
    ):
        out.append(images_by_steps[1].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu())
    return torch.cat(out)
