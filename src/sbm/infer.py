"""Inference exports of the SD-VAE-latent maps, and one-step sampling from them.

An export is one EMA copy of the DiT plus what is needed to use it:

    {"format": "sbm_sdvae_export_v1",
     "state_dict": {"model.<key>": fp32 tensor},      # the `model.` prefix of a training checkpoint is kept
     "model": {...},                                  # sbm.models.build_model arguments
     "cfg": {...},                                    # the training recipe (configs/sbm), store paths as ${store:}
     "ema_decay": 0.9999, "global_step": 355000, "latents_scale": 0.18215,
     "meta": {"model": "<one-line description>",
              "read": {"protocol": "xl_val50k", "omega": 0.95, "label_scale": 1.2, "ema": 0.9999, "step": 355000, "nfe": 1},
              "results": {"fid_adm": 3.56, ...}}}

Because `state_dict` keeps the training checkpoint's key layout, an export also works wherever a checkpoint does:
`sbm.models.load_weights` and `init.from=<export>` (fine-tuning) read it, and `sbm eval <export>` scores it.

`generate` reproduces the evaluator's sampling path (sbm.eval) for the export's protocol: one network call on Gaussian
noise in the 32 x 32 x 4 latent at the anchor time omega, the class rows of the label table extrapolated from the null
row by `label_scale` (one-pass guidance, no extra network call), then the SD-VAE decoder (stabilityai/sd-vae-ft-mse).
`b2_adm50k` runs the fp32 weights under bf16 autocast; `xl_val50k` casts the weights to bf16 and draws bf16 noise.
"""
from __future__ import annotations

from pathlib import Path

import torch

from sbm import run as R
from sbm.models import build_model

FORMAT = "sbm_sdvae_export_v1"
VAE_REPO, VAE_REVISION = "stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493"
MODEL_KEYS = ("name", "input_size", "in_channels", "label_dim", "mlp_ratio", "train_pos_embed", "tangent_only_jvp",
              "gw_embedder", "gw_default", "gw_input", "gw_interval")


def is_export(obj) -> bool:
    """True for a loaded export dict, or for a path whose file is one."""
    if isinstance(obj, (str, Path)):
        obj = torch.load(obj, map_location="cpu", weights_only=False, mmap=True)
    return isinstance(obj, dict) and obj.get("format") == FORMAT


def export_from_checkpoint(ckpt: str | Path, which: str = "ema", *, cfg: dict, read: dict, meta: dict | None = None) -> dict:
    """The export of training checkpoint `ckpt`: its EMA copy `which` (ema = the active copy, ema_<label> another one,
    e.g. ema_0_9999), the recipe `cfg` (a plain dict, no absolute paths), the read settings `read` ({"protocol",
    "omega", "label_scale", ...}) and `meta` ({"model": description, "results": {...}})."""
    from sbm.models import read_weights
    from sbm.train import LOSS_STATE_KEY
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if which == "ema":
        decay = float(ck["ema_active_decay"])
    elif which.startswith("ema_") and which[4:] in ck.get("ema_state_dicts", {}):
        decay = float(ck.get("ema_decays", {}).get(which[4:], which[4:].replace("_", ".")))
    else:
        raise ValueError(f"{ckpt} has no EMA copy {which!r} (ema, or ema_<label> of {sorted(ck.get('ema_state_dicts', {}))})")
    sd = {f"model.{k}": v.detach().to(torch.float32).contiguous().clone() for k, v in read_weights(ckpt, which).items()}
    model = {k: cfg["model"][k] for k in MODEL_KEYS if k in cfg["model"]}
    out = {"format": FORMAT, "state_dict": sd, "model": model, "cfg": cfg, "ema_decay": decay,
           "global_step": int(ck["global_step"]), "latents_scale": float(cfg["data"]["latents_scale"]),
           "meta": {"read": dict(read), **(meta or {})}}
    if LOSS_STATE_KEY in ck:   # the balancer's state (only loss.balance.mode=ema carries one), for fine-tuning
        out[LOSS_STATE_KEY] = ck[LOSS_STATE_KEY]
    return out


def load_export(path: str | Path, device: str | torch.device = "cuda"):
    """-> (model, cfg, meta). The model is in eval mode on `device`, in the precision its read uses (bf16 weights for the
    xl_val50k protocol), with the read settings attached as `model.read` (generate's defaults)."""
    from omegaconf import OmegaConf
    ex = torch.load(path, map_location="cpu", weights_only=False)
    if ex.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} file")
    model = build_model(**ex["model"])
    model.load_state_dict({k[len("model."):]: v for k, v in ex["state_dict"].items()}, strict=True)
    meta = {**ex["meta"], "ema_decay": ex["ema_decay"], "global_step": ex["global_step"], "latents_scale": ex["latents_scale"]}
    model = model.to(device).eval().requires_grad_(False)
    if meta["read"].get("protocol") == "xl_val50k":
        model = model.to(torch.bfloat16)
    model.read = dict(meta["read"])
    model.latents_scale = float(ex["latents_scale"])
    return model, OmegaConf.create(ex["cfg"]), meta


def vae_cache_dir() -> Path:
    """Where scripts/rae/fetch_assets.py --sdvae puts the SD-VAE (a Hugging Face hub cache)."""
    return R.store() / "rae" / "assets" / "hf_home" / "hub"


def load_vae(device: str | torch.device = "cuda"):
    """The SD-VAE decoder (fp32, eval), from the store's copy (offline), else from the Hugging Face hub."""
    from diffusers import AutoencoderKL
    try:
        vae = AutoencoderKL.from_pretrained(VAE_REPO, revision=VAE_REVISION, cache_dir=vae_cache_dir(), local_files_only=True)
    except OSError:
        vae = AutoencoderKL.from_pretrained(VAE_REPO, revision=VAE_REVISION)
    return vae.to(device).float().eval().requires_grad_(False)


class _LabelScale:
    """The class rows of the label table extrapolated from the null (last) row by `s` inside the block, restored after."""

    def __init__(self, model, s):
        self.table, self.s = model.y_embedder.embedding_table.weight, float(s)

    def __enter__(self):
        if self.s != 1.0:
            self.saved = self.table.data.clone()
            w = self.table.data
            w[:-1] = w[-1:] + self.s * (w[:-1] - w[-1:])

    def __exit__(self, *exc):
        if self.s != 1.0:
            self.table.data.copy_(self.saved)


@torch.no_grad()
def generate(model, cfg, labels, *, omega=None, label_scale=None, seed=0, device=None, vae=None, batch=64):
    """One-step class-conditional samples: uint8 images (N, 3, 256, 256) for the class ids `labels`.

    omega / label_scale default to the export's read settings (`model.read`); `seed` seeds one generator on the device
    for the whole call. `vae` (load_vae) can be passed in to reuse it across calls."""
    read = getattr(model, "read", {}) or {}
    omega = float(read.get("omega", 1.0) if omega is None else omega)
    label_scale = float(read.get("label_scale", 1.0) if label_scale is None else label_scale)
    xl = read.get("protocol") == "xl_val50k"
    device = torch.device(device) if device is not None else next(model.parameters()).device
    vae = vae if vae is not None else load_vae(device)
    scale = float(getattr(model, "latents_scale", None) or cfg.data.latents_scale)
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    g = torch.Generator(device=device.type).manual_seed(int(seed))
    dtype = next(model.parameters()).dtype
    shape = (model.in_channels, *model.x_embedder.img_size)
    out = []
    with _LabelScale(model, label_scale):
        for start in range(0, len(labels), int(batch)):
            y = labels[start:start + int(batch)].to(device)
            z = torch.randn((len(y), *shape), generator=g, device=device, dtype=torch.bfloat16 if xl else torch.float32)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lat = model(z.to(dtype) if not xl else z, y, omega=omega if xl else torch.tensor(omega, device=device))
            if xl:
                lat = lat.float() / scale
            else:   # divide by a fp32 tensor, as the B/2 evaluator does (bit-identical decodes)
                lat = lat / torch.full((1, lat.shape[1], 1, 1), scale, dtype=torch.float32, device=lat.device)
            vb = 8 if xl else 64   # the evaluator's decode chunks
            with torch.autocast(device.type, enabled=False):
                imgs = torch.cat([vae.decode(lat[i:i + vb].float()).sample.float() for i in range(0, len(lat), vb)])
            out.append(((imgs + 1.0) / 2.0).clamp(0.0, 1.0).mul(255).byte().cpu())
    return torch.cat(out)
