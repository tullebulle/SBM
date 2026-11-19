"""Inference: load a text-to-image model and generate images from prompts.

    from sbm_t2i.infer import load_model, generate
    model, cfg, step = load_model("store/ckpt/t2i/sbm_t2i_xl_step1200000_ema0_9999.pt")
    images = generate(model, cfg, ["a red fox in fresh snow", "a bowl of ramen"], seed=0)   # uint8 [2, 3, 256, 256]

One step per image: x = decode(T(z; prompt, omega)), z ~ N(0, I) in the RAE latent, the prompt's FLAN-T5 states moved
away from the mask token by `label_scale`, decoded by the RAE ViT-XL decoder at 256 px. The text encoder and the codec
are read from the store (paths in the checkpoint's config; `scripts/t2i/data/fetch.sh scripts/t2i/data/manifest_flan_t5.tsv`
and `scripts/rae/fetch_assets.py` put them there), offline.

The inference export of a training checkpoint holds one EMA copy and the run's config, without the optimizer, the guidance
teacher or the other EMA copies (a 16 GB XL training checkpoint -> 3.3 GB). Layout (a plain torch.save dict, readable as
a slim checkpoint by sbm_t2i.models.read_weights / load_weights):

    format             "sbm_t2i_export_v1"
    state_dict         the EMA copy, keys "model.<name>" (build_model(**cfg.model).load_state_dict(strict=True) after the prefix)
    ema_active_decay   its decay (so a reader asking for that decay gets state_dict)
    global_step        the checkpoint's step
    cfg                the run config (run.yaml) as a plain container; paths may be ${store:} / ${repo:} interpolations
    guided_cbm_loss    the guided-CBM loss state (frozen balance weight), for a fine-tune that starts here
    meta               a description and the read settings (meta["read"]: omega, label_scale) of the published numbers

`init.from=<export>` (init.resume_state=false) starts a fine-tune from it; sbm_t2i.train restores the loss state of an
export. scripts/t2i/geneval_sample.py samples from it (or from a training checkpoint next to its run.yaml).
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

FORMAT = "sbm_t2i_export_v1"

# the read settings of the released models, by network: the default of an export or checkpoint whose meta has none
READS = {"TextDiT-XL/1-DH": {"omega": 0.35, "label_scale": 1.125}, "TextDiT-B/1-DH-t15": {"omega": 0.3, "label_scale": 1.25}}
FALLBACK_READ = {"omega": 0.8, "label_scale": 1.0}


def is_export(obj) -> bool:
    """True for a loaded export dict, or for a path whose file is one."""
    if isinstance(obj, (str, Path)):
        try:
            obj = torch.load(str(obj), map_location="meta", mmap=True, weights_only=False)
        except Exception:
            return False
    return isinstance(obj, dict) and str(obj.get("format", "")).startswith("sbm_t2i_export")


def ema_copy(ck: dict, which) -> dict:
    """The checkpoint's EMA copy `which` with the "model." prefix kept: a decay (its state_dict when it is the active
    decay, else ema_state_dicts[label]), "active" (the state_dict) or "raw" (the online weights)."""
    if which == "raw":
        return ck["current_model_state"]
    if which == "active" or abs(float(which) - float(ck.get("ema_active_decay", -1))) < 1e-12:
        return ck["state_dict"]
    label = f"{float(which):.10g}".replace("-", "m").replace(".", "_")
    return ck["ema_state_dicts"][label]


def export_from_checkpoint(ckpt: str | Path, cfg, which="0.9999", meta: dict | None = None) -> dict:
    """The export of training checkpoint `ckpt` (its EMA copy `which`) with run config `cfg`."""
    ck = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=False)
    sd = {k: v.clone() for k, v in ema_copy(ck, which).items() if k.startswith("model.")}
    out = {"format": FORMAT, "state_dict": sd, "ema_active_decay": float(which), "global_step": int(ck.get("global_step", -1)),
           "cfg": OmegaConf.to_container(OmegaConf.create(cfg), resolve=False), "meta": dict(meta or {})}
    if ck.get("guided_cbm_loss") is not None:
        out["guided_cbm_loss"] = ck["guided_cbm_loss"]
    return out


def read_settings(cfg, meta: dict | None = None) -> dict:
    """{omega, label_scale}: the export's meta["read"], else the released model's of that network, else omega 0.8 and
    label scale 1."""
    read = dict(FALLBACK_READ)
    read.update(READS.get(str(cfg.model.get("name", "")), {}))
    read.update({k: float(v) for k, v in ((meta or {}).get("read") or {}).items() if k in FALLBACK_READ})
    return read


def load_model(path: str | Path, which="0.9999"):
    """(model, cfg, step) of an export or of a training checkpoint <run>/ckpt/<step>.ckpt (config: <run>/run.yaml). The
    model is on the CPU in eval mode; cfg.read holds its read settings (read_settings), which `generate` defaults to."""
    from sbm_t2i import run as _run  # noqa: F401  (registers ${store:} / ${repo:})
    from sbm_t2i.models import build_model

    path = Path(path)
    ck = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    meta = None
    if is_export(ck):
        cfg = OmegaConf.create(ck["cfg"])
        if which != "active" and abs(float(which) - float(ck["ema_active_decay"])) > 1e-12:
            raise ValueError(f"{path} holds EMA {ck['ema_active_decay']} only, not {which}")
        sd, meta = ck["state_dict"], ck.get("meta")
    else:
        cfg = OmegaConf.load(path.parent.parent / "run.yaml")
        sd = ema_copy(ck, which)
    model = build_model(**cfg.model)
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
    cfg.read = read_settings(cfg, meta)
    return model.eval(), cfg, int(ck.get("global_step", -1))


_ASSETS = {}


def assets(cfg):
    """(text encoder, tokenizer, codec) of a model's config, built once per process and shared by later calls."""
    from sbm_t2i.codec import build_codec
    from sbm_t2i.text import TextEncoder, tokenizer

    text_path, text_len = str(cfg.data.text_encoder), int(cfg.data.get("text_len", 256))
    codec_cfg = OmegaConf.to_container(cfg.codec, resolve=True)
    key = (text_path, text_len, str(sorted(codec_cfg.items())))
    if key not in _ASSETS:
        _ASSETS[key] = (TextEncoder(text_path, text_len), tokenizer(text_path), build_codec(OmegaConf.create(codec_cfg)))
    return _ASSETS[key]


@torch.no_grad()
def generate(model, cfg, prompts, *, omega=None, label_scale=None, seed=0, device="cuda", batch_size=64,
             generator=None) -> torch.Tensor:
    """uint8 images [N, 3, S, S] (on the CPU), one per prompt, in order (repeat a prompt for several images).

    omega / label_scale default to cfg.read (load_model). The noise comes from one CUDA (or CPU) generator seeded `seed`
    (or the given `generator`), drawn batch by batch: the same prompts in the same batches give the same images
    (scripts/t2i/geneval_sample.py draws 4 images x 16 prompts per batch, so batch_size=64 on its prompt order matches it)."""
    from sbm_t2i.dit_t2i import PromptBound
    from sbm_t2i.text import tokenize
    from sbm_t2i.text_eval import to_unit

    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    read = cfg.get("read") or read_settings(cfg)
    omega = float(read["omega"] if omega is None else omega)
    label_scale = float(read["label_scale"] if label_scale is None else label_scale)
    device = torch.device(device)
    model = model.to(device).eval()
    text, tok, codec = assets(cfg)
    text_len = int(cfg.data.get("text_len", 256))
    g = generator if generator is not None else torch.Generator(device=device).manual_seed(int(seed))
    out = []
    for i0 in range(0, len(prompts), int(batch_size)):
        chunk = prompts[i0:i0 + int(batch_size)]
        ids, mask = tokenize(tok, chunk, text_len)
        context, mask = text(ids.to(device), mask.to(device))
        z = torch.randn(len(chunk), *codec.latent_shape, generator=g, device=device)
        bound = PromptBound(model, context, mask, cfg.model.label_dim, label_scale=label_scale)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent = bound(z, torch.arange(z.shape[0], device=device), omega=omega).float()
        out.append((to_unit(codec.decode(latent)) * 255).round().to(torch.uint8).cpu())
    return torch.cat(out)
