"""FID-50k of a class-conditional map of the RAE line, the protocol of the published numbers: one network evaluation from
noise, labels 0..999 cycled, Inception features of the 50k decoded samples against the ImageNet-val statistics (`fid`) and
the ADM ImageNet-train reference (`fid_by_reference.adm`, the number the paper reports).

    python scripts/rae/eval_fid.py checkpoint_path=store/ckpt/imagenet/sbm_rae_xl_step1042500_ema0_9999.pt \
        real_dir=store/data/imagenet_1k/val
    torchrun --nproc_per_node 8 scripts/rae/eval_fid.py checkpoint_path=... real_dir=...       # the same, on 8 GPUs

The read settings (omega, label scale, seed offset) default to the export's meta["read"]; override with fid.omega=...,
fid.label_scale=..., fid.seed_offset=... (configs/rae/fid.yaml). One GPU reproduces the published sample stream; N GPUs draw
a different but equivalent one. Results land in outputs/fid/<time>/fid_metrics.json.
"""
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from omegaconf import open_dict, DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from sbm_rae.ema import ema_decay_label
from sbm_rae.generate import apply_label_scale, build_weighting_model, configure_model_cfg, default_w
from sbm_rae.infer import is_slim_export, load_slim_export, module_from_export
from sbm_rae.loss import get_loss_fn
from sbm_rae.models import build_model
from sbm_rae.paths import fid_cache_dir
from sbm_rae.sampling import build_class_labels, generate_image_batches, normalize_generation_steps, save_images
from sbm_rae.trainer import TrainingModule

_MP = os.environ.get("RAE_MATMUL_PRECISION", "high")   # "high" = TF32 matmuls (the published reads); "highest" = exact fp32
torch.set_float32_matmul_precision(_MP)
torch.backends.cudnn.allow_tf32 = _MP != "highest"

_TRAINING_CFG_CACHE = {}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_rank_zero(self):
        return self.rank == 0


@dataclass(frozen=True)
class WeightVariant:
    source: str
    label: str | None = None
    decay: float | None = None

    @property
    def suffix(self):
        if self.source == "active":
            return ""
        if self.source == "raw":
            return "_raw"
        return f"_ema_{self.label}"


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(False, 0, 0, 1)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(True, dist.get_rank(), local_rank, dist.get_world_size())


def _cleanup_distributed(distributed):
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _sum_across_ranks(value, device, distributed):
    total = torch.tensor(int(value), device=device, dtype=torch.long)
    if distributed.enabled:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total.item())


def _real_image_transform(resolution):
    return transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )


def _build_real_dataset(cfg):
    return ImageFolder(
        cfg.real_dir, transform=_real_image_transform(cfg.dataset.img_resolution)
    )


def _real_dataloader(cfg, distributed):
    dataset = _build_real_dataset(cfg)
    max_real = cfg.fid.get("num_real_samples", None)
    if max_real is not None:
        max_real = min(int(max_real), len(dataset))
        dataset = Subset(dataset, list(range(max_real)))
    if distributed.world_size > 1:
        dataset = Subset(dataset, list(range(distributed.rank, len(dataset), distributed.world_size)))

    return DataLoader(
        dataset,
        batch_size=cfg.fid.real_batch_size,
        shuffle=False,
        num_workers=cfg.fid.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.fid.num_workers > 0,
    )


def _load_checkpoint(checkpoint_path, map_location="meta"):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except RuntimeError:
        if map_location == "meta":
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raise


def _load_ema_state_dict(checkpoint_path, ema_label):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema_state_dicts = checkpoint.get("ema_state_dicts", {}) or {}
    if ema_label not in ema_state_dicts:
        available = sorted(ema_state_dicts)
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain inactive EMA '{ema_label}'. "
            f"Available inactive EMA labels: {available}"
        )
    return ema_state_dicts[ema_label]


def _load_raw_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("current_model_state", None)
    if not raw:
        raise ValueError(
            f"{checkpoint_path} has no current_model_state, so its raw (non-averaged) "
            "weights were never saved and fid.score_raw cannot be honoured"
        )
    return dict(raw)


def _load_module(cfg, device, weight_variant=None):
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must point to a Lightning checkpoint")

    if weight_variant is None:
        weight_variant = WeightVariant(source="active")

    configure_model_cfg(cfg)
    model = build_model(cfg.model)
    weighting_model = build_weighting_model(cfg)
    loss_fn = get_loss_fn(cfg)
    if is_slim_export(cfg.checkpoint_path):
        # the EMA-only export (sbm_rae.infer, scripts/rae/export_record.py): one weight copy, no optimizer / teacher /
        # EMA-callback state; the original Lightning checkpoint path keeps working through the branch below
        if weight_variant.source != "active":
            raise ValueError(f"{cfg.checkpoint_path} is a slim export holding one EMA copy; cannot score '{weight_variant.source}'")
        module = module_from_export(cfg, load_slim_export(cfg.checkpoint_path), model, loss_fn, weighting_model=weighting_model)
    else:
        module = TrainingModule.load_from_checkpoint(
            cfg.checkpoint_path,
            cfg=cfg,
            model=model,
            loss_fn=loss_fn,
            weighting_model=weighting_model,
            map_location="cpu",
        )
    if weight_variant.source == "ema_state_dicts":
        module.load_state_dict(
            _load_ema_state_dict(cfg.checkpoint_path, weight_variant.label),
            strict=True,
        )
    elif weight_variant.source == "raw":
        # A checkpoint's `state_dict` holds the ACTIVE EMA copy (the callback swaps the
        # weights in before saving); the optimizer's own weights live in
        # `current_model_state`. Scoring those is the only way to ask whether a gap
        # between two arms is learning or is the averaging doing the work.
        module.load_state_dict(_load_raw_state_dict(cfg.checkpoint_path), strict=True)
    apply_label_scale(module, cfg.fid.get("label_scale", None))
    module.eval()
    module.requires_grad_(False)
    module.to(device)
    module.setup("predict")
    return module


def _update_real_features(metric, dataloader, device, distributed):
    total = 0
    for images, _ in tqdm(dataloader, desc="Real batches", disable=not distributed.is_rank_zero):
        images = images.to(device, non_blocking=True)
        metric.update(images, real=True)
        total += images.shape[0]
    return total


def _update_fake_features(metrics, cfg, module, device, distributed):
    generation_steps = sorted(metrics)
    labels = build_class_labels(
        num_samples=cfg.fid.num_fake_samples,
        num_classes=cfg.dataset.num_classes,
        class_labels=cfg.fid.class_labels,
    )
    local_indices = torch.arange(distributed.rank, labels.shape[0], distributed.world_size)
    labels = labels[local_indices]

    saved = 0
    total = 0
    latent_shape = (cfg.model.in_channels, cfg.model.input_size, cfg.model.input_size)
    generator = generate_image_batches(
        module,
        labels=labels,
        latent_shape=latent_shape,
        sample_batch_size=cfg.fid.sample_batch_size,
        generation_steps=generation_steps,
        vae_batch_size=cfg.fid.vae_batch_size,
        seed=cfg.seed + cfg.fid.seed_offset + distributed.rank,
        inverse_scaler=lambda x: (x + 1.0) / 2.0,
        device=device,
        omega=cfg.fid.get("omega", None),
        w=cfg.fid.get("w", None),
    )

    total_batches = (len(labels) + cfg.fid.sample_batch_size - 1) // cfg.fid.sample_batch_size
    for images_by_steps, batch_labels in tqdm(generator, total=total_batches, desc="Fake batches", disable=not distributed.is_rank_zero):
        for steps, images in images_by_steps.items():
            metrics[steps].update(images.to(device, non_blocking=True), real=False)

        batch_size = int(batch_labels.shape[0])
        if cfg.fid.save_generated_dir:
            batch_indices = local_indices[saved : saved + batch_size]
            for steps, images in images_by_steps.items():
                save_dir = Path(cfg.fid.save_generated_dir)
                if len(generation_steps) > 1:
                    save_dir = save_dir / f"steps_{steps}"
                save_images(images.cpu(), save_dir, indices=batch_indices.cpu(), labels=batch_labels.cpu())
            saved += batch_size
        total += batch_size

    return total


def _config_value_is_set(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null"}
    return True


def _clone_config(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return []
        return [stripped]
    try:
        return list(value)
    except TypeError:
        return [value]


def _training_cfg_from_candidate(candidate):
    if candidate is None:
        return None

    cfg = OmegaConf.create(candidate)
    if "model" not in cfg or "loss" not in cfg:
        return None

    OmegaConf.set_struct(cfg, False)
    return cfg


def _training_cfg_from_checkpoint(checkpoint_path):
    checkpoint = _load_checkpoint(checkpoint_path)
    candidates = []

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if isinstance(hyper_parameters, dict):
        candidates.extend(
            hyper_parameters.get(key)
            for key in ("cfg", "config", "training_cfg")
            if key in hyper_parameters
        )

    candidates.extend(
        checkpoint.get(key)
        for key in ("cfg", "config", "training_cfg")
        if key in checkpoint
    )

    for candidate in candidates:
        cfg = _training_cfg_from_candidate(candidate)
        if cfg is not None:
            return cfg

    return None


def _requested_extra_ema_decays(fid_cfg):
    value = fid_cfg.fid.get("ema_decays", fid_cfg.fid.get("extra_ema_decays", []))
    return tuple(float(decay) for decay in _as_list(value))


def _requested_omegas(fid_cfg):
    values = fid_cfg.fid.get("omegas", [])
    if _as_list(values):
        return tuple(float(omega) for omega in _as_list(values))

    omega = fid_cfg.fid.get("omega", None)
    if _config_value_is_set(omega):
        return (float(omega),)

    return (None,)


def _requested_ws(fid_cfg):
    values = fid_cfg.fid.get("ws", [])
    if _as_list(values):
        return tuple(float(w) for w in _as_list(values))

    w = fid_cfg.fid.get("w", None)
    if _config_value_is_set(w):
        return (float(w),)

    return (None,)


def _omega_suffix(omega):
    if omega is None:
        return ""
    label = f"{float(omega):.6g}".replace("-", "m").replace(".", "p")
    return f"_omega_{label}"


def _w_suffix(w):
    if w is None:
        return ""
    label = f"{float(w):.6g}".replace("-", "m").replace(".", "p")
    return f"_w_{label}"


def _weight_variants_for_checkpoint(fid_cfg, checkpoint_path, distributed):
    checkpoint = _load_checkpoint(checkpoint_path)
    if is_slim_export(checkpoint):
        # a slim export carries exactly one EMA copy (sbm_rae.infer); it is scored as the "active" weights
        label, decay = checkpoint.get("ema_label"), checkpoint.get("ema_decay")
        requested = _requested_extra_ema_decays(fid_cfg)
        if requested and all(ema_decay_label(d) != label for d in requested):
            message = (f"{checkpoint_path} is a slim export holding only EMA {decay} (label '{label}'); "
                       f"requested fid.ema_decays={list(requested)}")
            if bool(fid_cfg.fid.get("strict_ema_decays", False)):
                raise ValueError(message)
            if getattr(distributed, "is_rank_zero", True):
                print(f"Note: {message}; scoring the exported copy")
        if bool(fid_cfg.fid.get("score_raw", False)) and getattr(distributed, "is_rank_zero", True):
            print(f"Note: {checkpoint_path} is a slim export without raw weights; fid.score_raw is ignored")
        return [WeightVariant(source="active", label=label, decay=decay)]
    active_decay = checkpoint.get("ema_active_decay", None)
    active_label = ema_decay_label(active_decay) if active_decay is not None else None
    variants = []

    ema_state_dicts = checkpoint.get("ema_state_dicts", {}) or {}
    strict_ema_decays = bool(fid_cfg.fid.get("strict_ema_decays", False))
    for decay in _requested_extra_ema_decays(fid_cfg):
        label = ema_decay_label(decay)
        if label == active_label:
            # The active decay is what the checkpoint's own weights already
            # carry, so it is scored in place rather than from ema_state_dicts.
            variants.append(WeightVariant(source="active", label=label, decay=decay))
            continue
        if label not in ema_state_dicts:
            message = (
                f"Checkpoint {checkpoint_path} does not contain inactive EMA {decay} "
                f"(label '{label}'); available inactive labels={sorted(ema_state_dicts)}"
            )
            if strict_ema_decays:
                raise ValueError(message)
            if distributed.is_rank_zero:
                print(f"Skipping requested EMA: {message}")
            continue
        variants.append(WeightVariant(source="ema_state_dicts", label=label, decay=decay))

    if bool(fid_cfg.fid.get("score_raw", False)):
        variants.append(WeightVariant(source="raw", label="raw"))

    if not variants:
        # Nothing was requested, or nothing requested is available: score the
        # weights exactly as saved rather than silently evaluating nothing.
        variants.append(WeightVariant(source="active", label=active_label, decay=active_decay))

    return variants


def _sidecar_training_config_path(checkpoint_path):
    candidate_roots = []
    checkpoint_parent = checkpoint_path.parent
    if checkpoint_parent.name == "checkpoints":
        candidate_roots.append(checkpoint_parent.parent)
    candidate_roots.append(checkpoint_parent)

    seen_roots = set()
    for candidate_root in candidate_roots:
        if candidate_root in seen_roots:
            continue
        seen_roots.add(candidate_root)
        config_path = candidate_root / ".hydra" / "config.yaml"
        if config_path.is_file():
            return config_path

    return None


def _load_training_cfg(checkpoint_path):
    config_path = _sidecar_training_config_path(checkpoint_path)
    if config_path in _TRAINING_CFG_CACHE:
        return _clone_config(_TRAINING_CFG_CACHE[config_path])

    cfg = _training_cfg_from_checkpoint(checkpoint_path)
    if cfg is not None:
        return cfg

    if config_path is not None:
        cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(cfg, False)
        _TRAINING_CFG_CACHE[config_path] = _clone_config(cfg)
        return cfg

    raise ValueError(
        "Could not find the training config in the checkpoint or in a sibling "
        f".hydra/config.yaml for {checkpoint_path}"
    )


def _resolve_real_dir(fid_cfg, training_cfg):
    real_dir = fid_cfg.get("real_dir", None)
    if _config_value_is_set(real_dir):
        return str(real_dir)

    data_dir = training_cfg.get("data_dir", None)
    if _config_value_is_set(data_dir):
        return str(Path(str(data_dir)) / "val")
    return None


def _build_eval_cfg(fid_cfg, checkpoint_path, output_dir, omega=None, w=None):
    training_cfg = _load_training_cfg(checkpoint_path)
    eval_cfg = _clone_config(training_cfg)
    OmegaConf.set_struct(eval_cfg, False)

    eval_cfg.checkpoint_path = str(checkpoint_path)
    eval_cfg.work_dir = str(output_dir)
    eval_cfg.real_dir = _resolve_real_dir(fid_cfg, eval_cfg)
    eval_cfg.fid = _clone_config(fid_cfg.fid)
    # Model overrides given on the evaluation command line, e.g.
    # `+model.attn_func=default` to score a Hopper-trained checkpoint on an A100.
    # The kernel choice does not change the map, only which implementation
    # computes its attention; everything else about the model still comes from
    # the checkpoint's training config.
    model_overrides = fid_cfg.get("model", None)
    if model_overrides:
        eval_cfg.model = OmegaConf.merge(eval_cfg.model, model_overrides)
    eval_cfg.fid.omega = None if omega is None else float(omega)
    eval_cfg.fid.w = None if w is None else float(w)
    if eval_cfg.fid.w is None and default_w(eval_cfg) is not None:
        # a guidance-scale-conditioned network read without fid.w would get w = 1 (no guidance): read at the guide's w0
        eval_cfg.fid.w = default_w(eval_cfg)
        print(f"guidance-scale-conditioned network without fid.w: reading at the guide's scale w0 = {eval_cfg.fid.w:g}")

    seed = fid_cfg.get("seed", None)
    if _config_value_is_set(seed):
        eval_cfg.seed = int(seed)
    elif not _config_value_is_set(eval_cfg.get("seed", None)):
        eval_cfg.seed = 42

    return eval_cfg


def _resolve_checkpoint_dir(path):
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    return path


def _discover_checkpoints(path, include_last):
    checkpoint_dir = _resolve_checkpoint_dir(path.expanduser())
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_paths = sorted(checkpoint_dir.glob("*.ckpt"))
    if not include_last:
        checkpoint_paths = [path for path in checkpoint_paths if path.name != "last.ckpt"]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return checkpoint_paths


def _select_latest_checkpoint(checkpoint_paths):
    return [max(checkpoint_paths, key=lambda path: path.stat().st_mtime)]


def _resolve_checkpoint_paths(cfg):
    include_last = bool(cfg.get("include_last", True))
    latest_only = bool(cfg.get("latest_only", False))
    checkpoint_path = cfg.get("checkpoint_path", None)
    if _config_value_is_set(checkpoint_path):
        resolved_path = Path(str(checkpoint_path)).expanduser()
        if resolved_path.is_dir():
            checkpoint_paths = _discover_checkpoints(resolved_path, include_last)
            return _select_latest_checkpoint(checkpoint_paths) if latest_only else checkpoint_paths
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
        return [resolved_path]

    checkpoint_dir = cfg.get("checkpoint_dir", None)
    if _config_value_is_set(checkpoint_dir):
        checkpoint_paths = _discover_checkpoints(Path(str(checkpoint_dir)), include_last)
        return _select_latest_checkpoint(checkpoint_paths) if latest_only else checkpoint_paths

    raise ValueError("Set checkpoint_dir to a checkpoint or run directory")


def _load_reference_stats(path):
    """(mu, sigma) in float64 from an .npz (`mu` / `sigma`, or MeanFlow's `ref_mu` / `ref_sigma`)."""
    z = np.load(str(path))
    pre = "ref_" if "ref_mu" in z.files else ""
    return (torch.from_numpy(np.asarray(z[pre + "mu"], np.float64)),
            torch.from_numpy(np.asarray(z[pre + "sigma"], np.float64)))


def _fid_of_fake_against(metric, mu_ref, sigma_ref):
    """FID of a torchmetrics FID object's accumulated fake features against fixed statistics, in float64 on the CPU:
    |mu - mu_r|^2 + tr S + tr S_r - 2 tr (S_r^1/2 S S_r^1/2)^1/2."""
    n = float(metric.fake_features_num_samples)
    total = metric.fake_features_sum.detach().double().cpu()
    mu = total / n
    sigma = (metric.fake_features_cov_sum.detach().double().cpu() - n * torch.outer(mu, mu)) / (n - 1.0)
    w, v = torch.linalg.eigh(sigma_ref)
    root_ref = (v * w.clamp_min(0).sqrt()) @ v.T
    cross = torch.linalg.eigvalsh(root_ref @ sigma @ root_ref).clamp_min(0).sqrt().sum()
    return float((mu - mu_ref).pow(2).sum() + torch.trace(sigma) + torch.trace(sigma_ref) - 2.0 * cross)


def _build_fid_metric(cfg, device):
    try:
        return FrechetInceptionDistance(feature=cfg.fid.feature_dim, normalize=True).to(device)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FID evaluation requires torch-fidelity. Install dependencies with `pip install -r requirements.txt` or `pip install torch-fidelity`."
        ) from exc


def _requested_generation_steps(fid_cfg):
    value = fid_cfg.fid.get("generation_steps", 1)
    steps = _as_list(value)
    if not steps:
        steps = [1]
    return tuple(normalize_generation_steps(steps))


def _copy_real_features(source, target):
    target.real_features_sum = source.real_features_sum.clone()
    target.real_features_cov_sum = source.real_features_cov_sum.clone()
    target.real_features_num_samples = source.real_features_num_samples.clone()


def _evaluate_checkpoint(fid_cfg, checkpoint_path, output_dir, device, distributed, weight_variant=None, omega=None, w=None):
    if weight_variant is None:
        weight_variant = WeightVariant(source="active")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_eval_cfg(fid_cfg, checkpoint_path, output_dir, omega=omega, w=w)
    seed_everything(cfg.seed, workers=True)

    if distributed.is_rank_zero:
        omega_label = "default" if omega is None else f"{float(omega):.6g}"
        w_label = "default" if w is None else f"{float(w):.6g}"
        print(f"Evaluating {checkpoint_path.name}{weight_variant.suffix} with omega={omega_label} w={w_label}")

    generation_steps = _requested_generation_steps(cfg)
    metrics = {steps: _build_fid_metric(cfg, device) for steps in generation_steps}
    module = _load_module(cfg, device, weight_variant=weight_variant)

    # The real-split Inception statistics do not depend on the checkpoint: they are cached in the store's
    # ckpt/imagenet/fid_cache (the ImageNet-val file comes with the checkpoints), so no real images are needed once it exists.
    # Rank 0 holds them; the metric sums its states over ranks.
    first_metric = metrics[generation_steps[0]]
    cache_file = None
    if fid_cache_dir().is_dir():
        cache_file = fid_cache_dir() / (
            f"real_{cfg.dataset.name}_{cfg.fid.real_split}_{cfg.fid.get('num_real_samples', None)}_{cfg.fid.feature_dim}.pt"
        )
    if cache_file is not None and cache_file.is_file():
        n_real = 0
        if distributed.is_rank_zero:
            state = torch.load(cache_file, map_location=device)
            first_metric.real_features_sum = state["sum"].to(first_metric.real_features_sum.dtype)
            first_metric.real_features_cov_sum = state["cov_sum"].to(first_metric.real_features_cov_sum.dtype)
            first_metric.real_features_num_samples = state["n"].to(first_metric.real_features_num_samples.dtype)
            n_real = int(state["n"].item())
            print(f"real-feature statistics loaded from {cache_file} (n={n_real})")
    else:
        if not cfg.real_dir:
            raise ValueError("no cached real statistics: set real_dir=<ImageNet val folder>")
        n_real = _update_real_features(first_metric, _real_dataloader(cfg, distributed), device, distributed)
        if cache_file is not None and distributed.world_size == 1:
            tmp = cache_file.with_suffix(f".tmp{os.getpid()}")
            torch.save({"sum": first_metric.real_features_sum.detach().cpu(),
                        "cov_sum": first_metric.real_features_cov_sum.detach().cpu(),
                        "n": first_metric.real_features_num_samples.detach().cpu()}, tmp)
            os.replace(tmp, cache_file)
    for steps in generation_steps[1:]:
        _copy_real_features(first_metric, metrics[steps])

    n_fake = _update_fake_features(metrics, cfg, module, device, distributed)
    n_real = _sum_across_ranks(n_real, device, distributed)
    n_fake = _sum_across_ranks(n_fake, device, distributed)
    fid_by_steps = {steps: float(metrics[steps].compute().item()) for steps in generation_steps}
    # the same samples' features against other reference statistics (fid.extra_refs: name -> .npz with mu / sigma),
    # e.g. the ADM ImageNet-train statistics the published one-step numbers use
    fid_by_reference = {}
    for ref_name, ref_path in (cfg.fid.get("extra_refs", None) or {}).items():
        if not _config_value_is_set(ref_path):
            continue
        mu_ref, sigma_ref = _load_reference_stats(ref_path)
        fid_by_reference[str(ref_name)] = {}
        for steps in generation_steps:
            # the fake-feature sums of every rank (compute() restores the rank-local states afterwards)
            with metrics[steps].sync_context(should_sync=distributed.enabled):
                fid_by_reference[str(ref_name)][str(steps)] = _fid_of_fake_against(metrics[steps], mu_ref, sigma_ref)

    results = {
        "fid": fid_by_steps[generation_steps[-1]],
        "fid_by_generation_steps": {str(steps): fid_by_steps[steps] for steps in generation_steps},
        "checkpoint_path": str(checkpoint_path),
        "real_dir": cfg.real_dir,
        "num_real_samples": n_real,
        "num_fake_samples": n_fake,
        "generation_steps": list(generation_steps),
        "omega": cfg.fid.get("omega", None),
        "w": cfg.fid.get("w", None),
        "label_scale": cfg.fid.get("label_scale", None),
        "weight_source": weight_variant.source,
        "ema_label": weight_variant.label,
        "ema_decay": weight_variant.decay,
    }
    if fid_by_reference:
        results["fid_by_reference"] = fid_by_reference

    _log_fid_to_wandb(fid_cfg, results, checkpoint_path, distributed)

    if distributed.is_rank_zero:
        output_path = output_dir / "fid_metrics.json"
        output_path.write_text(json.dumps(results, indent=2) + "\n")

        for steps in generation_steps:
            print(f"FID (generation_steps={steps}): {fid_by_steps[steps]:.4f}")
            for ref_name, by_steps in fid_by_reference.items():
                print(f"FID vs {ref_name} (generation_steps={steps}): {by_steps[str(steps)]:.4f}")
        print(json.dumps(results, indent=2))
        print(f"Saved FID metrics to {output_path}")

    del metrics, module
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def _training_step_of(checkpoint_path):
    """Global step the checkpoint was written at, for the FID curve's x-axis."""
    try:
        checkpoint = _load_checkpoint(checkpoint_path)
    except Exception:
        return None
    for key in ("global_step", "step"):
        value = checkpoint.get(key, None)
        if isinstance(value, int):
            return value
    # Fall back to the filename, which encodes the step either as
    # step0020000-... (current) or step-step=20000 (older checkpoints).
    match = re.search(r"step[-=]?(\d+)", Path(checkpoint_path).name)
    return int(match.group(1)) if match else None


def _log_fid_to_wandb(fid_cfg, results, checkpoint_path, distributed):
    """Append one point per evaluation to a per-training-run W&B run.

    The run id is a hash of the training run directory, so every evaluation of
    the same training run lands in the same W&B run and builds a curve instead
    of scattering across one-off runs.
    """
    if not distributed.is_rank_zero:
        return
    wandb_cfg = fid_cfg.fid.get("wandb", None) or {}
    if not wandb_cfg.get("enabled", False):
        return

    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent

    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping FID logging")
        return

    run_id = "fid-" + hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]
    name = wandb_cfg.get("run_name", None) or f"fid-{run_dir.name}"
    step = _training_step_of(checkpoint_path)

    payload = {}
    for steps, value in (results.get("fid_by_generation_steps", {}) or {}).items():
        payload[f"fid/{steps}_step"] = value
    payload["fid/ema_decay"] = results.get("ema_decay", None)
    payload["fid/num_fake_samples"] = results.get("num_fake_samples", None)
    if step is not None:
        payload["train_step"] = step
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "sbm_rae"),
            entity=wandb_cfg.get("entity", None),
            id=run_id,
            name=name,
            resume="allow",
            reinit=True,
            config={"training_run_dir": str(run_dir)},
        )
        run.log(payload, step=step)
        run.finish()
        print(f"Logged FID to W&B run {name} ({run_id}) at train_step={step}")
    except Exception as exc:  # never fail an evaluation because logging failed
        print(f"W&B FID logging failed: {exc}")


def _checkpoint_variant_output_dir(output_root, checkpoint_path, weight_variant, omega=None, w=None):
    return output_root / f"{checkpoint_path.stem}{weight_variant.suffix}{_omega_suffix(omega)}{_w_suffix(w)}"


def _fid_metrics_path(output_dir):
    return output_dir / "fid_metrics.json"


def _load_existing_fid_results(output_dir):
    metrics_path = _fid_metrics_path(output_dir)
    if not metrics_path.is_file():
        return None
    try:
        results = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(results, dict) or not isinstance(results.get("fid", None), (int, float)):
        return None
    return results


def _evaluation_label(checkpoint_path, weight_variant, omega=None, w=None):
    omega_label = "default" if omega is None else f"{float(omega):.6g}"
    w_label = "default" if w is None else f"{float(w):.6g}"
    return f"{checkpoint_path.name}{weight_variant.suffix} with omega={omega_label} w={w_label}"


def _load_or_evaluate_checkpoint(fid_cfg, checkpoint_path, output_dir, device, distributed, weight_variant, omega=None, w=None):
    if bool(fid_cfg.fid.get("skip_existing", True)):
        existing_results = _load_existing_fid_results(output_dir)
        if existing_results is not None:
            if distributed.is_rank_zero:
                fid_value = existing_results.get("fid", None)
                fid_text = f" (FID={float(fid_value):.4f})" if isinstance(fid_value, (int, float)) else ""
                print(
                    f"Skipping {_evaluation_label(checkpoint_path, weight_variant, omega, w)}{fid_text}; "
                    f"found existing metrics at {_fid_metrics_path(output_dir)}"
                )
            return existing_results

    return _evaluate_checkpoint(
        fid_cfg,
        checkpoint_path,
        output_dir,
        device,
        distributed,
        weight_variant,
        omega=omega,
        w=w,
    )


def _read_defaults_from_export(cfg, checkpoint_paths):
    """Unset read settings (fid.omega / omegas, fid.label_scale, fid.seed_offset) from the export's meta["read"]."""
    read = {}
    if len(checkpoint_paths) == 1 and is_slim_export(checkpoint_paths[0]):
        read = (_load_checkpoint(checkpoint_paths[0]).get("meta") or {}).get("read") or {}
    fid = cfg.fid
    if not _config_value_is_set(fid.get("omega", None)) and not _as_list(fid.get("omegas", [])) and read.get("omega") is not None:
        fid.omega = float(read["omega"])
    if not _config_value_is_set(fid.get("label_scale", None)) and read.get("label_scale") is not None:
        fid.label_scale = float(read["label_scale"])
    if not _config_value_is_set(fid.get("seed_offset", None)):
        fid.seed_offset = int(read.get("seed_offset", 2024))
    if read:
        print(f"read settings: omega {fid.get('omega')}, label scale {fid.get('label_scale')}, seed offset {fid.seed_offset}")


@hydra.main(config_path="../../configs/rae", config_name="fid", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    output_root = Path(HydraConfig.get().runtime.output_dir)
    distributed = _setup_distributed()
    device = torch.device(f"cuda:{distributed.local_rank}" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint_paths = _resolve_checkpoint_paths(cfg)
        _read_defaults_from_export(cfg, checkpoint_paths)
        checkpoint_variants = [
            (checkpoint_path, _weight_variants_for_checkpoint(cfg, checkpoint_path, distributed))
            for checkpoint_path in checkpoint_paths
        ]
        omegas = _requested_omegas(cfg)
        ws = _requested_ws(cfg)
        evaluations = [
            (checkpoint_path, weight_variant, omega, w)
            for checkpoint_path, variants in checkpoint_variants
            for weight_variant in variants
            for omega in omegas
            for w in ws
        ]
        num_evaluations = len(evaluations)

        if num_evaluations == 1:
            checkpoint_path, weight_variant, omega, w = evaluations[0]
            _load_or_evaluate_checkpoint(
                cfg,
                checkpoint_path,
                output_root,
                device,
                distributed,
                weight_variant,
                omega=omega,
                w=w,
            )
            return

        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "fid_summary.jsonl"
        summary_file = summary_path.open("w") if distributed.is_rank_zero else None
        try:
            for checkpoint_path, weight_variant, omega, w in evaluations:
                checkpoint_output_dir = _checkpoint_variant_output_dir(
                    output_root,
                    checkpoint_path,
                    weight_variant,
                    omega=omega,
                    w=w,
                )
                results = _load_or_evaluate_checkpoint(
                    cfg,
                    checkpoint_path,
                    checkpoint_output_dir,
                    device,
                    distributed,
                    weight_variant,
                    omega=omega,
                    w=w,
                )
                if distributed.is_rank_zero:
                    summary_file.write(json.dumps(results, sort_keys=True) + "\n")
                    summary_file.flush()
        finally:
            if summary_file is not None:
                summary_file.close()

        if distributed.is_rank_zero:
            print(f"Saved FID summary to {summary_path}")
    finally:
        _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()