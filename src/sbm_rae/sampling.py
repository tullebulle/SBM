from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision


def build_class_labels(num_samples, num_classes, class_labels=None):
    if class_labels:
        labels = torch.tensor(class_labels, dtype=torch.long)
    else:
        labels = torch.arange(num_samples, dtype=torch.long) % num_classes

    if labels.numel() < num_samples:
        repeat = (num_samples + labels.numel() - 1) // labels.numel()
        labels = labels.repeat(repeat)

    return labels[:num_samples]


def build_overfitting_class_labels(num_samples, first_batch_label=None):
    if first_batch_label is None:
        return None

    class_label = int(first_batch_label)
    return torch.full((num_samples,), class_label, dtype=torch.long)


def _autocast_context(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_generation_steps(generation_steps):
    if isinstance(generation_steps, (int, float)):
        steps = [int(generation_steps)]
    else:
        steps = [int(step) for step in generation_steps]
    steps = sorted(set(steps))
    if not steps:
        raise ValueError("generation_steps must contain at least one step count")
    if steps[0] < 1:
        raise ValueError(f"generation_steps must all be >= 1, got {steps}")
    return steps


def _map_output(model, x, labels, omega, w):
    output = model(x, labels, omega=omega, w=w)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def run_generation_step_captures(module, noise, labels, capture_steps, omega=None, w=None, autocast_enabled=True):
    """Apply the map repeatedly, T(z), T(T(z)), ..., and return {k: the output of call k} for every k in
    `capture_steps` (the released models are read with one call)."""
    capture_steps = normalize_generation_steps(capture_steps)
    wanted = set(capture_steps)
    captured = {}
    current = noise
    for step in range(1, capture_steps[-1] + 1):
        with _autocast_context(current.device, enabled=autocast_enabled):
            current = _map_output(module.model, current, labels, omega, w)
        if step in wanted:
            captured[step] = current
    return captured


@torch.no_grad()
def decode_latents(module, latents, vae_batch_size, inverse_scaler=None):
    latents = latents / module.latents_scale + module.latents_bias

    # Pixel-space models emit images directly; there is no VAE to run.
    if not getattr(module, "vae_enabled", True):
        images = latents
        if inverse_scaler is not None:
            images = inverse_scaler(images)
        return images.clamp(0.0, 1.0)

    images = []

    for start in range(0, latents.shape[0], vae_batch_size):
        end = min(start + vae_batch_size, latents.shape[0])
        with torch.amp.autocast("cuda", enabled=False):
            batch = latents[start:end].to(dtype=module.vae.dtype)
            decoded = module.vae.decode(batch).sample
        images.append(decoded.to(dtype=latents.dtype))

    images = torch.cat(images, dim=0)
    if inverse_scaler is not None:
        images = inverse_scaler(images)
    return images.clamp(0.0, 1.0)


@torch.no_grad()
def generate_image_batches(
    module,
    labels,
    latent_shape,
    sample_batch_size,
    generation_steps,
    vae_batch_size,
    seed,
    inverse_scaler=None,
    device=None,
    omega=None,
    w=None,
    autocast_enabled=True,
):
    if device is None:
        device = next(module.model.parameters()).device

    generation_steps = normalize_generation_steps(generation_steps)
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    for start in range(0, labels.shape[0], sample_batch_size):
        end = min(start + sample_batch_size, labels.shape[0])
        batch_labels = labels[start:end].to(device)
        noise = torch.randn((end - start, *latent_shape), generator=generator, device=device)
        latents_by_steps = run_generation_step_captures(
            module,
            noise,
            batch_labels,
            capture_steps=generation_steps,
            omega=omega,
            w=w,
            autocast_enabled=autocast_enabled,
        )
        images_by_steps = {
            steps: decode_latents(
                module,
                latents,
                vae_batch_size=vae_batch_size,
                inverse_scaler=inverse_scaler,
            )
            for steps, latents in latents_by_steps.items()
        }
        yield images_by_steps, batch_labels


def save_images(images, output_dir, start_index=0, labels=None, indices=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for offset, image in enumerate(images):
        index = int(indices[offset]) if indices is not None else start_index + offset
        suffix = ""
        if labels is not None:
            suffix = f"_class{int(labels[offset])}"
        torchvision.utils.save_image(image, output_path / f"{index:06d}{suffix}.png")