"""Sample from the released SBM checkpoints (one network evaluation per image / frame).

    python sample.py t2i "a red fox curled up in fresh snow, soft morning light"      # text-to-image (SBM-XL)
    python sample.py imagenet --classes 207 360 387 974                               # ImageNet 256 (SBM-XL, RAE)
    python sample.py wm                                                               # world model rollouts (Bridge-V2)

Images land in samples/<task>/ (one PNG each plus grid.png); world-model rollouts as mp4 (ground truth | SBM).
Checkpoints are read from <store>/ckpt/<task>/ (`bash download.sh <task>`); `--ckpt` takes any other path.
"""
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
STORE = Path(os.environ.setdefault("SBM_STORE", str(REPO / "store"))).resolve()
os.environ.setdefault("HF_HOME", str(STORE / "rae" / "assets" / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(STORE / "rae" / "assets" / "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# model name -> checkpoint file prefix in <store>/ckpt/<task>/ (the file is <prefix>_step<N>_ema<decay>.pt)
MODELS = {
    "imagenet": {"rae-xl": "sbm_rae_xl", "rae-b": "sbm_rae_b", "sdvae-xl": "sbm_sdvae_xl", "sdvae-b": "sbm_sdvae_b"},
    "t2i": {"xl": "sbm_t2i_xl", "b": "sbm_t2i_b"},
    "wm": {"bridge": "sbm_wm_bridge", "rt1": "sbm_wm_rt1", "bridge-dino": "sbm_wm_bridge_dino", "rt1-dino": "sbm_wm_rt1_dino"},
}
DEFAULT_CLASSES = [88, 207, 279, 360, 387, 417, 933, 974]   # macaw, golden retriever, arctic fox, otter, red panda,
                                                            # balloon, cheeseburger, geyser
DEFAULT_PROMPTS = [
    "a red fox curled up asleep in fresh snow, soft morning light",
    "a cozy wooden cabin by a lake at sunset, mountains in the background",
    "a bowl of ramen with a soft-boiled egg, steam rising, close-up",
    "a hummingbird hovering next to a pink flower, shallow depth of field",
]


def find_checkpoint(task, model, ckpt=None):
    if ckpt:
        return Path(ckpt)
    prefix = MODELS[task][model]
    pattern = re.compile(re.escape(prefix) + r"_step(\d+)_ema[0-9_]+\.pt$")
    folder = STORE / "ckpt" / task
    hits = sorted((int(m.group(1)), p) for p in folder.glob("*.pt") if (m := pattern.match(p.name)))
    if not hits:
        sys.exit(f"no {prefix}_step*.pt in {folder}: run `bash download.sh {task}` (or pass --ckpt)")
    return hits[-1][1]


def save_grid(images, out, nrow):
    """images: uint8 (N, 3, H, W) -> out/000.png ... and out/grid.png"""
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(out / f"{i:03d}.png")
    grid = make_grid(images.float(), nrow=nrow, padding=2, pad_value=255).to(torch.uint8)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out / "grid.png")
    print(f"{len(images)} images -> {out}/ (grid.png)")


def run_imagenet(a):
    import torch

    path = find_checkpoint("imagenet", a.model, a.ckpt)
    labels = torch.tensor(a.classes).repeat(a.per_class)        # one column per class
    if a.model.startswith("sdvae"):
        from sbm.infer import generate, load_export

        model, cfg, meta = load_export(path, device=a.device)
        images = generate(model, cfg, labels, omega=a.omega, label_scale=a.label_scale, seed=a.seed, device=a.device)
    else:
        from sbm_rae.generate import generate, load_generator

        gen = load_generator(path, device=a.device, label_scale=a.label_scale)
        images = generate(gen, labels, seed=a.seed, omega=a.omega)
    print(f"{path.name}: {len(labels)} samples, classes {a.classes}")
    save_grid(images, Path(a.out or "samples/imagenet"), nrow=len(a.classes))


def run_t2i(a):
    from sbm_t2i.infer import generate, load_model

    path = find_checkpoint("t2i", a.model, a.ckpt)
    prompts = list(a.prompts or DEFAULT_PROMPTS)
    if a.prompt_file:
        prompts += [line.strip() for line in Path(a.prompt_file).read_text().splitlines() if line.strip()]
    model, cfg, _ = load_model(path)
    model.to(a.device)
    batch = prompts * a.per_prompt                               # one column per prompt
    images = generate(model, cfg, batch, omega=a.omega, label_scale=a.label_scale, seed=a.seed, device=a.device)
    print(f"{path.name}: {len(batch)} samples for {len(prompts)} prompts")
    save_grid(images, Path(a.out or "samples/t2i"), nrow=min(len(prompts), 8))


def run_wm(a):
    import imageio.v2 as imageio
    import numpy as np
    import torch

    task_dir = STORE / "ckpt" / "wm"
    path = find_checkpoint("wm", a.model, a.ckpt)
    spec = importlib.util.spec_from_file_location("eval_bridge", REPO / "scripts" / "rae" / "eval_bridge.py")
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    from sbm_rae.bridge_data import BridgeClips

    device = torch.device(a.device)
    model, cfg, step, _ = eb.load_checkpoint(str(path), None, device)
    read = (torch.load(str(path), map_location="meta", weights_only=False).get("meta") or {}).get("read") or {}
    omega = a.omega if a.omega is not None else read.get("omega", 0.8)
    frame_size = read.get("frame_size")
    if frame_size:
        os.environ["BRIDGE_FRAME_SIZE"] = str(frame_size)
    dataset = "rt1" if a.model.startswith("rt1") else "bridge"
    root = Path(a.episodes) if a.episodes else task_dir / "examples" / dataset
    clips = BridgeClips(root, "val", n_frames=1, action_dim=int(cfg.dataset.get("action_dim", 7)), full_video=True)
    codec = eb.Codec(cfg, device)
    n = int(cfg.dataset.get("n_history", 4))
    out = Path(a.out or f"samples/wm_{a.model}")
    out.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(clips.stems):
        item = clips[i]
        frames, act = item["x"], item["act"]
        L = min(frames.shape[0], act.shape[0], n + a.frames) if a.frames else min(frames.shape[0], act.shape[0])
        frames, act = frames[:L], act[:L]
        obs = codec.encode(frames)
        gen = codec.decode(eb.rollout(model, obs, act.to(device), n, 8, omega, None, seed=a.seed + i))
        pred = eb.to_u8(gen).transpose(0, 2, 3, 1)          # (L, H, W, 3); the first frames are the given history
        real = frames.permute(0, 2, 3, 1).numpy()
        gap = np.full((real.shape[1], 4, 3), 255, np.uint8)
        video = [np.concatenate([r, gap, p], axis=1) for r, p in zip(real, pred)]
        text = (root / "val" / f"{stem}.txt")
        imageio.mimwrite(out / f"{stem}.mp4", video, fps=a.fps, macro_block_size=1)
        print(f"{stem}: {L} frames ({n} given) -> {out / (stem + '.mp4')}"
              + (f"  [{text.read_text().strip()}]" if text.is_file() else ""))
    print(f"{path.name} (step {step}), omega {omega}: left ground truth, right SBM (8-frame chunks from the true history)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", help="checkpoint path (default: the newest matching file in <store>/ckpt/<task>/)")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--omega", type=float, help="anchor omega (default: the checkpoint's read setting)")
    common.add_argument("--out", help="output folder")
    common.add_argument("--device", default="cuda")

    p = sub.add_parser("imagenet", parents=[common], help="class-conditional ImageNet 256")
    p.add_argument("--model", choices=list(MODELS["imagenet"]), default="rae-xl")
    p.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="ImageNet class ids (0-999)")
    p.add_argument("--per-class", type=int, default=1)
    p.add_argument("--label-scale", type=float, help="class-label scale (default: the checkpoint's read setting)")

    p = sub.add_parser("t2i", parents=[common], help="text-to-image 256")
    p.add_argument("prompts", nargs="*", help="prompts (default: a few examples)")
    p.add_argument("--prompt-file", help="a text file with one prompt per line")
    p.add_argument("--model", choices=list(MODELS["t2i"]), default="xl")
    p.add_argument("--per-prompt", type=int, default=1)
    p.add_argument("--label-scale", type=float, help="text-condition scale (default: the checkpoint's read setting)")

    p = sub.add_parser("wm", parents=[common], help="action-conditioned world-model rollouts")
    p.add_argument("--model", choices=list(MODELS["wm"]), default="bridge")
    p.add_argument("--episodes", help="a folder with val/<episode>.{mp4,npz} (default: the downloaded examples)")
    p.add_argument("--frames", type=int, default=0, help="frames to generate per episode (default: the whole episode)")
    p.add_argument("--fps", type=int, default=5)

    a = ap.parse_args()
    {"imagenet": run_imagenet, "t2i": run_t2i, "wm": run_wm}[a.task](a)


if __name__ == "__main__":
    main()
