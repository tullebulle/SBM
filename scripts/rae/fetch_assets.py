"""Download the public frozen networks the models need into the store layout of sbm_rae.paths (needs internet once;
sampling, training and scoring then run offline with HF_HUB_OFFLINE=1). `bash download.sh` calls it.

    <store>/rae/assets/rae/decoders/dinov2/wReg_small/ViTXL_n08/model.pt   RAE ViT-XL decoder   (nyu-visionx/RAE-collections)
    <store>/rae/assets/rae/stats/dinov2/wReg_small/imagenet1k/stat.pt      latent normalization (nyu-visionx/RAE-collections)
    <store>/rae/assets/rae/decoder_config_ViTXL/config.json                the decoder config   (third_party/RAE/configs)
    <store>/rae/assets/hf_home/hub/models--facebook--dinov2-with-registers-small   the frozen encoder
    <store>/rae/assets/torch_home/hub/checkpoints/weights-inception-2015-12-05-*.pth  FID Inception (torch-fidelity)
    with --sdvae: <store>/rae/assets/hf_home/hub/models--stabilityai--sd-vae-ft-mse (the SD-VAE line)
    with --wm (the world models' evaluator, scripts/rae/eval_bridge.py, and its FVD step):
      <store>/rae/assets/hf_home/hub/models--stabilityai--stable-diffusion-3-medium-diffusers  the SD3 VAE (vae/ only; a
          gated repo: accept its license on huggingface.co and `hf auth login` first)
      <store>/rae/assets/torch_home/hub/checkpoints/{pt_inception-2015-12-05-6726825d,alexnet-owt-7be5be79}.pth  FID (pytorch-fid), LPIPS
      <store>/rae/assets/dnnlib/downloads/*_i3d_torchscript.pt   the I3D detector of stylegan-v's FVD (DNNLIB_CACHE_DIR)
      <store>/refs/code/stylegan-v                                stylegan-v at the pinned commit (its FVD script), with the
          one-line scipy >= 1.18 fix (sqrtm has no disp= argument)

The revisions and sha256 sums are the ones the released checkpoints were trained and scored with; a mismatch fails.
The FID reference statistics (<store>/ckpt/imagenet/fid_cache/) come with the ImageNet checkpoints (download.sh imagenet).

    python scripts/rae/fetch_assets.py [--store <dir>] [--sdvae] [--wm]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sbm_rae.paths import fid_cache_dir, store_root  # noqa: E402

RAE_REPO = "nyu-visionx/RAE-collections"
RAE_FILES = {
    "decoders/dinov2/wReg_small/ViTXL_n08/model.pt": "d1124c1681ada94975c6350ee32b29641f85d599c78bdb744f20873a10bb544c",
    "stats/dinov2/wReg_small/imagenet1k/stat.pt": "fda0194c492bce64ffa8a12202ecef6484dd41256ab0204319c019d3dfc3cb69",
}
HF_MODELS = {  # repo -> pinned revision
    "facebook/dinov2-with-registers-small": "0d9846e56b43a21fa46d7f3f5070f0506a5795a9",
}
SDVAE = {"stabilityai/sd-vae-ft-mse": "31f26fdeee1355a5c34592e401dd41e45d25a493"}
INCEPTION_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"

# --wm: what the world models' paper reads used
SD3 = ("stabilityai/stable-diffusion-3-medium-diffusers", "ea42f8cef0f178587cf766dc8129abd379c90671", {
    "model_index.json": "f6b0490bd731d8467e7b478d7f7d13b447575d6fe8b0ae9948af526820823137",
    "vae/config.json": "2b82cf4cf6728bb2b411199c437c16ce61a82265ebd8461d66600073c8470ee4",
    "vae/diffusion_pytorch_model.safetensors": "f9b67a279283625caee39d61eacb5324243848477b4eb535355eaaa8423d4e09",
})
TORCH_WEIGHTS = {  # url -> sha256 (pytorch-fid's Inception, torchvision's AlexNet under LPIPS)
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth":
        "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
    "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth":
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
}
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_FILE = "5780f6fd48bed6b4f055c5cac089dbee_i3d_torchscript.pt"   # dnnlib's cache name: md5(url) + "_" + file name
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"
STYLEGAN_V = ("https://github.com/universome/stylegan-v", "3fecd69c602e1cda204357201461c0fb0a634909", {
    "src/metrics/frechet_inception_distance.py": "8cb496366491a745ea9880e7a0e534c005025796bfa6418095637dfff53bf3c4",
    "src/metrics/frechet_video_distance.py": "88b09b7345e95679ba8f10f85a2befed5e3f3977a84669adc7b55944ef96e99e",
})
SQRTM_OLD = "s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) #"
SQRTM_NEW = "s = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real))  # scipy >= 1.18 has no disp= #"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check(path, expected):
    got = sha256(path)
    if got != expected:
        sys.exit(f"sha256 mismatch for {path}: {got} (expected {expected})")
    print(f"ok  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="store root (default: $SBM_STORE or <repo>/store)")
    ap.add_argument("--sdvae", action="store_true", help="also fetch the SD-VAE (stabilityai/sd-vae-ft-mse)")
    ap.add_argument("--wm", action="store_true", help="also fetch the world models' evaluation assets (SD3 VAE, FID / LPIPS / FVD networks, stylegan-v)")
    args = ap.parse_args()
    if args.store:
        os.environ["SBM_STORE"] = args.store
    assets = store_root() / "rae" / "assets"
    hf_hub = assets / "hf_home" / "hub"
    torch_home = assets / "torch_home"
    for d in (assets / "rae", hf_hub, torch_home):
        d.mkdir(parents=True, exist_ok=True)

    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download, snapshot_download

    for name, digest in RAE_FILES.items():
        target = assets / "rae" / name
        if not target.is_file():
            hf_hub_download(RAE_REPO, name, local_dir=assets / "rae")
        check(target, digest)
    config = assets / "rae" / "decoder_config_ViTXL" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "third_party" / "RAE" / "configs" / "decoder" / "ViTXL" / "config.json", config)
    print(f"ok  {config}")

    for repo, rev in {**HF_MODELS, **(SDVAE if args.sdvae else {})}.items():
        path = snapshot_download(repo, revision=rev, cache_dir=hf_hub)
        # a download by commit hash does not write refs/main, which the offline from_pretrained(<repo id>) resolves
        ref = hf_hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(rev)
        print(f"ok  {repo}@{rev[:10]} -> {path} (refs/main)")

    os.environ["TORCH_HOME"] = str(torch_home)
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3

    torch.hub.load_state_dict_from_url(URL_INCEPTION_V3, progress=True)
    check(Path(torch.hub.get_dir()) / "checkpoints" / Path(URL_INCEPTION_V3).name, INCEPTION_SHA256)

    if args.wm:
        fetch_wm(assets, hf_hub, torch_home)

    adm = fid_cache_dir() / "adm_jit_in256_stats.npz"
    print(f"\nassets ready under {assets}" + ("" if adm.is_file() else
          f"\nfor the ADM-reference FID: bash download.sh imagenet (puts adm_jit_in256_stats.npz at {adm})"))


def fetch_wm(assets, hf_hub, torch_home):
    from huggingface_hub import snapshot_download

    repo, rev, files = SD3
    path = Path(snapshot_download(repo, revision=rev, cache_dir=hf_hub, allow_patterns=list(files)))
    ref = hf_hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(rev)
    for name, digest in files.items():
        check(path / name, digest)

    import torch

    for url, digest in TORCH_WEIGHTS.items():
        torch.hub.load_state_dict_from_url(url, progress=True)
        check(torch_home / "hub" / "checkpoints" / Path(url).name, digest)

    i3d = assets / "dnnlib" / "downloads" / I3D_FILE
    if not i3d.is_file():
        i3d.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(I3D_URL, i3d)
    check(i3d, I3D_SHA256)

    url, commit, patched = STYLEGAN_V
    dest = store_root() / "refs" / "code" / "stylegan-v"
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    for name, digest in patched.items():
        f = dest / name
        f.write_text(f.read_text().replace(SQRTM_OLD, SQRTM_NEW))
        check(f, digest)


if __name__ == "__main__":
    main()
