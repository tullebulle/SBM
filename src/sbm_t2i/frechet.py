"""The Frechet feature term (loss.frechet): the Frechet distance between DINOv2 features of fresh one-step samples and
of decoded TRAIN latents, pushed down through the map.

Features: DINOv2 ViT-B/14 at 224 px, mean patch token (+) cls token, 1536-d, of the SD-VAE decoding of the samples.
Per-sample direction v_b = (mu_g - mu_r) + W (f_b - mu_g), W = dFD/dSigma_g = I - A (A Sigma_g A)^{-1/2} A,
A = Sigma_r^{1/2}; mu_g and Sigma_g are running estimates (cumulative mean for the first 1/(1 - ema) calls, then the
EMA), all-reduced over ranks; W and FD are refreshed every `w_every` calls. The unit loss mean_b <sg(v_b), f_b> has
exactly the finite-sample FD gradient direction. `anneal` = min(1, sqrt(FD / fd_ref)) fades the term as FD falls;
fd_ref = the first FD after warm-up unless given (a run resumed from one with the term on must pass that run's value:
this state is not checkpointed, unless loss.frechet.save_state puts `state_dict()` into the checkpoint).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

DINO_HUB = os.environ.get("SBM_DINO_HUB") or str(Path(os.environ.get("SBM_STORE") or Path(__file__).resolve().parents[2] / "store")
                                                 / "refs/torch_home/hub/facebookresearch_dinov2_main")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrechetTerm:
    def __init__(self, stats_path, vae_name="stabilityai/sd-vae-ft-mse", latents_scale=0.18215, model_name="dinov2_vitb14",
                 hub_dir=DINO_HUB, ema=0.99, resolution=224, w_every=10, eig_floor=1e-5, fd_ref=None, nets=None):
        state = torch.load(stats_path, map_location="cpu") if isinstance(stats_path, (str, os.PathLike)) else stats_path
        if "cov" not in state:
            raise ValueError(f"the Frechet term needs the real feature covariance ('cov') in {stats_path}")
        self.mu_r, self.sig_r, self.n_real = state["mean"].float(), state["cov"].double(), int(state.get("n", 0))
        self.vae_name, self.latents_scale, self.model_name, self.hub_dir = str(vae_name), float(latents_scale), str(model_name), str(hub_dir)
        self.ema, self.resolution, self.w_every, self.eig_floor = float(ema), int(resolution), int(w_every), float(eig_floor)
        self.fd_ref = None if fd_ref is None else float(fd_ref)
        self._nets, self._device = nets, None     # [vae, dino]; tests pass stand-ins
        self.mu_g = self.S_g = self._W = self._A = self.fd = None
        self._calls = 0

    def build(self, device):
        if self._nets is None:
            from diffusers import AutoencoderKL
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            vae = AutoencoderKL.from_pretrained(self.vae_name, local_files_only=True).to(device).eval().requires_grad_(False)
            dino = torch.hub.load(self.hub_dir, self.model_name, source="local", pretrained=True).to(device).eval().requires_grad_(False)
            self._nets = [vae, dino]
        self._device = device
        self.mu_r = self.mu_r.to(device)
        if self.mu_g is not None:          # restored by load_state_dict
            self.mu_g = self.mu_g.to(device)
        if self.S_g is not None:
            self.S_g = self.S_g.to(device)
        sig = 0.5 * (self.sig_r + self.sig_r.T).to(device)
        lr, Vr = torch.linalg.eigh(sig)
        self._A = (Vr * lr.clamp_min(0).sqrt()) @ Vr.T          # Sigma_r^{1/2}
        self.sig_r, self._tr_r = sig, float(sig.trace())
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def state_dict(self):
        """The running state a resume needs; W is recomputed on the first call after load_state_dict."""
        cpu = lambda v: None if v is None else v.detach().cpu()
        return {"mu_g": cpu(self.mu_g), "S_g": cpu(self.S_g), "fd": self.fd, "fd_ref": self.fd_ref, "calls": self._calls}

    def load_state_dict(self, state):
        self.mu_g, self.S_g, self.fd, self.fd_ref = state["mu_g"], state["S_g"], state["fd"], state["fd_ref"]
        self._calls = int(state["calls"])
        self._W, self._A = None, None      # rebuild on the next call (moves the moments to the device, recomputes W)

    def features(self, latents):
        """latents (B, 4, 32, 32) in model space -> DINO features (B, 1536), differentiable."""
        vae, dino = self._nets
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            img = vae.decode(latents.float() / self.latents_scale).sample
        img = ((img.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        img = torch.nn.functional.interpolate(img, size=(self.resolution, self.resolution), mode="bicubic", align_corners=False)
        img = (img - self._mean) / self._std
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = dino.forward_features(img)
        return torch.cat([out["x_norm_patchtokens"].float().mean(1), out["x_norm_clstoken"].float()], dim=1)

    def __call__(self, latents):
        """-> (unit loss, aux) for a batch of generated latents at the read point."""
        if self._A is None or self._device != latents.device:
            self.build(latents.device)
        f = self.features(latents)
        with torch.no_grad():
            m = f.mean(0).detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(m, op=dist.ReduceOp.AVG)
            self._calls += 1
            warm = int(round(1.0 / (1.0 - self.ema))) if 0.0 < self.ema < 1.0 else 1
            rate = max(1.0 - self.ema, 1.0 / self._calls) if self.ema > 0 else 1.0
            self.mu_g = m if self.mu_g is None else (1.0 - rate) * self.mu_g + rate * m
            gap = (self.mu_g - self.mu_r).norm()
            fd64 = f.detach().double()
            S = fd64.T @ fd64 / fd64.shape[0]
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(S, op=dist.ReduceOp.AVG)
            if torch.isfinite(S).all():   # every rank sees the same all-reduced S, so every rank skips together
                self.S_g = S if self.S_g is None else (1.0 - rate) * self.S_g + rate * S
            mu = self.mu_g.double()
            if (self._W is None or self._calls % self.w_every == 0) and self.S_g is not None:
                sig_g = self.S_g - torch.outer(mu, mu); sig_g = 0.5 * (sig_g + sig_g.T)
                M = self._A @ sig_g @ self._A; M = 0.5 * (M + M.T)
                # a degenerate Sigma_g (from scratch every sample decodes to about the same image) makes the eigensolver
                # fail on repeated eigenvalues: a tiny ridge breaks the degeneracy, LAPACK on the CPU as the fallback
                ridge = 1e-9 * self._tr_r / M.shape[0]
                eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
                try:
                    lm, Vm = torch.linalg.eigh(M + ridge * eye)
                except RuntimeError:
                    lm, Vm = torch.linalg.eigh((M + ridge * eye).cpu())
                    lm, Vm = lm.to(M.device), Vm.to(M.device)
                lm = lm - ridge
                self.fd = float((mu - self.mu_r.double()).pow(2).sum() + sig_g.trace() + self._tr_r - 2.0 * lm.clamp_min(0).sqrt().sum())
                lmc = lm.clamp_min(max(self.eig_floor * float(lm.max()), 1e-6 * self._tr_r / M.shape[0]))
                self._W = eye - self._A @ ((Vm * lmc.rsqrt()) @ Vm.T) @ self._A
                if self.fd_ref is None and self._calls >= warm:
                    self.fd_ref = max(self.fd, 1e-8)
            v = ((mu - self.mu_r.double()).unsqueeze(0) + (fd64 - mu) @ self._W).float()   # W symmetric
            anneal = (torch.ones((), device=f.device) if self.fd_ref is None
                      else torch.tensor(min(1.0, (max(self.fd, 0.0) / self.fd_ref) ** 0.5), device=f.device))
        loss = (v * f).sum(1).mean()
        return loss, {"frechet_gap": gap.detach(), "frechet_anneal": anneal.detach(),
                      "frechet_fd": torch.tensor(float(self.fd), device=f.device)}
