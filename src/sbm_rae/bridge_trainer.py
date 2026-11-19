"""Lightning module for the Bridge-V2 / RT-1 world models: sbm_rae.trainer.TrainingModule with the batch of
sbm_rae.bridge_data (uint8 frames + actions) encoded on the GPU by the codec (SD-VAE / SD3 VAE / RAE, `vae.kind`) and split
into the target x = the latent of frame n and the condition {"obs": the n history latents, "act": their actions}
(DriftWorld's layout: n_history = 4, clip = 5 frames, action i is the one taken at frame i). Everything else (EMA, the
guide's frozen teacher, Muon / AdamW, checkpoints) is the parent's.
"""
import torch

from sbm_rae.trainer import TrainingModule, _ema_log_prefix


class BridgeTrainingModule(TrainingModule):
    def __init__(self, cfg, model, loss_fn, weighting_model=None):
        tangent_only = bool(getattr(loss_fn, "tangent_only_jvp", False))
        if tangent_only:
            loss_fn.tangent_only_jvp = False      # the parent would convert a DiT's layers; the U-Net converts its own
        super().__init__(cfg, model, loss_fn, weighting_model=weighting_model)
        if tangent_only:
            loss_fn.tangent_only_jvp = True
            if hasattr(self.model, "use_tangent_only_jvp"):
                self.model.use_tangent_only_jvp()
        self.n_history = int(cfg.dataset.get("n_history", 4))
        self.encode_chunk = int(cfg.dataset.get("encode_chunk", 64))

    # ------------------------------------------------------------------ frames -> latents -> (x, cond)
    def encode_frames(self, frames):
        """(B, T, 3, H, W) uint8 -> (B, T, C, h, w) float32 latents (scaled as the parent's _encode_to_latents)."""
        B, T = frames.shape[:2]
        if frames.dtype != torch.uint8:          # cached latents (dataset.latent_cache_dir): already encoded and scaled
            return frames.float()
        flat = frames.flatten(0, 1).float().div_(127.5).sub_(1.0)
        if not self.vae_enabled:
            return flat.reshape(B, T, *flat.shape[1:])
        outs = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            for i in range(0, flat.shape[0], self.encode_chunk):
                z = self.vae.encode(flat[i : i + self.encode_chunk].to(self.vae.dtype)).latent_dist.sample()
                outs.append(((z - self.latents_bias) * self.latents_scale).float())
        z = torch.cat(outs)
        return z.reshape(B, T, *z.shape[1:])

    def split_clip(self, latents, actions):
        n = self.n_history
        x = latents[:, n]
        cond = {"obs": latents[:, :n].flatten(1, 2), "act": actions[:, :n].float()}
        return x, cond

    def _clip_losses(self, batch):
        latents = self.encode_frames(batch["x"])
        x, cond = self.split_clip(latents, batch["act"])
        return self._compute_losses(x, cond)

    # ------------------------------------------------------------------ steps (the parent's logging)
    def training_step(self, batch, batch_idx):
        del batch_idx
        losses, aux_losses = self._clip_losses(batch)
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for name, value in aux_losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False, prog_bar=name.endswith("_mse"), logger=True)
        self._log_loss_grad_norms(losses)
        total_loss = sum(losses.values())
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        val_loss, _ = self._clip_losses(batch)
        val_prefix = _ema_log_prefix("val", self.cfg)
        for name, loss in val_loss.items():
            self.log(f"{val_prefix}/{name}", loss, on_step=False, on_epoch=True, sync_dist=True)
        return sum(val_loss.values())

    # ------------------------------------------------------------------ one-pass prediction (the scorer and the rollouts)
    @torch.no_grad()
    def predict_next(self, obs_latents, actions, omega=1.0, w=None, noise=None, model=None):
        """T(z; history, actions) for a batch: obs_latents (B, n, C, h, w), actions (B, n, A) -> (B, C, h, w)."""
        net = self.model if model is None else model
        B, n, C, h, w_ = obs_latents.shape
        z = torch.randn(B, C, h, w_, device=obs_latents.device, dtype=obs_latents.dtype) if noise is None else noise
        cond = {"obs": obs_latents.flatten(1, 2), "act": actions.float()}
        return net(z, cond, omega=omega, w=w)
