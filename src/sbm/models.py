"""DiT with class and anchor-time (omega) conditioning: the map T(x; y, omega).

The DiT of Peebles & Xie (2023) with the class label and the anchor time omega as conditioning; no dropout, no learned
sigma, no qk-norm. Attention is the plain implementation, which torch.func.jvp differentiates directly.

`tangent_only_jvp` (off by default; on in the XL recipes): under torch.func.jvp the stock forward-mode formula of a
Linear (and of the patch convolution) also multiplies the primal by the weight's tangent. The weights are not dual
here, so that tangent is materialised as zeros and costs a full GEMM per layer, about one forward per JVP. With the
flag, the token-path Linears and the patch convolution run the stock op on the primal (same value, same reverse-mode
graph) and compute the tangent as the layer applied to the input's tangent, without the bias and outside the
reverse-mode graph (the loss detaches the tangent). Primal, tangent and gradients are bit-identical
(tests/sbm/test_switches.py).

`gw_embedder` (off by default): a guidance-strength input w, the map T(x; y, omega, w). A TimestepEmbedder of w added
to the conditioning like omega's, its output layer zero-initialised, registered after every other module and
initialised after them: every existing parameter keeps its init and its optimizer index, and a checkpoint written
without it loads with the function unchanged (`fill_gw_keys`: only the embedder's own keys may be missing). A caller
that passes no w gets `gw_default`; passing w to a map without the input is an error. `gw_input: kappa` feeds the
embedder 1 - 1/w (iMF's Omega-conditioning; w = 1 is unguided) instead of w itself. `gw_interval` (needs gw_embedder):
two more zero-init embedders for the guidance interval [s_lo, s_hi] (defaults 0, 1), iMF's interval conditioning.
Used by loss.guide.omega_cond and read by the evaluator's gw / gw_lo / gw_hi; the paper's maps do not use them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q * self.head_dim ** -0.5) @ k.transpose(-2, -1)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class _TangentOnly:
    """Mixin for nn.Linear / nn.Conv2d: inside a forward-AD level the output's tangent is `_tangent(input tangent)`."""

    def forward(self, x):
        primal, tangent = fwAD.unpack_dual(x)
        if tangent is None:
            return super().forward(x)
        out = super().forward(primal)
        with torch.no_grad():
            out_tangent = self._tangent(tangent)
        return fwAD.make_dual(out, out_tangent)


class TangentOnlyLinear(_TangentOnly, nn.Linear):
    def _tangent(self, t):
        return F.linear(t, self.weight)


class TangentOnlyConv2d(_TangentOnly, nn.Conv2d):
    def _tangent(self, t):
        return self._conv_forward(t, self.weight, None)


def use_tangent_only_jvp(model: "DiT") -> "DiT":
    """Switch the token-path layers (patch conv; qkv, proj, fc1, fc2 per block; the final linear) to the tangent-only
    forward. Parameters and state-dict keys are unchanged; the adaLN Linears see no tangent and are left alone."""
    model.x_embedder.proj.__class__ = TangentOnlyConv2d
    layers = [model.final_layer.linear]
    for block in model.blocks:
        layers += [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2]
    for layer in layers:
        layer.__class__ = TangentOnlyLinear
    return model


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar (here the anchor time omega) followed by an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """num_classes + 1 rows: the last is the null class (trained by loss.label_dropout, read by the guide)."""

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16,
                 mlp_ratio=4.0, label_dim=1000, train_pos_embed=False, tangent_only_jvp=False, gw_embedder=False,
                 gw_default=1.0, gw_input="raw", gw_interval=False):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.label_dim = label_dim
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = LabelEmbedder(label_dim, hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size)
        # fixed sin-cos embedding (train_pos_embed: trained)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size),
                                      requires_grad=bool(train_pos_embed))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.gw_embedder, self.gw_default = None, float(gw_default)
        if gw_embedder:   # last, and after the init above: nothing that existed before moves or re-draws
            self.gw_embedder = TimestepEmbedder(hidden_size)
            nn.init.normal_(self.gw_embedder.mlp[0].weight, std=0.02)
            nn.init.constant_(self.gw_embedder.mlp[0].bias, 0)
            nn.init.constant_(self.gw_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.gw_embedder.mlp[2].bias, 0)
        if gw_input not in ("raw", "kappa"):
            raise ValueError(f"gw_input must be raw or kappa, got {gw_input}")
        self.gw_input = str(gw_input)
        self.gi_lo_embedder = self.gi_hi_embedder = None
        if gw_interval:
            if not gw_embedder:
                raise ValueError("gw_interval needs gw_embedder")
            self.gi_lo_embedder, self.gi_hi_embedder = TimestepEmbedder(hidden_size), TimestepEmbedder(hidden_size)
            for e in (self.gi_lo_embedder, self.gi_hi_embedder):
                nn.init.normal_(e.mlp[0].weight, std=0.02)
                nn.init.constant_(e.mlp[0].bias, 0)
                nn.init.constant_(e.mlp[2].weight, 0)
                nn.init.constant_(e.mlp[2].bias, 0)
        if tangent_only_jvp:
            use_tangent_only_jvp(self)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.omega_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.omega_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.omega_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.omega_embedder.mlp[2].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for m in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):   # zero readout: the output starts at 0
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def unpatchify(self, x):
        c, p = self.out_channels, self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = torch.einsum("nhwpqc->nchpwq", x.reshape(x.shape[0], h, w, p, p, c))
        return x.reshape(x.shape[0], c, h * p, h * p)

    @staticmethod
    def _per_example(v, B, device):
        if not torch.is_tensor(v):
            v = torch.full((B,), float(v), device=device, dtype=torch.float32)
        return v.to(device).reshape(-1).float().expand(B) if v.numel() == 1 else v.to(device).reshape(-1).float()

    def forward(self, x, y, omega=None, w=None, s_lo=None, s_hi=None, **_):
        """T(x; y, omega[, w]). omega: None (= 1), a scalar, or one value per example; w likewise (None = gw_default),
        and only for a map built with gw_embedder; s_lo / s_hi likewise, only with gw_interval."""
        B = x.shape[0]
        omega = self._per_example(1.0 if omega is None else omega, B, x.device)
        h = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y) + self.omega_embedder(omega)
        if self.gw_embedder is not None:
            wv = self._per_example(self.gw_default if w is None else w, B, x.device)
            c = c + self.gw_embedder(1.0 - 1.0 / wv if self.gw_input == "kappa" else wv)
        elif w is not None:
            raise ValueError("this map has no guidance-strength input (model.gw_embedder is off); w must be None")
        if self.gi_lo_embedder is not None:
            c = c + self.gi_lo_embedder(self._per_example(0.0 if s_lo is None else s_lo, B, x.device))
            c = c + self.gi_hi_embedder(self._per_example(1.0 if s_hi is None else s_hi, B, x.device))
        elif s_lo is not None or s_hi is not None:
            raise ValueError("this map has no guidance-interval input (model.gw_interval is off)")
        c = c.reshape(B, 1, -1)
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.final_layer(h, c))


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.stack(np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)), axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)


def _sincos_1d(embed_dim, pos):
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


GEOMETRY = {  # name -> (depth, hidden, patch, heads)
    "DiT-XL/2": (28, 1152, 2, 16), "DiT-L/2": (24, 1024, 2, 16), "DiT-B/2": (12, 768, 2, 12), "DiT-S/2": (12, 384, 2, 6),
    "DiT-tiny": (2, 64, 2, 4),   # tests only
}


def build_model(name="DiT-B/2", input_size=32, in_channels=4, label_dim=1000, mlp_ratio=4.0, train_pos_embed=False,
                tangent_only_jvp=False, gw_embedder=False, gw_default=1.0, gw_input="raw", gw_interval=False, **_):
    """**_ swallows model keys of older configs that no longer exist."""
    depth, hidden, patch, heads = GEOMETRY[name]
    return DiT(input_size=input_size, patch_size=patch, in_channels=in_channels, hidden_size=hidden, depth=depth,
               num_heads=heads, mlp_ratio=mlp_ratio, label_dim=label_dim, train_pos_embed=train_pos_embed,
               tangent_only_jvp=bool(tangent_only_jvp), gw_embedder=bool(gw_embedder),
               gw_default=float(gw_default), gw_input=str(gw_input), gw_interval=bool(gw_interval))


GW_PREFIX = ("gw_embedder.", "gi_lo_embedder.", "gi_hi_embedder.")


def fill_gw_keys(own: dict, sd: dict, prefix: str = "") -> list[str]:
    """Add to the state dict `sd` the guidance embedder's keys it lacks (a checkpoint written before the input existed),
    taken from `own` (the freshly built model's state: output layer zero, so the function is unchanged). `prefix` is
    the keys' prefix in both ('model.' for a Lightning module). Any other difference is left for the strict load to
    report. Returns the keys added."""
    added = [k for k in own if k.startswith(tuple(prefix + g for g in GW_PREFIX)) and k not in sd]
    for k in added:
        sd[k] = own[k].detach().clone()
    return added


def read_weights(path: str | Path, which: str = "ema") -> dict:
    """The DiT weights of a checkpoint, keys without the `model.` prefix.

    - a training checkpoint (sbm.train): `state_dict` is the active EMA copy, `current_model_state` the raw weights;
      which = ema | raw | ema_<label> (e.g. ema_0_9999, from `ema_state_dicts`);
    - a raw payload whose `lora_state_dict` is the full DiT state;
    - an export (sbm.infer) or any other single-copy file: `state_dict` only, whatever `which` asks for.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "lora_state_dict" in ck:
        return dict(ck["lora_state_dict"])
    if which == "raw" and "current_model_state" in ck:
        sd = ck["current_model_state"]
    elif which.startswith("ema_") and which[4:] in ck.get("ema_state_dicts", {}):
        sd = ck["ema_state_dicts"][which[4:]]
    else:
        sd = ck["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def load_weights(model: DiT, path: str | Path, which: str = "ema") -> DiT:
    sd = {k: v.to(torch.float32) for k, v in read_weights(path, which).items()}
    if getattr(model, "gw_embedder", None) is not None:
        fill_gw_keys(model.state_dict(), sd)
    model.load_state_dict(sd, strict=True)
    return model
