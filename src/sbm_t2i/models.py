"""The networks of the text-to-image package, and the class-conditional DiT they grew from.

DiT (Peebles & Xie, 2023) with class and anchor-time (omega) conditioning: the map T(x; y, omega). No REPA projector,
no auxiliary velocity head, no dropout, no learned sigma, no qk-norm; DiT's state-dict keys (`load_weights`).
The latent text-to-image network, TextDiT, is in sbm_t2i.dit_t2i; `build_model` builds every network by name.

Attention is the plain implementation: torch.func.jvp differentiates it directly.

`tangent_only_jvp` (off by default): under torch.func.jvp the stock forward-mode formula of a Linear (and of the patch
convolution) also multiplies the primal by the weight's tangent. The weights are not dual here, so that tangent is
materialised as zeros and costs a full GEMM per layer, about one forward per JVP. With the flag, the token-path Linears
and the patch convolution run the stock op on the primal (same value, same reverse-mode graph) and compute the
tangent as the layer applied to the input's tangent, without the bias and outside the reverse-mode graph (the loss
detaches the tangent). Primal, tangent and gradients are bit-identical.

`gw_embedder` (off by default): a guidance-strength input w, the map T(x; y, omega, w).
A TimestepEmbedder of w added to the conditioning like omega's, its output layer zero-initialised, registered after
every other module and initialised after them: every existing parameter keeps its init and its optimizer index, and
a checkpoint written without it loads with the function unchanged (`fill_gw_keys`: only the embedder's own keys may be
missing). A caller that passes no w gets `gw_default`; passing w to a map without the input is an error.
`gw_input: kappa` feeds the embedder 1 - 1/w (iMF's Omega-conditioning; w = 1 is unguided) instead of w itself.
`gw_interval` (needs gw_embedder): two more zero-init embedders for the guidance interval [s_lo, s_hi] (defaults 0, 1),
iMF's interval conditioning (Geng et al., improved MeanFlow).

MM-JiT (`MMJiT`, names `MMJiT-*`): MiniT2I's pixel-space text-to-image transformer as the map T(x; text, omega). MiniT2I's
released network is a denoiser D(x; text) with no time input, so it is the map's diagonal read as it stands; the anchor
omega enters as `omega_tokens` prefix tokens of the text stream (the conditioning of MiniT2I's Mean Flow student), the
only parameters a MiniT2I checkpoint lacks (`load_weights` leaves them at their init). State-dict keys are MiniT2I's, so
`minit2i-*/transformer/diffusion_pytorch_model.safetensors` loads directly.
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
    forward. Parameters and state-dict keys are unchanged; the adaLN Linears see no tangent and are left alone.
    MM-JiT: every Linear and the patch convolutions (the text stream carries the tangent from the first joint block on)."""
    if isinstance(model, MMJiT):
        for m in model.modules():
            if type(m) is nn.Linear:
                m.__class__ = TangentOnlyLinear
            elif type(m) is nn.Conv2d:
                m.__class__ = TangentOnlyConv2d
        return model
    model.x_embedder.proj.__class__ = TangentOnlyConv2d
    layers = [model.final_layer.linear] + [head.linear for head in getattr(model, "aux_heads", [])]
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
    """num_classes + 1 rows: the last is the null class (read by guidance and label dropout)."""

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
                 mlp_ratio=4.0, label_dim=1000, train_pos_embed=False, tangent_only_jvp=False, aux_taps=(), gw_embedder=False,
                 gw_default=1.0, gw_input="raw", gw_interval=False):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.label_dim = label_dim
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = LabelEmbedder(label_dim, hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size)
        # fixed sin-cos embedding (train_pos_embed: trained as well)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size),
                                      requires_grad=bool(train_pos_embed))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # intermediate supervision (aux_taps): a side readout after each tapped block count; registered only when
        # taps are given, so the default model (parameters, state-dict keys, their order) is unchanged
        self.aux_taps = tuple(int(k) for k in aux_taps)
        if any(not 1 <= k < depth for k in self.aux_taps) or list(self.aux_taps) != sorted(set(self.aux_taps)):
            raise ValueError(f"aux_taps must be increasing block counts in [1, {depth - 1}], got {self.aux_taps}")
        if self.aux_taps:   # built under a forked RNG (every readout parameter is zeroed below), so the trunk draws the same
            with torch.random.fork_rng(devices=[]):
                self.aux_heads = nn.ModuleList([FinalLayer(hidden_size, patch_size, self.out_channels) for _ in self.aux_taps])
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
        for name, child in self.named_children():   # the side readouts are zeroed below and draw nothing from the RNG,
            if name != "aux_heads":                  # so the trunk's initialisation is the same with or without them
                child.apply(_basic_init)
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
        for head in [self.final_layer, *getattr(self, "aux_heads", [])]:   # zero readouts: every output starts at 0
            for m in (head.adaLN_modulation[-1], head.linear):
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

    def forward(self, x, y, omega=None, w=None, return_aux=False, s_lo=None, s_hi=None, **_):
        """T(x; y, omega[, w]). omega: None (= 1), a scalar, or one value per example; w likewise (None = gw_default),
        and only for a map built with gw_embedder. return_aux: (T, [q_k]) with q_k the side readouts after blocks
        aux_taps, each of T's shape."""
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
        taps = []
        for i, block in enumerate(self.blocks, start=1):
            h = block(h, c)
            if return_aux and i in self.aux_taps:
                taps.append(self.unpatchify(self.aux_heads[self.aux_taps.index(i)](h, c)))
        out = self.unpatchify(self.final_layer(h, c))
        return (out, taps) if return_aux else out


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.stack(np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)), axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)


def _sincos_1d(embed_dim, pos):
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class RMSNorm(nn.Module):
    """x / rms(x) with a learned per-channel gain (MiniT2I trains the gain)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = (hidden + 7) // 8 * 8
        self.w1, self.w3 = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BottleneckPatchEmbed(nn.Module):
    """Patch convolution to `pca_channels`, then a 1x1 convolution to the width (JiT's bottleneck embedding)."""

    def __init__(self, patch_size, in_channels, hidden_size, pca_channels):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


def _rope_1d(n, dim, theta=10000.0):
    """Rotary angles [n, dim] of the text stream: position p turns the pair (i, i + dim/2) by p * theta^(-2i/dim)."""
    a = torch.arange(n, dtype=torch.float32)[:, None] / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)[None]
    return torch.cat([a, a], dim=-1)


def _rope_2d(side, dim, theta=10000.0):
    """Rotary angles [side^2, dim] of the image stream: half of the pairs turn with the row, half with the column."""
    half = dim // 2
    a = torch.arange(side, dtype=torch.float32)[:, None] / theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half)[None]
    a = torch.cat([a[:, None, :].expand(side, side, -1), a[None, :, :].expand(side, side, -1)], dim=-1)
    return torch.cat([a, a], dim=-1).reshape(side * side, dim)


def _rotary_attention(q, k, v, rope, fused=True):
    """Attention over [B, heads, L, head_dim] after turning q and k by the rotary tables rope = (cos, sin) [L, head_dim],
    scale head_dim^-1/2. Inside a forward-AD level (the transport JVP) it is the plain product, which torch.func.jvp
    differentiates; elsewhere (`fused`: the endpoint pass, the guide's and the teacher's reads) the fused kernel, which
    computes the same function with ~2.5x less memory at 1280 tokens but has no forward-mode derivative."""
    cos, sin = rope

    def turn(x):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    q, k = turn(q), turn(k)
    if fused and fwAD.unpack_dual(q).tangent is None and fwAD.unpack_dual(v).tangent is None:
        return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
    return ((q * q.shape[-1] ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1) @ v


class TextBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU on the text stream alone (the preamble blocks)."""

    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio, fused=True):
        super().__init__()
        self.num_heads, self.head_dim, self.fused = num_heads, head_dim, fused
        self.norm1, self.norm2 = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim)
        self.attn_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm, self.k_norm = RMSNorm(head_dim), RMSNorm(head_dim)

    def forward(self, txt, rope):
        B, L, _ = txt.shape
        q, k, v = self.qkv(self.norm1(txt)).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = _rotary_attention(self.q_norm(q), self.k_norm(k), v, rope, self.fused)
        txt = txt + self.attn_proj(out.transpose(1, 2).reshape(B, L, -1))
        return txt + self.mlp(self.norm2(txt))


class DoubleBlock(nn.Module):
    """MM-DiT block without adaLN: per-stream norms, qkv, projections and MLPs; one attention over [text | image].
    `last`: the block's text output is read by nothing (the readout takes the image tokens), so it is not computed and
    the layers that only make it are frozen (the release trains them on no gradient; DDP rejects such parameters)."""

    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio, fused=True, last=False):
        super().__init__()
        self.num_heads, self.head_dim, self.fused, self.last = num_heads, head_dim, fused, last
        self.img_norm1, self.img_norm2 = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.txt_norm1, self.txt_norm2 = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim)
        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim)
        self.q_norm, self.k_norm = RMSNorm(head_dim), RMSNorm(head_dim)
        self.img_attn_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.txt_attn_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        if last:
            for m in (self.txt_attn_proj, self.txt_mlp, self.txt_norm2):
                m.requires_grad_(False)

    def forward(self, img, txt, rope):
        B, Lt = txt.shape[:2]
        qkv = torch.cat([self.txt_qkv(self.txt_norm1(txt)), self.img_qkv(self.img_norm1(img))], dim=1)
        q, k, v = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = _rotary_attention(self.q_norm(q), self.k_norm(k), v, rope, self.fused)
        out = out.transpose(1, 2).reshape(B, qkv.shape[1], -1)
        img = img + self.img_attn_proj(out[:, Lt:])
        img = img + self.img_mlp(self.img_norm2(img))
        if self.last:
            return img, txt
        txt = txt + self.txt_attn_proj(out[:, :Lt])
        return img, txt + self.txt_mlp(self.txt_norm2(txt))


class JiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class MMJiT(nn.Module):
    """MiniT2I's MM-JiT as the map T(x; text, omega): x-prediction in pixel space, `text_depth` text-only blocks, then
    `depth` double-stream blocks, no adaLN and no time input. `omega_tokens` > 0 prepends that many tokens carrying the
    anchor omega to the text stream (learned tokens + an embedding of omega; not rotated, the text keeps its positions);
    0 is the released denoiser, which ignores omega. `fused_attention`: the fused kernel outside the JVP (off: plain
    attention everywhere, the reference arithmetic)."""

    def __init__(self, input_size=512, patch_size=16, in_channels=3, hidden_size=768, depth=17, text_depth=2, num_heads=12,
                 head_dim=64, mlp_ratio=2.6667, txt_dim=1024, txt_len=256, pca_channels=128, omega_tokens=4,
                 tangent_only_jvp=False, fused_attention=True):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.patch_size, self.grid, self.txt_len = patch_size, input_size // patch_size, txt_len
        self.img_embedder = BottleneckPatchEmbed(patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embedder = nn.Linear(txt_dim, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, txt_dim))   # stands in for every dropped token: the null prompt
        fused = bool(fused_attention)
        self.txt_preamble_blocks = nn.ModuleList([TextBlock(hidden_size, num_heads, head_dim, mlp_ratio, fused)
                                                  for _ in range(text_depth)])
        self.double_blocks = nn.ModuleList([DoubleBlock(hidden_size, num_heads, head_dim, mlp_ratio, fused, last=i == depth - 1)
                                            for i in range(depth)])
        self.final_layer = JiTFinalLayer(hidden_size, patch_size, in_channels)
        pos = get_2d_sincos_pos_embed(hidden_size, self.grid)
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False)
        self.register_buffer("rope_txt", _rope_1d(txt_len, head_dim), persistent=False)
        self.register_buffer("rope_img", _rope_2d(self.grid, head_dim), persistent=False)
        self.n_omega_tokens = int(omega_tokens)
        if self.n_omega_tokens:
            self.omega_embedder = TimestepEmbedder(hidden_size)
            self.omega_tokens = nn.Parameter(torch.zeros(1, self.n_omega_tokens, hidden_size))
        self.initialize_weights()
        if tangent_only_jvp:
            use_tangent_only_jvp(self)

    def initialize_weights(self):
        """MiniT2I's init: torch defaults for the attention Linears, Xavier for the MLPs and the patch embedding, a zero
        readout; the omega tokens and their embedder as in MiniT2I's Mean Flow student (normal 0.02, zero biases)."""
        for block in [*self.txt_preamble_blocks, *self.double_blocks]:
            for name, m in block.named_modules():
                if isinstance(m, nn.Linear) and "mlp" in name:
                    nn.init.xavier_uniform_(m.weight)
        for conv in (self.img_embedder.proj1, self.img_embedder.proj2):
            nn.init.xavier_uniform_(conv.weight.data.view(conv.weight.shape[0], -1))
        nn.init.constant_(self.img_embedder.proj2.bias, 0)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.n_omega_tokens:
            nn.init.normal_(self.omega_tokens, std=0.02)
            for m in (self.omega_embedder.mlp[0], self.omega_embedder.mlp[2]):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def unpatchify(self, x):
        c, p, g = self.out_channels, self.patch_size, self.grid
        return torch.einsum("nhwpqc->nchpwq", x.reshape(x.shape[0], g, g, p, p, c)).reshape(x.shape[0], c, g * p, g * p)

    def _rope(self, n_txt):
        """(cos, sin) [prefix + n_txt + grid^2, head_dim]: the omega tokens unrotated, text from position 0, image 2-D."""
        txt = self.rope_txt[:n_txt]
        if n_txt > txt.shape[0]:
            txt = _rope_1d(n_txt, txt.shape[1]).to(txt)
        angles = torch.cat([txt.new_zeros(self.n_omega_tokens, txt.shape[1]), txt, self.rope_img])
        return angles.cos(), angles.sin()

    def forward(self, x, y, omega=None, **_):
        """T(x; y, omega). y = (context, mask): the text encoder's token states [B, L, txt_dim] and the 0/1 token mask
        [B, L] (padding; an all-zero row is the null prompt). omega: None (= 1), a scalar, or one value per example."""
        context, mask = y
        B, n_txt = context.shape[:2]
        context = torch.where(mask[:, :, None] > 0.5, context, self.mask_token.to(context.dtype))
        img = self.img_embedder(x) + self.pos_embed
        txt = self.txt_embedder(context).float()   # both residual streams stay fp32 under autocast, like the DiT's
        if self.n_omega_tokens:
            omega = DiT._per_example(1.0 if omega is None else omega, B, x.device)
            txt = torch.cat([self.omega_tokens + self.omega_embedder(omega).float()[:, None], txt], dim=1)
        cos, sin = self._rope(n_txt)
        n_stream = txt.shape[1]
        for block in self.txt_preamble_blocks:
            txt = block(txt, (cos[:n_stream], sin[:n_stream]))
        for block in self.double_blocks:
            img, txt = block(img, txt, (cos, sin))
        return self.unpatchify(self.final_layer(img)).float()   # 786k pixels per example: the losses sum them in fp32


GEOMETRY = {  # name -> (depth, hidden, patch, heads)
    "DiT-XL/2": (28, 1152, 2, 16), "DiT-L/2": (24, 1024, 2, 16), "DiT-B/2": (12, 768, 2, 12), "DiT-S/2": (12, 384, 2, 6),
    "DiT-tiny": (2, 64, 2, 4),   # tests only
}


MMJIT_GEOMETRY = {  # name -> (double blocks, hidden, patch, heads, head dim, mlp ratio): MiniT2I's variants
    "MMJiT-B/16": (17, 768, 16, 12, 64, 2.6667), "MMJiT-B/32": (17, 768, 32, 12, 64, 2.6667),
    "MMJiT-L/16": (23, 1248, 16, 24, 52, 2.7),
    "MMJiT-tiny": (2, 64, 16, 4, 16, 2.6667),   # tests only
}


def build_model(name="DiT-B/2", input_size=32, in_channels=4, label_dim=1000, mlp_ratio=4.0, train_pos_embed=False,
                tangent_only_jvp=False, aux_taps=(), gw_embedder=False, gw_default=1.0, gw_input="raw", gw_interval=False,
                txt_dim=1024, txt_len=256, omega_tokens=4, fused_attention=True, **_):
    if str(name).startswith("TextDiT"):   # the latent text-to-image map (sbm_t2i.dit_t2i): input_size the codec's token grid
        from sbm_t2i.dit_t2i import build_text_dit
        return build_text_dit(name, input_size=input_size, in_channels=in_channels, mlp_ratio=mlp_ratio, txt_dim=txt_dim,
                              txt_len=txt_len, tangent_only_jvp=tangent_only_jvp, **_)
    if name in MMJIT_GEOMETRY:   # input_size is the image side in pixels (512), in_channels 3
        depth, hidden, patch, heads, head_dim, ratio = MMJIT_GEOMETRY[name]
        return MMJiT(input_size=input_size, patch_size=patch, in_channels=in_channels, hidden_size=hidden, depth=depth,
                     num_heads=heads, head_dim=head_dim, mlp_ratio=ratio, txt_dim=int(txt_dim), txt_len=int(txt_len),
                     omega_tokens=int(omega_tokens), tangent_only_jvp=bool(tangent_only_jvp),
                     fused_attention=bool(fused_attention))
    depth, hidden, patch, heads = GEOMETRY[name]
    return DiT(input_size=input_size, patch_size=patch, in_channels=in_channels, hidden_size=hidden, depth=depth,
               num_heads=heads, mlp_ratio=mlp_ratio, label_dim=label_dim, train_pos_embed=train_pos_embed,
               tangent_only_jvp=bool(tangent_only_jvp), aux_taps=tuple(aux_taps or ()), gw_embedder=bool(gw_embedder),
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
    """The network weights in any checkpoint format this package reads, keys without a `model.` prefix.

    - a Lightning checkpoint of this trainer: `state_dict` is the active EMA copy, `current_model_state` the raw weights;
      which = ema | raw | ema_<label> (e.g. ema_0_9999, from `ema_state_dicts`);
    - a raw payload: `lora_state_dict` is the full network state;
    - an inference export (sbm_t2i.infer) or another slim copy (state_dict only);
    - MiniT2I's release (`*.safetensors`, diffusers layout): the MM-JiT under `model.net.`.
    """
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return {k[len("model.net."):]: v for k, v in load_file(str(path)).items() if k.startswith("model.net.")}
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


MMJIT_UNREAD = ("t_embedder.", "pooled_embedder.")   # in MiniT2I's release, never read by its forward (adaLN removed)
MMJIT_OMEGA = ("omega_embedder.", "omega_tokens")


def load_weights(model: DiT | MMJiT, path: str | Path, which: str = "ema") -> DiT | MMJiT:
    sd = {k: v.to(torch.float32) for k, v in read_weights(path, which).items()}
    if isinstance(model, MMJiT):   # a checkpoint of the time-blind base: the omega input stays at its init
        sd = {k: v for k, v in sd.items() if not k.startswith(MMJIT_UNREAD)}
        sd.update({k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(MMJIT_OMEGA) and k not in sd})
    if getattr(model, "gw_embedder", None) is not None:
        fill_gw_keys(model.state_dict(), sd)
    model.load_state_dict(sd, strict=True)
    return model
