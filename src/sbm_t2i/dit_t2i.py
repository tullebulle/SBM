"""TextDiT: the ImageNet DiT-XL/1-DH map with text conditioning, the map T(x; prompt, omega) on a representation-
autoencoder latent (sbm_t2i.codec).

The ImageNet network is the class-conditional DiT-XL/1-DH of the RAE line (sbm_rae.models.dit): a DiT trunk (adaLN-Zero
blocks, patch 1 on the encoder's 16x16 token grid, fixed sin-cos positions) and RAE's wide DDT head (2 blocks at width
2048, conditioned per token by the trunk output plus the anchor embedding). What changes for text, and nothing else:

    class -> prompt   the class embedding is replaced by (i) a pooled-prompt vector added to the anchor embedding in
                      every trunk adaLN (the global conditioning the class gave; `pooled_cond`) and (ii) cross-attention
                      from the image tokens to the prompt's tokens in every trunk block, after the self-attention and
                      before the MLP (PixArt-alpha's placement), with qk-norm and a zero-initialised output projection,
                      so each block starts as the ImageNet block. The prompt tokens are the frozen FLAN-T5-Large states
                      through a linear map, a fixed 1-D sin-cos position and `text_depth` text-only refiner blocks
                      (MiniT2I's preamble). The DDT head sees the prompt only through the trunk, as it saw the class.
    null prompt       MiniT2I's: every token replaced by a learned mask token (the padding too), so the loss's null index
                      and the read-time label scale (sbm_t2i.text.TextBound) work as they do for MM-JiT.

Under torch.func.jvp only the image tokens carry a tangent: the prompt path (refiner, the cross-attention's keys and
values, the pooled vector) is a constant of the JVP. Attention outside a forward-AD level is torch's fused kernel,
inside it the plain product, which forward AD differentiates: the same function (tests/t2i/test_t2i_latent.py).
`tangent_only_jvp` switches the token-path layers to sbm_t2i.models' tangent-only forms, as for the ImageNet DiT.
`guidance_enabled` adds the guidance-scale input w (loss.guided_cbm.guide.scale_cond), zero-initialised, as there.

`trunk_split` k (model.trunk_split, default 1 = the design above): the trunk runs on a k-times finer token
grid than the latent's. Each latent token is embedded as a k x k block of trunk tokens (one Linear to k^2 x width, each
slice initialised as the k = 1 embedding), with sin-cos positions on the (k * 16)^2 grid; before the head each k x k
block is merged back (concatenated, one Linear to the head width: the trunk projection). The head, its re-embedding of
the input and the readout stay on the latent's grid. At k = 2 the trunk sees 1,024 image tokens, the token count of
MiniT2I's B/16 at 512 px (32 x 32 patches), so its per-forward compute is near theirs at the same parameter count.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed

from sbm_t2i.models import (RMSNorm, TangentOnlyConv2d, TangentOnlyLinear, TimestepEmbedder, _sincos_1d,
                        get_2d_sincos_pos_embed, modulate)

TEXT_DIT_GEOMETRY = {  # name -> (trunk depth, hidden, heads, head width, head depth, head heads)
    "TextDiT-XL/1-DH": (21, 1152, 16, 2048, 2, 16),   # the ImageNet DiT-XL/1-DH trunk and head
    "TextDiT-B/1-DH": (12, 768, 12, 2048, 2, 16),     # DiT-B trunk, same head
    # SBM-B: the DiT-B trunk with a 2 x 1024 DDT head, the trunk depth set so the parameter count matches MiniT2I's
    # MM-JiT-B/16 (258M): 15 blocks, 253.8M
    "TextDiT-B/1-DH-t15": (15, 768, 12, 1024, 2, 16),
    "TextDiT-tiny": (2, 64, 4, 96, 1, 4),             # tests only
}


def _gelu():
    return nn.GELU(approximate="tanh")


def _tangent(x):
    return fwAD.unpack_dual(x).tangent is not None


def attention(q, k, v, mode="auto", fp32=False):
    """softmax(q k^T / sqrt(d)) v over [B, heads, L, d]. mode auto (eager): the fused kernel, or the plain product when an
    input carries a forward-AD tangent (the transport JVP); fused / plain: always that one (the torch.compile'd calls of
    sbm_t2i.speed, which know whether they differentiate forward). fp32 (model.attn_fp32, off by default): q, k, v upcast
    and the logits, softmax and value product computed outside autocast, cast back to v's dtype."""
    if fp32:
        with torch.autocast(q.device.type, enabled=False):
            return attention(q.float(), k.float(), v.float(), mode).to(v.dtype)
    if mode == "fused" or (mode == "auto" and not (_tangent(q) or _tangent(k) or _tangent(v))):
        return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)   # (qk-norm's fp32 gain promotes q, k)
    return ((q * q.shape[-1] ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1) @ v


def set_attention_mode(model, mode):
    """Every attention of `model` to `mode` (auto | fused | plain); returns the model."""
    for m in model.modules():
        if isinstance(m, (SelfAttention, CrossAttention)):
            m.attn_mode = mode
    return model


def set_attention_fp32(model, on=True):
    """model.attn_fp32: every attention of `model` computes in fp32 (see `attention`); returns the model."""
    for m in model.modules():
        if isinstance(m, (SelfAttention, CrossAttention)):
            m.attn_fp32 = bool(on)
    return model


class AffineLayerNorm(nn.LayerNorm):
    """nn.LayerNorm (same parameters and state-dict keys) computed as the weight-free layer_norm and an elementwise
    affine: dynamo cannot trace an affine layer_norm under torch.func.jvp ("InferenceMode::is_enabled() &&
    self.is_inference()"), and the ImageNet DiT's norms are all weight-free."""

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps) * self.weight + self.bias


class SelfAttention(nn.Module):
    """The ImageNet DiT's attention (qkv with bias, no qk-norm by default; parameter names qkv / proj)."""
    attn_fp32 = False   # set_attention_fp32 (model.attn_fp32)

    def __init__(self, dim, num_heads, qk_norm=False):
        super().__init__()
        self.num_heads, self.head_dim, self.attn_mode = num_heads, dim // num_heads, "auto"
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = attention(self.q_norm(q), self.k_norm(k), v, self.attn_mode, self.attn_fp32)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class CrossAttention(nn.Module):
    """Image tokens (queries) attend to the prompt tokens (keys, values); qk-norm, zero-initialised output."""
    attn_fp32 = False   # set_attention_fp32 (model.attn_fp32)

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim, self.attn_mode = num_heads, dim // num_heads, "auto"
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, txt):
        B, L, C = x.shape
        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(txt).reshape(B, txt.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = attention(self.q_norm(q), self.k_norm(k), v, self.attn_mode, self.attn_fp32)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class Block(nn.Module):
    """The ImageNet DiT block (adaLN-Zero: shift, scale, gate for the attention and the MLP), with `cross` a
    cross-attention to the prompt between them (pre-norm, ungated: its zero-initialised output is the gate)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, cross=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.cross = bool(cross)
        if self.cross:
            self.norm_cross = AffineLayerNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x, c, txt=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross:
            x = x + self.cross_attn(self.norm_cross(x), txt)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class RefinerBlock(nn.Module):
    """A pre-norm transformer block on the prompt tokens alone."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = AffineLayerNorm(hidden_size, eps=1e-6), AffineLayerNorm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, qk_norm=True)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=_gelu, drop=0)

    def forward(self, txt):
        txt = txt + self.attn(self.norm1(txt))
        return txt + self.mlp(self.norm2(txt))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SplitEmbed(nn.Module):
    """trunk_split k: each latent token [C] -> a k x k block of trunk tokens [width] on the (k H) x (k W) grid, row-major
    (PatchEmbed's token order). `img_size` is the latent grid (the evaluator's noise shape)."""

    def __init__(self, input_size, in_channels, hidden_size, k):
        super().__init__()
        self.k, self.hidden = int(k), int(hidden_size)
        self.img_size = (int(input_size), int(input_size))
        self.num_patches = (int(input_size) * self.k) ** 2
        self.proj = nn.Linear(in_channels, self.k * self.k * self.hidden)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k
        t = self.proj(x.flatten(2).transpose(1, 2))   # [B, H W, k k width]
        return t.reshape(B, H, W, k, k, self.hidden).permute(0, 1, 3, 2, 4, 5).reshape(B, H * k * W * k, self.hidden)


class SplitMerge(nn.Module):
    """trunk_split k: each k x k block of trunk tokens [width] concatenated and mapped to one head-grid token."""

    def __init__(self, hidden_size, out_size, k):
        super().__init__()
        self.k = int(k)
        self.proj = nn.Linear(self.k * self.k * hidden_size, out_size)

    def forward(self, t):
        B, L, D = t.shape
        k, side = self.k, int(L ** 0.5) // self.k
        t = t.reshape(B, side, k, side, k, D).permute(0, 1, 3, 2, 4, 5).reshape(B, side * side, k * k * D)
        return self.proj(t)


def _per_example(v, B, device):
    v = torch.as_tensor(1.0 if v is None else v, device=device, dtype=torch.float32)
    return v.reshape(-1).expand(B) if v.numel() == 1 else v.reshape(B)


class TextDiT(nn.Module):
    def __init__(self, input_size=16, patch_size=1, in_channels=768, hidden_size=1152, depth=21, num_heads=16, mlp_ratio=4.0,
                 head_hidden_size=2048, head_depth=2, head_num_heads=16, txt_dim=1024, txt_len=256, text_depth=2,
                 pooled_cond=True, qk_norm=False, guidance_enabled=False, tangent_only_jvp=False, attn_fp32=False,
                 trunk_split=1):
        super().__init__()
        self.in_channels = self.out_channels = int(in_channels)
        self.patch_size, self.txt_len = int(patch_size), int(txt_len)
        self.trunk_split = int(trunk_split)
        if self.trunk_split > 1:
            if self.patch_size != 1 or not head_depth:
                raise ValueError("TextDiT: trunk_split needs patch 1 and the DDT head (the readout is on the latent grid)")
            self.x_embedder = SplitEmbed(input_size, in_channels, hidden_size, self.trunk_split)
        else:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        n = self.x_embedder.num_patches
        self.register_buffer("pos_embed", torch.from_numpy(get_2d_sincos_pos_embed(hidden_size, int(n ** 0.5))).float()[None],
                             persistent=False)
        # the prompt: frozen encoder states -> width, positions, refiner; the mask token is the null prompt and the padding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, txt_dim))
        self.txt_embedder = nn.Linear(txt_dim, hidden_size)
        pos = _sincos_1d(hidden_size, np.arange(self.txt_len, dtype=np.float64))
        self.register_buffer("txt_pos", torch.from_numpy(pos).float()[None], persistent=False)
        self.refiner = nn.ModuleList([RefinerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(int(text_depth))])
        self.pooled_cond = bool(pooled_cond)
        if self.pooled_cond:
            self.pool_embedder = nn.Sequential(nn.Linear(txt_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.omega_embedder = TimestepEmbedder(hidden_size)
        self.guidance_enabled = bool(guidance_enabled)
        if self.guidance_enabled:
            self.w_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio, qk_norm, cross=True) for _ in range(depth)])
        # RAE's DDT head (the ImageNet DiT-XL/1-DH's): the input re-embedded at the head width, adaLN per token from the
        # trunk output plus the anchor embedding
        self.head_depth = int(head_depth or 0)
        width = hidden_size
        if self.head_depth > 0:
            width = self.head_hidden_size = int(head_hidden_size)
            self.head_x_embedder = PatchEmbed(input_size, patch_size, in_channels, width, bias=True)
            n_head = self.head_x_embedder.num_patches
            self.register_buffer("head_pos_embed",
                                 torch.from_numpy(get_2d_sincos_pos_embed(width, int(n_head ** 0.5))).float()[None],
                                 persistent=False)
            if self.trunk_split > 1:
                self.s_projector = SplitMerge(hidden_size, width, self.trunk_split)
            else:
                self.s_projector = nn.Linear(hidden_size, width) if width != hidden_size else nn.Identity()
            self.head_blocks = nn.ModuleList([Block(width, int(head_num_heads), mlp_ratio, qk_norm)
                                              for _ in range(self.head_depth)])
        self.final_layer = FinalLayer(width, patch_size, self.out_channels)
        self.initialize_weights()
        if tangent_only_jvp:
            use_tangent_only_jvp(self)
        if attn_fp32:
            set_attention_fp32(self)

    def initialize_weights(self):
        """The ImageNet DiT's init (Xavier Linears, zero biases, zero adaLN outputs and readout, a zero-output anchor
        embedder) plus: zero cross-attention outputs (each block starts as the ImageNet block), the mask token as
        MiniT2I's (normal 0.02)."""
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(basic)
        embedders = [self.x_embedder] + ([self.head_x_embedder] if self.head_depth > 0 else [])
        for e in embedders:
            if isinstance(e, SplitEmbed):   # each sub-token's slice as the k = 1 patch embedding
                for w in e.proj.weight.data.view(e.k * e.k, e.hidden, -1):
                    nn.init.xavier_uniform_(w)
            else:
                nn.init.xavier_uniform_(e.proj.weight.data.view(e.proj.weight.shape[0], -1))
            nn.init.constant_(e.proj.bias, 0)
        nn.init.normal_(self.mask_token, std=0.02)
        for emb in [self.omega_embedder] + ([self.w_embedder] if self.guidance_enabled else []):
            nn.init.normal_(emb.mlp[0].weight, std=0.02)
            nn.init.constant_(emb.mlp[0].bias, 0)
            nn.init.constant_(emb.mlp[2].weight, 0)
            nn.init.constant_(emb.mlp[2].bias, 0)
        for block in [*self.blocks, *(self.head_blocks if self.head_depth > 0 else [])]:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.cross:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)
        for m in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = torch.einsum("nhwpqc->nchpwq", x.reshape(x.shape[0], h, w, p, p, c))
        return x.reshape(x.shape[0], c, h * p, w * p)

    def prompt(self, context, mask):
        """(prompt tokens [B, L, hidden], pooled conditioning [B, hidden] or None) of (encoder states, 0/1 mask)."""
        context = torch.where(mask[:, :, None] > 0.5, context, self.mask_token.to(context.dtype))
        if context.shape[1] > self.txt_pos.shape[1]:
            raise ValueError(f"TextDiT: {context.shape[1]} prompt tokens, txt_len {self.txt_pos.shape[1]}")
        txt = self.txt_embedder(context) + self.txt_pos[:, :context.shape[1]]
        for block in self.refiner:
            txt = block(txt)
        pooled = None
        if self.pooled_cond:   # mean over the prompt's own tokens (the null prompt: over its mask tokens)
            keep = torch.where(mask.sum(1, keepdim=True) > 0, mask, torch.ones_like(mask)).to(context.dtype)
            pooled = self.pool_embedder((context * keep[:, :, None]).sum(1) / keep.sum(1, keepdim=True))
        return txt, pooled

    def forward(self, x, y, omega=None, w=None, **_):
        """T(x; y, omega). y = (context, mask): the text encoder's token states [B, L, txt_dim] and the 0/1 token mask
        [B, L] (an all-zero row is the null prompt). omega (and w): None (= 1), a scalar, or one value per example."""
        return self.image_path(x, *self.prompt(*y), omega=omega, w=w)

    def image_path(self, x, txt, pooled, omega=None, w=None):
        """T(x; prompt, omega) from the prompt's tokens and pooled vector (`prompt`): everything the anchor, the input or
        the guidance scale reach."""
        B = x.shape[0]
        omega_emb = self.omega_embedder(_per_example(omega, B, x.device))
        c = omega_emb if pooled is None else omega_emb + pooled
        if self.guidance_enabled:
            c = c + self.w_embedder(_per_example(w, B, x.device))
        elif w is not None:
            raise ValueError("TextDiT: a guidance scale w was passed to a map without the input (guidance_enabled)")
        c = c[:, None]
        h = self.x_embedder(x) + self.pos_embed
        for block in self.blocks:
            h = block(h, c, txt)
        if self.head_depth > 0:
            s = self.s_projector(F.silu(h + omega_emb[:, None]))
            g = self.head_x_embedder(x) + self.head_pos_embed
            for block in self.head_blocks:
                g = block(g, s)
            out = self.final_layer(g, s)
        else:
            out = self.final_layer(h, c)
        return self.unpatchify(out).float()


def use_tangent_only_jvp(model: TextDiT) -> TextDiT:
    """The token-path layers to sbm_t2i.models' tangent-only forms (patch convolutions; per trunk block qkv, proj, the
    cross-attention's query and output, fc1, fc2; the readout; in the DDT head also the per-token adaLN Linears and the
    trunk projection, as for the ImageNet DiT-DH). The prompt path sees no tangent and keeps the stock layers."""
    convs, layers = [], [model.final_layer.linear]
    (layers if isinstance(model.x_embedder, SplitEmbed) else convs).append(model.x_embedder.proj)
    for block in model.blocks:
        layers += [block.attn.qkv, block.attn.proj, block.cross_attn.q, block.cross_attn.proj, block.mlp.fc1, block.mlp.fc2]
    if model.head_depth > 0:
        convs.append(model.head_x_embedder.proj)
        layers.append(model.final_layer.adaLN_modulation[-1])
        if isinstance(model.s_projector, nn.Linear):
            layers.append(model.s_projector)
        elif isinstance(model.s_projector, SplitMerge):
            layers.append(model.s_projector.proj)
        for block in model.head_blocks:
            layers += [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2, block.adaLN_modulation[-1]]
    for conv in convs:
        conv.__class__ = TangentOnlyConv2d
    for layer in layers:
        layer.__class__ = TangentOnlyLinear
    return model


def build_text_dit(name, input_size=16, in_channels=768, mlp_ratio=4.0, txt_dim=1024, txt_len=256, text_depth=2,
                   pooled_cond=True, qk_norm=False, guidance_enabled=False, tangent_only_jvp=False, attn_fp32=False,
                   trunk_split=1, **_):
    depth, hidden, heads, head_width, head_depth, head_heads = TEXT_DIT_GEOMETRY[name]
    return TextDiT(input_size=int(input_size), patch_size=1, in_channels=int(in_channels), hidden_size=hidden, depth=depth,
                   num_heads=heads, mlp_ratio=float(mlp_ratio), head_hidden_size=head_width, head_depth=head_depth,
                   head_num_heads=head_heads, txt_dim=int(txt_dim), txt_len=int(txt_len), text_depth=int(text_depth),
                   pooled_cond=bool(pooled_cond), qk_norm=bool(qk_norm), guidance_enabled=bool(guidance_enabled),
                   tangent_only_jvp=bool(tangent_only_jvp), attn_fp32=bool(attn_fp32), trunk_split=int(trunk_split))


class PromptBound:
    """sbm_t2i.text.TextBound for a TextDiT, with the prompt path run once per bound batch: `prompts()` computes every row's
    prompt tokens and pooled vector and the null prompt's (the refiner and the pooled MLP act per example, so a row's
    read is the same whichever batch it sits in), and each call indexes them and runs `image_path`. The guided step
    reads the teacher 4 times and the student twice per batch; with TextBound each read re-ran the prompt path.
    Call `prompts()` outside any function transform (the student's under the step's grad mode: its graph then feeds
    the gradients of both student reads; the frozen readers' under no_grad). `model` may be a sbm_t2i.speed reader."""

    def __init__(self, model, context, mask, null, label_scale=1.0):
        self.model, self.null, self.label_scale = model, int(null), float(label_scale)
        if self.label_scale != 1.0:   # the kept tokens moved away from the mask token (TextBound's label scale)
            token = model.mask_token.to(context.dtype)
            context = token + self.label_scale * (context - token)
        self.context, self.mask, self._cache = context, mask, None

    def __getattr__(self, name):   # parameters, sub-modules (the balancer's final_layer), geometry
        return getattr(self.model, name)

    def prompts(self):
        if self._cache is None:
            txt, pooled = self.model.prompt(self.context, self.mask)
            null_txt, null_pooled = self.model.prompt(self.context[:1], torch.zeros_like(self.mask[:1]))
            self._cache = (txt, pooled, null_txt, null_pooled)
        return self._cache

    def rows(self, y):
        """(prompt tokens, pooled vector) of rows y; the index `null` is the null prompt."""
        txt, pooled, null_txt, null_pooled = self.prompts()
        keep = y != self.null
        r = torch.where(keep, y, torch.zeros_like(y))
        t = torch.where(keep[:, None, None], txt[r], null_txt.to(txt.dtype))
        return t, None if pooled is None else torch.where(keep[:, None], pooled[r], null_pooled.to(pooled.dtype))

    def __call__(self, x, y, omega=None, w=None, **_):
        return self.model.image_path(x, *self.rows(y), omega=omega, w=w)
