# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, PatchEmbed

from sbm_rae.models.attn import Attention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def build_mlp(hidden_size, projector_dim, output_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


#################################################################################
#                         Embedding Layers for Class Labels                      #
#################################################################################

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size, use_cfg_embedding: bool = True):
        super().__init__()
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )

    def forward(self, labels, train=None, force_drop_ids=None):
        return self.embedding_table(labels)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def expand_scalar_condition(value, batch_size, device, name):
    if value is None:
        return torch.ones(batch_size, device=device, dtype=torch.float32)

    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    else:
        value = value.to(device=device)

    value = value.reshape(-1).float()
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or have one value per batch item; "
            f"got {value.numel()} values for batch size {batch_size}"
        )
    return value


#################################################################################
#                                Core DiT Model                                 #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_func="fa3",
        dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        # Optional dropout (0 for every released model): on the MLP's hidden activations and on
        # both residual branches, which is where DiT implementations put it.
        self.dropout_rate = float(dropout)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_func=attn_func,
            **block_kwargs,
        )
        self.modulation_size = 6

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=self.dropout_rate,
        )
        self.resid_drop = (
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self.modulation_size * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(self.modulation_size, dim=-1)
        )
        
        x = x + self.resid_drop(
            gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        )
        x = x + self.resid_drop(
            gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Transformer backbone with class, anchor-time, and guidance conditioning.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_dim=1000,
        dropout=0.0,
        learn_sigma=False,
        qk_norm=False,
        attn_func="fa3",
        is_zero_data=True,
        guidance_enabled=False,
        repa_enabled=False,
        repa_encoder_depth=8,
        repa_z_dim=768,
        repa_projector_dim=2048,
        head_hidden_size=None,
        head_depth=0,
        head_num_heads=16,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.is_zero_data = is_zero_data
        self.guidance_enabled = bool(guidance_enabled)
        self.repa_enabled = bool(repa_enabled)
        self.repa_encoder_depth = int(repa_encoder_depth)
        self.repa_z_dim = int(repa_z_dim)

        if self.repa_enabled and not (1 <= self.repa_encoder_depth <= depth):
            raise ValueError(
                f"repa_encoder_depth must be in [1, {depth}], got {self.repa_encoder_depth}"
            )

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )

        self.label_dim = label_dim
        self.y_embedder = LabelEmbedder(label_dim, hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size)
        if self.guidance_enabled:
            self.w_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    attn_func=attn_func,
                )
                for _ in range(depth)
            ]
        )
        # Wide DDT head (RAE's DiT^DH, stage2/models/DDT.py DiTwDDTHead): the trunk output s, with the anchor-time
        # embedding added (silu(omega_emb + s)) and projected to the head width, conditions a few wide blocks per token
        # (adaLN from s, not from the per-sample c) that run on the input re-embedded at the head width; the final layer
        # is modulated by s too. RAE's head uses RoPE; with standard DiT blocks here, the head input gets the fixed
        # sin-cos position embedding instead.
        self.head_depth = int(head_depth or 0)
        if self.head_depth > 0:
            # REPA reads a trunk block, before the head, so it combines with the head.
            if learn_sigma:
                raise ValueError("the DDT head does not combine with learn_sigma")
            hh = int(head_hidden_size)
            self.head_hidden_size = hh
            self.head_x_embedder = PatchEmbed(input_size, patch_size, in_channels, hh, bias=True)
            self.head_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hh), requires_grad=False)
            self.s_projector = nn.Linear(hidden_size, hh) if hh != hidden_size else nn.Identity()
            self.head_blocks = nn.ModuleList(
                [
                    DiTBlock(hh, int(head_num_heads), mlp_ratio=mlp_ratio, dropout=dropout, qk_norm=qk_norm,
                             attn_func=attn_func)
                    for _ in range(self.head_depth)
                ]
            )
        self.final_layer = FinalLayer(
            self.head_hidden_size if self.head_depth > 0 else hidden_size, patch_size, self.out_channels
        )
        if self.repa_enabled:
            self.repa_projector = build_mlp(hidden_size, int(repa_projector_dim), self.repa_z_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fixed sin-cos position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.omega_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.omega_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.omega_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.omega_embedder.mlp[2].bias, 0)
        if self.guidance_enabled:
            nn.init.normal_(self.w_embedder.mlp[0].weight, std=0.02)
            nn.init.constant_(self.w_embedder.mlp[0].bias, 0)
            nn.init.constant_(self.w_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.w_embedder.mlp[2].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.head_depth > 0:
            head_pos = get_2d_sincos_pos_embed(self.head_hidden_size, int(self.head_x_embedder.num_patches**0.5))
            self.head_pos_embed.data.copy_(torch.from_numpy(head_pos).float().unsqueeze(0))
            w = self.head_x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.head_x_embedder.proj.bias, 0)
            for block in self.head_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, y, omega=None, w=None, return_repa=False, **kwargs):
        """
        Forward pass with class, anchor-time, and guidance conditioning.
        """
        if return_repa and not self.repa_enabled:
            raise ValueError("return_repa=True requires model.repa_enabled=true")

        x_in = x
        x_emb = self.x_embedder(x)
        B, L, D = x_emb.shape

        # Embed label attributes
        class_emb = self.y_embedder(y, self.training)
        omega = expand_scalar_condition(omega, B, x.device, "omega")
        omega_emb = self.omega_embedder(omega)
        cond_emb = omega_emb
        if self.guidance_enabled:
            w = expand_scalar_condition(w, B, x.device, "w")
            cond_emb = cond_emb + self.w_embedder(w)
        y_emb = class_emb + cond_emb
        y_emb = y_emb.reshape(B, -1, D)  # (N, 1, D)

        x = x_emb + self.pos_embed

        # Compute static conditioning vector across all sequential blocks
        c = y_emb
        repa_projection = None

        # Execute all layers directly sequentially
        for block_idx, block in enumerate(self.blocks, start=1):
            x = block(x, c)
            if return_repa and block_idx == self.repa_encoder_depth:
                repa_projection = self.repa_projector(x.reshape(-1, D)).reshape(B, L, self.repa_z_dim)

        if return_repa and repa_projection is None:
            raise RuntimeError("REPA projection was not produced")

        trunk = x
        if self.head_depth > 0:
            s = self.s_projector(nn.functional.silu(trunk + omega_emb.reshape(B, 1, D)))
            h = self.head_x_embedder(x_in) + self.head_pos_embed
            for block in self.head_blocks:
                h = block(h, s)
            x = self.final_layer(h, s)
        else:
            x = self.final_layer(x, c)
        image = self.unpatchify(x)

        if self.learn_sigma:
            image, _ = image.chunk(2, dim=1)

        if return_repa:
            return image, repa_projection

        return image


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL_1(**kwargs):
    # patch size 1: for latents that are already a token grid (e.g. RAE's 16x16 DINOv2 tokens)
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def DiT_XL_1_DH(**kwargs):
    # DiT-XL/1 with RAE's wide DDT head (2 blocks at width 2048, 16 heads); the trunk is cut from 28 to 21 blocks so the
    # trainable parameters stay within DiT-XL/1's (for 768-channel tokens: 670.6M vs 676.6M; about 12% more compute per token)
    kwargs.setdefault("head_hidden_size", 2048)
    kwargs.setdefault("head_depth", 2)
    kwargs.setdefault("head_num_heads", 16)
    return DiT(depth=21, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B_1_DH(**kwargs):
    # DiT-B/1 with a wide DDT head, budgeted like DiT-XL/1-DH: the trunk is cut from 12 to 8 blocks to pay for 2 head blocks
    # at width 1024 (16 heads; RAE's 2048 head alone outweighs DiT-B/1). For 384-channel tokens: 128.3M vs 130.9M for
    # DiT-B/1, 26.2 vs 23.0 GMACs per sample.
    kwargs.setdefault("head_hidden_size", 1024)
    kwargs.setdefault("head_depth", 2)
    kwargs.setdefault("head_num_heads", 16)
    kwargs.setdefault("depth", 8)
    return DiT(hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/1": DiT_XL_1,
    "DiT-XL/1-DH": DiT_XL_1_DH,
    "DiT-B/1": DiT_B_1,
    "DiT-B/1-DH": DiT_B_1_DH,
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}