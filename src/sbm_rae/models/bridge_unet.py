"""DriftWorld's action-conditioned U-Net (Lu, Chen, Ye, Du, arXiv 2607.15065; their bridge_driftworld/unet, itself adapted
from han20192019/gpc_code) as a one-step map T(x; cond, omega, w) for the Bridge-V2 and RT-1 world models.

Kept as theirs: the blocks (GroupNorm, AdaGroupNorm FiLM, self-attention and action cross-attention at the coarse levels,
down/up-sampling), the action tokenizer (per-frame-component tokens with a learned null token and Fourier features of the
scale), the Bridge_UNet shape (16-channel latent, 4 conditioning frames concatenated with the input, channels
[160, 320, 640, 640], two residual blocks per level, attention at the two coarse levels, 256-channel FiLM vector; 160 M
parameters in their paper). Changes for the one-step map, each zero-initialised so the net starts as their net:
  * omega (the anchor time, a network input of every one-step map) enters as Fourier features + MLP added to the FiLM vector;
  * w (the guidance scale of loss.guided_cbm.guide.scale_cond) takes the slot of their action-accentuation scale alpha
    (FiLM and tokenizer); w = 1 when the loss passes none;
  * the null condition (what label dropout's null class is on ImageNet) replaces the action sequence by a learned null
    action vector, so the map also learns the action-marginal prediction (their no-action negative plays this role);
  * the action cross-attention is written out as matmul + softmax (their F.scaled_dot_product_attention has no forward-mode
    derivative on the fused backends; same arithmetic).
The input is the interpolant I_t in their noise slot, concatenated with the history latents; the output is the predicted
clean latent (their net predicts samples directly, so nothing changes there).

cond = {"obs": (B, n*C, H, W) history latents (oldest first), "act": (B, n, A) actions, "null": (B,) bool (optional)}.
"""
import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# world_vector(3) + rotation_delta(3) + gripper(1)
BRIDGE_COMPONENT_SLICES = ((0, 3), (3, 6), (6, 7))
# the per-component layout keyed by the action dim: Bridge 7-D (world, rotation, gripper) and RT-1's 10-D from DriftWorld's
# conversion (world 3, rotation 3, gripper 1, base displacement 2, base rotation 1).
COMPONENT_SLICES = {7: BRIDGE_COMPONENT_SLICES, 10: ((0, 3), (3, 6), (6, 7), (7, 9), (9, 10))}


# ---------------------------------------------------------------- blocks (theirs)
class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [t for t in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ActionCrossAttention2d(nn.Module):
    """Each spatial location independently attends over the action token sequence."""

    def __init__(self, in_channels: int, context_dim: int, head_dim: int = ATTN_HEAD_DIM, max_h: int = 16, max_w: int = 16) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.max_h = max_h
        self.max_w = max_w
        self.norm = GroupNorm(in_channels)
        self.pos_embed = nn.Parameter(torch.randn(1, in_channels, max_h, max_w) * 0.02)
        self.q_proj = Conv1x1(in_channels, in_channels)
        self.kv_proj = nn.Linear(context_dim, in_channels * 2)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, action_tokens: Tensor) -> Tensor:
        n, c, h, w = x.shape
        assert h <= self.max_h and w <= self.max_w, f"feature map {h}x{w} exceeds pos_embed {self.max_h}x{self.max_w}"
        x_norm = self.norm(x) + self.pos_embed[:, :, :h, :w]
        q = self.q_proj(x_norm)
        q = q.view(n, self.n_head, c // self.n_head, h * w).transpose(2, 3)          # (n, heads, h*w, d)
        kv = self.kv_proj(action_tokens)                                              # (n, seq, 2c)
        kv = kv.view(n, -1, self.n_head, 2 * (c // self.n_head)).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)                                                    # (n, heads, seq, d)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))                       # their sdpa, written out
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class ResBlockActionCond(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, context_dim: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        self.action_attn = ActionCrossAttention2d(out_channels, context_dim) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        if not isinstance(self.action_attn, nn.Identity):
            x = self.action_attn(x, action_tokens)
        return x


class ResBlocksActionCond(nn.Module):
    def __init__(self, list_in_channels: List[int], list_out_channels: List[int], cond_channels: int, context_dim: int, attn: bool) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList([ResBlockActionCond(i, o, cond_channels, context_dim, attn)
                                        for (i, o) in zip(list_in_channels, list_out_channels)])

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor, to_cat: Optional[List[Tensor]] = None):
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond, action_tokens)
            outputs.append(x)
        return x, outputs


class UNetActionCond(nn.Module):
    def __init__(self, cond_channels: int, context_dim: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1
        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(ResBlocksActionCond([c1] + [c2] * (n - 1), [c2] * n, cond_channels, context_dim, attn_depths[i]))
            u_blocks.append(ResBlocksActionCond([2 * c2] * n + [c1 + c2], [c2] * n + [c1], cond_channels, context_dim, attn_depths[i]))
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))
        self.mid_blocks = ResBlocksActionCond([channels[-1]] * 2, [channels[-1]] * 2, cond_channels, context_dim, True)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]])
        self.upsamples = nn.ModuleList([nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])])

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor):
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond, action_tokens)
            d_outputs.append((x_down, *block_outputs))
        x, _ = self.mid_blocks(x, cond, action_tokens)
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, _ = block(x_up, cond, action_tokens, skip[::-1])
        return x[..., :h, :w]


class ActionTokenizer(nn.Module):
    """Their tokenizer: per-frame-component tokens (Bridge's 7-D layout) with frame position embeddings, a token MLP, a
    learned null token, and the scale (their alpha, our w) added through Fourier features."""

    def __init__(self, action_dim: int, token_dim: int, mode: str = "per_frame_component", max_frames: int = 16, use_cfg: bool = True) -> None:
        super().__init__()
        assert mode in ("per_frame_component", "per_frame"), f"unknown action_token_mode {mode!r}"
        self.mode = mode
        self.max_frames = max_frames
        self.use_cfg = use_cfg
        if mode == "per_frame_component":
            assert action_dim in COMPONENT_SLICES, f"per_frame_component tokenization knows the 7-D (Bridge) and 10-D (RT-1) layouts, not {action_dim}"
            self.slices = COMPONENT_SLICES[action_dim]
            self.component_projs = nn.ModuleList([nn.Linear(hi - lo, token_dim) for lo, hi in self.slices])
            self.component_type_embed = nn.Parameter(torch.randn(len(self.slices), token_dim) * 0.02)
        else:
            self.frame_proj = nn.Sequential(nn.Linear(action_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.frame_pos_embed = nn.Parameter(torch.randn(max_frames, token_dim) * 0.02)
        self.token_mlp = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.null_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        if use_cfg:
            self.cfg_fourier = FourierFeatures(token_dim)
            self.cfg_proj = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
            nn.init.zeros_(self.cfg_proj[-1].weight)
            nn.init.zeros_(self.cfg_proj[-1].bias)

    def forward(self, act: Tensor, cfg_scale: Optional[Tensor] = None) -> Tensor:
        b, n, _ = act.shape
        assert n <= self.max_frames, f"got {n} history steps but max_frames={self.max_frames}"
        assert not (self.use_cfg and cfg_scale is None), "use_cfg tokenizer needs a cfg_scale"
        if self.mode == "per_frame_component":
            per_component = [proj(act[..., lo:hi]) + self.component_type_embed[ci]
                             for ci, (proj, (lo, hi)) in enumerate(zip(self.component_projs, self.slices))]
            tokens = torch.stack(per_component, dim=2)                                   # (B, n, 3, D)
            tokens = tokens + self.frame_pos_embed[:n][None, :, None, :]
            tokens = tokens.reshape(b, -1, tokens.size(-1))                              # (B, 3n, D)
        else:
            tokens = self.frame_proj(act) + self.frame_pos_embed[:n][None]
        tokens = tokens + self.token_mlp(tokens)
        tokens = torch.cat([tokens, self.null_token.expand(b, -1, -1)], dim=1)
        if self.use_cfg:
            tokens = tokens + self.cfg_proj(self.cfg_fourier(cfg_scale)).unsqueeze(1)
        return tokens


# ---------------------------------------------------------------- the one-step map
@dataclass
class BridgeUNetConfig:
    img_channels: int = 16               # latent channels (SD3 VAE 16; SD-VAE 4)
    num_steps_conditioning: int = 4      # history frames given to the net as context
    cond_channels: int = 256
    depths: tuple = (2, 2, 2, 2)
    channels: tuple = (160, 320, 640, 640)
    attn_depths: tuple = (0, 0, 1, 1)
    num_actions: int = 7
    action_token_dim: int = 256
    action_token_mode: str = "per_frame_component"
    film_actions: bool = True


class BridgeUNet(nn.Module):
    """T(x; cond, omega, w): x (B, C, H, W) the interpolant frame, cond the history and actions (docstring at the top)."""

    def __init__(self, cfg: BridgeUNetConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.img_channels
        self.out_channels = cfg.img_channels
        self.n_history = cfg.num_steps_conditioning
        self.num_actions = cfg.num_actions
        self.film_actions = cfg.film_actions
        if cfg.film_actions:
            self.act_emb = nn.Sequential(nn.Linear(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning), nn.ReLU(), nn.Flatten())
            self.cond_proj = nn.Sequential(nn.Linear(cfg.cond_channels, cfg.cond_channels), nn.SiLU(), nn.Linear(cfg.cond_channels, cfg.cond_channels))
        # their alpha slot: our guidance scale w
        self.cfg_fourier = FourierFeatures(cfg.cond_channels)
        self.cfg_proj = nn.Sequential(nn.Linear(cfg.cond_channels, cfg.cond_channels), nn.SiLU(), nn.Linear(cfg.cond_channels, cfg.cond_channels))
        nn.init.zeros_(self.cfg_proj[-1].weight)
        nn.init.zeros_(self.cfg_proj[-1].bias)
        # ours: the anchor time omega (zero-initialised: no effect at start)
        self.omega_fourier = FourierFeatures(cfg.cond_channels)
        self.omega_proj = nn.Sequential(nn.Linear(cfg.cond_channels, cfg.cond_channels), nn.SiLU(), nn.Linear(cfg.cond_channels, cfg.cond_channels))
        nn.init.zeros_(self.omega_proj[-1].weight)
        nn.init.zeros_(self.omega_proj[-1].bias)
        # ours: the null condition's action sequence (a learned vector per action dimension; zero = "no motion" at start)
        self.null_action = nn.Parameter(torch.zeros(1, 1, cfg.num_actions))
        self.action_tokenizer = ActionTokenizer(action_dim=cfg.num_actions, token_dim=cfg.action_token_dim, mode=cfg.action_token_mode,
                                                max_frames=cfg.num_steps_conditioning, use_cfg=True)
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])
        self.unet = UNetActionCond(cond_channels=cfg.cond_channels, context_dim=cfg.action_token_dim, depths=list(cfg.depths),
                                   channels=list(cfg.channels), attn_depths=list(cfg.attn_depths))
        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)
        # the balancer's "final layer" (loss.guided_cbm.balance.parameter_path): the output convolution
        self.final_layer = self.conv_out

    @staticmethod
    def _scalar(v, B, device, default):
        if v is None:
            return torch.full((B,), float(default), device=device, dtype=torch.float32)
        if not torch.is_tensor(v):
            return torch.full((B,), float(v), device=device, dtype=torch.float32)
        v = v.to(device).reshape(-1).float()
        return v.expand(B) if v.numel() == 1 else v

    def forward(self, x: Tensor, cond: dict, omega=None, w=None, **_) -> Tensor:
        B = x.shape[0]
        obs, act = cond["obs"], cond["act"]
        if obs.ndim == 5:                       # (B, n, C, H, W) -> channel stack, oldest first (their layout)
            obs = obs.reshape(B, -1, *obs.shape[-2:])
        null = cond.get("null", None)
        act = act.to(x.dtype)
        if null is None:
            null = torch.zeros(B, dtype=torch.bool, device=x.device)
        # always through the where: DDP counts null_action as used on every step (a conditional use raises DDP's
        # unused-parameter error)
        act = torch.where(null.view(B, 1, 1), self.null_action.to(act.dtype).expand_as(act), act)
        omega_t = self._scalar(omega, B, x.device, 1.0)
        w_t = self._scalar(w, B, x.device, 1.0)
        c = self.cfg_proj(self.cfg_fourier(w_t)) + self.omega_proj(self.omega_fourier(omega_t))
        if self.film_actions:
            c = c + self.cond_proj(self.act_emb(act))
        c = c.to(x.dtype)
        action_tokens = self.action_tokenizer(act, w_t).to(x.dtype)
        h = self.conv_in(torch.cat((obs.to(x.dtype), x), dim=1))
        h = self.unet(h, c, action_tokens)
        return self.conv_out(F.silu(self.norm_out(h)))

    def use_tangent_only_jvp(self):
        """sbm_rae.models.tangent_only for this net: every convolution on the feature path computes its forward-mode tangent as
        the layer applied to the input's tangent (the weights carry none). The cond / action Linears see no tangent."""
        from sbm_rae.models.tangent_only import TangentOnlyConv2d

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and type(m) is nn.Conv2d:
                m.__class__ = TangentOnlyConv2d
        return self


def Bridge_UNet(**kwargs):
    """Their Bridge_UNet shape; `in_channels` (the latent's channels) and the other fields override the defaults."""
    kw = {}
    if "in_channels" in kwargs:
        kw["img_channels"] = int(kwargs["in_channels"])
    for k in ("num_steps_conditioning", "cond_channels", "num_actions", "action_token_dim", "action_token_mode", "film_actions"):
        if k in kwargs:
            kw[k] = kwargs[k]
    for k in ("depths", "channels", "attn_depths"):
        if k in kwargs:
            kw[k] = tuple(kwargs[k])
    return BridgeUNet(BridgeUNetConfig(**kw))


Bridge_models = {"Bridge_UNet": Bridge_UNet}
