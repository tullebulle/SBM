"""The latent text-to-image map: TextDiT (sbm_t2i.dit_t2i), the guided CBM objective
(sbm_t2i.guided_cbm) on text rows, the RAE codec (sbm_t2i.codec; `-m real`: loads the DINOv2-S encoder and the 1.7 GB decoder)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sbm_t2i.dit_t2i import PromptBound
from sbm_t2i.guided_cbm import GuidedCBMLoss
from sbm_t2i.models import build_model
from sbm_t2i.text import TextBound

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "t2i"
KW = dict(name="TextDiT-tiny", input_size=8, in_channels=6, txt_dim=16, txt_len=8, text_depth=1)


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    model = build_model(**{**KW, **kw})
    with torch.no_grad():   # zero-initialised outputs (readout, adaLN, cross-attention): give them weights
        for name, p in model.named_parameters():
            if p.ndim >= 1 and float(p.abs().sum()) == 0 and "bias" not in name:
                p.normal_(std=0.05)
    return model


def batch(B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.arange(8)[None] < torch.tensor([3, 8, 5, 1])[:B, None]).float()
    return torch.randn(B, 6, 8, 8, generator=g), torch.randn(B, 8, 16, generator=g), mask


def test_geometry_and_parameter_count():
    """TextDiT-XL/1-DH on the 16 x 16 x 384 DINOv2-S grid: the ImageNet DiT-XL/1-DH's trunk and head, plus the prompt."""
    with torch.device("meta"):
        xl = build_model(name="TextDiT-XL/1-DH", input_size=16, in_channels=384, txt_dim=1024, txt_len=256)
    n = sum(p.numel() for p in xl.parameters())
    cross = sum(p.numel() for name, p in xl.named_parameters() if ".cross_attn." in name or ".norm_cross." in name)
    prompt = sum(p.numel() for name, p in xl.named_parameters()
                 if name.startswith(("refiner.", "txt_embedder.", "pool_embedder.", "mask_token")))
    assert len(xl.blocks) == 21 and xl.head_depth == 2 and xl.head_hidden_size == 2048
    assert 660e6 < n - cross - prompt < 680e6   # the ImageNet network (668.6M with its 1001-row class table)
    assert 100e6 < cross < 120e6 and 30e6 < prompt < 40e6


def test_forward_null_prompt_and_label_scale():
    model = tiny()
    x, ctx, mask = batch()
    out = model(x, (ctx, mask), omega=torch.tensor([0.4, 0.6, 0.8, 0.95]))
    assert out.shape == x.shape and out.dtype == torch.float32
    # the null prompt is read as all mask tokens, whatever the states hold
    null = torch.zeros_like(mask)
    assert torch.allclose(model(x, (ctx, null), omega=0.8), model(x, (torch.randn_like(ctx), null), omega=0.8), atol=1e-6)
    # padding is read as mask tokens too: states past the mask do not matter
    noisy = torch.where(mask[:, :, None] > 0, ctx, torch.randn_like(ctx))
    assert torch.allclose(model(x, (ctx, mask), omega=0.8), model(x, (noisy, mask), omega=0.8), atol=1e-6)
    # the prompt is read (cross-attention and the pooled vector), and so is the anchor
    assert not torch.allclose(model(x, (ctx, mask), omega=0.8), model(x, (ctx.roll(1, 0), mask.roll(1, 0)), omega=0.8))
    assert not torch.allclose(model(x, (ctx, mask), omega=0.4), model(x, (ctx, mask), omega=0.9))
    bound = TextBound(model, ctx, mask, null=10, label_scale=1.5)   # moves the states away from the mask token
    rows = torch.tensor([0, 1, 10, 3])
    out = bound(x, rows, omega=0.8)
    assert torch.allclose(out[2], model(x[2:3], (ctx[:1], null[:1]), omega=0.8)[0], atol=1e-6)


def test_jvp_tangent_only_and_attention_switch():
    """The tangent-only layers and the plain-product attention under the JVP: the same primal and tangent as the stock
    layers, and the tangent agrees with a central difference. Outside a JVP the fused kernel computes the
    same function."""
    x, ctx, mask = batch()
    v, om = torch.randn_like(x), torch.tensor([0.4, 0.6, 0.8, 0.95])
    stock = tiny()
    fast = build_model(**KW, tangent_only_jvp=True)
    fast.load_state_dict(stock.state_dict())
    p0, j0 = torch.func.jvp(lambda i: stock(i, (ctx, mask), omega=om), (x,), (v,))
    p1, j1 = torch.func.jvp(lambda i: fast(i, (ctx, mask), omega=om), (x,), (v,))
    assert torch.allclose(p0, p1, atol=1e-6) and torch.allclose(j0, j1, atol=1e-5) and float(j0.abs().max()) > 0
    assert torch.allclose(stock(x, (ctx, mask), omega=om), p0, atol=1e-5)
    eps = 1e-3   # a wrong attention tangent would be off by O(1), not by the difference's O(eps^2)
    fd = (stock(x + eps * v, (ctx, mask), omega=om) - stock(x - eps * v, (ctx, mask), omega=om)) / (2 * eps)
    assert torch.allclose(j0, fd, atol=2e-3, rtol=1e-2)


def test_every_trainable_parameter_is_used():
    model = tiny()
    x, ctx, mask = batch()
    model(x, (ctx, mask), omega=0.7).square().sum().backward()
    assert not [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


def guided_cfg(**over):
    base = OmegaConf.load(CONFIGS / "train.yaml").loss.guided_cbm
    return OmegaConf.merge(base, OmegaConf.create(over))


def test_guided_cbm_on_text_rows():
    """loss.guided_cbm with the prompts bound to the student, the teacher and the weak model: finite terms, gradients
    on the student only, the guide moves the tangent, null rows stay unguided; the balance freezes at the guide start."""
    model = tiny()
    g = torch.Generator().manual_seed(4)
    x, ctx = torch.randn(16, 6, 8, 8, generator=g), torch.randn(16, 8, 16, generator=g)
    mask = (torch.arange(8)[None] < torch.randint(1, 9, (16, 1), generator=g)).float()
    teacher, weak = copy.deepcopy(model).requires_grad_(False), tiny(seed=7).requires_grad_(False)
    bind = lambda m: TextBound(m, ctx, mask, 100)   # noqa: E731
    cfg = guided_cfg(time_shift=4.899, balance={"mode": "freeze"}, transport={"weight": 1.0},
                     guide={"start_step": 5, "kappa": 0.8, "autoguide": {"enabled": True, "ckpt": "unused", "weight": 0.09}})
    loss = GuidedCBMLoss(cfg, label_dim=100)
    rows = torch.arange(16)
    torch.manual_seed(0)
    losses, aux = loss(bind(model), None, x, rows, 0)            # before the guide: the balancer
    assert all(torch.isfinite(v) for v in losses.values()) and "balance_ratio_ema" in aux
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.zero_grad()
    loss.teacher, loss.weak = bind(teacher), bind(weak)
    torch.manual_seed(1)
    losses, aux = loss(bind(model), None, x, rows, 5)            # guided: frozen balance, teacher and weak model read
    assert loss._frozen_weight is not None and float(aux["guide_frac"]) > 0 and float(aux["guide_shift_rms"]) > 0
    assert float(aux["autoguide_shift_rms"]) > 0
    sum(losses.values()).backward()
    assert all(p.grad is None for p in [*teacher.parameters(), *weak.parameters()])
    assert loss.state_dict()["frozen_weight"] == loss._frozen_weight


def test_experiment_config_composes():
    from sbm_t2i.cli import compose_exp
    cfg = compose_exp("t2i_rae_xl1dh", [])
    assert cfg.loss.name == "guided_cbm" and cfg.codec.kind == "rae" and cfg.model.name == "TextDiT-XL/1-DH"
    assert cfg.loss.guided_cbm.time_shift == pytest.approx((256 * 384 / 4096) ** 0.5, abs=1e-3)
    assert cfg.loss.guided_cbm.balance.mode == "freeze" and cfg.loss.guided_cbm.transport.weight == 1.0
    assert cfg.loss.guided_cbm.guide.start_step == 35000 and cfg.train.per_rank_seed
    assert cfg.loss.guided_cbm.label_dropout > 0 and cfg.model.label_dim >= cfg.train.per_device_batch
    GuidedCBMLoss(cfg.loss.guided_cbm, label_dim=cfg.model.label_dim)
    for key in ("rae_src", "encoder", "decoder", "decoder_config", "stats"):
        assert Path(cfg.codec[key]).exists(), key


@pytest.mark.real
def test_codec_round_trip():
    """The DINOv2-S RAE as the ImageNet runs use it: 256 px images -> 16 x 16 x 384, unit-scale latent -> 256 px."""
    from sbm_t2i.cli import compose_exp
    from sbm_t2i.codec import build_codec
    codec = build_codec(compose_exp("t2i_rae_xl1dh", []).codec)
    assert codec.latent_shape == (384, 16, 16) and codec.out_size == 256
    import io
    from PIL import Image
    from sbm_t2i.data import TextImageShards, center_square
    cfg = compose_exp("t2i_rae_xl1dh", []).data
    shards = TextImageShards(cfg.shards, 256, cfg.text_encoder, caption_dir=cfg.caption_dir)
    reader = shards._read(shards.shards[0])
    img = torch.stack([center_square(Image.open(io.BytesIO(next(reader)[0])), 256) for _ in range(4)])
    z = codec.encode(img)
    assert z.shape == (4, 384, 16, 16) and torch.isfinite(z).all() and 0.3 < float(z.std()) < 3
    rec = codec.decode(z)   # unclamped (the readers clamp to [-1, 1])
    err = float((rec.clamp(-1, 1) - (img.float() / 127.5 - 1)).abs().mean())   # RAE: PSNR ~19 dB on ImageNet
    print(f"codec: latent std {float(z.std()):.3f}, reconstruction mean |error| {err:.3f} on [-1, 1]")
    assert rec.shape == (4, 3, 256, 256) and torch.isfinite(rec).all() and err < 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile parity runs on a GPU")
def test_compiled_calls_match_eager():
    """train.compile (sbm_t2i.speed): the compiled teacher read, student forward and student JVP give the eager step's terms
    and gradients (fp32, up to kernel order); the twins share the student's parameters."""
    from sbm_t2i.speed import CompiledCalls
    torch.backends.cuda.matmul.allow_tf32 = False
    model = tiny(tangent_only_jvp=True).cuda()
    teacher = copy.deepcopy(model).requires_grad_(False)
    g = torch.Generator().manual_seed(4)
    x, ctx = torch.randn(16, 6, 8, 8, generator=g).cuda(), torch.randn(16, 8, 16, generator=g).cuda()
    mask = (torch.arange(8)[None] < torch.randint(1, 9, (16, 1), generator=g)).float().cuda()
    cfg = guided_cfg(time_shift=4.899, transport={"weight": 1.0}, guide={"start_step": 0, "kappa": 0.8})

    def bind(m, grad=True):
        b = PromptBound(m, ctx, mask, 100)
        with torch.set_grad_enabled(grad):
            b.prompts()
        return b

    def step(calls):
        loss = GuidedCBMLoss(cfg, label_dim=100)
        reader = teacher if calls is None else calls.reader(copy.deepcopy(teacher))
        loss.teacher = bind(reader, grad=False)
        if calls is not None:
            calls.install(loss)
        model.zero_grad()
        torch.manual_seed(0)
        losses, _ = loss(bind(model), None, x, torch.arange(16, device="cuda"), 10)
        sum(losses.values()).backward()
        return {k: v.detach() for k, v in losses.items()}, [p.grad.clone() for p in model.parameters()]

    eager, g0 = step(None)
    calls = CompiledCalls({"teacher": True, "student_fwd": True, "student_jvp": True}, model)
    compiled, g1 = step(calls)
    for k in eager:   # the compiled forward's fused attention picks another fp32 kernel: rounding-level differences
        assert torch.allclose(eager[k], compiled[k], rtol=1e-4), (k, eager[k], compiled[k])
    rel = [float((a - b).norm() / a.norm().clamp_min(1e-12)) for a, b in zip(g0, g1)]
    assert max(rel) < 1e-3, max(rel)   # typically ~2e-4 at worst


def test_batched_muon_matches_muon():
    from sbm_t2i.speed import BatchedMuon
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(*s)) for s in [(8, 16), (16, 8), (8, 16), (12, 12)]]
    qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    kw = dict(lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.7750, 2.0315),
              eps=1e-7, ns_steps=5, adjust_lr_fn="original")
    a, b = torch.optim.Muon(ps, **kw), BatchedMuon(qs, **kw)
    for _ in range(3):
        for p, q in zip(ps, qs):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        a.step()
        b.step()
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(ps, qs))


def test_prompt_bound_matches_text_bound():
    """PromptBound (the prompt path once per batch, rows indexed) reads what TextBound reads, eager and under the JVP,
    label scale included; the gradient reaches the prompt path through the cached prompts."""
    model = tiny(tangent_only_jvp=True)
    x, ctx, mask = batch()
    rows, om, v = torch.tensor([2, 10, 0, 3]), torch.tensor([0.4, 0.6, 0.8, 0.95]), torch.randn_like(x)
    for ls in (1.0, 1.25):
        a, b = TextBound(model, ctx, mask, 10, ls), PromptBound(model, ctx, mask, 10, ls)
        assert torch.allclose(a(x, rows, omega=om), b(x, rows, omega=om), atol=1e-6)
        pa, ja = torch.func.jvp(lambda i: a(i, rows, omega=om), (x,), (v,))
        b = PromptBound(model, ctx, mask, 10, ls)
        b.prompts()
        pb, jb = torch.func.jvp(lambda i: b(i, rows, omega=om), (x,), (v,))
        assert torch.allclose(pa, pb, atol=1e-6) and torch.allclose(ja, jb, atol=1e-5)
    pb.square().sum().backward()
    assert model.refiner[0].attn.qkv.weight.grad is not None and model.pool_embedder[0].weight.grad is not None
