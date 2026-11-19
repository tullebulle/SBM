"""Text-to-image: the MM-JiT map, text rows behind the loss's label indices, an external
frozen teacher for the guide, the image-caption shard stream."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import torch

from sbm_t2i.loss import CBMLoss
from sbm_t2i.models import build_model, load_weights
from sbm_t2i.text import Frozen, TextBound
from omegaconf import OmegaConf

from sbm_t2i import run as R

DEFAULTS = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs" / "t2i" / "train.yaml")


def loss_cfg(**over):
    """The default `loss` block of configs/t2i/train.yaml with `over` merged in."""
    return OmegaConf.merge(DEFAULTS.loss, OmegaConf.create(over))

T5 = R.store() / "t2i" / "weights" / "flan-t5-large"   # the frozen FLAN-T5-Large (tokenizer + encoder)
KW = dict(name="MMJiT-tiny", input_size=64, in_channels=3, txt_dim=16, txt_len=8)


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    model = build_model(**{**KW, **kw})
    with torch.no_grad():   # the readout starts at zero: give it weights so the reads differ
        model.final_layer.linear.weight.normal_(std=0.05)
    return model


def batch(B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.arange(8)[None] < torch.tensor([3, 8, 5, 1])[:B, None]).float()
    return torch.randn(B, 3, 64, 64, generator=g), torch.randn(B, 8, 16, generator=g), mask


def test_jvp_layers_and_fused_attention():
    x, ctx, mask = batch()
    v, om = torch.randn_like(x), torch.tensor([0.4, 0.6, 0.8, 0.95])
    plain, fast = tiny(fused_attention=False), tiny(fused_attention=False, tangent_only_jvp=True)
    p0, j0 = torch.func.jvp(lambda i: plain(i, (ctx, mask), omega=om), (x,), (v,))
    p1, j1 = torch.func.jvp(lambda i: fast(i, (ctx, mask), omega=om), (x,), (v,))
    assert torch.equal(p0, p1) and torch.equal(j0, j1) and float(j0.detach().abs().max()) > 0
    fused = tiny(tangent_only_jvp=True)
    p2, j2 = torch.func.jvp(lambda i: fused(i, (ctx, mask), omega=om), (x,), (v,))   # the JVP keeps the plain product
    assert torch.allclose(p2, p0, atol=1e-5) and torch.allclose(j2, j0, atol=1e-5)
    assert torch.allclose(fused(x, (ctx, mask), omega=om), p0, atol=1e-5)             # outside it: the fused kernel
    assert not torch.allclose(plain(x, (ctx, mask), omega=0.4), plain(x, (ctx, mask), omega=0.95), atol=1e-6)


def test_every_trainable_parameter_is_used():
    """DDP needs a gradient on every trainable parameter: the last block's text-only layers are frozen, nothing else."""
    model = tiny()
    x, ctx, mask = batch()
    model(x, (ctx, mask), omega=0.7).square().sum().backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    frozen = {n.split(".")[2] for n, p in model.named_parameters() if not p.requires_grad}
    assert not unused and frozen == {"txt_attn_proj", "txt_mlp", "txt_norm2"}


def test_release_checkpoint_loads(tmp_path):
    from safetensors.torch import save_file
    base = tiny(omega_tokens=0)
    release = {f"model.net.{k}": v for k, v in base.state_dict().items()}
    release["model.net.t_embedder.mlp.0.weight"] = torch.zeros(4, 4)    # in the release, never read by its forward
    release["model.net.pooled_embedder.weight"] = torch.zeros(4, 4)
    save_file(release, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    x, ctx, mask = batch()
    same = load_weights(tiny(seed=5, omega_tokens=0), tmp_path / "diffusion_pytorch_model.safetensors")
    assert torch.equal(same(x, (ctx, mask), omega=0.3), base(x, (ctx, mask)))        # the base ignores omega
    student = load_weights(tiny(seed=5), tmp_path / "diffusion_pytorch_model.safetensors")   # omega tokens stay at their init
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in base.state_dict().items())


def test_text_rows_and_the_null_prompt():
    model = tiny()
    x, ctx, mask = batch()
    bound = TextBound(model, ctx, mask, null=10)
    rows = torch.tensor([2, 10, 0, 10])
    want = model(x, (ctx[[2, 0, 0, 0]], mask[[2, 0, 0, 0]] * torch.tensor([1.0, 0, 1, 0])[:, None]), omega=0.8)
    assert torch.equal(bound(x, rows, omega=0.8), want)
    null = model(x[1:2], (torch.randn(1, 8, 16), torch.zeros(1, 8)), omega=0.8)       # no token kept: the text is not read
    assert torch.allclose(bound(x, rows, omega=0.8)[1:2], null, atol=1e-6)
    assert bound.final_layer is model.final_layer
    scaled = TextBound(model, ctx, mask, null=10, label_scale=1.5)
    assert torch.equal(scaled(x, rows, omega=0.8)[1], want[1]) and not torch.equal(scaled(x, rows, omega=0.8)[0], want[0])


def test_loss_with_an_external_teacher():
    """loss.guide.teacher_from: the tangent's guidance is read from the frozen base; noise 2 N(0, I); the null prompt's
    rows stay unguided; gradients reach the map only."""
    model, teacher = tiny(tangent_only_jvp=True), Frozen(tiny(seed=3, omega_tokens=0))
    x, ctx, mask = batch()
    rows = torch.arange(4)
    guide = {"enabled": True, "teacher_from": "base", "kappa": 5.0, "t_lo": 0.0, "t_hi": 0.95}
    cfg = loss_cfg(noise_scale=2.0, label_dropout=0.1, anchor_alpha=0.65, time={"p_mean": -0.8, "p_std": 0.8}, guide=guide)
    loss = CBMLoss(cfg, label_dim=10)

    def run(kappa=5.0, t=teacher.module):
        loss.g_kappa = kappa
        torch.manual_seed(2)
        losses, aux = loss(TextBound(model, ctx, mask, 10), x, rows, 0, teacher=TextBound(t, ctx, mask, 10))
        return losses, aux

    losses, aux = run()
    assert all(torch.isfinite(v) for v in losses.values()) and float(aux["guide_frac"]) > 0
    sum(losses.values()).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    assert all(p.grad is None for p in teacher.module.parameters())
    off, aux0 = run(kappa=0.0)
    assert float(aux0["guide_shift_rms"]) == 0.0 and float(aux["guide_shift_rms"]) > 0
    assert float(off["transport"]) != float(losses["transport"])
    with pytest.raises(ValueError, match="interval"):   # without the teacher the reads are the map's: anchors must reach t_lo
        CBMLoss(loss_cfg(label_dropout=0.1, anchor_alpha=0.65, guide={**guide, "teacher_from": None}), label_dim=10)


def test_diag_time_moves_the_denoiser_rows():
    """loss.guide.diag_time: the diagonal endpoint rows read the map at omega = omega' = their own logit-normal time,
    clamped into the anchors; the other endpoint rows keep omega ~ U[1 - anchor_alpha, 1] and omega' >= omega."""
    x, ctx, mask = batch()
    rows = torch.arange(4)
    guide = {"enabled": True, "kappa": 0.8, "t_lo": 0.02, "t_hi": 0.5, "endpoint": True, "diag_frac": 0.5}

    def endpoint_times(diag_time):
        loss = CBMLoss(loss_cfg(noise_scale=2.0, label_dropout=0.1, anchor_alpha=0.98,
                                guide={**guide, "diag_time": diag_time}), label_dim=10)
        seen, interp = [], loss._interp
        loss._interp = lambda z, x, t: (seen.append(("I", t)), interp(z, x, t))[1]
        bound = TextBound(tiny(tangent_only_jvp=True), ctx, mask, 10)
        call = bound.__class__.__call__
        spy = type("Spy", (TextBound,), {"__call__": lambda self, *a, **kw: (seen.append(("T", kw.get("omega"))), call(self, *a, **kw))[1]})
        torch.manual_seed(3)
        losses, _ = loss(spy(bound.model, ctx, mask, 10), x, rows, 50_000)
        assert all(torch.isfinite(v) for v in losses.values())
        omega_prime = [t for k, t in seen if k == "I"][-1]
        e_omega = [t for k, t in seen if k == "T"][-1]       # the last read is the endpoint term's own forward
        return omega_prime, e_omega

    op, om = endpoint_times({"p_mean": -6.0, "p_std": 0.01})   # sigmoid(-6) < 0.02: every diagonal row lands on the floor
    diag = om == 0.02
    assert 0 < int(diag.sum()) < 4 and torch.all(op[diag] == 0.02) and torch.all(op >= om)
    op0, om0 = endpoint_times(None)                              # off: the diagonal rows keep omega' = omega ~ U[0.02, 1]
    assert torch.all(op0 >= om0) and not torch.any(om0 == 0.02)


def test_flow_control_loss():
    """loss.flow: MiniT2I's v-loss on the time-blind denoiser, |(D(I_t) - x) / max(1 - t, a_min)|^2, whatever the map
    switches (guide, anchors) say; one forward, the null prompt on the dropped rows."""
    model = tiny(omega_tokens=0)
    x, ctx, mask = batch()
    rows = torch.arange(4)
    guide = {"enabled": True, "kappa": 2 / 3, "t_lo": 0.35, "t_hi": 0.6, "endpoint": True, "diag_frac": 0.5}
    cfg = loss_cfg(noise_scale=2.0, label_dropout=0.5, anchor_alpha=0.65, time={"p_mean": -0.8, "p_std": 0.8}, guide=guide,
                   flow={"enabled": True, "a_min": 0.01})
    loss, bound = CBMLoss(cfg, label_dim=10), TextBound(model, ctx, mask, 10)
    torch.manual_seed(5)
    losses, aux = loss(bound, x, rows, 50_000)
    assert set(losses) == {"flow"} and torch.isfinite(losses["flow"])
    torch.manual_seed(5)   # the same draws by hand: the dropout, t, then z
    drop = torch.rand(4) < 0.5
    assert 0 < int(drop.sum()) < 4
    rows = torch.where(drop, torch.full_like(rows, 10), rows)
    t = torch.sigmoid(torch.randn(4) * 0.8 - 0.8)
    t = t * (1 - 2 * loss.eps) + loss.eps if loss.eps > 0 else t
    z = 2.0 * torch.randn_like(x)
    I_t = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * x
    want = ((bound(I_t, rows) - x) / (1 - t).clamp_min(0.01)[:, None, None, None]).pow(2).mean()
    assert torch.allclose(losses["flow"], want, rtol=1e-5, atol=1e-6)
    losses["flow"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


@pytest.mark.skipif(not (T5 / "tokenizer.json").exists(), reason="the FLAN-T5 tokenizer is not on this machine")
def test_shard_stream(tmp_path):
    from PIL import Image
    from sbm_t2i.data import TextImageShards
    for s in range(2):
        with tarfile.open(tmp_path / f"{s:05d}.tar", "w") as tar:
            for i in range(6):
                key = f"{s}{i:04d}"
                buf = io.BytesIO()
                Image.new("RGB", (90 + 10 * i, 70), (40 * i, 10, 200)).save(buf, "JPEG")
                for name, data in ((f"{key}.jpg", buf.getvalue()), (f"{key}.txt", f"a picture number {i}".encode())):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
    stream = iter(TextImageShards(str(tmp_path / "*.tar"), 64, T5, text_len=16, buffer=4))
    x, ids, mask = next(stream)
    assert x.shape == (3, 64, 64) and x.dtype == torch.uint8 and ids.shape == mask.shape == (16,)
    assert 0 < int(mask.sum()) < 16 and int(ids[int(mask.sum()) - 1]) == 1          # ends with T5's </s>, then padding
    seen = {tuple(next(stream)[1].tolist()) for _ in range(40)}                      # an endless stream over both shards
    assert len(seen) == 6                                                            # the six captions, each seen
