"""The networks of sbm_rae.models: DiT, the DDT-head variants and the Bridge U-Net."""
import pytest
import torch
from omegaconf import OmegaConf

from sbm_rae.models import DiT_models, build_model
from sbm_rae.models.dit import DiT, DiTBlock


def _tiny_dit(**kwargs):
    model_kwargs = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=32, depth=1, num_heads=4, label_dim=10,
                        attn_func="default")
    model_kwargs.update(kwargs)
    return DiT(**model_kwargs)


def _count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_dit_forward_returns_image_tensor():
    model = _tiny_dit()
    x = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 3])
    omega = torch.tensor([1.0, 0.75])
    image = model(x, labels, omega=omega)
    assert isinstance(image, torch.Tensor)
    assert image.shape == x.shape


def test_dit_forward_supports_jvp():
    model = _tiny_dit()
    x = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 3])
    omega = torch.tensor([1.0, 0.75])
    tangent = torch.randn_like(x)
    image, image_jvp = torch.func.jvp(lambda inp: model(inp, labels, omega=omega), (x,), (tangent,))
    assert image.shape == x.shape
    assert image_jvp.shape == x.shape
    assert torch.isfinite(image_jvp).all()


def test_conditioning_embedders_final_projection_start_zero():
    model = _tiny_dit(guidance_enabled=True)
    for embedder in (model.omega_embedder, model.w_embedder):
        torch.testing.assert_close(embedder.mlp[2].weight, torch.zeros_like(embedder.mlp[2].weight))
        torch.testing.assert_close(embedder.mlp[2].bias, torch.zeros_like(embedder.mlp[2].bias))


def test_guidance_disabled_omits_w_embedder():
    assert not hasattr(_tiny_dit(guidance_enabled=False), "w_embedder")
    assert hasattr(_tiny_dit(guidance_enabled=True), "w_embedder")


def test_build_model_registers_the_kept_families():
    assert {"DiT-XL/1-DH", "DiT-B/1-DH", "DiT-XL/1", "DiT-XL/2", "DiT-B/2", "Bridge_UNet"} <= set(DiT_models)


def _dh_cfg(name):
    return OmegaConf.create({"name": name, "input_size": 16, "in_channels": 384, "mlp_ratio": 4.0, "label_dim": 1000,
                             "learn_sigma": False, "qk_norm": False, "attn_func": "default", "is_zero_data": True, "dropout": 0.0})


def test_dit_xl_1_dh_is_the_released_network():
    """The released ImageNet DiT-XL/1-DH (RAE DINOv2-S latent, 16x16x384): 247 tensors, 669.43 M parameters."""
    with torch.device("meta"):
        model = build_model(_dh_cfg("DiT-XL/1-DH"))
    assert len(model.state_dict()) == 247
    assert abs(_count_parameters(model) / 1e6 - 669.43) < 0.05
    assert model.head_depth == 2 and model.head_hidden_size == 2048 and len(model.blocks) == 21


def test_dit_b_1_dh_parameter_count():
    with torch.device("meta"):
        model = build_model(_dh_cfg("DiT-B/1-DH"))
    assert abs(_count_parameters(model) / 1e6 - 128.79) < 0.05   # 128.79 M incl. the fixed position embeddings
    assert len(model.blocks) == 8 and model.head_hidden_size == 1024


def test_dropout_is_off_by_default_and_fires_only_in_training():
    import torch.nn as nn

    x = torch.randn(2, 16, 64)
    c = torch.randn(2, 1, 64)
    plain = DiTBlock(64, 4, mlp_ratio=4.0, attn_func="default")
    plain.train()
    assert torch.allclose(plain(x, c), plain(x, c)), "default must be deterministic"
    assert not [m for m in plain.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    block = DiTBlock(64, 4, mlp_ratio=4.0, attn_func="default", dropout=0.2)
    block.train()
    assert not torch.allclose(block(x, c), block(x, c)), "dropout must fire in training"
    block.eval()
    assert torch.allclose(block(x, c), block(x, c)), "dropout must be off at sampling"


def test_dit_threads_dropout_into_every_block():
    model = DiT(input_size=8, patch_size=2, in_channels=3, hidden_size=32, depth=3, num_heads=4, label_dim=10,
                attn_func="default", dropout=0.15)
    assert [b.dropout_rate for b in model.blocks] == [0.15, 0.15, 0.15]
    default = DiT(input_size=8, patch_size=2, in_channels=3, hidden_size=32, depth=2, num_heads=4, label_dim=10,
                  attn_func="default")
    assert [b.dropout_rate for b in default.blocks] == [0.0, 0.0]
