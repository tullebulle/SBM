import pytest
import torch


def _raw_attn_func(q, k, v):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn_probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_probs, v)
    return out.permute(0, 2, 1, 3)


def _load_fa3_func():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash-attention numerical tests")
    try:
        from sbm_rae.models.flash_attn.fa3_jvp import flash_attn_func
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"flash-attn-3 is not available: {exc}")
    return flash_attn_func


@pytest.mark.parametrize("seq_len", [256, 264], ids=["aligned", "with_class_tokens"])
@pytest.mark.parametrize("head_dim", [64, 72], ids=["plain_head", "split_head"])
def test_fa3_forward_and_jvp_support_nonaligned_sequence_lengths(seq_len, head_dim):
    flash_attn_func = _load_fa3_func()
    torch.manual_seed(1234 + seq_len + head_dim)
    torch.cuda.empty_cache()

    batch_size = 1
    num_heads = 4
    dtype = torch.bfloat16
    q, k, v, dq, dk, dv = (
        torch.empty(
            (batch_size, seq_len, num_heads, head_dim),
            dtype=dtype,
            device="cuda",
        ).normal_(mean=0.0, std=0.5)
        for _ in range(6)
    )

    out_ref = _raw_attn_func(q, k, v)
    out_tri = flash_attn_func(q, k, v)
    torch.testing.assert_close(out_ref, out_tri, atol=1.5e-3, rtol=0.0)

    jvp_ref, tangent_ref = torch.func.jvp(_raw_attn_func, (q, k, v), (dq, dk, dv))
    jvp_tri, tangent_tri = torch.func.jvp(
        flash_attn_func,
        (q, k, v),
        (dq, dk, dv),
    )

    torch.testing.assert_close(jvp_ref, jvp_tri, atol=1.5e-3, rtol=0.0)
    torch.testing.assert_close(tangent_ref, tangent_tri, atol=1.5e-3, rtol=0.0)