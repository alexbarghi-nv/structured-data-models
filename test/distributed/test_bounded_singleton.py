"""External review tests; no changes to the active PR worktree."""

import pytest
import torch
import torch.nn.functional as F

import sdm.distributed.attention as attention


def tensor(shape, offset=0):
    return (
        torch.arange(torch.Size(shape).numel()).reshape(shape) * 0.13 + offset
    ).sin()


@pytest.mark.parametrize("keys", [3, 5])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("scale", [None, 0.37])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_singleton_score_limits(monkeypatch, keys, heads, scale, length_dtype):
    q = tensor((2, 3, 7, 4, 8))
    k = tensor((1, 3, keys, heads, 8), 0.4)
    v = tensor((1, 3, keys, heads, 6), 0.8)
    lengths = torch.tensor([[0, keys, 1], [keys, 2, keys]], dtype=length_dtype)
    mask = torch.arange(keys) < lengths[..., None, None, None]
    expected = F.scaled_dot_product_attention(
        q.transpose(-3, -2),
        k.transpose(-3, -2),
        v.transpose(-3, -2),
        attn_mask=mask,
        enable_gqa=heads != 4,
        scale=scale,
    ).transpose(-3, -2)
    shapes = []
    original = F.scaled_dot_product_attention

    def native(query, key, value, **kwargs):
        shapes.append((query.size(0), query.size(-2), key.size(-2)))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", native)
    original_partial = attention._partial_attention

    def partial(query, key, **kwargs):
        shapes.append((query.size(0), query.size(-2), key.size(-2)))
        return original_partial(query=query, key=key, **kwargs)

    monkeypatch.setattr(attention, "_partial_attention", partial)
    actual = attention.sharded_attention(
        q,
        k,
        v,
        seqused_key_value=lengths,
        scale=scale,
        query_chunk_size=2,
        key_chunk_size=3,
        batch_chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (2, 3, 7, 4, 6)
    assert shapes
    assert all(b <= 2 and q <= 2 and k <= 3 for b, q, k in shapes)
    assert not actual.requires_grad
