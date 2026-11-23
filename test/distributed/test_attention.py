# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native attention references, bounded projections and sharded replay."""

from functools import partial
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

import sdm.distributed.attention as attention
from sdm.distributed import distributed_transformer_block, sharded_attention
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale, RotaryEmbedding
from test.distributed._runtime import distributed_session, spawn_workers


def _tensor(shape: tuple[int, ...], offset: float = 0) -> Tensor:
    return (
        torch.arange(torch.Size(shape).numel()).reshape(shape) * 0.13 + offset
    ).sin()


def _reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lengths: Tensor,
    scale: float | None = None,
) -> Tensor:
    mask = (
        torch.arange(k.size(-3), device=k.device)
        < lengths[..., None, None, None]
    )
    return F.scaled_dot_product_attention(
        q.transpose(-3, -2),
        k.transpose(-3, -2),
        v.transpose(-3, -2),
        attn_mask=mask,
        enable_gqa=q.size(-2) != k.size(-2),
        scale=scale,
    ).transpose(-3, -2)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("scale", [None, 0.37])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_local_attention(
    heads: int, scale: float | None, length_dtype: torch.dtype
) -> None:
    """Chunked attention matches native masked and grouped-query attention."""
    q = _tensor((2, 5, 4, 8))
    k = _tensor((2, 7, heads, 8), 0.4)
    v = _tensor((2, 7, heads, 6), 0.8)
    # One batch has no valid keys; its result must still be defined.
    lengths = torch.tensor([0, 5], dtype=length_dtype)
    actual = sharded_attention(
        q,
        k,
        v,
        seqused_key_value=lengths,
        scale=scale,
        query_chunk_size=2,
        key_chunk_size=3,
        batch_chunk_size=1,
    )
    torch.testing.assert_close(
        actual, _reference(q, k, v, lengths, scale), rtol=1e-5, atol=1e-6
    )
    assert not actual.requires_grad


@pytest.mark.parametrize(
    ("batch", "queries", "keys"), [(2, 5, 0), (2, 0, 7), (0, 5, 7)]
)
def test_empty_attention(batch: int, queries: int, keys: int) -> None:
    """Empty batches, queries or keys return correctly shaped zeros."""
    q = _tensor((batch, queries, 4, 8))
    k = _tensor((batch, keys, 2, 8))
    v = _tensor((batch, keys, 2, 6))
    actual = sharded_attention(q, k, v)
    assert actual.shape == (batch, queries, 4, 6)
    assert torch.count_nonzero(actual) == 0


def test_local_attention_broadcast() -> None:
    """Leading query and K/V batch dimensions broadcast correctly."""
    q = _tensor((2, 1, 5, 4, 8))
    k = _tensor((3, 7, 2, 8), 0.4)
    v = _tensor((3, 7, 2, 6), 0.8)
    lengths = torch.tensor([[0, 3, 7], [7, 1, 5]], dtype=torch.int32)
    actual = sharded_attention(q, k, v, seqused_key_value=lengths, scale=0.37)
    expected = _reference(q, k, v, lengths, scale=0.37)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _projection_block(device):
    """Include rotary positions and learned scaling in projection checks."""
    block = KumoTabularTransformerBlock(
        channels=16,
        num_heads=2,
        query_scaling=LogScale(num_heads=2, device=device),
        rope=RotaryEmbedding(channels=8, layout="split_half", device=device),
        device=device,
    ).eval()
    with torch.no_grad():
        for index, parameter in enumerate(block.parameters()):
            parameter.copy_(
                _tensor(tuple(parameter.shape), index * 0.2).to(device) * 0.15
            )
    return block


@pytest.mark.parametrize("shared_keys", [False, True])
def test_projection_batches(shared_keys: bool) -> None:
    """Small projection batches preserve outputs and cached K/V."""
    block = _projection_block(torch.device("cpu"))
    query = _tensor((2, 3, 5, 16))
    key_value = _tensor((1 if shared_keys else 2, 3, 7, 16), 0.3)
    positions = torch.arange(5).expand(2, 3, -1)
    key_positions = torch.arange(7).expand(1 if shared_keys else 2, 3, -1)
    project = partial(
        distributed_transformer_block,
        block,
        query,
        group=None,
        global_key_length=7,
        query_positions=positions,
    )
    # A single large projection is the reference for bounded batch projection.
    expected, expected_cache = project(
        key_value,
        key_positions=key_positions,
        return_key_value=True,
        projection_batch_size=65535,
    )
    actual, cache = project(
        key_value,
        key_positions=key_positions,
        return_key_value=True,
        projection_batch_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        cache.key, expected_cache.key, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        cache.value, expected_cache.value, rtol=1e-5, atol=1e-6
    )
    assert cache.key.shape[:-3] == key_value.shape[:-2]
    # Replay can chunk a broadcast cache without expanding its persistent form.
    replay = project(cache, projection_batch_size=2)
    torch.testing.assert_close(replay, expected, rtol=1e-5, atol=1e-6)


def _check_readout_only_attention(device, group, local_rank, lengths):
    """Only the readout owner has queries; every owner still supplies K/V."""
    # Final row attention has queries only on the readout owner. Other ranks
    # still supply KV. Preserve global prefix order across uneven owners.
    q = _tensor((5, 2, 4, 8)).to(device)
    k = _tensor((5, 7, 4, 8), 0.4).to(device)
    v = _tensor((5, 7, 4, 6), 0.8).to(device)
    qs = slice(None) if local_rank == 0 else slice(0, 0)
    ks = slice(0, 3) if local_rank == 0 else slice(3, 7)
    actual = sharded_attention(
        q[:, qs],
        k[:, ks],
        v[:, ks],
        group=group,
        seqused_key_value=lengths,
        query_chunk_size=1,
        key_chunk_size=7,
        batch_chunk_size=2,
    )
    torch.testing.assert_close(
        actual, _reference(q, k, v, lengths)[:, qs], rtol=1e-5, atol=1e-6
    )


def _check_short_global_keys(
    device: torch.device, group: ProcessGroup | None, local_rank: int
) -> None:
    lengths = torch.tensor([0, 1, 3, 7, 9], dtype=torch.int32, device=device)
    for heads in (1, 2, 4):
        q = _tensor((5, 5, 4, 8)).to(device)
        k = _tensor((5, 7, heads, 8), 0.4).to(device)
        v = _tensor((5, 7, heads, 6), 0.8).to(device)
        for scale in (None, 0.37):
            expected = _reference(q, k, v, lengths, scale)
            for q_split, k_split in ((2, 3), (0, 0), (5, 7), (0, 7), (5, 0)):
                qs = (
                    slice(0, q_split) if local_rank == 0 else slice(q_split, 5)
                )
                ks = (
                    slice(0, k_split) if local_rank == 0 else slice(k_split, 7)
                )
                actual = sharded_attention(
                    q[:, qs],
                    k[:, ks],
                    v[:, ks],
                    group=group,
                    seqused_key_value=lengths,
                    scale=scale,
                    query_chunk_size=2,
                    key_chunk_size=7,
                    batch_chunk_size=2,
                )
                torch.testing.assert_close(
                    actual, expected[:, qs], rtol=1e-5, atol=1e-6
                )
                assert actual.shape == (5, q[:, qs].size(-3), 4, 6)
                assert torch.count_nonzero(actual[0]) == 0
                # FP32 inputs keep FP32 attention arithmetic under autocast.
                dtype = (
                    torch.float16 if device.type == "cuda" else torch.bfloat16
                )
                with torch.autocast(device.type, dtype=dtype):
                    autocast = sharded_attention(
                        q[:, qs],
                        k[:, ks],
                        v[:, ks],
                        group=group,
                        seqused_key_value=lengths,
                        scale=scale,
                        query_chunk_size=2,
                        key_chunk_size=7,
                        batch_chunk_size=2,
                    )
                torch.testing.assert_close(autocast, actual, rtol=0, atol=0)

    _check_readout_only_attention(device, group, local_rank, lengths)


def _check_sharded_queries(device, group, local_rank):
    """Partitioned and replicated queries match full masked attention."""
    q = _tensor((2, 5, 4, 8)).to(device)
    k = _tensor((2, 7, 2, 8), 0.4).to(device)
    v = _tensor((2, 7, 2, 6), 0.8).to(device)
    lengths = torch.tensor([0, 5], dtype=torch.int32, device=device)
    expected = _reference(q, k, v, lengths)
    # Uneven partitions, no keys on one rank, and no queries on one rank.
    for q_split, k_split in [(2, 3), (0, 0), (5, 7)]:
        qs = slice(0, q_split) if local_rank == 0 else slice(q_split, 5)
        ks = slice(0, k_split) if local_rank == 0 else slice(k_split, 7)
        actual = sharded_attention(
            q[:, qs],
            k[:, ks],
            v[:, ks],
            group=group,
            seqused_key_value=lengths,
            query_chunk_size=2,
            key_chunk_size=3,
            batch_chunk_size=1,
        )
        torch.testing.assert_close(
            actual, expected[:, qs], rtol=1e-5, atol=1e-6
        )
        replicated = sharded_attention(
            q,
            k[:, ks],
            v[:, ks],
            group=group,
            seqused_key_value=lengths,
            replicated_query=True,
            query_chunk_size=2,
            key_chunk_size=3,
            batch_chunk_size=1,
        )
        torch.testing.assert_close(replicated, expected, rtol=1e-5, atol=1e-6)
    empty = sharded_attention(q[:, :0], k[:, ks], v[:, ks], group=group)
    assert empty.shape == (2, 0, 4, 6)


def _check_transformer_cache_replay(device, group, local_rank):
    """Projected K/V stays sharded and is not rotated again on replay."""
    block = _projection_block(device)
    x = _tensor((2, 5, 16)).to(device)
    kv = _tensor((2, 7, 16), 0.3).to(device)
    qs = slice(0, 2) if local_rank == 0 else slice(2, 5)
    ks = slice(0, 3) if local_rank == 0 else slice(3, 7)
    with torch.inference_mode():
        expected, cache = block(query=x, key_value=kv, return_key_value=True)
        actual, local_cache = distributed_transformer_block(
            block,
            x[:, qs],
            kv[:, ks],
            group=group,
            global_key_length=7,
            query_positions=torch.arange(5, device=device)[qs],
            key_positions=torch.arange(7, device=device)[ks],
            return_key_value=True,
            query_chunk_size=2,
            key_chunk_size=3,
            batch_chunk_size=1,
            projection_batch_size=1,
        )
        torch.testing.assert_close(
            actual, expected[:, qs], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            local_cache.key, cache.key[:, ks], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            local_cache.value, cache.value[:, ks], rtol=1e-5, atol=1e-6
        )
        # Cached keys must not be rotated again during prediction.
        expected = block(query=x, key_value=cache)
        actual = distributed_transformer_block(
            block,
            x[:, qs],
            local_cache,
            group=group,
            global_key_length=7,
            query_positions=torch.arange(5, device=device)[qs],
            projection_batch_size=1,
        )
        torch.testing.assert_close(
            actual, expected[:, qs], rtol=1e-5, atol=1e-6
        )
        # Final feature attention has readout queries on only one rank.
        expected = block(query=x[:, :2], key_value=kv)
        local_query = x[:, :2] if local_rank == 0 else x[:, :0]
        actual = distributed_transformer_block(
            block,
            local_query,
            kv[:, ks],
            group=group,
            global_key_length=7,
            query_positions=torch.arange(local_query.size(-2), device=device),
            key_positions=torch.arange(7, device=device).expand(2, -1)[:, ks],
            projection_batch_size=1,
        )
        torch.testing.assert_close(
            actual,
            expected if local_rank == 0 else expected[:, :0],
            rtol=1e-5,
            atol=1e-6,
        )


def _worker(
    rank: int, world: int, rendezvous: str, cuda: bool = False
) -> None:
    with distributed_session(
        rank,
        world,
        rendezvous,
        backend="nccl" if cuda else "gloo",
        timeout_seconds=90,
    ) as device:
        group: ProcessGroup | None = None
        local_rank = rank
        if world == 4:
            groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            group = cast(ProcessGroup, groups[rank % 2])
            local_rank = rank // 2
        _check_short_global_keys(device, group, local_rank)
        _check_sharded_queries(device, group, local_rank)
        _check_transformer_cache_replay(device, group, local_rank)


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_attention(world: int) -> None:
    """Uneven shards and subgroups match full native attention."""
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    spawn_workers(_worker, world)


@pytest.mark.cuda
def test_cuda_float32_attention() -> None:
    """CUDA FP32 attention matches the native GPU implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    q = _tensor((2, 129, 4, 32)).cuda()
    k = _tensor((2, 257, 2, 32), 0.4).cuda()
    v = _tensor((2, 257, 2, 32), 0.8).cuda()
    expected = F.scaled_dot_product_attention(
        q.transpose(-3, -2),
        k.transpose(-3, -2),
        v.transpose(-3, -2),
        enable_gqa=True,
    ).transpose(-3, -2)
    actual = sharded_attention(q, k, v, key_chunk_size=67)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_cuda_autocast_accumulation() -> None:
    """Autocast leaves the explicit attention accumulation unchanged."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    q = _tensor((2, 129, 4, 32)).to(device="cuda", dtype=torch.bfloat16)
    k = _tensor((2, 257, 2, 32), 0.4).to(device="cuda", dtype=torch.bfloat16)
    v = _tensor((2, 257, 2, 32), 0.8).to(device="cuda", dtype=torch.bfloat16)
    expected = sharded_attention(q, k, v, key_chunk_size=67)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = sharded_attention(q, k, v, key_chunk_size=67)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
def test_distributed_attention_cuda() -> None:
    """Real NCCL shards preserve attention and cache replay."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    spawn_workers(_worker, 2, True)


@pytest.fixture
def score_shapes(monkeypatch):
    """Observe score extents in both native and split-key attention."""
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
    return shapes


@pytest.mark.parametrize("keys", [3, 5])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("scale", [None, 0.37])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_singleton_score_limits(
    score_shapes, keys, heads, scale, length_dtype
):
    """Broadcast attention respects every configured score-buffer bound."""
    q = _tensor((2, 3, 7, 4, 8))
    k = _tensor((1, 3, keys, heads, 8), 0.4)
    v = _tensor((1, 3, keys, heads, 6), 0.8)
    lengths = torch.tensor([[0, keys, 1], [keys, 2, keys]], dtype=length_dtype)
    expected = _reference(q, k, v, lengths, scale)
    # Exclude the unchunked oracle from the production score-bound checks.
    score_shapes.clear()
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
    assert score_shapes
    assert all(b <= 2 and q <= 2 and k <= 3 for b, q, k in score_shapes)
    assert not actual.requires_grad
