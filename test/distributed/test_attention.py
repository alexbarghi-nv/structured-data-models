# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

import sdm.distributed.attention as attention
from sdm.distributed import distributed_transformer_block, sharded_attention
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale, RotaryEmbedding


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
    mask = torch.arange(k.size(-3), device=k.device).unsqueeze(
        0
    ) < lengths.unsqueeze(-1)
    return F.scaled_dot_product_attention(
        q.transpose(-3, -2),
        k.transpose(-3, -2),
        v.transpose(-3, -2),
        attn_mask=mask[:, None, None, :],
        enable_gqa=q.size(-2) != k.size(-2),
        scale=scale,
    ).transpose(-3, -2)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("scale", [None, 0.37])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_local_attention(
    heads: int, scale: float | None, length_dtype: torch.dtype
) -> None:
    q = _tensor((2, 5, 4, 8))
    k = _tensor((2, 7, heads, 8), 0.4)
    v = _tensor((2, 7, heads, 6), 0.8)
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
    q = _tensor((batch, queries, 4, 8))
    k = _tensor((batch, keys, 2, 8))
    v = _tensor((batch, keys, 2, 6))
    actual = sharded_attention(q, k, v)
    assert actual.shape == (batch, queries, 4, 6)
    assert torch.count_nonzero(actual) == 0


def test_local_attention_broadcast() -> None:
    q = _tensor((2, 1, 5, 4, 8))
    k = _tensor((3, 7, 2, 8), 0.4)
    v = _tensor((3, 7, 2, 6), 0.8)
    lengths = torch.tensor([[0, 3, 7], [7, 1, 5]], dtype=torch.int32)
    actual = sharded_attention(q, k, v, seqused_key_value=lengths, scale=0.37)
    mask = torch.arange(7) < lengths[..., None, None, None]
    expected = F.scaled_dot_product_attention(
        q.transpose(-3, -2),
        k.transpose(-3, -2),
        v.transpose(-3, -2),
        attn_mask=mask,
        enable_gqa=True,
        scale=0.37,
    ).transpose(-3, -2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shared_keys", [False, True])
def test_projection_batches(shared_keys: bool) -> None:
    block = KumoTabularTransformerBlock(
        channels=16,
        num_heads=2,
        query_scaling=LogScale(num_heads=2),
        rope=RotaryEmbedding(channels=8, layout="split_half"),
    ).eval()
    with torch.no_grad():
        for index, parameter in enumerate(block.parameters()):
            parameter.copy_(
                _tensor(tuple(parameter.shape), index * 0.2) * 0.15
            )
    query = _tensor((2, 3, 5, 16))
    key_value = _tensor((1 if shared_keys else 2, 3, 7, 16), 0.3)
    positions = torch.arange(5).expand(2, 3, -1)
    key_positions = torch.arange(7).expand(1 if shared_keys else 2, 3, -1)
    expected, expected_cache = distributed_transformer_block(
        block,
        query,
        key_value,
        group=None,
        global_key_length=7,
        query_positions=positions,
        key_positions=key_positions,
        return_key_value=True,
        projection_batch_size=65535,
    )
    actual, cache = distributed_transformer_block(
        block,
        query,
        key_value,
        group=None,
        global_key_length=7,
        query_positions=positions,
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
    replay = distributed_transformer_block(
        block,
        query,
        cache,
        group=None,
        global_key_length=7,
        query_positions=positions,
        projection_batch_size=2,
    )
    torch.testing.assert_close(replay, expected, rtol=1e-5, atol=1e-6)


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


def _worker(
    rank: int, world: int, rendezvous: str, cuda: bool = False
) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        rank=rank,
        world_size=world,
        init_method=f"file://{rendezvous}",
    )
    try:
        group: ProcessGroup | None = None
        local_rank = rank
        if world == 4:
            groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            group = cast(ProcessGroup, groups[rank % 2])
            local_rank = rank // 2
        _check_short_global_keys(device, group, local_rank)
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
            torch.testing.assert_close(
                replicated, expected, rtol=1e-5, atol=1e-6
            )
        empty = sharded_attention(q[:, :0], k[:, ks], v[:, ks], group=group)
        assert empty.shape == (2, 0, 4, 6)

        block = KumoTabularTransformerBlock(
            channels=16,
            num_heads=2,
            query_scaling=LogScale(num_heads=2, device=device),
            rope=RotaryEmbedding(
                channels=8, layout="split_half", device=device
            ),
            device=device,
        ).eval()
        with torch.no_grad():
            for index, parameter in enumerate(block.parameters()):
                parameter.copy_(
                    _tensor(tuple(parameter.shape), index * 0.2).to(device)
                    * 0.15
                )
        x = _tensor((2, 5, 16)).to(device)
        kv = _tensor((2, 7, 16), 0.3).to(device)
        qs = slice(0, 2) if local_rank == 0 else slice(2, 5)
        ks = slice(0, 3) if local_rank == 0 else slice(3, 7)
        with torch.inference_mode():
            expected, cache = block(
                query=x, key_value=kv, return_key_value=True
            )
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
                query_positions=torch.arange(
                    local_query.size(-2), device=device
                ),
                key_positions=torch.arange(7, device=device).expand(2, -1)[
                    :, ks
                ],
                projection_batch_size=1,
            )
            torch.testing.assert_close(
                actual,
                expected if local_rank == 0 else expected[:, :0],
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_attention(world: int) -> None:
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(world, str(Path(directory) / "rendezvous")),
            nprocs=world,
        )


@pytest.mark.cuda
def test_cuda_float32_attention() -> None:
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
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(2, str(Path(directory) / "rendezvous"), True),
            nprocs=2,
        )


@pytest.mark.parametrize("keys", [3, 5])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("scale", [None, 0.37])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_singleton_score_limits(monkeypatch, keys, heads, scale, length_dtype):
    q = _tensor((2, 3, 7, 4, 8))
    k = _tensor((1, 3, keys, heads, 8), 0.4)
    v = _tensor((1, 3, keys, heads, 6), 0.8)
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
