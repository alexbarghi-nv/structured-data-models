# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Literal, overload

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from sdm.cache import KVCacheEntry
from sdm.nn import SDPA, TransformerBlock
from sdm.nn.attention import _chunk


def _partial_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    # Q: [B, Hkv, G, Q, D], KV: [B, Hkv, 1, K, D].
    with torch.autocast(query.device.type, enabled=False):
        scores = query.float() @ key.float().transpose(-1, -2)
        scores.mul_(scale)
        if mask is not None:
            scores.masked_fill_(mask, -math.inf)
        maximum = scores.amax(-1)
        safe_max = maximum.masked_fill(maximum.isneginf(), 0)
        weights = (scores - safe_max.unsqueeze(-1)).exp()
        return maximum, weights.sum(-1), weights @ value.float()


@torch.inference_mode()
def sharded_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group: ProcessGroup | None = None,
    scale: float | None = None,
    seqused_key_value: Tensor | None = None,
    replicated_query: bool = False,
    query_chunk_size: int = 128,
    key_chunk_size: int = 256,
    batch_chunk_size: int = 16,
) -> Tensor:
    """Attend local queries to all sequence shards without gathering KV.

    All group members must call this function in the same order, with the
    same broadcastable batch shape, head counts, dtypes and chunk sizes.
    Query and key sequence lengths may differ between ranks, including zero.
    Global key order is group-rank order followed by local sequence order.
    Each KV chunk is broadcast once and merged with online softmax statistics
    in float32. There is no dropout or gradient support.

    Args:
        query: Local queries of shape ``[..., Q, Hq, D]``.
        key: Local keys of shape ``[..., K, Hkv, D]``.
        value: Local values of shape ``[..., K, Hkv, Dv]``.
        group: Group sharing KV shards; ``None`` uses the default group when
            initialized, or executes locally when distributed is uninitialized.
        scale: Score multiplier; defaults to ``1 / sqrt(D)``.
        seqused_key_value: Valid global key prefix lengths, shape ``[...]``.
        replicated_query: Queries are identical on every group member. Merge
            partial statistics with reductions instead of broadcasting KV.
        query_chunk_size: Maximum queries in a score block.
        key_chunk_size: Maximum keys in a broadcast and score block.
        batch_chunk_size: Maximum flattened batch entries per broadcast.

    Returns:
        Local query outputs of shape ``[..., Q, Hq, Dv]``, in query dtype.
        Queries without valid keys produce zeros.
    """
    distributed = dist.is_initialized()
    world_size = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    lengths = [key.size(-3)]
    query_lengths = [query.size(-3)]
    if world_size > 1:
        length = torch.tensor(
            [lengths[0], query_lengths[0]],
            dtype=torch.int64,
            device=key.device,
        )
        gathered = [torch.empty_like(length) for _ in range(world_size)]
        dist.all_gather(gathered, length, group=group)
        # The small metadata exchange is an explicit scheduling boundary.
        lengths = [int(item[0].item()) for item in gathered]
        query_lengths = [int(item[1].item()) for item in gathered]

    batch_shapes = [query.shape[:-3], key.shape[:-3], value.shape[:-3]]
    if seqused_key_value is not None:
        batch_shapes.append(seqused_key_value.shape)
    batch_shape = torch.broadcast_shapes(*batch_shapes)
    batch_size = math.prod(batch_shape)
    q_len, q_heads, channels = query.shape[-3:]
    kv_heads = key.size(-2)
    if q_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    groups = q_heads // kv_heads
    value_channels = value.size(-1)
    if not batch_size or not sum(lengths) or not sum(query_lengths):
        return query.new_zeros((*batch_shape, q_len, q_heads, value_channels))
    if world_size == 1:
        mask = None
        if seqused_key_value is not None:
            indices = torch.arange(key.size(-3), device=key.device)
            mask = indices < seqused_key_value[..., None]
            mask = mask[..., None, None, :]
        return SDPA(
            num_query_heads=q_heads,
            num_key_value_heads=kv_heads,
            scale=scale,
        )(
            query,
            key,
            value,
            attn_mask=mask,
        )
    out = query.new_zeros((*batch_shape, q_len, q_heads, value_channels))

    query = query.expand(*batch_shape, *query.shape[-3:]).reshape(
        batch_size, q_len, q_heads, channels
    )
    key = key.expand(*batch_shape, *key.shape[-3:]).reshape(
        batch_size, key.size(-3), kv_heads, channels
    )
    value = value.expand(*batch_shape, *value.shape[-3:]).reshape(
        batch_size, value.size(-3), kv_heads, value_channels
    )
    out_flat = out.reshape(batch_size, q_len, q_heads, value_channels)
    valid_lengths = (
        None
        if seqused_key_value is None
        else seqused_key_value.expand(batch_shape).reshape(batch_size)
    )
    scale = channels**-0.5 if scale is None else scale

    for batch_start in range(0, batch_size, batch_chunk_size):
        batch_end = min(batch_start + batch_chunk_size, batch_size)
        b = batch_end - batch_start
        # [B, Hkv, G, Q, D]; KV heads broadcast across GQA groups.
        q = (
            query[batch_start:batch_end]
            .reshape(b, q_len, kv_heads, groups, channels)
            .permute(0, 2, 3, 1, 4)
        )
        maximum = q.new_full(
            (b, kv_heads, groups, q_len), -math.inf, dtype=torch.float32
        )
        normalizer = torch.zeros_like(maximum)
        numerator = q.new_zeros(
            (b, kv_heads, groups, q_len, value_channels), dtype=torch.float32
        )
        key_offset = 0
        for source, length in enumerate(lengths):
            if replicated_query and source != rank:
                key_offset += length
                continue
            source_rank = (
                dist.get_global_rank(group, source)
                if group is not None
                else source
            )
            for key_start in range(0, length, key_chunk_size):
                key_end = min(key_start + key_chunk_size, length)
                k_len = key_end - key_start
                if rank == source:
                    k = key[
                        batch_start:batch_end, key_start:key_end
                    ].contiguous()
                    v = value[
                        batch_start:batch_end, key_start:key_end
                    ].contiguous()
                else:
                    k = key.new_empty((b, k_len, kv_heads, channels))
                    v = value.new_empty((b, k_len, kv_heads, value_channels))
                if world_size > 1 and not replicated_query:
                    dist.broadcast(k, src=source_rank, group=group)
                    dist.broadcast(v, src=source_rank, group=group)
                k = k.permute(0, 2, 1, 3).unsqueeze(2)
                v = v.permute(0, 2, 1, 3).unsqueeze(2)
                mask = None
                if valid_lengths is not None:
                    indices = torch.arange(
                        key_offset + key_start,
                        key_offset + key_end,
                        device=query.device,
                    )
                    mask = indices.unsqueeze(0) >= valid_lengths[
                        batch_start:batch_end
                    ].unsqueeze(-1)
                    mask = mask[:, None, None, None, :]
                for query_start in range(0, q_len, query_chunk_size):
                    query_end = min(query_start + query_chunk_size, q_len)
                    index = (..., slice(query_start, query_end))
                    block_max, block_sum, block_value = _partial_attention(
                        query=q[..., query_start:query_end, :],
                        key=k,
                        value=v,
                        mask=mask,
                        scale=scale,
                    )
                    merged_max = torch.maximum(maximum[index], block_max)
                    # Empty/all-masked blocks have max=-inf. Subtract zero
                    # there so masked exponentials are zero, never NaN.
                    safe_max = merged_max.masked_fill(merged_max.isneginf(), 0)
                    old_weight = (maximum[index] - safe_max).exp()
                    block_weight = (block_max - safe_max).exp()
                    normalizer[index] = (
                        normalizer[index] * old_weight
                        + block_sum * block_weight
                    )
                    numerator[..., query_start:query_end, :] = numerator[
                        ..., query_start:query_end, :
                    ] * old_weight.unsqueeze(
                        -1
                    ) + block_value * block_weight.unsqueeze(-1)
                    maximum[index] = merged_max
            key_offset += length
        if replicated_query and world_size > 1 and q_len:
            global_max = maximum.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
            safe_max = global_max.masked_fill(global_max.isneginf(), 0)
            weight = (maximum - safe_max).exp()
            normalizer.mul_(weight)
            numerator.mul_(weight.unsqueeze(-1))
            dist.all_reduce(normalizer, group=group)
            dist.all_reduce(numerator, group=group)
        result = numerator / normalizer.clamp_min(
            torch.finfo(torch.float32).tiny
        ).unsqueeze(-1)
        out_flat[batch_start:batch_end] = result.permute(
            0, 3, 1, 2, 4
        ).reshape(b, q_len, q_heads, value_channels)
    return out


@overload
def distributed_transformer_block(
    block: TransformerBlock,
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None = None,
    *,
    group: ProcessGroup | None,
    global_key_length: int,
    query_positions: Tensor | None = None,
    key_positions: Tensor | None = None,
    return_key_value: Literal[False] = False,
    seqused_key_value: Tensor | None = None,
    replicated_query: bool = False,
    query_chunk_size: int = 128,
    key_chunk_size: int = 256,
    batch_chunk_size: int = 16,
    projection_batch_size: int | Literal["auto"] = "auto",
) -> Tensor: ...


@overload
def distributed_transformer_block(
    block: TransformerBlock,
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None = None,
    *,
    group: ProcessGroup | None,
    global_key_length: int,
    query_positions: Tensor | None = None,
    key_positions: Tensor | None = None,
    return_key_value: Literal[True],
    seqused_key_value: Tensor | None = None,
    replicated_query: bool = False,
    query_chunk_size: int = 128,
    key_chunk_size: int = 256,
    batch_chunk_size: int = 16,
    projection_batch_size: int | Literal["auto"] = "auto",
) -> tuple[Tensor, KVCacheEntry]: ...


@overload
def distributed_transformer_block(
    block: TransformerBlock,
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None = None,
    *,
    group: ProcessGroup | None,
    global_key_length: int,
    query_positions: Tensor | None = None,
    key_positions: Tensor | None = None,
    return_key_value: bool,
    seqused_key_value: Tensor | None = None,
    replicated_query: bool = False,
    query_chunk_size: int = 128,
    key_chunk_size: int = 256,
    batch_chunk_size: int = 16,
    projection_batch_size: int | Literal["auto"] = "auto",
) -> Tensor | tuple[Tensor, KVCacheEntry]: ...


@torch.inference_mode()
def distributed_transformer_block(
    block: TransformerBlock,
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None = None,
    *,
    group: ProcessGroup | None,
    global_key_length: int,
    query_positions: Tensor | None = None,
    key_positions: Tensor | None = None,
    return_key_value: bool = False,
    seqused_key_value: Tensor | None = None,
    replicated_query: bool = False,
    query_chunk_size: int = 128,
    key_chunk_size: int = 256,
    batch_chunk_size: int = 16,
    projection_batch_size: int | Literal["auto"] = "auto",
) -> Tensor | tuple[Tensor, KVCacheEntry]:
    """Run an inference transformer block with sequence-sharded KV.

    Parameters follow :class:`sdm.nn.TransformerBlock`. ``global_key_length``
    is used for query scaling, and positions refer to the unsharded sequence.
    All ranks participate even when their local query sequence is empty.
    Returned KV contains only this rank's transformed key/value projections.
    ``projection_batch_size="auto"`` uses the block's native memory budget
    for projection, output and MLP work over independent batch entries. This
    is separate from score chunking. Full output and requested cache storage
    remain allocated. When returning K/V with a shared broadcast batch, the
    block runs without projection chunking to retain its original cache shape.
    """
    batch_shapes = [query.shape[:-2]]
    if isinstance(key_value, Tensor):
        batch_shapes.append(key_value.shape[:-2])
    elif isinstance(key_value, KVCacheEntry):
        batch_shapes.extend(
            [key_value.key.shape[:-3], key_value.value.shape[:-3]]
        )
    if seqused_key_value is not None:
        batch_shapes.append(seqused_key_value.shape)
    for positions in (query_positions, key_positions):
        if positions is not None:
            batch_shapes.append(positions.shape[:-1])
    batch_shape = torch.broadcast_shapes(*batch_shapes)
    batch_size = math.prod(batch_shape)
    if projection_batch_size == "auto":
        projection_batch_size = block.auto_batch_size_limit(
            device=query.device,
            element_size=(
                torch.get_autocast_dtype(query.device.type).itemsize
                if torch.is_autocast_enabled(query.device.type)
                else query.element_size()
            ),
            query_length=query.size(-2),
            key_value_length=global_key_length,
        )
        if (
            batch_size > 1
            and dist.is_initialized()
            and dist.get_world_size(group) > 1
        ):
            limit = torch.tensor(
                projection_batch_size, device=query.device, dtype=torch.int64
            )
            dist.all_reduce(limit, op=dist.ReduceOp.MIN, group=group)
            projection_batch_size = int(limit.item())
    # Preserve the cache's original batch shape when queries broadcast it.
    cache_batch_shape = (
        key_value.key.shape[:-3]
        if isinstance(key_value, KVCacheEntry)
        else query.shape[:-2]
        if key_value is None
        else key_value.shape[:-2]
    )
    options: dict[str, Any] = {
        "group": group,
        "global_key_length": global_key_length,
        "return_key_value": return_key_value,
        "replicated_query": replicated_query,
        "query_chunk_size": query_chunk_size,
        "key_chunk_size": key_chunk_size,
        "batch_chunk_size": batch_chunk_size,
    }
    if batch_size <= projection_batch_size or (
        return_key_value and batch_shape != cache_batch_shape
    ):
        return _transformer_block_forward(
            block,
            query,
            key_value,
            query_positions=query_positions,
            key_positions=key_positions,
            seqused_key_value=seqused_key_value,
            **options,
        )
    flat_output: Tensor | None = None
    flat_key: Tensor | None = None
    flat_value: Tensor | None = None
    for start in range(0, batch_size, projection_batch_size):
        end = min(start + projection_batch_size, batch_size)
        result = _transformer_block_forward(
            block,
            query=_chunk(query, batch_shape, 2, start, end),
            key_value=_chunk(key_value, batch_shape, 2, start, end),
            query_positions=(
                query_positions
                if query_positions is None or query_positions.dim() == 1
                else _chunk(query_positions, batch_shape, 1, start, end)
            ),
            key_positions=(
                key_positions
                if key_positions is None or key_positions.dim() == 1
                else _chunk(key_positions, batch_shape, 1, start, end)
            ),
            seqused_key_value=_chunk(
                seqused_key_value, batch_shape, 0, start, end
            ),
            **options,
        )
        if isinstance(result, tuple):
            chunk, cache = result
            if flat_key is None:
                flat_key = cache.key.new_empty(
                    batch_size, *cache.key.shape[-3:]
                )
                flat_value = cache.value.new_empty(
                    batch_size, *cache.value.shape[-3:]
                )
            flat_key[start:end] = cache.key
            assert flat_value is not None
            flat_value[start:end] = cache.value
            del cache
        else:
            chunk = result
        if flat_output is None:
            flat_output = chunk.new_empty(batch_size, *chunk.shape[-2:])
        flat_output[start:end] = chunk
        del chunk, result
    assert flat_output is not None
    output = flat_output.reshape(*batch_shape, *flat_output.shape[-2:])
    if return_key_value:
        assert flat_key is not None
        assert flat_value is not None
        return output, KVCacheEntry(
            flat_key.reshape(*batch_shape, *flat_key.shape[-3:]),
            flat_value.reshape(*batch_shape, *flat_value.shape[-3:]),
        )
    return output


@torch.inference_mode()
def _transformer_block_forward(
    block: TransformerBlock,
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    *,
    group: ProcessGroup | None,
    global_key_length: int,
    query_positions: Tensor | None,
    key_positions: Tensor | None,
    return_key_value: bool,
    seqused_key_value: Tensor | None,
    replicated_query: bool,
    query_chunk_size: int,
    key_chunk_size: int,
    batch_chunk_size: int,
) -> Tensor | tuple[Tensor, KVCacheEntry]:
    attn = block.attn
    if key_value is None and key_positions is None:
        key_positions = query_positions
    if block.key_value_norm is not None:
        if key_value is None and block.key_value_norm is not block.query_norm:
            key_value = block.key_value_norm(query)
        elif isinstance(key_value, Tensor):
            key_value = block.key_value_norm(key_value)
    q = query if block.query_norm is None else block.query_norm(query)
    if isinstance(key_value, KVCacheEntry):
        q = F.linear(
            q,
            attn.qkv_lin.weight[: attn.q_dim],
            None
            if attn.qkv_lin.bias is None
            else attn.qkv_lin.bias[: attn.q_dim],
        )
        k, v = key_value.key, key_value.value
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Cached KV dtype must match the projected query")
    elif key_value is None:
        q, k, v = attn.qkv_lin(q).split(
            [attn.q_dim, attn.kv_dim, attn.kv_dim], dim=-1
        )
    else:
        q = F.linear(
            q,
            attn.qkv_lin.weight[: attn.q_dim],
            None
            if attn.qkv_lin.bias is None
            else attn.qkv_lin.bias[: attn.q_dim],
        )
        k, v = F.linear(
            key_value,
            attn.qkv_lin.weight[attn.q_dim :],
            None
            if attn.qkv_lin.bias is None
            else attn.qkv_lin.bias[attn.q_dim :],
        ).chunk(2, dim=-1)
    cached = isinstance(key_value, KVCacheEntry)
    del key_value
    q = q.unflatten(-1, (attn.num_query_heads, attn.head_dim))
    if not cached:
        k = k.unflatten(-1, (attn.num_key_value_heads, attn.head_dim))
        v = v.unflatten(-1, (attn.num_key_value_heads, attn.head_dim))
    if attn.query_transform is not None:
        q = (
            attn.query_transform(q)
            if query_positions is None
            else attn.query_transform(q, pos=query_positions)
        )
    if not cached and attn.key_transform is not None:
        k = (
            attn.key_transform(k)
            if key_positions is None
            else attn.key_transform(k, pos=key_positions)
        )
    if attn.sdpa.query_scaling is not None:
        key_len = (
            global_key_length
            if seqused_key_value is None
            else seqused_key_value.unsqueeze(-1)
        )
        q = attn.sdpa.query_scaling(q, key_len=key_len)
    output = sharded_attention(
        query=q,
        key=k,
        value=v,
        group=group,
        scale=attn.sdpa.scale,
        seqused_key_value=seqused_key_value,
        replicated_query=replicated_query,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        batch_chunk_size=batch_chunk_size,
    ).flatten(-2)
    cache = (
        KVCacheEntry(k.to(v.dtype).contiguous(), v.contiguous())
        if return_key_value
        else None
    )
    del q, k, v
    output = attn.out_lin(output)
    if block.post_attn_norm is not None:
        output = block.post_attn_norm(output)
    output = output + query
    if block.mlp is not None:
        output = output + block.mlp(output)
    return output if cache is None else (output, cache)
