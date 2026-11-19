# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental pure-row column-cache storage; no change to fit arithmetic."""

from sdm.distributed.tensor_control import TensorControl, fingerprint
from dataclasses import dataclass, replace
from math import prod
import re

import torch
import torch.distributed as dist

from sdm.cache import Cache, KVCacheEntry
from sdm.tensor.mixin import DeviceMixin


def _dense(shape, stride):
    """Accept dense permutations, including singleton axes, but not holes/overlap."""
    if prod(shape) == 0:
        return True
    expected = 1
    for step, size in sorted(zip(stride, shape)):
        if size > 1:
            if step != expected:
                return False
            expected *= size
    return True


@dataclass(frozen=True)
class ColumnCacheShard(DeviceMixin):
    session: object
    generation: int
    member: int
    rank: int
    shapes: tuple
    strides: tuple
    payloads: tuple

    def _tensors(self):
        yield from self.payloads

    def _apply_tensor(self, fn):
        return replace(self, payloads=tuple(fn(t) for t in self.payloads))

    def __reduce_ex__(self, protocol):
        raise TypeError(
            "Sharded column caches are local-session state; fit again after loading"
        )


class ShardedColumnOffload:
    """Store balanced physical-storage slices; reconstruct on the current stream.

    Only replicated induced-column K/V is transformed. Full member caches are
    reconstructed before replay, matching the baseline's lifetime. Broadcasts
    are synchronous API calls, use no new streams and have bounded wire chunks.
    A failed operation poisons this instance and retains reconstruction buffers.
    """

    def __init__(
        self, mesh, *, chunk_bytes=32 << 20, max_column_bytes=8 << 30
    ):
        if mesh.feature_parts != 1:
            raise ValueError("Sharded column offload requires a pure-row mesh")
        if chunk_bytes < 16 or max_column_bytes < 0:
            raise ValueError("Invalid sharded column offload bounds")
        self.group = mesh.group
        self.members = tuple(dist.get_process_group_ranks(self.group))
        self.rank = dist.get_rank(self.group)
        self.world = len(self.members)
        self.chunk_bytes = chunk_bytes
        self.max_column_bytes = max_column_bytes
        self.session = object()
        self.generation = 0
        self.poisoned = False
        self._inflight = []

    def clear(self):
        # Invalidates old references; never makes a failed collective reusable.
        self.generation += 1

    def _bounds(self, count, rank):
        return count * rank // self.world, count * (rank + 1) // self.world

    def _map(self, value, fn, path=()):
        if isinstance(value, Cache):
            result = Cache()
            for key, item in value.items():
                here = path + (key,)
                if isinstance(key, str) and re.fullmatch(
                    r"row_embedding\.col_block\d+", key
                ):
                    result[key] = fn(item, here)
                else:
                    result[key] = self._map(item, fn, here)
            return result.freeze() if value.is_replaying else result
        if isinstance(value, list):
            return [self._map(x, fn, path + (i,)) for i, x in enumerate(value)]
        if isinstance(value, tuple) and not isinstance(value, DeviceMixin):
            return tuple(
                self._map(x, fn, path + (i,)) for i, x in enumerate(value)
            )
        if isinstance(value, dict):
            return {k: self._map(x, fn, path + (k,)) for k, x in value.items()}
        return value

    def _validate(self, cache, member, restoring, device=None):
        if self.poisoned:
            raise RuntimeError(
                "Sharded column offload failed; use a fresh model/session"
            )
        metadata = []
        error = None
        control_device = device
        try:
            if tuple(dist.get_process_group_ranks(self.group)) != self.members:
                raise RuntimeError("Column-cache communicator changed")
            if dist.get_rank(self.group) != self.rank:
                raise RuntimeError("Column-cache rank changed")
            if restoring:
                backend = str(dist.get_backend(self.group))
                if backend == "nccl" and device.type != "cuda":
                    raise ValueError(
                        "NCCL cache reconstruction requires a CUDA device"
                    )
                if backend == "gloo" and device.type != "cpu":
                    raise ValueError(
                        "Gloo cache reconstruction supports CPU tests only"
                    )

            def visit(value, path):
                nonlocal control_device
                if restoring:
                    if not isinstance(value, ColumnCacheShard):
                        raise TypeError("Expected a column-cache shard")
                    if (
                        value.session is not self.session
                        or value.generation != self.generation
                        or value.member != member
                        or value.rank != self.rank
                    ):
                        raise ValueError(
                            "Stale or misassigned column-cache shard"
                        )
                    shapes, strides, tensors = (
                        value.shapes,
                        value.strides,
                        value.payloads,
                    )
                    if (
                        len(shapes) != 2
                        or len(strides) != 2
                        or len(tensors) != 2
                    ):
                        raise ValueError("Invalid K/V shard descriptor")
                else:
                    if not isinstance(value, KVCacheEntry):
                        raise TypeError("Expected induced column K/V")
                    tensors = tuple(value)
                    if tensors and tensors[0].is_cuda:
                        control_device = tensors[0].device
                    shapes = tuple(tuple(t.shape) for t in tensors)
                    strides = tuple(tuple(t.stride()) for t in tensors)
                for shape, stride, tensor in zip(shapes, strides, tensors):
                    if len(shape) != len(stride) or not _dense(shape, stride):
                        raise ValueError(
                            "Column K/V must be nonoverlapping dense tensors"
                        )
                    if restoring:
                        a, b = self._bounds(prod(shape), self.rank)
                        if (
                            tensor.device.type != "cpu"
                            or tensor.ndim != 1
                            or tensor.numel() != b - a
                            or not tensor.is_contiguous()
                        ):
                            raise ValueError("Invalid CPU shard payload")
                    elif tensor.layout != torch.strided:
                        raise ValueError("Column K/V must be strided tensors")
                metadata.append(
                    (path, shapes, strides, tuple(t.dtype for t in tensors))
                )
                return value

            self._map(cache, visit)
            if not metadata:
                raise ValueError(
                    "No column K/V found; refusing a silent no-op"
                )
            total = sum(
                prod(s) * torch.empty((), dtype=d).element_size()
                for _, shapes, _, dtypes in metadata
                for s, d in zip(shapes, dtypes)
            )
            if total > self.max_column_bytes:
                raise ValueError("Column cache exceeds reconstruction budget")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Propagate local descriptor errors before any payload collectives.
        if dist.get_backend(self.group) == "nccl":
            if (
                control_device is None
                or torch.device(control_device).type != "cuda"
            ):
                control_device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
        else:
            control_device = torch.device("cpu")
        control = TensorControl(self.group, control_device)
        if control.reduce(int(error is not None)):
            errors = control.gather_bytes((error or "").encode())
            raise ValueError(
                "; ".join(value.decode() for value in errors if value)
            )
        digest = control.local_result(
            lambda: fingerprint(
                (
                    member,
                    self.generation,
                    restoring,
                    self.chunk_bytes,
                    self.max_column_bytes,
                    metadata,
                )
            )
        )
        signature = torch.tensor(
            list(digest), dtype=torch.uint8, device=control.device
        )
        signatures = [torch.empty_like(signature) for _ in range(self.world)]
        dist.all_gather(signatures, signature, group=self.group)
        all_signatures = torch.stack(signatures)
        if not bool((all_signatures == all_signatures[:1]).all().item()):
            raise ValueError(
                "Column-cache metadata/order differs across ranks"
            )

    def offload(self, cache, member):
        try:
            self._validate(cache, member, False)

            def split(entry, path):
                payloads = []
                for tensor in entry:
                    # Physical order preserves dense strides on reconstruction.
                    # Narrow BEFORE copying, so no full-column D2H occurs here.
                    flat = tensor.as_strided((tensor.numel(),), (1,))
                    a, b = self._bounds(tensor.numel(), self.rank)
                    payloads.append(flat[a:b].to("cpu", copy=True))
                return ColumnCacheShard(
                    self.session,
                    self.generation,
                    member,
                    self.rank,
                    tuple(tuple(t.shape) for t in entry),
                    tuple(tuple(t.stride()) for t in entry),
                    tuple(payloads),
                )

            # Non-column state follows the original offload, including ICL K/V.
            return self._map(cache, split).to("cpu")
        except Exception:
            self.poisoned = True
            raise

    def restore(self, cache, member, device):
        try:
            device = torch.device(device)
            self._validate(cache, member, True, device)

            def gather(shard, path):
                values = []
                for shape, stride, payload in zip(
                    shard.shapes, shard.strides, shard.payloads
                ):
                    tensor = torch.empty_strided(
                        shape, stride, dtype=payload.dtype, device=device
                    )
                    self._inflight.append(tensor)
                    flat = tensor.as_strided((tensor.numel(),), (1,))
                    step = self.chunk_bytes // tensor.element_size()
                    for owner, source in enumerate(self.members):
                        a, b = self._bounds(tensor.numel(), owner)
                        if owner == self.rank:
                            flat[a:b].copy_(payload, non_blocking=False)
                        for start in range(a, b, step):
                            dist.broadcast(
                                flat[start : min(start + step, b)],
                                src=source,
                                group=self.group,
                            )
                    values.append(tensor)
                return KVCacheEntry(*values)

            restored = self._map(cache, gather).to(device)
            self._inflight.clear()
            return restored
        except Exception:
            self.poisoned = True
            raise
