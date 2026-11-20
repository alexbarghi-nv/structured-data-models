# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/Gloo correctness and ownership tests; no GPU/timing instrumentation."""

import os
import pickle
import tempfile
from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.parallel import _DistributedKumoTabular
from sdm.models.kumo.tabular.sharded_offload import (
    ColumnCacheShard,
    ShardedColumnOffload,
)


def init(rank, world, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )
    return TableMesh("mesh", mesh_shape=(world, 1))


def run(worker, world, *args):
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(
            worker,
            args=(world, os.path.join(d, "rdzv"), *args),
            nprocs=world,
            join=True,
        )


def storage_worker(rank, world, path):
    mesh = init(rank, world, path)
    try:
        owner = ShardedColumnOffload(mesh, chunk_bytes=16)
        # Nonzero offset + dense transpose, unequal K/V lengths, empty tensor,
        # nested ECOC-like caches, and fewer elements than ranks.
        k = (
            torch.arange(36, dtype=torch.float32)[6:30]
            .view(2, 3, 4)
            .transpose(0, 2)
        )
        v = torch.arange(13, dtype=torch.float64)
        cache = Cache(
            nested=[Cache({"row_embedding.col_block0": KVCacheEntry(k, v)})],
            **{
                "row_embedding.col_block1": KVCacheEntry(
                    torch.empty(0, 3), torch.ones(1)
                ),
                "icl_block.layer0": KVCacheEntry(
                    torch.ones(3) * rank, torch.ones(3) * (rank + 1)
                ),
            },
        )
        cache.freeze()
        original_size = cache.size()
        stored = owner.offload(cache, 2)
        shard = cast(list[Cache], stored["nested"])[0][
            "row_embedding.col_block0"
        ]
        assert isinstance(shard, ColumnCacheShard)
        assert stored.is_replaying
        expected = sum(
            (t.numel() * (rank + 1) // world - t.numel() * rank // world)
            * t.element_size()
            for t in (k, v, torch.empty(0, 3), torch.ones(1))
        )
        actual = sum(t.numel() * t.element_size() for t in shard.payloads)
        actual += sum(
            t.numel() * t.element_size()
            for t in cast(
                ColumnCacheShard, stored["row_embedding.col_block1"]
            ).payloads
        )
        assert actual == expected
        # Physical host allocations own only their shard, not a view of the
        # full cache.
        for s in [
            shard,
            cast(ColumnCacheShard, stored["row_embedding.col_block1"]),
        ]:
            for t in s.payloads:
                assert (
                    t.untyped_storage().nbytes()
                    == t.numel() * t.element_size()
                )
        assert stored.size() == expected + 24
        counts = [None] * world
        dist.all_gather_object(counts, actual)
        assert sum(counts) == original_size - 24
        for _ in range(2):
            restored = owner.restore(stored.cpu(), 2, "cpu")
            entry = cast(
                KVCacheEntry,
                cast(list[Cache], restored["nested"])[0][
                    "row_embedding.col_block0"
                ],
            )
            for a, b in zip(entry, (k, v)):
                assert torch.equal(a, b)
                assert a.stride() == b.stride()
            assert torch.equal(
                cast(KVCacheEntry, restored["icl_block.layer0"]).key,
                torch.ones(3) * rank,
            )
            assert torch.equal(
                cast(KVCacheEntry, restored["icl_block.layer0"]).value,
                torch.ones(3) * (rank + 1),
            )
            assert restored.is_replaying
            assert not owner._inflight
        nested = cast(list[Cache], cache["nested"])
        assert (
            cast(KVCacheEntry, nested[0]["row_embedding.col_block0"]).key is k
        )
        assert cache.size() == original_size
        with pytest.raises(TypeError, match="local-session"):
            pickle.dumps(stored)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_storage_partition_roundtrip(world):
    run(storage_worker, world)


def failure_worker(rank, world, path):
    mesh = init(rank, world, path)
    try:

        def fresh():
            return ShardedColumnOffload(mesh, chunk_bytes=16), Cache(
                {
                    "row_embedding.col_block0": KVCacheEntry(
                        torch.arange(13.0).view(1, 13), torch.ones(7)
                    )
                }
            )

        owner, cache = fresh()
        stored = owner.offload(cache, 0)
        if rank == 0:
            owner.clear()
        with pytest.raises(ValueError, match="Stale"):
            owner.restore(stored, 0, "cpu")
        assert owner.poisoned

        owner, cache = fresh()
        if rank == 0:
            cache["row_embedding.col_block0"] = KVCacheEntry(
                torch.ones(3), torch.ones(7)
            )
        with pytest.raises(ValueError, match="metadata/order"):
            owner.offload(cache, 0)

        owner, cache = fresh()
        if rank == 0:
            cache["row_embedding.col_block0"] = KVCacheEntry(
                torch.ones(12)[::2], torch.ones(7)
            )
        with pytest.raises(ValueError, match="dense"):
            owner.offload(cache, 0)

        owner, cache = fresh()
        stored = owner.offload(cache, 0)
        if rank == 0:
            shard = stored["row_embedding.col_block0"]
            stored["row_embedding.col_block0"] = replace(
                shard, payloads=(torch.ones(1), shard.payloads[1])
            )
        with pytest.raises(ValueError, match="payload"):
            owner.restore(stored, 0, "cpu")

        owner, cache = fresh()
        stored = owner.offload(cache, 0)
        broadcast = dist.broadcast

        def fail(*args, **kwargs):
            raise RuntimeError("injected transfer failure")

        dist.broadcast = fail
        try:
            with pytest.raises(RuntimeError, match="injected"):
                owner.restore(stored, 0, "cpu")
        finally:
            dist.broadcast = broadcast
        assert owner.poisoned
        assert owner._inflight
        retained = owner._inflight[0]
        owner.clear()
        with pytest.raises(RuntimeError, match="fresh model/session"):
            owner.restore(stored, 0, "cpu")
        assert owner._inflight[0] is retained
    finally:
        dist.destroy_process_group()


def test_collective_failures():
    run(failure_worker, 2)


def model_worker(rank, world, path, rows):
    mesh = init(rank, world, path)
    try:
        torch.manual_seed(42)
        model = _KumoTabular(
            num_classes=0,
            num_quantiles=1,
            cell_channels=8,
            num_embedding_layers=2,
            num_embedding_heads=2,
            num_inducing_points=3,
            group_size=2,
            num_frequencies=3,
            num_readout_tokens=2,
            icl_channels=16,
            num_icl_layers=2,
            num_icl_heads=2,
        )
        mode = _DistributedKumoTabular(model, mesh)
        mode._fused_pure_row = True
        mode._fused_column_pipeline = True
        mode._fused_replay_original = True
        mode._fused_chunk_rows = 512
        mode._fused_feature_tile = 64
        x = torch.randn(1, rows, 500)
        y = torch.randn(1, rows)
        mask = torch.zeros(1, 500, dtype=torch.bool)
        start, end = mesh.rows(rows)
        cache = Cache(
            cell_impute_mean=x.nanmean(dim=-2, keepdim=True),
            **{"distributed.feature_count": 500},
        )
        owner = ShardedColumnOffload(mesh, chunk_bytes=256)
        with torch.inference_mode():
            embedding = mode._embed_rows(
                x[..., start:end, :], y[..., start:end], mask, rows, cache
            )
            mode._icl(embedding, y[..., start:end], rows, cache)
            cache.freeze()
            stored = owner.offload(cache, 0)
            q = torch.randn(1, 3, 500)

            def predict(c):
                emb = mode._embed_rows(q, y[..., :0], mask, rows, c)
                return mode._icl(emb, y[..., :0], rows, c)

            reference = predict(cache)
            for _ in range(2):
                restored = owner.restore(stored, 0, "cpu")
                for key, entry in cache.items():
                    if isinstance(entry, KVCacheEntry):
                        for a, b in zip(
                            cast(KVCacheEntry, restored[key]), entry
                        ):
                            assert torch.equal(a, b)
                            assert a.stride() == b.stride()
                result = predict(restored)
                torch.testing.assert_close(
                    result, reference, atol=1e-6, rtol=1e-5
                )
                assert torch.equal(result, reference)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("world", "rows"), [(1, 7), (2, 7), (4, 2)])
def test_actual_embedding_icl_replay(world, rows):
    run(model_worker, world, rows)
