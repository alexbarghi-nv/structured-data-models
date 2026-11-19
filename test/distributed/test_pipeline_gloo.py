# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare tiled column recording and original cache replay."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sdm.models.kumo.tabular.parallel as parallel
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.parallel import _DistributedKumoTabular


def compare_cache(actual, reference):
    assert set(actual) == set(reference)
    for key in actual:
        a, b = actual[key], reference[key]
        if isinstance(a, KVCacheEntry):
            torch.testing.assert_close(a.key, b.key, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(a.value, b.value, atol=1e-6, rtol=1e-5)
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        else:
            assert a == b


def worker(rank, world, path, rows):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=world
    )
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
            num_icl_layers=1,
            num_icl_heads=2,
        )
        mesh = TableMesh(layout="mesh", mesh_shape=(world, 1))
        modes = [_DistributedKumoTabular(model, mesh) for _ in range(2)]
        for mode in modes:
            mode._fused_pure_row = True
            mode._fused_chunk_rows = 512
            mode._fused_feature_tile = 64
            mode._fused_replay_original = True
        modes[1]._fused_column_pipeline = True
        x = torch.randn(1, rows, 1000)[..., ::2]
        y = torch.randn(1, rows * 2)[..., ::2]
        x_before = x.clone()
        y_before = y.clone()
        mask = torch.zeros(1, 500, dtype=torch.bool)
        start, end = mesh.rows(rows)
        caches = [Cache() for _ in modes]
        for cache in caches:
            cache["cell_impute_mean"] = x.nanmean(dim=-2, keepdim=True)
            cache["distributed.feature_count"] = 500
        with torch.inference_mode():
            values = []
            observed = []
            for mode, cache in zip(modes, caches):
                reads = []
                originals = []
                for col in model.row_embedding.col_blocks:
                    original = col.summarize
                    originals.append(original)

                    def record(
                        value,
                        *args,
                        _original=original,
                        _reads=reads,
                        **kwargs,
                    ):
                        _reads.append(value.detach().clone())
                        return _original(value, *args, **kwargs)

                    col.summarize = record
                try:
                    values.append(
                        mode._embed_rows(
                            x[..., start:end, :],
                            y[..., start:end],
                            mask,
                            rows,
                            cache,
                        )
                    )
                finally:
                    for col, original in zip(
                        model.row_embedding.col_blocks, originals
                    ):
                        col.summarize = original
                observed.append(reads)
            assert len(observed[0]) == len(observed[1])
            for before, after in zip(*observed):
                torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(
                values[1], values[0], atol=1e-6, rtol=1e-5
            )
            compare_cache(caches[1], caches[0])
            assert torch.equal(x, x_before)
            assert torch.equal(y, y_before)
            for key in caches[1]:
                if isinstance(caches[1][key], KVCacheEntry):
                    assert caches[1][key].key.shape == caches[0][key].key.shape
                    assert (
                        caches[1][key].value.shape
                        == caches[0][key].value.shape
                    )
                    assert caches[1][key].key.is_contiguous()
                    assert caches[1][key].value.is_contiguous()
                    assert (
                        caches[1][key].key.stride()
                        == caches[0][key].key.stride()
                    )
                    assert (
                        caches[1][key].value.stride()
                        == caches[0][key].value.stride()
                    )
            for cache in caches:
                cache.freeze()
            preserved = {
                k: (v, v.key, v.value, v.key.clone(), v.value.clone())
                for k, v in caches[1].items()
                if isinstance(v, KVCacheEntry)
            }
            q = torch.randn(1, 3, 500)
            empty = y[..., :0]
            values = [
                mode._embed_rows(q, empty, mask, rows, cache)
                for mode, cache in zip(modes, caches)
            ]
            torch.testing.assert_close(
                values[1], values[0], atol=1e-6, rtol=1e-5
            )
            again = modes[1]._embed_rows(q, empty, mask, rows, caches[1])
            torch.testing.assert_close(again, values[1], atol=1e-6, rtol=1e-5)
            for k, (
                entry,
                key,
                value,
                key_copy,
                value_copy,
            ) in preserved.items():
                assert caches[1][k] is entry
                assert entry.key is key
                assert entry.value is value
                assert torch.equal(entry.key, key_copy)
                assert torch.equal(entry.value, value_copy)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "rows"), [(1, 7), (2, 7), (4, 7), (4, 2), (1, 1250)]
)
def test_replay_dispatch(world, rows):
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(
            worker,
            args=(world, os.path.join(d, "rdzv"), rows),
            nprocs=world,
            join=True,
        )


def test_fake_ticket_order_and_failure(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(tmp_path / "ledger"),
        rank=0,
        world_size=1,
    )
    ledger = []
    queues = []

    class Ticket:
        def __init__(self, out, lse, group, stream):
            self.value = (out, lse)
            self.consumed = False

    class Queue:
        def __init__(self):
            self.ticket = None
            self.poisoned = False
            queues.append(self)

        def submit(self, factory, *args):
            if self.ticket is not None:
                assert self.ticket.consumed
            self.ticket = factory(*args)
            ledger.append("submit")

        def drain(self):
            assert self.ticket is not None
            assert not self.ticket.consumed
            self.ticket.consumed = True
            ledger.append("drain")
            return self.ticket.value

        def close(self):
            ledger.append("close")
            self.ticket = None

    try:
        monkeypatch.setattr(
            torch.Tensor, "is_cuda", property(lambda self: True)
        )
        monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: None)
        monkeypatch.setattr(
            parallel, "chunk_memory_limit", lambda device: 2**30
        )
        monkeypatch.setattr(parallel, "OneTicketQueue", Queue)
        monkeypatch.setattr(parallel, "CudaSummaryTicket", Ticket)
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
            num_icl_layers=1,
            num_icl_heads=2,
        )
        mode = _DistributedKumoTabular(
            model, TableMesh(layout="mesh", mesh_shape=(1, 1))
        )
        mode._fused_pure_row = True
        mode._fused_replay_original = True
        mode._fused_column_pipeline = True
        mode._fused_async = True
        mode._fused_chunk_rows = 512
        mode._fused_feature_tile = 64
        for col in model.row_embedding.col_blocks:
            col.register_forward_pre_hook(
                lambda m, args: ledger.append("column")
            )
        x = torch.randn(1, 2, 130)
        y = torch.randn(1, 2)
        mask = torch.zeros(1, 130, dtype=torch.bool)
        cache = Cache()
        cache["cell_impute_mean"] = x.nanmean(dim=-2, keepdim=True)
        cache["distributed.feature_count"] = 130
        with torch.inference_mode():
            mode._embed_rows(x, y, mask, 2, cache)
        assert (
            ledger
            == [
                "submit",
                "drain",
                "submit",
                "column",
                "drain",
                "submit",
                "column",
                "drain",
                "close",
                "column",
            ]
            * 2
        )
        assert mode._fused_async_queue is None
        ledger.clear()
        cache = Cache()
        cache["cell_impute_mean"] = x.nanmean(dim=-2, keepdim=True)
        cache["distributed.feature_count"] = 130

        def fail(*args, **kwargs):
            raise ValueError("projection failure")

        column0 = model.row_embedding.col_blocks[0]
        original_induced = column0.induced_key_value
        monkeypatch.setattr(column0, "induced_key_value", fail)
        with (
            torch.inference_mode(),
            pytest.raises(ValueError, match="projection failure"),
        ):
            mode._embed_rows(x, y, mask, 2, cache)
        assert ledger[:3] == ["submit", "drain", "submit"]
        assert mode._fused_async_queue is queues[-1]
        assert queues[-1].ticket is not None
        assert queues[-1].poisoned
        mode._fused_async_queue = None
        ledger.clear()
        cache = Cache()
        cache["cell_impute_mean"] = x.nanmean(dim=-2, keepdim=True)
        cache["distributed.feature_count"] = 130
        monkeypatch.setattr(column0, "induced_key_value", original_induced)
        original_forward = column0.forward
        monkeypatch.setattr(column0, "forward", fail)
        with (
            torch.inference_mode(),
            pytest.raises(ValueError, match="projection failure"),
        ):
            mode._embed_rows(x, y, mask, 2, cache)
        assert ledger[:3] == ["submit", "drain", "submit"]
        assert mode._fused_async_queue is queues[-1]
        assert queues[-1].ticket is not None
        assert queues[-1].poisoned
        mode._fused_async_queue = None
        ledger.clear()
        cache = Cache()
        cache["cell_impute_mean"] = x.nanmean(dim=-2, keepdim=True)
        cache["distributed.feature_count"] = 130
        monkeypatch.setattr(column0, "forward", original_forward)
        original_cat = torch.cat

        def fail_assembly(values, *args, **kwargs):
            if (
                kwargs.get("dim") == -4
                and len(values) == 3
                and all(v.ndim == 5 for v in values)
            ):
                raise ValueError("assembly failure")
            return original_cat(values, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", fail_assembly)
        with (
            torch.inference_mode(),
            pytest.raises(ValueError, match="assembly failure"),
        ):
            mode._embed_rows(x, y, mask, 2, cache)
        assert mode._fused_async_queue is queues[-1]
        assert queues[-1].poisoned
    finally:
        dist.destroy_process_group()
