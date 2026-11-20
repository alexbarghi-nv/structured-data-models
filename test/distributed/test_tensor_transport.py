# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed protocol, failure ownership and real small-model lifecycle checks."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pyarrow as pa
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sdm.models.kumo.tabular.prepared as prepared_module
from sdm import StringTensor, TableTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.distributed.tensor_control import (
    _FAILED_TRANSFERS,
    TensorControl,
    pack,
    unpack,
)
from sdm.models.kumo.tabular.distributed import DistributedKumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS
from sdm.models.kumo.tabular.prepared import PreparedInputs
from sdm.models.kumo.tabular.sharded_offload import ShardedColumnOffload
from sdm.models.kumo.tabular.typed_exchange import exchange_member
from test.distributed.test_column_metadata import host_value, legacy_fit


def forbidden(*args, **kwargs):
    raise AssertionError("Object collective reached on typed path")


def test_codec():
    x = torch.arange(30, dtype=torch.float64).view(5, 6).t()[1:5]
    value = {
        "x": x,
        "empty": torch.empty(0, 3),
        "scalar": torch.tensor(2),
        "strings": StringTensor.from_arrow(
            pa.array(["é", None, "long column"])
        ),
        "nested": (None, True, ["héllo", 2, torch.float32]),
    }
    actual = unpack(pack(value, torch.device("cpu")))
    assert torch.equal(actual["x"], x)
    assert actual["empty"].shape == (0, 3)
    assert actual["scalar"].item() == 2
    assert actual["nested"] == value["nested"]
    assert actual["strings"].to_arrow().to_pylist() == [
        "é",
        None,
        "long column",
    ]


def worker(rank, world, path, backend, subgroup, model):
    torch.set_num_threads(1)
    device = (
        torch.device("cuda", rank)
        if backend == "nccl"
        else torch.device("cpu")
    )
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method="file://" + path,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        group = dist.group.WORLD
        if subgroup:
            group = dist.new_group(
                [1, 3], backend=backend, timeout=timedelta(seconds=90)
            )
            if rank not in (1, 3):
                return
        group = cast(dist.ProcessGroup, group)
        size = dist.get_world_size(group)
        local = dist.get_rank(group)
        mesh = TableMesh("mesh", group=group, mesh_shape=(size, 1))
        source = dist.get_process_group_ranks(group)[-1]
        control = TensorControl(group, device)
        with (
            patch.object(dist, "all_gather_object", forbidden),
            patch.object(dist, "broadcast_object_list", forbidden),
            patch.object(dist, "scatter_object_list", forbidden),
        ):
            assert control.local_result(lambda: local) == local

            def failure():
                if local == size - 1:
                    raise ValueError("deliberate callback failure")

            with pytest.raises(
                RuntimeError, match="deliberate callback failure"
            ):
                control.local_result(failure)
            assert control.local_result(lambda: None) is None
            result = control.broadcast(
                {
                    "tensor": torch.arange(7, device=device),
                    "name": "cat__count",
                }
                if rank == source
                else None,
                source,
            )
            assert result["name"] == "cat__count"
            assert torch.equal(
                result["tensor"], torch.arange(7, device=device)
            )
            result = control.scatter(
                [torch.arange(i * 3, dtype=torch.int64) for i in range(size)]
                if rank == source
                else None,
                source,
            )
            assert result.device == device
            assert torch.equal(result, torch.arange(local * 3, device=device))
            with pytest.raises(
                RuntimeError, match="one payload per group rank"
            ):
                control.scatter([] if rank == source else None, source)
            with pytest.raises(
                RuntimeError, match="Unsupported typed control"
            ):
                control.broadcast(object() if rank == source else None, source)
            # Batched feature payloads, empty row destinations and NaN means.
            prepared = PreparedInputs(mesh, "sharded_cpu", device, 1)
            rows = max(1, size - 1)
            full = (
                torch.arange(2 * rows * 3, device=device)
                .view(2, rows, 3)
                .float()
            )
            full[..., 0] = torch.nan
            all_names = ("a", "b", "c")
            owners = tuple(
                all_names[i * 3 // size : (i + 1) * 3 // size]
                for i in range(size)
            )
            left, right = local * 3 // size, (local + 1) * 3 // size
            bounds = [mesh.bounds(rows, i, size) for i in range(size)]
            data, mean = exchange_member(
                prepared,
                full[..., left:right],
                owners[local],
                owners,
                [all_names] * size,
                bounds,
            )
            a, b = bounds[local]
            expected_mean = full.nanmean(-2, keepdim=True).nan_to_num(nan=0)
            torch.testing.assert_close(
                data, full[..., a:b, :], atol=0, rtol=0, equal_nan=True
            )
            torch.testing.assert_close(mean, expected_mean, atol=0, rtol=0)
            with pytest.raises(RuntimeError, match="schema"):
                exchange_member(
                    prepared,
                    full[..., left:right],
                    ("bad",) if local == size - 1 else owners[local],
                    owners,
                    [all_names] * size,
                    bounds,
                )
            # Cache descriptors use tensor signatures; no object gather on
            # fit/replay.
            owner = ShardedColumnOffload(mesh, chunk_bytes=32)
            cache = Cache(
                {
                    "row_embedding.col_block0": KVCacheEntry(
                        torch.arange(12, device=device).view(3, 4).t(),
                        torch.ones(7, device=device),
                    )
                }
            ).freeze()
            stored = owner.offload(cache, 0)
            restored = owner.restore(stored, 0, device)
            assert torch.equal(
                cast(KVCacheEntry, restored["row_embedding.col_block0"]).key,
                cast(KVCacheEntry, cache["row_embedding.col_block0"]).key,
            )
            bad_owner = ShardedColumnOffload(mesh)
            bad_cache = Cache(
                {
                    "row_embedding.col_block0": KVCacheEntry(
                        torch.ones(3 + local, device=device),
                        torch.ones(7, device=device),
                    )
                }
            ).freeze()
            if size > 1:
                with pytest.raises(ValueError, match="metadata/order"):
                    bad_owner.offload(bad_cache, 0)
                assert bad_owner.poisoned

            # Failure after admission must retain Work and buffers and prohibit
            # reuse.
            class FailedWork:
                def wait(self):
                    raise RuntimeError("injected wait failure")

            broken = TensorControl(group, device)
            receive = torch.empty(size, device=device)
            send = torch.ones(size, device=device)
            with (
                patch.object(
                    dist, "all_to_all_single", return_value=FailedWork()
                ),
                pytest.raises(RuntimeError, match="injected wait"),
            ):
                broken.exchange(receive, send, [1] * size, [1] * size)
            assert broken.poisoned
            assert _FAILED_TRANSFERS[-1]["send"] is send
            with pytest.raises(
                RuntimeError, match="fresh distributed session"
            ):
                broken.reduce(0)
        if not model:
            return
        original_kwargs = MODEL_KWARGS["small"]
        MODEL_KWARGS["small"] = {
            "cell_channels": 8,
            "num_embedding_layers": 2,
            "num_embedding_heads": 2,
            "num_inducing_points": 3,
            "group_size": 3,
            "num_frequencies": 2,
            "num_readout_tokens": 2,
            "icl_channels": 16,
            "num_icl_layers": 3,
            "num_icl_heads": 4,
            "num_icl_key_value_heads_for_query": 2,
        }
        candidate = prepared_module.fit_sharded
        candidate_transform = prepared_module.transform_sharded
        baseline = legacy_fit()
        broadcast, scatter = PreparedInputs.broadcast, PreparedInputs.scatter
        x = TableTensor.from_tensor(
            torch.arange(19 * 17).view(19, 17).float().sin()
        )
        query = TableTensor.from_tensor(
            torch.arange(51).view(3, 17).float().cos()
        )
        x.numerical[2, 0] = torch.nan
        try:
            for task in ["regression", "classification"]:
                outputs = []
                for implementation in [baseline, candidate]:
                    with (
                        patch.object(
                            prepared_module, "fit_sharded", implementation
                        ),
                        patch.object(
                            prepared_module,
                            "transform_sharded",
                            implementation.__globals__["transform_sharded"],
                        ),
                        patch.object(
                            PreparedInputs,
                            "broadcast",
                            broadcast
                            if implementation is candidate
                            else lambda self, *a, **kw: host_value(
                                broadcast(self, *a, **kw)
                            ),
                        ),
                        patch.object(
                            PreparedInputs,
                            "scatter",
                            scatter
                            if implementation is candidate
                            else lambda self, *a, **kw: host_value(
                                scatter(self, *a, **kw)
                            ),
                        ),
                    ):
                        torch.manual_seed(42)
                        m = DistributedKumoTabular(
                            task=task,
                            size="small",
                            pretrained=False,
                            device=device,
                            layout="mesh",
                            mesh_shape=(size, 1),
                            preprocessing="sharded_cuda"
                            if backend == "nccl"
                            else "sharded_cpu",
                            sharded_column_offload=True,
                        )
                        # Constructor uses WORLD; lifecycle cases run on WORLD
                        # only.
                        guards = []
                        if implementation is candidate:
                            for name in [
                                "all_gather_object",
                                "broadcast_object_list",
                                "scatter_object_list",
                            ]:
                                guard = patch.object(dist, name, forbidden)
                                guard.start()
                                guards.append(guard)
                        try:
                            y = torch.arange(19).view(-1, 1)
                            y = (
                                y % 3
                                if task == "classification"
                                else y.float().sin()
                            )
                            m.fit(
                                x if rank == 0 else None,
                                y if rank == 0 else None,
                                num_estimators=2,
                                generator=torch.Generator(
                                    device=device
                                ).manual_seed(37),
                            )
                            first = (
                                m.predict(query if rank == 0 else None)
                                .numerical.cpu()
                                .clone()
                            )
                            repeat = (
                                m.predict(query if rank == 0 else None)
                                .numerical.cpu()
                                .clone()
                            )
                            torch.testing.assert_close(
                                first, repeat, atol=0, rtol=0
                            )
                            outputs.append(first)
                            m.clear()
                        finally:
                            for guard in reversed(guards):
                                guard.stop()
                torch.testing.assert_close(
                    outputs[0], outputs[1], atol=1e-6, rtol=1e-5
                )
                if backend == "gloo":
                    assert torch.equal(outputs[0], outputs[1])
        finally:
            prepared_module.fit_sharded = candidate
            prepared_module.transform_sharded = candidate_transform
            PreparedInputs.broadcast, PreparedInputs.scatter = (
                broadcast,
                scatter,
            )
            MODEL_KWARGS["small"] = original_kwargs
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "subgroup", "model"),
    [(1, False, True), (2, False, True), (4, True, False)],
)
def test_protocol_and_model(world, subgroup, model):
    backend = os.environ.get("STATUS_BACKEND", "gloo")
    assert backend in ("gloo", "nccl")
    if backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.device_count() >= world
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            worker,
            args=(world, str(Path(tmp) / "rdzv"), backend, subgroup, model),
            nprocs=world,
            join=True,
        )
