# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed protocol, failure ownership and real small-model lifecycle checks."""

import os
from datetime import timedelta
from typing import cast
from unittest.mock import patch

import pyarrow as pa
import pytest
import torch
import torch.distributed as dist

from sdm import StringTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.distributed.tensor_control import (
    _FAILED_TRANSFERS,
    TensorControl,
    pack,
    unpack,
)
from sdm.models.kumo.tabular.prepared import PreparedInputs
from sdm.models.kumo.tabular.sharded_offload import ShardedColumnOffload
from sdm.models.kumo.tabular.typed_exchange import exchange_member
from test.distributed._kumo import check_public_offload
from test.distributed._runtime import (
    distributed_session,
    forbid_object_collectives,
    spawn_workers,
)


def test_codec():
    """Typed packing preserves tensors, nested values and Unicode strings."""
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


def _check_failed_transfer_ownership(group, device):
    """An admitted transfer failure retains its buffers and rejects reuse."""
    size = dist.get_world_size(group)

    # Failure after admission must retain Work and buffers and prohibit
    # reuse.
    class FailedWork:
        def wait(self):
            raise RuntimeError("injected wait failure")

    broken = TensorControl(group, device)
    receive = torch.empty(size, device=device)
    send = torch.ones(size, device=device)
    with (
        patch.object(dist, "all_to_all_single", return_value=FailedWork()),
        pytest.raises(RuntimeError, match="injected wait"),
    ):
        broken.exchange(receive, send, [1] * size, [1] * size)
    assert broken.poisoned
    assert _FAILED_TRANSFERS[-1]["send"] is send
    with pytest.raises(RuntimeError, match="fresh distributed session"):
        broken.reduce(0)


def _check_cache_descriptor_transport(mesh, device):
    """Cache descriptors are typed and mismatched ranks fail together."""
    size, local = mesh.world_size, mesh.rank
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


def _check_feature_payloads(mesh, device):
    """Exchange uneven features/rows and agree on global imputation means."""
    size, local = mesh.world_size, mesh.rank
    # Batched feature payloads, empty row destinations and NaN means.
    prepared = PreparedInputs(mesh, device, 1)
    rows = max(1, size - 1)
    full = torch.arange(2 * rows * 3, device=device).view(2, rows, 3).float()
    full[..., 0] = torch.nan
    all_names = ("a", "b", "c")
    owners = tuple(
        all_names[i * 3 // size : (i + 1) * 3 // size] for i in range(size)
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


def _check_control_messages(control, source, rank, local, size, device):
    """Status, errors, broadcast and scatter use the existing communicator."""
    assert control.local_result(lambda: local) == local

    def failure():
        if local == size - 1:
            raise ValueError("deliberate callback failure")

    with pytest.raises(RuntimeError, match="deliberate callback failure"):
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
    assert torch.equal(result["tensor"], torch.arange(7, device=device))
    result = control.scatter(
        [torch.arange(i * 3, dtype=torch.int64) for i in range(size)]
        if rank == source
        else None,
        source,
    )
    assert result.device == device
    assert torch.equal(result, torch.arange(local * 3, device=device))
    with pytest.raises(RuntimeError, match="one payload per group rank"):
        control.scatter([] if rank == source else None, source)
    with pytest.raises(RuntimeError, match="Unsupported typed control"):
        control.broadcast(object() if rank == source else None, source)


def worker(rank, world, path, backend, subgroup, model):
    with distributed_session(
        rank, world, path, backend=backend, timeout_seconds=90
    ) as device:
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
        with forbid_object_collectives():
            _check_control_messages(control, source, rank, local, size, device)
            _check_feature_payloads(mesh, device)
            _check_cache_descriptor_transport(mesh, device)
            _check_failed_transfer_ownership(group, device)
        if model:
            with forbid_object_collectives():
                check_public_offload(rank, size, device)


@pytest.mark.parametrize(
    ("world", "subgroup", "model"),
    [(1, False, True), (2, False, True), (4, True, False)],
)
def test_protocol_and_model(world, subgroup, model):
    """Typed control and public offload run without object collectives."""
    backend = os.environ.get("STATUS_BACKEND", "gloo")
    assert backend in ("gloo", "nccl")
    if backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.device_count() >= world
    spawn_workers(worker, world, backend, subgroup, model)
