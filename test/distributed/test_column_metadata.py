# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer metadata codecs and tensor gathers, with explicit NCCL opt-in."""

import os
from datetime import timedelta
from typing import cast

import pytest
import torch
import torch.distributed as dist

from sdm import Stype
from sdm.models.kumo.tabular.column_metadata import (
    ColumnMetadata,
    column_dictionary,
    decode_column_metadata,
    encode_column_metadata,
    gather_column_metadata,
)
from test.distributed._runtime import distributed_session, spawn_workers


def test_dictionary_and_codec():
    """Stable IDs round-trip Unicode, duplicate and empty metadata."""
    # Include Unicode and a real name that resembles a generated count name.
    schema = {
        Stype.numerical: ("z", "a__count"),
        Stype.categorical: ("é", "a"),
    }
    names = column_dictionary(schema)
    assert names == tuple(sorted({"z", "a__count", "é", "a", "é__count"}))
    metadata: ColumnMetadata = [
        (("z", "a", "z", "a__count"), ("z",), ("a__count",)),
        ((), (), ()),
    ]
    packed = encode_column_metadata(metadata, names)
    assert decode_column_metadata(packed.tolist(), names) == metadata
    assert (
        decode_column_metadata(encode_column_metadata([], ()).tolist(), ())
        == []
    )
    with pytest.raises(KeyError):
        encode_column_metadata([(("unknown",), (), ())], names)


def worker(rank, world, path, backend, subgroup):
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
        local_rank = dist.get_rank(group)
        size = dist.get_world_size(group)
        names = ("a", "b", "z", "é")

        def metadata_for(i: int) -> ColumnMetadata:
            return [
                (tuple(names[: i % 4]), ("z",) * i, ()),
                (("é", "a"), (), ()),
            ]

        # Exchange lengths before gathering uneven ID payloads.
        metadata = metadata_for(local_rank)
        # No object collective in the transport helper, even for uneven
        # lengths.
        original = dist.all_gather_object

        def forbidden(*args, **kwargs):
            raise AssertionError("Object gather in ID transport")

        dist.all_gather_object = forbidden
        try:
            actual = gather_column_metadata(
                encode_column_metadata(metadata, names),
                names,
                group=group,
                device=device,
            )
            assert actual == [metadata_for(i) for i in range(size)]
            assert gather_column_metadata(
                encode_column_metadata([], ()), (), group=group, device=device
            ) == [[] for _ in range(size)]
        finally:
            dist.all_gather_object = original


@pytest.mark.parametrize(
    ("world", "subgroup"), [(1, False), (2, False), (4, False), (4, True)]
)
def test_distributed_metadata(world, subgroup):
    """Uneven metadata gathers work in full groups and subgroups."""
    backend = os.environ.get("STATUS_BACKEND", "gloo")
    assert backend in ("gloo", "nccl")
    if backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.device_count() >= world
    spawn_workers(worker, world, backend, subgroup)
