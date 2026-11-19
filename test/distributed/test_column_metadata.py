# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed metadata tests with explicit NCCL opt-in."""

import importlib.machinery
import importlib.util
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sdm.models.kumo.tabular.prepared as prepared_module
from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.column_metadata import (
    column_dictionary,
    decode_column_metadata,
    encode_column_metadata,
    gather_column_metadata,
)
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.models.kumo.tabular.prepared import PreparedInputs
from sdm.models.kumo.tabular.recipe import default_recipe


def test_dictionary_and_codec():
    schema = {
        Stype.numerical: ("z", "a__count"),
        Stype.categorical: ("é", "a"),
    }
    names = column_dictionary(schema)
    assert names == tuple(sorted({"z", "a__count", "é", "a", "é__count"}))
    metadata = [
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


def legacy_fit():
    path = Path(__file__).with_name("baseline_sharded_recipe.py.reference")
    loader = importlib.machinery.SourceFileLoader(
        "baseline_sharded_recipe", str(path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.fit_sharded


def host_value(value):
    """Stage legacy transport on CPU, including in CUDA tests."""
    if isinstance(value, (torch.Tensor, EnsembleTable)):
        return value.cpu()
    if isinstance(value, list):
        return [host_value(x) for x in value]
    if isinstance(value, tuple):
        return tuple(host_value(x) for x in value)
    if isinstance(value, dict):
        return {k: host_value(v) for k, v in value.items()}
    return value


def cases():
    numeric = torch.arange(67 * 5).view(67, 5).float().sin()
    numeric[:, 1] = 1
    numeric[2, 3] = torch.nan
    mixed = TableTensor(
        columns={
            "numerical": ["z", "cat__count", "a", "é", "b"],
            "categorical": ["cat", "smallcat"],
        },
        numerical=numeric,
        categorical=CategoricalTensor(
            torch.stack([torch.arange(67) % 60, torch.arange(67) % 3], -1),
            categories=[torch.arange(60), torch.arange(3)],
        ),
    )
    varied = torch.arange(4 * 7 * 8).view(4, 7, 8).float().sin()
    for i in range(4):
        varied[i, :, :i] = 0
    return [
        (mixed, torch.arange(67).view(-1, 1).float().sin(), 2),
        (
            TableTensor.from_tensor(varied),
            torch.arange(28).view(4, 7, 1).float(),
            None,
        ),
        (
            TableTensor.from_tensor(
                torch.arange(9 * 517).view(9, 517).float().sin()
            ),
            torch.arange(9).view(-1, 1).float(),
            2,
        ),
        (
            TableTensor.from_tensor(torch.arange(9).view(9, 1).float()),
            torch.arange(9).view(-1, 1).float(),
            2,
        ),
    ]


def snapshot(contexts, queries, prepared, generator):
    result = [generator.get_state().cpu().clone()]
    names = []
    for context, query in zip(contexts, queries, strict=True):
        result.extend(
            [
                context.context.x.numerical.clone(),
                context.context.y.numerical.clone(),
                context.feature_indices.clone(),
                context.categorical_mask.clone(),
                context.mean.clone(),
                query.x.numerical.clone(),
            ]
        )
        names.append(
            (context.context.x.columns, query.x.columns, context.feature_count)
        )
    for numeric, signs, chosen in prepared.plans:
        names.append((numeric, chosen))
        result.append(signs.clone())
    return [tensor.detach().cpu().clone() for tensor in result], names


def make_shard(table, mesh):
    names = [name for stype in Stype for name in table.columns[stype]]
    owned = []
    for stype in Stype:
        columns = table.columns[stype]
        start, end = mesh.bounds(len(columns), mesh.rank, mesh.world_size)
        owned.extend(columns[start:end])
    rows = torch.arange(table.size(-2)).expand(table.shape[:-1])
    return ColumnShard(
        table.select_columns(owned).clone(),
        tuple(names.index(name) for name in owned),
        dict(table.columns),
        rows,
    )


def worker(rank, world, path, backend, subgroup, integration, direct):
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
        local_rank = dist.get_rank(group)
        size = dist.get_world_size(group)
        names = ("a", "b", "z", "é")

        def metadata_for(i):
            return [
                (tuple(names[: i % 4]), ("z",) * i, ()),
                (("é", "a"), (), ()),
            ]

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
        if not integration:
            return
        mesh = TableMesh("mesh", group=group, mesh_shape=(size, 1))
        candidate = prepared_module.fit_sharded
        candidate_transform = prepared_module.transform_sharded
        baseline = legacy_fit()
        broadcast, scatter = PreparedInputs.broadcast, PreparedInputs.scatter
        try:
            for x, y, estimators in cases():
                snapshots = []
                for implementation in [baseline, candidate]:
                    prepared_module.fit_sharded = implementation
                    prepared_module.transform_sharded = (
                        implementation.__globals__["transform_sharded"]
                    )
                    PreparedInputs.broadcast = (
                        broadcast
                        if implementation is candidate
                        else (
                            lambda self, *a, **kw: host_value(
                                broadcast(self, *a, **kw)
                            )
                        )
                    )
                    PreparedInputs.scatter = (
                        scatter
                        if implementation is candidate
                        else (
                            lambda self, *a, **kw: host_value(
                                scatter(self, *a, **kw)
                            )
                        )
                    )
                    prepared = PreparedInputs(mesh, "sharded_cpu", device, 3)
                    generator = torch.Generator(device=device).manual_seed(37)
                    fit_x = (
                        make_shard(x, mesh)
                        if direct
                        else (x if local_rank == 0 else None)
                    )
                    query = x[..., :3, :]
                    query_x = (
                        make_shard(query, mesh)
                        if direct
                        else (query if local_rank == 0 else None)
                    )
                    contexts = prepared.fit(
                        fit_x,
                        y if local_rank == 0 else None,
                        default_recipe(),
                        estimators,
                        generator,
                    )
                    queries = prepared.transform(query_x)
                    snapshots.append(
                        snapshot(contexts, queries, prepared, generator)
                    )
                left, right = snapshots
                assert left[1] == right[1]
                for a, b in zip(left[0], right[0], strict=True):
                    torch.testing.assert_close(
                        a, b, atol=0, rtol=0, equal_nan=True
                    )
        finally:
            prepared_module.fit_sharded = candidate
            prepared_module.transform_sharded = candidate_transform
            PreparedInputs.broadcast, PreparedInputs.scatter = (
                broadcast,
                scatter,
            )
        # Unknown names must be agreed as failures before any rank enters the
        # size/payload gathers. This exercises the real encoding callback path.
        encoder = candidate.__globals__["encode_column_metadata"]

        def invalid_name(metadata, dictionary):
            if local_rank == size - 1:
                return encoder(
                    [(("injected_unknown_column",), (), ())], dictionary
                )
            return encoder(metadata, dictionary)

        candidate.__globals__["encode_column_metadata"] = invalid_name
        try:
            x, y, estimators = cases()[-1]
            prepared = PreparedInputs(mesh, "sharded_cpu", device, 3)
            with pytest.raises(RuntimeError, match="injected_unknown_column"):
                prepared.fit(
                    x if local_rank == 0 else None,
                    y if local_rank == 0 else None,
                    default_recipe(),
                    estimators,
                    torch.Generator(device=device).manual_seed(37),
                )
        finally:
            candidate.__globals__["encode_column_metadata"] = encoder
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "subgroup", "integration", "direct"),
    [
        (1, False, True, False),
        (2, False, True, False),
        (4, False, True, False),
        (2, False, True, True),
        (4, False, True, True),
        (4, True, False, False),
    ],
)
def test_distributed_metadata(world, subgroup, integration, direct):
    backend = os.environ.get("STATUS_BACKEND", "gloo")
    assert backend in ("gloo", "nccl")
    if backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.device_count() >= world
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            worker,
            args=(
                world,
                str(Path(tmp) / "rdzv"),
                backend,
                subgroup,
                integration,
                direct,
            ),
            nprocs=world,
            join=True,
        )
