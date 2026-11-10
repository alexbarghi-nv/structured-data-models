# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular import ColumnShard, DistributedKumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS, KumoTabular
from sdm.models.kumo.tabular.prepared import PreparedInputs
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.processing.execution import RecipeExecution

_TINY = {
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


def _data() -> tuple[TableTensor, TableTensor, torch.Tensor]:
    values = torch.arange(61 * 5).reshape(61, 5).float().mul(0.13).sin()
    values[:, 1] = 1
    values[2, 3] = torch.nan
    table = TableTensor(
        columns={
            "numerical": ["n0", "a__count", "n2", "n3", "n4"],
            "categorical": ["a", "b"],
        },
        numerical=values,
        categorical=CategoricalTensor(
            torch.stack((torch.arange(61) % 51, torch.arange(61) % 2), -1),
            categories=[torch.arange(51), torch.tensor([100, 200])],
        ),
    )
    query = TableTensor(
        columns={str(k): v for k, v in table.columns.items()},
        numerical=torch.arange(15).reshape(3, 5).float().cos(),
        categorical=CategoricalTensor(
            torch.tensor([[0, 0], [1, 1], [2, 0]]),
            categories=[torch.tensor([20, 0, 100]), torch.tensor([200, 100])],
        ),
    )
    query.numerical[1, 3] = torch.nan
    return table, query, torch.arange(61).reshape(-1, 1).float().sin()


def _ordered_columns(table: TableTensor, names: list[str]) -> TableTensor:
    numerical = [
        name for name in names if name in table.columns[Stype.numerical]
    ]
    categorical = [
        name for name in names if name in table.columns[Stype.categorical]
    ]
    num_indices = [
        table.columns[Stype.numerical].index(name) for name in numerical
    ]
    cat_indices = [
        table.columns[Stype.categorical].index(name) for name in categorical
    ]
    return TableTensor(
        columns={"numerical": numerical, "categorical": categorical},
        numerical=table.numerical.index_select(
            -1, torch.tensor(num_indices, dtype=torch.long)
        ),
        categorical=CategoricalTensor(
            table.categorical.code.index_select(
                -1, torch.tensor(cat_indices, dtype=torch.long)
            ),
            categories=[
                table.categorical.categories[index] for index in cat_indices
            ],
        ),
    )


def _indices(width: int, rank: int) -> torch.Tensor:
    start, end = TableMesh.bounds(width, rank % 2, 2)
    return (
        ((torch.arange(start, end)[:, None] + torch.tensor([0, 1, 3])) % width)
        .flatten()
        .unique()
    )


def _write_fixture(directory: Path, world: int) -> None:
    x, query, y = _data()
    columns = dict(x.columns)
    names = [name for stype in Stype for name in columns[stype]]
    execution = RecipeExecution(default_recipe())
    generator = torch.Generator().manual_seed(37)
    contexts = execution.fit_transform(
        x, y, None, num_members=8, generator=generator
    )
    queries = execution.transform(query, None)
    for rank in range(world):
        # Four-rank fixtures deliberately leave the source's feature shard
        # empty. Other workers own uneven, noncontiguous complete columns.
        ids = [
            index
            for index in range(len(names))
            if (
                index % world == rank
                if world == 2
                else index % (world - 1) + 1 == rank
            )
        ]
        ids = [
            *reversed([index for index in ids if index < 5]),
            *reversed([index for index in ids if index >= 5]),
        ]
        local_names = [names[index] for index in ids]
        context = ColumnShard(
            cast(TableTensor, _ordered_columns(x, local_names).clone()),
            tuple(ids),
            columns,
            torch.arange(61, dtype=torch.int64) + 1000,
        )
        query_shard = ColumnShard(
            cast(TableTensor, _ordered_columns(query, local_names).clone()),
            tuple(ids),
            columns,
            torch.arange(3, dtype=torch.int64) + 5000,
        )
        torch.save(
            {"context": context, "query": query_shard, "targets": y},
            directory / f"input.rank{rank}.pt",
        )
        start, end = TableMesh.bounds(61, rank // 2, world // 2)
        expected = []
        for original, test in zip(contexts, queries, strict=True):
            indices = _indices(original.x.size(-1), rank)
            mean = original.x.numerical.nanmean(-2, keepdim=True)
            expected.append(
                {
                    "x": original.x.numerical[start:end].index_select(
                        -1, indices
                    ),
                    "y": original.y.numerical[start:end],
                    "mean": mean.masked_fill(mean.isnan(), 0).index_select(
                        -1, indices
                    ),
                    "query": test.x.numerical.index_select(-1, indices),
                }
            )
        torch.save(
            {"members": expected, "generator": generator.get_state()},
            directory / f"expected.rank{rank}.pt",
        )
    original_kwargs = MODEL_KWARGS["small"]
    MODEL_KWARGS["small"] = _TINY
    try:
        for task in ("regression", "classification"):
            targets = (
                y
                if task == "regression"
                else torch.arange(61).reshape(-1, 1) % 12
            )
            reference = KumoTabular(task=task, size="small", pretrained=False)
            generator = torch.Generator().manual_seed(37)
            reference.fit(
                x,
                targets,
                recipe=default_recipe(),
                num_estimators=8,
                generator=generator,
            )
            torch.save(
                {
                    "weights": reference.models[task].state_dict(),
                    "output": reference.predict(query),
                },
                directory / f"model.{task}.pt",
            )
    finally:
        MODEL_KWARGS["small"] = original_kwargs


def _parity(rank: int, mesh: TableMesh, directory: Path) -> None:
    # No worker opens another worker's raw input file or a full input table.
    data = torch.load(directory / f"input.rank{rank}.pt", weights_only=False)
    expected = torch.load(
        directory / f"expected.rank{rank}.pt", weights_only=False
    )
    context, query, targets = data["context"], data["query"], data["targets"]
    for mode in ("sharded_cpu", "sharded_cuda"):
        prepared = PreparedInputs(mesh, mode, torch.device("cpu"), 3)
        generator = torch.Generator().manual_seed(37)
        members = prepared.fit(
            context, targets, default_recipe(), 8, generator
        )
        if rank == 0:
            torch.testing.assert_close(
                generator.get_state(), expected["generator"]
            )
        for actual, original in zip(members, expected["members"], strict=True):
            torch.testing.assert_close(
                actual.context.x.numerical,
                original["x"],
                equal_nan=True,
                atol=2e-6,
                rtol=2e-6,
            )
            torch.testing.assert_close(
                actual.context.y.numerical, original["y"]
            )
            torch.testing.assert_close(
                actual.mean, original["mean"], atol=2e-6, rtol=2e-6
            )
        for _ in range(2):
            for actual, original in zip(
                prepared.transform(query), expected["members"], strict=True
            ):
                torch.testing.assert_close(
                    actual.x.numerical,
                    original["query"],
                    equal_nan=True,
                    atol=2e-6,
                    rtol=2e-6,
                )
        # Labels may live only on an empty-feature source worker.
        prepared.fit(
            context,
            targets if rank == 0 else None,
            default_recipe(),
            8,
            torch.Generator().manual_seed(37),
        )
    original_kwargs = MODEL_KWARGS["small"]
    MODEL_KWARGS["small"] = _TINY
    try:
        for task in ("regression", "classification"):
            reference = torch.load(
                directory / f"model.{task}.pt", weights_only=False
            )
            targets = (
                data["targets"]
                if task == "regression"
                else torch.arange(61).reshape(-1, 1) % 12
            )
            for mode in ("sharded_cpu", "sharded_cuda"):
                model = DistributedKumoTabular(
                    task=task,
                    size="small",
                    pretrained=False,
                    layout="mesh",
                    preprocessing=mode,
                )
                model.models[task].model.load_state_dict(reference["weights"])
                model.fit(
                    context,
                    targets if rank == 0 else None,
                    recipe=default_recipe(),
                    num_estimators=8,
                    generator=torch.Generator().manual_seed(37),
                )
                actual = model.predict(query)
                assert actual.columns == reference["output"].columns
                torch.testing.assert_close(
                    actual.numerical,
                    reference["output"].numerical,
                    atol=2e-5,
                    rtol=2e-4,
                )
                small = replace(
                    query, table=query.table[:1], row_ids=query.row_ids[:1]
                )
                torch.testing.assert_close(
                    model.predict(small).numerical,
                    reference["output"].numerical[:1],
                    atol=2e-5,
                    rtol=2e-4,
                )
    finally:
        MODEL_KWARGS["small"] = original_kwargs


def _errors(rank: int, mesh: TableMesh, directory: Path) -> None:
    data = torch.load(directory / f"input.rank{rank}.pt", weights_only=False)
    shard, targets = data["context"], data["targets"]
    for failure, match in (
        ("rows", "row identities/cohort"),
        ("schema", "global schema"),
        ("targets", "targets differ"),
        ("ownership", "exactly once"),
        ("count_name", "count names collide"),
        ("duplicates", "unique"),
        ("missing", "Every worker"),
    ):
        local, y = shard, targets
        if failure == "count_name":
            columns = dict(shard.columns)
            columns[Stype.numerical] = ("n0", "n1", "n2", "n3", "n4")
            columns[Stype.categorical] = ("a", "a__count")
            local_columns = dict(shard.table.columns)
            local_columns[Stype.numerical] = tuple(
                "n1" if name == "a__count" else name
                for name in local_columns[Stype.numerical]
            )
            categories = list(local_columns[Stype.categorical])
            codes = shard.table.categorical.code.clone()
            if "b" in categories:
                codes[..., categories.index("b")] = 0
                categories[categories.index("b")] = "a__count"
            local_columns[Stype.categorical] = tuple(categories)
            table = TableTensor(
                columns=local_columns,
                numerical=shard.table.numerical,
                categorical=CategoricalTensor(
                    codes, categories=shard.table.categorical.categories
                ),
            )
            local = replace(shard, table=table, columns=columns)
        elif rank == mesh.world_size - 1:
            if failure == "rows":
                local = replace(shard, row_ids=shard.row_ids.flip(-1))
            elif failure == "schema":
                columns = dict(shard.columns)
                numerical = list(columns[Stype.numerical])
                unused = next(
                    index
                    for index in range(5)
                    if index not in shard.column_ids
                )
                numerical[unused] += "_wrong"
                columns[Stype.numerical] = tuple(numerical)
                local = replace(shard, columns=columns)
            elif failure == "ownership":
                duplicate = next(
                    index
                    for index in range(5)
                    if index not in shard.column_ids
                )
                ids = list(shard.column_ids)
                ids[0] = duplicate
                columns = dict(shard.table.columns)
                numerical = list(columns[Stype.numerical])
                numerical[0] = shard.columns[Stype.numerical][duplicate]
                columns[Stype.numerical] = tuple(numerical)
                table = TableTensor(
                    columns=columns,
                    numerical=shard.table.numerical,
                    categorical=shard.table.categorical,
                )
                local = replace(shard, table=table, column_ids=tuple(ids))
            elif failure == "targets":
                y = targets + 1
            elif failure == "duplicates":
                ids = shard.row_ids.clone()
                ids[0] = ids[1]
                local = replace(shard, row_ids=ids)
            elif failure == "missing":
                local = None
        prepared = PreparedInputs(mesh, "sharded_cpu", torch.device("cpu"), 3)
        with pytest.raises(RuntimeError, match=match):
            prepared.fit(local, y, default_recipe(), 1, torch.Generator())
    prepared = PreparedInputs(mesh, "sharded_cpu", torch.device("cpu"), 3)
    prepared.fit(shard, targets, default_recipe(), 1, torch.Generator())
    query = data["query"]
    if rank == mesh.world_size - 1:
        query = replace(query, row_ids=query.row_ids.flip(-1))
    with pytest.raises(RuntimeError, match="row identities/cohort"):
        prepared.transform(query)
    query = data["query"]
    if rank == 1:
        numeric = len(query.table.columns[Stype.numerical])
        ids = (
            *reversed(query.column_ids[:numeric]),
            *query.column_ids[numeric:],
        )
        global_names = [
            name for stype in Stype for name in query.columns[stype]
        ]
        table = _ordered_columns(
            query.table, [global_names[index] for index in ids]
        )
        query = replace(query, table=table, column_ids=ids)
    with pytest.raises(RuntimeError, match="column ownership must match fit"):
        prepared.transform(query)
    with pytest.raises(RuntimeError, match="same source-input or ColumnShard"):
        prepared.transform(None)
    assert prepared.transform(data["query"])[0].x.size(-2) == 3


def _worker(rank: int, world: int, rendezvous: str, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"file://{rendezvous}"
    )
    try:
        mesh = TableMesh("mesh")
        _parity(rank, mesh, Path(directory))
        _errors(rank, mesh, Path(directory))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_prepartitioned_column_files(world: int, tmp_path: Path) -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _write_fixture(tmp_path, world)
    finally:
        torch.set_num_threads(threads)
    with tempfile.TemporaryDirectory() as rendezvous:
        mp.spawn(
            _worker,
            args=(world, str(Path(rendezvous) / "store"), str(tmp_path)),
            nprocs=world,
        )


def _cuda_reference_inputs(
    rank: int,
    mesh: TableMesh,
    device: torch.device,
) -> tuple[PreparedInputs, list[Any], list[Any], torch.Tensor]:
    # The reference alone loads the complete source-host table. Its local
    # raw variables are released before workers load direct shard files.
    x, query, y = _data() if rank == 0 else (None, None, None)
    prepared = PreparedInputs(mesh, "rank0_cpu", device, 3)
    generator = torch.Generator(device=device).manual_seed(37)
    contexts = prepared.fit(x, y, default_recipe(), 8, generator)
    queries = prepared.transform(query)
    return prepared, contexts, queries, generator.get_state()


def _cuda_reference_model(
    rank: int,
    task: str,
    device: torch.device,
    weights: dict[str, Any],
) -> TableTensor:
    x, query, y = _data() if rank == 0 else (None, None, None)
    if rank == 0 and task == "classification":
        y = torch.arange(61).reshape(-1, 1) % 12
    model = DistributedKumoTabular(
        task=task,
        size="small",
        pretrained=False,
        device=device,
        layout="mesh",
        preprocessing="rank0_cpu",
    )
    model.models[task].model.load_state_dict(weights)
    model.fit(
        x,
        y,
        recipe=default_recipe(),
        num_estimators=8,
        generator=torch.Generator(device=device).manual_seed(37),
    )
    return model.predict(query)


def _cuda_worker(
    rank: int, world: int, rendezvous: str, directory: str
) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, init_method=f"file://{rendezvous}"
    )
    original_kwargs = MODEL_KWARGS["small"]
    MODEL_KWARGS["small"] = _TINY
    try:
        mesh = TableMesh("mesh")
        _, contexts, queries, state = _cuda_reference_inputs(
            rank, mesh, device
        )
        data = torch.load(
            Path(directory) / f"input.rank{rank}.pt", weights_only=False
        )
        context, query, y = data["context"], data["query"], data["targets"]
        for mode in ("sharded_cpu", "sharded_cuda"):
            prepared = PreparedInputs(mesh, mode, device, 3)
            generator = torch.Generator(device=device).manual_seed(37)
            actual = prepared.fit(context, y, default_recipe(), 8, generator)
            if rank == 0:
                torch.testing.assert_close(generator.get_state(), state)
            for member, reference in zip(actual, contexts, strict=True):
                torch.testing.assert_close(
                    member.context.x.numerical,
                    reference.context.x.numerical,
                    equal_nan=True,
                    atol=2e-5,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    member.mean, reference.mean, atol=2e-5, rtol=2e-5
                )
            for member, reference in zip(
                prepared.transform(query), queries, strict=True
            ):
                torch.testing.assert_close(
                    member.x.numerical,
                    reference.x.numerical,
                    equal_nan=True,
                    atol=2e-5,
                    rtol=2e-5,
                )
        for task in ("regression", "classification"):
            weights = torch.load(
                Path(directory) / f"model.{task}.pt", weights_only=False
            )["weights"]
            expected = _cuda_reference_model(rank, task, device, weights)
            targets = (
                y
                if task == "regression"
                else torch.arange(61).reshape(-1, 1) % 12
            )
            for mode in ("sharded_cpu", "sharded_cuda"):
                model = DistributedKumoTabular(
                    task=task,
                    size="small",
                    pretrained=False,
                    device=device,
                    layout="mesh",
                    preprocessing=mode,
                )
                model.models[task].model.load_state_dict(weights)
                model.fit(
                    context,
                    targets if rank == 0 else None,
                    recipe=default_recipe(),
                    num_estimators=8,
                    generator=torch.Generator(device=device).manual_seed(37),
                )
                actual = model.predict(query)
                assert actual.columns == expected.columns
                torch.testing.assert_close(
                    actual.numerical, expected.numerical, atol=2e-5, rtol=2e-4
                )
                small = replace(
                    query, table=query.table[:1], row_ids=query.row_ids[:1]
                )
                torch.testing.assert_close(
                    model.predict(small).numerical,
                    expected.numerical[:1],
                    atol=2e-5,
                    rtol=2e-4,
                )
    finally:
        MODEL_KWARGS["small"] = original_kwargs
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices",
)
def test_prepartitioned_column_files_cuda(tmp_path: Path) -> None:
    world = int(os.environ.get("SDM_TEST_CUDA_WORLD_SIZE", "2"))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _write_fixture(tmp_path, world)
    finally:
        torch.set_num_threads(threads)
    with tempfile.TemporaryDirectory() as rendezvous:
        mp.spawn(
            _cuda_worker,
            args=(world, str(Path(rendezvous) / "store"), str(tmp_path)),
            nprocs=world,
        )
