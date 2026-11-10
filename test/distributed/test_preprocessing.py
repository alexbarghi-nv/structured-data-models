# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ProcessGroup

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, TableTensor
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.distributed import DistributedKumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS, KumoTabular
from sdm.models.kumo.tabular.prepared import PreparedInputs, Preprocessing
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.processing.execution import RecipeExecution

_MODES: tuple[Preprocessing, ...] = (
    "rank0_cpu",
    "sharded_cpu",
    "sharded_cuda",
)


def _cases() -> list[
    tuple[TableTensor, TableTensor, torch.Tensor, int | None, int]
]:
    numerical = torch.arange(67 * 7).reshape(67, 7).float().sin()
    numerical[:, 1] = 1
    numerical[2, 3] = torch.nan
    numerical[3, 4] = torch.inf
    categorical = torch.stack(
        (torch.arange(67) % 60, torch.arange(67) % 3), -1
    )
    mixed = TableTensor(
        columns={
            "numerical": [f"n{i}" for i in range(7)],
            "categorical": ["cat", "smallcat"],
        },
        numerical=numerical,
        categorical=CategoricalTensor(
            categorical,
            categories=[torch.arange(60), torch.arange(3)],
        ),
    )
    categories = TableTensor(
        columns={"categorical": ["a"]},
        categorical=CategoricalTensor(
            torch.tensor([[0], [1], [2]]),
            categories=[torch.tensor([10, 20, 30])],
        ),
    )
    unseen = TableTensor(
        columns={"categorical": ["a"]},
        categorical=CategoricalTensor(
            torch.tensor([[0], [1]]),
            categories=[torch.tensor([20, 90])],
        ),
    )
    varied = torch.arange(8 * 7 * 8).reshape(8, 7, 8).float().sin()
    for member in range(8):
        varied[member, :, :member] = 0
    varied = TableTensor.from_tensor(varied)
    wide = TableTensor.from_tensor(
        torch.arange(9 * 517).reshape(9, 517).float().mul(0.13).sin()
    )
    return [
        (
            mixed,
            mixed[:3],
            torch.arange(67).reshape(-1, 1).float().sin(),
            8,
            8,
        ),
        (categories, unseen, torch.arange(3).reshape(-1, 1), 8, 8),
        (
            varied,
            varied[:, :2],
            torch.arange(56).reshape(8, 7, 1).float().sin(),
            None,
            8,
        ),
        (wide, wide[:2], torch.arange(9).reshape(-1, 1).float().sin(), 2, 500),
    ]


def _preprocessing_parity(
    rank: int,
    mesh: TableMesh,
    device: torch.device | None = None,
) -> None:
    device = torch.device("cpu") if device is None else device
    tolerance = 2e-6 if device.type == "cpu" else 2e-5
    for x, query, y, estimators, max_columns in _cases():
        reference = RecipeExecution(default_recipe())
        reference_generator = torch.Generator(device=device).manual_seed(37)
        expected = reference.fit_transform(
            x.to(device),
            y.to(device),
            None,
            num_members=estimators,
            generator=reference_generator,
        )
        expected_queries = reference.transform(query.to(device), None)
        for mode in _MODES:
            prepared = PreparedInputs(mesh, mode, device, 3)
            generator = torch.Generator(device=device).manual_seed(37)
            actual = prepared.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                default_recipe(),
                estimators,
                generator,
            )
            if rank == 0:
                torch.testing.assert_close(
                    generator.get_state(), reference_generator.get_state()
                )
            start, end = mesh.rows(x.size(-2))
            for member, original in zip(actual, expected, strict=True):
                if max_columns == 500:
                    assert member.feature_count == max_columns
                selected = (
                    original.x.numerical[..., start:end, :]
                    .index_select(
                        -1,
                        member.feature_indices.to(device),
                    )
                    .cpu()
                )
                torch.testing.assert_close(
                    member.context.x.numerical,
                    selected,
                    equal_nan=True,
                    atol=tolerance,
                    rtol=tolerance,
                )
                torch.testing.assert_close(
                    member.context.y.numerical,
                    original.y.numerical[..., start:end, :].cpu(),
                )
                torch.testing.assert_close(
                    member.context.y.categorical.code,
                    original.y.categorical.code[..., start:end, :].cpu(),
                )
            for _ in range(2):
                queries = prepared.transform(query if rank == 0 else None)
                for member, original, context in zip(
                    queries,
                    expected_queries,
                    actual,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        member.x.numerical,
                        original.x.numerical.index_select(
                            -1, context.feature_indices.to(device)
                        ).cpu(),
                        equal_nan=True,
                        atol=tolerance,
                        rtol=tolerance,
                    )


def _lifecycle(
    rank: int,
    world: int,
    device: torch.device | None = None,
) -> None:
    device = torch.device("cpu") if device is None else device
    original = MODEL_KWARGS["small"]
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
    try:
        x = TableTensor.from_tensor(
            torch.arange(19 * 17).reshape(19, 17).float().sin()
        )
        query = TableTensor.from_tensor(
            torch.arange(51).reshape(3, 17).float().cos()
        )
        x.numerical[2, 0] = torch.nan
        query.numerical[1, 0] = torch.nan
        for task in ("regression", "classification"):
            reference = KumoTabular(
                task=task,
                size="small",
                pretrained=False,
                device=device,
            )
            with torch.no_grad():
                for index, parameter in enumerate(reference.parameters()):
                    parameter.copy_(
                        torch.arange(parameter.numel(), device=device)
                        .reshape(parameter.shape)
                        .mul(0.13)
                        .add(index * 0.2)
                        .sin()
                        .mul(0.2),
                    )
            targets = torch.arange(19).reshape(-1, 1)
            y = (
                targets % 12
                if task == "classification"
                else targets.float().sin()
            )
            reference.fit(
                x.to(device),
                y.to(device),
                num_estimators=8,
                estimator_batch_size=2,
                generator=torch.Generator(device=device).manual_seed(37),
            )
            expected = reference.predict(query.to(device))
            for mode in _MODES:
                model = DistributedKumoTabular(
                    task=task,
                    size="small",
                    pretrained=False,
                    device=device,
                    layout="mesh",
                    preprocessing=mode,
                )
                with pytest.raises(ValueError, match="context-only"):
                    model.fit(
                        x if rank == 0 else None,
                        y if rank == 0 else None,
                        context_only_columns=["x0"],
                    )
                model.models[task].model.load_state_dict(
                    reference.models[task].state_dict()
                )
                for _ in range(2):
                    model.fit(
                        x if rank == 0 else None,
                        y if rank == 0 else None,
                        num_estimators=8,
                        estimator_batch_size=2,
                        generator=torch.Generator(device=device).manual_seed(
                            37
                        ),
                    )
                    actual = model.predict(query if rank == 0 else None)
                    assert actual.columns == expected.columns
                    torch.testing.assert_close(
                        actual.numerical,
                        expected.numerical,
                        atol=2e-5,
                        rtol=2e-4,
                    )
                    repeat = model.predict(query[:1] if rank == 0 else None)
                    torch.testing.assert_close(
                        repeat.numerical,
                        expected.numerical[:1],
                        atol=2e-5,
                        rtol=2e-4,
                    )
                model.clear()
                with pytest.raises(RuntimeError, match="Call fit"):
                    model.predict(query if rank == 0 else None)
                with pytest.raises(RuntimeError, match=r"fit.*predict"):
                    model(x, y, query)
            if task == "regression":
                recipe = Recipe(
                    features=sp.Standardize(),
                    output=sp.AverageEstimators(),
                )
                reference.fit(x.to(device), y.to(device), recipe=recipe)
                model = DistributedKumoTabular(
                    task=task,
                    size="small",
                    pretrained=False,
                    device=device,
                    layout="mesh",
                    preprocessing="rank0_cpu",
                )
                model.models[task].model.load_state_dict(
                    reference.models[task].state_dict()
                )
                model.fit(
                    x if rank == 0 else None,
                    y if rank == 0 else None,
                    recipe=recipe,
                )
                actual = model.predict(query if rank == 0 else None)
                expected = reference.predict(query.to(device))
                assert actual.numerical.dtype == torch.float32
                torch.testing.assert_close(
                    actual.numerical,
                    expected.numerical,
                    atol=2e-5,
                    rtol=2e-4,
                )
                reference.fit(
                    x.to(device),
                    y.to(device),
                    num_estimators=1,
                    generator=torch.Generator(device=device).manual_seed(43),
                )
                expected = reference.predict(query.to(device))
    finally:
        MODEL_KWARGS["small"] = original


def _fail_partition(*args: object, **kwargs: object) -> None:
    raise ValueError("partition failed")


def _failure_propagation(rank: int, mesh: TableMesh) -> None:
    x = torch.arange(12).reshape(3, 4).float()
    y = torch.arange(3).reshape(-1, 1).float()
    for mode in ("sharded_cpu", "sharded_cuda"):
        prepared = PreparedInputs(mesh, mode, torch.device("cpu"), 3)
        with pytest.raises(RuntimeError, match="default_recipe"):
            prepared.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                Recipe(features=sp.Standardize()),
                1,
                torch.Generator(),
            )
    custom = PreparedInputs(mesh, "rank0_cpu", torch.device("cpu"), 3)
    contexts = custom.fit(
        x if rank == 0 else None,
        y if rank == 0 else None,
        Recipe(features=sp.Standardize()),
        1,
        torch.Generator(),
    )
    expected = sp.Standardize().fit_transform(TableTensor.from_tensor(x))
    start, end = mesh.rows(x.size(0))
    torch.testing.assert_close(
        contexts[0].context.x.numerical,
        expected.numerical[start:end].index_select(
            -1, contexts[0].feature_indices
        ),
    )
    failing = PreparedInputs(mesh, "rank0_cpu", torch.device("cpu"), 3)
    with pytest.MonkeyPatch.context() as patch:
        if rank == 0:
            patch.setattr(failing, "partition", _fail_partition)
        with pytest.raises(RuntimeError, match="partition failed"):
            failing.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                default_recipe(),
                1,
                torch.Generator(),
            )
    prepared = PreparedInputs(mesh, "rank0_cpu", torch.device("cpu"), 3)
    prepared.fit(
        x if rank == 0 else None,
        y if rank == 0 else None,
        default_recipe(),
        1,
        torch.Generator(),
    )
    with pytest.raises(RuntimeError, match="same schema"):
        prepared.transform(x[:, :2] if rank == 0 else None)


def _subgroup_worker(rank: int, world: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    # Keep references across session teardown to detect stale cached groups.
    prior: list[PreparedInputs] = []
    for session in range(2):
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world,
            init_method=f"file://{rendezvous}-{session}",
        )
        try:
            parents = [
                dist.new_group([0, 2], backend="gloo"),
                dist.new_group([1, 3], backend="gloo"),
            ]
            parent = cast(ProcessGroup, parents[rank % 2])
            mesh = TableMesh("mesh", parent, mesh_shape=(2, 1))
            source = rank % 2
            x = torch.arange(24).reshape(6, 4).float().sin() + source
            y = torch.arange(6).reshape(-1, 1).float()
            for mode in ("rank0_cpu", "sharded_cpu"):
                prepared = PreparedInputs(mesh, mode, torch.device("cpu"), 3)
                members = prepared.fit(
                    x if rank == source else None,
                    y if rank == source else None,
                    default_recipe(),
                    1,
                    torch.Generator().manual_seed(31),
                )
                expected = RecipeExecution(default_recipe()).fit_transform(
                    TableTensor.from_tensor(x),
                    TableTensor.from_tensor(y),
                    None,
                    num_members=1,
                    generator=torch.Generator().manual_seed(31),
                )[0]
                start, end = mesh.rows(len(x))
                torch.testing.assert_close(
                    members[0].context.x.numerical,
                    expected.x.numerical[start:end].index_select(
                        -1, members[0].feature_indices
                    ),
                )
                queries = prepared.transform(x[:2] if rank == source else None)
                assert len(queries) == 1
                prior.append(prepared)
            groups = dist.get_pg_count()
            for _ in range(32):
                current = TableMesh("mesh", parent, mesh_shape=(2, 1))
                value = torch.tensor([rank])
                dist.all_reduce(value, group=current.row_group)
                assert value.item() == (2 if source == 0 else 4)
            assert dist.get_pg_count() == groups
            dist.barrier()
        finally:
            dist.destroy_process_group()


def test_preprocessing_subgroups_and_session_reuse() -> None:
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _subgroup_worker,
            args=(4, str(Path(directory) / "rendezvous")),
            nprocs=4,
        )


def _worker(rank: int, world: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"file://{rendezvous}",
    )
    try:
        mesh = TableMesh("mesh")
        _preprocessing_parity(rank, mesh)
        _lifecycle(rank, world)
        _failure_propagation(rank, mesh)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_preprocessing(world: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(world, str(Path(directory) / "rendezvous")),
            nprocs=world,
        )


def _cuda_worker(rank: int, world: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        init_method=f"file://{rendezvous}",
    )
    try:
        _preprocessing_parity(rank, TableMesh("mesh"), device)
        _lifecycle(rank, world, device)
        if world == 4:
            parents = [
                dist.new_group([0, 2], backend="nccl"),
                dist.new_group([1, 3], backend="nccl"),
            ]
            parent = cast(ProcessGroup, parents[rank % 2])
            mesh = TableMesh("mesh", parent, mesh_shape=(1, 2))
            _preprocessing_parity(mesh.rank, mesh, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices",
)
def test_distributed_preprocessing_cuda() -> None:
    world = int(os.environ.get("SDM_TEST_CUDA_WORLD_SIZE", "2"))
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _cuda_worker,
            args=(world, str(Path(directory) / "rendezvous")),
            nprocs=world,
        )
