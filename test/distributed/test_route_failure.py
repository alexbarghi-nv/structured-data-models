"""Two-rank CPU/Gloo tests for synchronized route failures."""

import datetime
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sdm.models.kumo.tabular.prepared as prepared_module
import sdm.models.kumo.tabular.sharded_recipe as candidate
from sdm import TableTensor
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.prepared import PreparedInputs
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.processing.execution import RecipeExecution


def worker(rank, rendezvous, folder, scenario):

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=12),
    )
    try:
        prepared_module.fit_sharded = candidate.fit_sharded
        if scenario == "fault":
            original = candidate._deterministic_route

            class FailingRoute:
                def to(self, device):
                    raise RuntimeError("injected route transfer failure")

            candidate._deterministic_route = lambda recipe: (
                FailingRoute() if rank == 1 else original(recipe)
            )
        mesh = TableMesh("mesh", mesh_shape=(1, 2))
        x = TableTensor.from_tensor(
            torch.arange(96).reshape(12, 8).float().sin()
        )
        y = torch.arange(12).reshape(12, 1).float().cos()
        query = x[:3]
        prepared = PreparedInputs(mesh, "sharded_cpu", torch.device("cpu"), 3)
        if scenario == "fault":
            try:
                prepared.fit(
                    x if rank == 0 else None,
                    y if rank == 0 else None,
                    default_recipe(),
                    2,
                    torch.Generator().manual_seed(37),
                )
            except RuntimeError as error:
                text = str(error)
            else:
                raise AssertionError("injected failure was swallowed")
            assert (
                "rank 1: RuntimeError: injected route transfer failure" in text
            ), text
            # Both workers must still be able to enter the next collective.
            completed = torch.tensor([1])
            dist.all_reduce(completed)
            assert completed.item() == 2
            result = {
                "rank": rank,
                "error": text,
                "post_failure_collective": 2,
            }
        else:
            actual = prepared.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                default_recipe(),
                2,
                torch.Generator().manual_seed(37),
            )
            queries = prepared.transform(query if rank == 0 else None)
            reference = RecipeExecution(default_recipe())
            expected = reference.fit_transform(
                x,
                TableTensor.from_tensor(y),
                None,
                num_members=2,
                generator=torch.Generator().manual_seed(37),
            )
            expected_query = reference.transform(query, None)
            assert len(actual) == len(queries) == len(expected) == 2
            start, end = mesh.rows(x.size(-2))
            for member, observed_query, native, native_query in zip(
                actual, queries, expected, expected_query
            ):
                expected_context = native.x.numerical[
                    ..., start:end, :
                ].index_select(-1, member.feature_indices)
                expected_query_values = native_query.x.numerical.index_select(
                    -1, member.feature_indices
                )
                torch.testing.assert_close(
                    member.context.x.numerical,
                    expected_context,
                    rtol=1e-5,
                    atol=1e-6,
                    equal_nan=True,
                )
                torch.testing.assert_close(
                    observed_query.x.numerical,
                    expected_query_values,
                    rtol=1e-5,
                    atol=1e-6,
                    equal_nan=True,
                )
            result = {
                "rank": rank,
                "contexts": len(actual),
                "queries": len(queries),
            }
        Path(folder, f"{scenario}-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["fault", "success"])
def test_two_rank_route_failure_and_success(scenario):
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(
            worker,
            args=(str(Path(folder, "rendezvous")), folder, scenario),
            nprocs=2,
            join=True,
        )
        rows = [
            json.loads(Path(folder, f"{scenario}-{rank}.json").read_text())
            for rank in range(2)
        ]
        if scenario == "fault":
            assert rows[0]["error"] == rows[1]["error"]
            assert all(row["post_failure_collective"] == 2 for row in rows)
        else:
            assert all(row["contexts"] == row["queries"] == 2 for row in rows)
