# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model equivalence, unique cache ownership and public fit/predict APIs."""

import os
from typing import Literal, cast

import pytest
import torch
import torch.distributed as dist

from sdm import TableTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.distributed import (
    ContextParallelKumoTabular,
    _ContextParallelKumoTabular,
)
from sdm.models.kumo.tabular.model import (
    KumoTabular,
    _KumoTabular,
)
from test.distributed._kumo import initialize_parameters, tiny_model_config
from test.distributed._runtime import distributed_session, spawn_workers


class _CheckpointKumo(ContextParallelKumoTabular):
    @torch.no_grad()
    def _load_from_pretrained(
        self,
        size: Literal["small", "medium", "large"],
        device: torch.device | str | None,
    ) -> KumoTabular:
        model = cast(_KumoTabular, self.models["regression"])
        model.to_empty(device=device)
        for parameter in model.parameters():
            parameter.copy_(
                torch.arange(parameter.numel(), device=device)
                .reshape(parameter.shape)
                .add(dist.get_rank())
            )
            if parameter.ndim == 2:
                parameter.data = parameter.T.contiguous().T
        model.register_buffer(
            "broadcast_probe",
            torch.arange(6, device=device)
            .reshape(2, 3)
            .T.add(dist.get_rank()),
        )
        return self


def _assert_context_cache_shards(
    mesh, cache, reference_cache, context_rows, world
):
    """Each context row has one ICL owner and the correct cached K/V."""
    start, end = mesh.context(context_rows)
    row_start, row_end = mesh.rows(context_rows)
    assert row_start <= start <= end <= row_end
    if world == 8:
        intervals: list[tuple[int, int] | None] = [None] * world
        # Test-only census of ownership, outside the model transport path.
        dist.all_gather_object(intervals, (start, end))
        ownership = torch.zeros(context_rows, dtype=torch.int64)
        for interval in intervals:
            assert interval is not None
            ownership[interval[0] : interval[1]] += 1
        torch.testing.assert_close(ownership, torch.ones_like(ownership))
    for index in range(3):
        local = cache[f"icl_block.layer{index}"]
        reference = reference_cache[f"icl_block.layer{index}"]
        assert isinstance(local, KVCacheEntry)
        assert isinstance(reference, KVCacheEntry)
        assert local.key.size(-3) == end - start
        assert local.key.size(-2) == 2
        torch.testing.assert_close(
            local.key,
            reference.key[..., start:end, :, :],
            atol=2e-5,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            local.value,
            reference.value[..., start:end, :, :],
            atol=2e-5,
            rtol=2e-4,
        )
    assert cache.is_replaying


def _model_inputs(device, context_rows, features, classes):
    """Use batched, mixed-type features with missing context/query values."""
    context = (
        torch.arange(2 * context_rows * features, device=device)
        .reshape(2, context_rows, features)
        .float()
        .sin()
    )
    context[:, 2, 0] = torch.nan
    query = (
        torch.arange(2 * 3 * features, device=device)
        .reshape(2, 3, features)
        .float()
        .cos()
    )
    query[:, 1, 0] = torch.nan
    y = (
        torch.arange(2 * context_rows, device=device).reshape(2, context_rows)
        % 3
    )
    if classes == 0:
        y = y.float()
    categorical_mask = (torch.arange(features, device=device) % 2 == 1).expand(
        2, -1
    )
    return context, query, y, categorical_mask


def _check_model_cache_case(
    device, world, mesh, classes, features, context_rows
):
    """Compare predictions and rank-local cached context ownership."""
    model = _KumoTabular(
        num_classes=classes,
        num_quantiles=7 if classes == 0 else 0,
        cell_channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=3,
        group_size=3,
        num_frequencies=2,
        num_readout_tokens=2,
        icl_channels=16,
        num_icl_layers=3,
        num_icl_heads=4,
        num_icl_key_value_heads_for_query=2,
        device=device,
    ).eval()
    with torch.inference_mode():
        initialize_parameters(model, device)
        distributed = _ContextParallelKumoTabular(
            model,
            mesh,
            attention_options={"projection_batch_size": 2}
            if world == 8
            else None,
        ).eval()
        context, query, y, categorical_mask = _model_inputs(
            device, context_rows, features, classes
        )
        reference_cache, cache = Cache(), Cache()
        model(context.clone(), y, categorical_mask, cache=reference_cache)
        distributed(context, y, categorical_mask, cache=cache)
        reference_cache.freeze()
        cache.freeze()
        empty_y = y[..., :0]
        expected = model(
            query.clone(),
            empty_y,
            categorical_mask,
            cache=reference_cache,
        )
        actual = distributed(query, empty_y, categorical_mask, cache=cache)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        # Repeated prediction must leave the fitted shard unchanged.
        repeat = distributed(
            query[:, :1], empty_y, categorical_mask, cache=cache
        )
        torch.testing.assert_close(
            repeat, expected[:, :1], atol=2e-5, rtol=2e-4
        )
        if world == 8:
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        _assert_context_cache_shards(
            mesh, cache, reference_cache, context_rows, world
        )


def _worker(
    rank: int,
    world: int,
    rendezvous: str,
    cuda: bool,
    mesh_shape: tuple[int, int] | None = None,
) -> None:
    with distributed_session(
        rank,
        world,
        rendezvous,
        backend="nccl" if cuda else "gloo",
        timeout_seconds=90,
    ) as device:
        with pytest.raises(ValueError, match="parent group"):
            TableMesh("mesh", mesh_shape=(1, world + 1))
        mesh = TableMesh(
            "hybrid" if world == 2 else "mesh", mesh_shape=mesh_shape
        )
        cases = [(0, 5, 9), (3, 5, 9), (0, 1, 9)]
        if world == 8:
            # Fewer rows than mesh rows gives empty embedding and KV owners.
            cases.extend([(0, 5, 3), (3, 1, 3)])
        for classes, features, context_rows in cases:
            _check_model_cache_case(
                device, world, mesh, classes, features, context_rows
            )
        _public_lifecycle(device, world, mesh_shape)


def _check_checkpoint_broadcast(device, world, mesh_shape):
    """Checkpoint loading broadcasts noncontiguous weights and buffers."""
    checkpoint = _CheckpointKumo(
        task="regression",
        size="small",
        pretrained=True,
        device=device,
        layout="hybrid" if world == 2 else "mesh",
        mesh_shape=mesh_shape,
    )
    underlying = checkpoint.models["regression"].model
    weight = underlying.row_embedding.cell_embedding.num_lin.weight
    assert not weight.is_contiguous()
    assert not weight.requires_grad
    torch.testing.assert_close(
        weight,
        torch.arange(weight.numel(), device=device)
        .reshape(weight.shape)
        .to(weight.dtype),
    )
    assert not underlying.broadcast_probe.is_contiguous()
    torch.testing.assert_close(
        underlying.broadcast_probe,
        torch.arange(6, device=device).reshape(2, 3).T,
    )


def _check_public_refit(reference, model, x, y, query, context_size, device):
    """Refitting replaces prior fitted state and uses source-rank sampling."""
    generator = torch.Generator(device=device).manual_seed(37)
    reference.fit(
        x[:context_size].to(device),
        y[:context_size].to(device),
        num_estimators=2,
        estimator_batch_size=2,
        generator=generator,
    )
    model.fit(
        x[:context_size] if dist.get_rank() == 0 else None,
        y[:context_size] if dist.get_rank() == 0 else None,
        num_estimators=2,
        estimator_batch_size=2,
        # Deliberately disagree off-source: only the source may sample.
        generator=torch.Generator(device=device).manual_seed(
            37 + dist.get_rank()
        ),
    )
    expected = reference.predict(query.to(device))
    actual = model.predict(query if dist.get_rank() == 0 else None)
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=2e-5, rtol=2e-4
    )
    assert model._cache is not None
    assert model._cache.is_replaying


def _public_lifecycle(
    device: torch.device,
    world: int,
    mesh_shape: tuple[int, int] | None = None,
) -> None:
    with tiny_model_config():
        for task in ["regression", "classification"]:
            reference = KumoTabular(
                task=task, size="small", pretrained=False, device=device
            )
            model = ContextParallelKumoTabular(
                task=task,
                size="small",
                pretrained=False,
                device=device,
                layout="hybrid" if world == 2 else "mesh",
                mesh_shape=mesh_shape,
            )
            expected_shape = mesh_shape or (world // 2, 2)
            assert model.distributed_layout["mesh_shape"] == expected_shape
            with torch.no_grad():
                initialize_parameters(reference, device)
            model.models[task].model.load_state_dict(
                reference.models[task].state_dict()
            )
            x = TableTensor.from_tensor(
                torch.arange(18 * 5).reshape(18, 5).float().sin()
            )
            query = TableTensor.from_tensor(
                torch.arange(3 * 5).reshape(3, 5).float().cos()
            )
            targets = torch.arange(18).reshape(18, 1)
            y = TableTensor.from_tensor(
                targets % 12
                if task == "classification"
                else targets.float().sin()
            )
            with pytest.raises(ValueError, match="does not support 'offload'"):
                model.fit(x, y, offload="cpu")
            with pytest.raises(RuntimeError, match=r"fit.*predict"):
                model(x, y, query)
            for context_size in [18, 13]:
                _check_public_refit(
                    reference, model, x, y, query, context_size, device
                )
        _check_checkpoint_broadcast(device, world, mesh_shape)


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_kumo(world: int) -> None:
    """Distributed models match native predictions and public lifecycles."""
    spawn_workers(_worker, world, False)


@pytest.mark.parametrize(
    "mesh_shape",
    [None, (2, 4), (8, 1), (1, 8)],
    ids=["default-4x2", "2x4", "8x1", "1x8"],
)
def test_eight_rank_mesh(mesh_shape: tuple[int, int] | None) -> None:
    """Every mesh layout preserves predictions and unique cache ownership."""
    spawn_workers(_worker, 8, False, mesh_shape)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices",
)
def test_distributed_kumo_cuda() -> None:
    """Run model and lifecycle equivalence on real CUDA/NCCL ranks."""
    world = int(os.environ.get("SDM_TEST_CUDA_WORLD_SIZE", "2"))
    spawn_workers(_worker, world, True)
