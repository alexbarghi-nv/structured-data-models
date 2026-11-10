# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

import sdm.processing as sp
from sdm import Recipe, Stype, TableTensor
from sdm.distributed.layout import TableMesh, _get_group
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.models.kumo.tabular.partition import (
    PreparedContext,
    halo_indices,
)
from sdm.models.kumo.tabular.sampling import sampling_recipe
from sdm.models.kumo.tabular.sharded_recipe import (
    fit_sharded,
    transform_sharded,
)
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)
from sdm.tensor.table import TableSchema

_T = TypeVar("_T")

Preprocessing = Literal[
    "replicated", "rank0_cpu", "sharded_cuda", "sharded_cpu"
]


class PreparedInputs:
    """Fit preprocessing on host/source, then transfer only model-owned shards.

    A shared Gloo group transports CPU tensors; an existing Gloo parent is
    reused directly. Companion groups belong to the distributed session and
    are released by the caller's distributed teardown, not by clearing a fit.
    Model inputs contain only owned rows/features and their required halo,
    which can span tiny feature axes.
    Source preprocessing retains complete context tables on the host.
    """

    def __init__(
        self,
        mesh: TableMesh,
        mode: Preprocessing,
        device: torch.device,
        group_size: int,
    ) -> None:
        self.mesh = mesh
        self.mode = mode
        self.device = device
        self.group_size = group_size
        self.group = cast(
            ProcessGroup,
            _get_group(
                mesh.group,
                dist.get_process_group_ranks(mesh.group),
                backend="gloo",
            ),
        )
        self.execution: RecipeExecution | None = None
        self.members: list[PreparedContext] = []
        self.raw_schema: TableSchema | None = None
        self.raw_columns: Mapping[Stype, tuple[str, ...]] = {}
        self.num_estimators: int | None = None
        self.prepartitioned = False
        self.column_ids: tuple[int, ...] = ()
        self.raw_dtype: torch.dtype | None = None
        self.local_raw_stypes: tuple[Mapping[str, Stype], ...] = ()
        self.local_route: sp.EnsembleProcessor = (
            sp.EnsembleProcessor.as_processor(sp.Identity())
        )
        self.plans: list[tuple[tuple[str, ...], Tensor, tuple[str, ...]]] = []
        self.local_signs: list[Tensor] = []

    @property
    def is_source(self) -> bool:
        """Whether this is the source within the caller's process group."""
        return self.mesh.rank == 0

    def broadcast(self, value: Any, *, source: int | None = None) -> Any:
        """Broadcast host metadata without NCCL tensor staging."""
        objects = [value]
        dist.broadcast_object_list(
            objects,
            src=self.mesh.source if source is None else source,
            group=self.group,
        )
        return objects[0]

    def scatter(
        self, values: list[Any] | None, *, source: int | None = None
    ) -> Any:
        """Transfer one destination's host payload to each rank."""
        output: list[Any] = [None]
        dist.scatter_object_list(
            output,
            values,
            src=self.mesh.source if source is None else source,
            group=self.group,
        )
        return output[0]

    def source_result(
        self, operation: Callable[[], _T], *, source: int | None = None
    ) -> _T:
        """Propagate source failures before peers enter transport."""
        value = None
        error = None
        source = self.mesh.source if source is None else source
        if dist.get_rank() == source:
            try:
                value = operation()
            except Exception as exc:  # noqa: BLE001 - synchronize failures
                error = f"{type(exc).__name__}: {exc}"
        error = self.broadcast(error, source=source)
        if error is not None:
            raise RuntimeError(error)
        return cast(_T, value)

    def local_result(self, operation: Callable[[], _T]) -> _T:
        """Propagate local failures before entering the next collective."""
        result = None
        error = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - synchronize rank failures
            error = f"rank {self.mesh.rank}: {type(exc).__name__}: {exc}"
        errors: list[Any] = [None] * self.mesh.world_size
        dist.all_gather_object(errors, error, group=self.group)
        if any(error is not None for error in errors):
            raise RuntimeError(
                next(error for error in errors if error is not None)
            )
        return cast(_T, result)

    def fit(
        self,
        x: Tensor | TableTensor | ColumnShard | None,
        y: Tensor | TableTensor | None,
        recipe: Recipe,
        num_estimators: int | None,
        generator: torch.Generator,
    ) -> list[PreparedContext]:
        """Fit source inputs and retain only model-owned context shards."""
        if self.mode != "rank0_cpu":
            return fit_sharded(self, x, y, recipe, num_estimators, generator)

        def fit_source() -> tuple[MemberContext, ...]:
            if isinstance(x, ColumnShard):
                raise ValueError(
                    "ColumnShard requires sharded_cpu or sharded_cuda"
                )
            assert x is not None
            assert y is not None
            if x.device.type != "cpu" or y.device.type != "cpu":
                raise ValueError(
                    "Distributed preprocessing expects host inputs "
                    "on the source rank"
                )
            raw = (
                x if isinstance(x, TableTensor) else TableTensor.from_tensor(x)
            )
            self.raw_schema = raw.schema
            self.execution = RecipeExecution(sampling_recipe(recipe))
            contexts = self.execution.fit_transform(
                x=x,
                y=y,
                related_tables=None,
                num_members=num_estimators,
                generator=generator,
            )
            for context in contexts:
                self._validate_features(context)
            return contexts

        contexts = self.source_result(fit_source)
        count = self.broadcast(len(contexts) if self.is_source else None)
        self.members = []
        for member in range(count):

            def partition_source(
                member: int = member,
            ) -> list[PreparedContext]:
                context = contexts[member]
                mean = context.x.numerical.nanmean(dim=-2, keepdim=True)
                mean = mean.masked_fill(mean.isnan(), 0)
                return [
                    self.partition(context, rank, mean=mean)
                    for rank in range(self.mesh.world_size)
                ]

            payloads = self.source_result(partition_source)
            self.members.append(self.scatter(payloads))
        return self.members

    @staticmethod
    def _validate_features(context: MemberContext) -> None:
        if context.x.numerical.size(-1) == 0:
            raise ValueError(
                "Recipe must retain at least one numerical feature"
            )
        if context.x.size(-1) != context.x.numerical.size(-1):
            raise ValueError(
                "Distributed model inputs must be numerical "
                "after preprocessing"
            )

    def partition(
        self,
        context: MemberContext,
        rank: int,
        *,
        mean: Tensor | None = None,
    ) -> PreparedContext:
        """Select owned embedding rows and the circular feature halo."""
        x = context.x.numerical
        features, rows = x.size(-1), x.size(-2)
        indices = halo_indices(features, rank, self.mesh, self.group_size)
        row_rank = rank // self.mesh.feature_parts
        start, end = self.mesh.bounds(rows, row_rank, self.mesh.row_parts)
        columns = [
            context.x.columns[Stype.numerical][i] for i in indices.tolist()
        ]
        mask = torch.tensor(
            [
                context.input_stypes.get(name) == Stype.categorical
                for name in columns
            ],
            dtype=torch.bool,
            device="cpu",
        )
        local_x = TableTensor.from_tensor(
            x[..., start:end, :].index_select(-1, indices),
            columns=columns,
        )
        if mean is None:
            mean = x.nanmean(dim=-2, keepdim=True)
            mean = mean.masked_fill(mean.isnan(), 0)
        mean = mean.index_select(-1, indices)
        return PreparedContext(
            context=MemberContext(
                x=local_x,
                y=cast(TableTensor, context.y[..., start:end, :].clone()),
                x_context_only=None,
                related_tables=None,
                input_stypes=context.input_stypes,
            ),
            categorical_mask=mask,
            context_rows=rows,
            feature_count=features,
            feature_indices=indices,
            mean=mean,
        )

    def transform(
        self, x: Tensor | TableTensor | ColumnShard | None
    ) -> list[MemberQuery]:
        """Transform queries using the fitted context state."""
        if self.mode != "rank0_cpu":
            return transform_sharded(self, x)

        def transform_source() -> tuple[MemberQuery, ...]:
            if isinstance(x, ColumnShard):
                raise ValueError(
                    "ColumnShard requires sharded_cpu or sharded_cuda"
                )
            assert self.execution is not None
            assert x is not None
            if x.device.type != "cpu":
                raise ValueError(
                    "Distributed preprocessing expects host query inputs"
                )
            raw = (
                x if isinstance(x, TableTensor) else TableTensor.from_tensor(x)
            )
            if raw.schema != self.raw_schema:
                raise ValueError(
                    "Expected context and query features to share "
                    "the same schema"
                )
            return self.execution.transform(x, None)

        queries = self.source_result(transform_source)
        outputs = []
        for member, prepared in enumerate(self.members):

            def partition_query(
                member: int = member,
                prepared: PreparedContext = prepared,
            ) -> list[MemberQuery]:
                table = queries[member].x
                payloads = []
                for rank in range(self.mesh.world_size):
                    indices = halo_indices(
                        prepared.feature_count,
                        rank,
                        self.mesh,
                        self.group_size,
                    )
                    names = [
                        table.columns[Stype.numerical][i]
                        for i in indices.tolist()
                    ]
                    payloads.append(
                        MemberQuery(
                            TableTensor.from_tensor(
                                table.numerical.index_select(-1, indices),
                                columns=names,
                            ),
                            None,
                        )
                    )
                return payloads

            payloads = self.source_result(partition_query)
            outputs.append(self.scatter(payloads))
        return outputs

    def output(
        self, outputs: list[TableTensor], regression: bool
    ) -> TableTensor:
        """Invert targets and apply the original output recipe on source."""

        def finish() -> TableTensor:
            assert self.execution is not None
            host = [cast(TableTensor, out.cpu()) for out in outputs]
            if regression:
                host = list(self.execution.inverse_transform_target(host))
            return self.execution.transform_output(host)

        result = self.source_result(finish)
        return cast(TableTensor, self.broadcast(result).to(self.device))
