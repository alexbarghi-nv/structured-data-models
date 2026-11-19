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
from sdm.distributed.tensor_control import TensorControl
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

    Sharded modes use typed control and tensor payloads on the mesh backend
    (NCCL on CUDA, Gloo for CPU sessions). The legacy rank0_cpu recipe path
    retains its host object transport. Groups belong to the distributed
    session and are released by the caller's teardown, not by clearing a fit.
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
        # rank0_cpu retains its original generic host recipe path. Sharded
        # preprocessing uses the mesh backend for typed control and payloads.
        self.control = (
            None if mode == "rank0_cpu" else TensorControl(mesh.group, device)
        )
        self.group = (
            self.control.group
            if self.control is not None
            else cast(
                ProcessGroup,
                _get_group(
                    mesh.group,
                    dist.get_process_group_ranks(mesh.group),
                    backend="gloo",
                ),
            )
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
        """Broadcast control values using the selected preprocessing transport."""
        if self.control is not None:
            return self.control.broadcast(
                value, self.mesh.source if source is None else source
            )
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
        """Transfer only each destination's payload to that rank."""
        if self.control is not None:
            return self.control.scatter(
                values, self.mesh.source if source is None else source
            )
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
        source = self.mesh.source if source is None else source
        if self.control is not None:
            return self.control.local_result(
                lambda: operation() if dist.get_rank() == source else None
            )
        value = None
        error = None
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
        """Agree on failures with a scalar; gather details only on failure.

        NCCL uses the model device, including for CPU preprocessing. CPU/Gloo
        sessions use the existing host group. A broken process or CUDA context
        still requires the process-group timeout and caller's teardown.
        """
        if self.control is not None:
            return self.control.local_result(operation)
        use_nccl = dist.get_backend(self.mesh.group) == "nccl"
        status_group = self.mesh.group if use_nccl else self.group
        status_device = self.device if use_nccl else torch.device("cpu")
        # Allocate before the callback so an ordinary callback failure does not
        # require a fresh device allocation to report its status.
        failed = torch.empty(1, dtype=torch.int32, device=status_device)
        result = None
        error = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - synchronize rank failures
            error = f"rank {self.mesh.rank}: {type(exc).__name__}: {exc}"
        failed.fill_(int(error is not None))
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=status_group)
        # Reading the scalar waits for device completion before branching, so
        # every rank takes the same error-detail collective path.
        if failed.item():
            errors: list[Any] = [None] * self.mesh.world_size
            dist.all_gather_object(errors, error, group=self.group)
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
