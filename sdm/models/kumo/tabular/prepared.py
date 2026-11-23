# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar, cast

import torch
import torch.distributed as dist
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, Stype, TableTensor
from sdm.distributed.layout import TableMesh
from sdm.distributed.tensor_control import TensorControl
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.models.kumo.tabular.partition import PreparedContext
from sdm.models.kumo.tabular.sharded_recipe import (
    fit_sharded,
    transform_sharded,
)
from sdm.processing.execution import (
    MemberQuery,
    RecipeExecution,
)
from sdm.tensor.table import TableSchema

_T = TypeVar("_T")

Preprocessing = Literal["replicated", "sharded_cuda"]


class PreparedInputs:
    """Preprocess owned columns on the model device and exchange model shards.

    Preprocessing runs on the model device. Typed control and tensor payloads
    use the mesh backend (NCCL on CUDA, Gloo in CPU protocol tests).
    Groups belong to the distributed session and are released by the caller's
    teardown, not by clearing a fit.
    Model inputs contain only owned rows/features and their required halo,
    which can span tiny feature axes.
    Source preprocessing retains complete context tables on the host.
    """

    def __init__(
        self,
        mesh: TableMesh,
        device: torch.device,
        group_size: int,
    ) -> None:
        self.mesh = mesh
        self.device = device
        self.group_size = group_size
        self.control = TensorControl(mesh.group, device)
        self.group = self.control.group
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
        self.transport_owner_names: tuple[
            tuple[tuple[str, ...], ...], ...
        ] = ()

    @property
    def is_source(self) -> bool:
        """Whether this is the source within the caller's process group."""
        return self.mesh.rank == 0

    def broadcast(self, value: Any, *, source: int | None = None) -> Any:
        """Broadcast control values over the preprocessing transport."""
        return self.control.broadcast(
            value, self.mesh.source if source is None else source
        )

    def scatter(
        self, values: list[Any] | None, *, source: int | None = None
    ) -> Any:
        """Transfer only each destination's payload to that rank."""
        return self.control.scatter(
            values, self.mesh.source if source is None else source
        )

    def source_result(
        self, operation: Callable[[], _T], *, source: int | None = None
    ) -> _T:
        """Propagate source failures before peers enter transport."""
        source = self.mesh.source if source is None else source
        # Only the source consumes the result; peers retain a placeholder.
        return cast(
            _T,
            self.control.local_result(
                lambda: operation() if dist.get_rank() == source else None
            ),
        )

    def local_result(self, operation: Callable[[], _T]) -> _T:
        """Agree on failures with a scalar; gather typed details on failure."""
        return self.control.local_result(operation)

    def fit(
        self,
        x: Tensor | TableTensor | ColumnShard | None,
        y: Tensor | TableTensor | None,
        recipe: Recipe,
        num_estimators: int | None,
        generator: torch.Generator,
    ) -> list[PreparedContext]:
        """Fit source inputs and retain only model-owned context shards."""
        return fit_sharded(self, x, y, recipe, num_estimators, generator)

    def transform(
        self, x: Tensor | TableTensor | ColumnShard | None
    ) -> list[MemberQuery]:
        """Transform queries using the fitted context state."""
        return transform_sharded(self, x)

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
