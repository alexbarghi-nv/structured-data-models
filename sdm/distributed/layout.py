# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast
from weakref import WeakKeyDictionary

import torch.distributed as dist
from torch.distributed import ProcessGroup

_GROUPS: WeakKeyDictionary[
    ProcessGroup, dict[tuple[tuple[int, ...], str], ProcessGroup]
] = WeakKeyDictionary()


def _get_group(
    parent: ProcessGroup,
    ranks: list[int],
    *,
    backend: str | None = None,
) -> ProcessGroup | None:
    """Reuse caller-owned session groups across model instances."""
    backend = str(dist.get_backend(parent)) if backend is None else backend
    if ranks == dist.get_process_group_ranks(
        parent
    ) and backend == dist.get_backend(parent):
        return parent
    if dist.get_rank() not in ranks:
        return None
    groups = _GROUPS.setdefault(parent, {})
    key = (tuple(ranks), backend)
    if key not in groups:
        groups[key] = cast(
            ProcessGroup,
            dist.new_group(
                ranks=ranks, backend=backend, use_local_synchronization=True
            ),
        )
    return groups[key]


class TableMesh:
    """Partition row embeddings across rows and feature positions.

    Args:
        layout: ``"hybrid"`` uses two feature ranks; ``"mesh"`` defaults to
            two feature ranks and divides the remaining ranks across rows.
            Four ranks use a two-by-two mesh and eight use four-by-two.
            ICL context is sharded over every rank in either layout.
        group: Parent process group; defaults to the initialized world group.
            Mesh subgroups are reused within this parent and owned by the
            distributed session. Destroy the session after all models finish.
            Do not destroy shared child groups individually; doing so requires
            a new distributed session before constructing another model.
        mesh_shape: Optional ``(row_parts, feature_parts)`` for ``"mesh"``.
            Both factors must be positive integers and their product must
            equal the parent group's size.
    """

    def __init__(
        self,
        layout: Literal["hybrid", "mesh"],
        group: ProcessGroup | None = None,
        *,
        mesh_shape: tuple[int, int] | None = None,
    ) -> None:
        parent = dist.group.WORLD if group is None else group
        assert parent is not None
        self.group = parent
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        if layout == "hybrid":
            if mesh_shape is not None:
                raise ValueError("mesh_shape requires layout='mesh'")
            self.row_parts, self.feature_parts = 1, 2
        elif layout == "mesh":
            self.row_parts, self.feature_parts = mesh_shape or (
                self.world_size // 2,
                2,
            )
        else:
            raise ValueError(f"Unknown table layout: {layout!r}")
        if (
            min(self.row_parts, self.feature_parts) <= 0
            or self.world_size != self.row_parts * self.feature_parts
        ):
            raise ValueError(
                f"Mesh shape {(self.row_parts, self.feature_parts)} requires "
                f"{self.row_parts * self.feature_parts} ranks, but the "
                f"parent group has {self.world_size}"
            )
        self.layout = layout
        self.row_rank = self.rank // self.feature_parts
        self.feature_rank = self.rank % self.feature_parts
        ranks = dist.get_process_group_ranks(self.group)
        self.feature_group: ProcessGroup | None = None
        self.row_group: ProcessGroup | None = None
        for row in range(self.row_parts):
            members = ranks[
                row * self.feature_parts : (row + 1) * self.feature_parts
            ]
            subgroup = _get_group(self.group, members)
            if self.row_rank == row:
                self.feature_group = cast(ProcessGroup, subgroup)
        if self.row_parts > 1:
            for feature in range(self.feature_parts):
                members = ranks[feature :: self.feature_parts]
                subgroup = _get_group(self.group, members)
                if self.feature_rank == feature:
                    self.row_group = cast(ProcessGroup, subgroup)
        self.readout_source = ranks[self.row_rank * self.feature_parts]
        self.source = ranks[0]

    def rows(self, length: int) -> tuple[int, int]:
        """Return this mesh row's global row interval."""
        return self.bounds(length, self.row_rank, self.row_parts)

    def features(self, length: int) -> tuple[int, int]:
        """Return the owned global feature interval."""
        return self.bounds(length, self.feature_rank, self.feature_parts)

    def context(self, length: int) -> tuple[int, int]:
        """Return this rank's final ICL context interval."""
        return self.bounds(length, self.rank, self.world_size)

    @staticmethod
    def bounds(length: int, rank: int, parts: int) -> tuple[int, int]:
        """Contiguous balanced partition, preserving global sequence order."""
        return length * rank // parts, length * (rank + 1) // parts
