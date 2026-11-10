# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sdm.distributed.layout import TableMesh
from sdm.processing.execution import MemberContext


@dataclass
class PreparedContext:
    """One estimator's local rows and circular feature halo."""

    context: MemberContext
    categorical_mask: Tensor
    context_rows: int
    feature_count: int
    feature_indices: Tensor
    mean: Tensor


def halo_indices(
    features: int, rank: int, mesh: TableMesh, group_size: int
) -> Tensor:
    """Return the unique source columns needed by owned circular groups."""
    feature_rank = rank % mesh.feature_parts
    start, end = mesh.bounds(features, feature_rank, mesh.feature_parts)
    positions = torch.arange(start, end, device="cpu").unsqueeze(-1)
    shifts = 2 ** torch.arange(group_size, device="cpu") - 1
    return ((positions + shifts) % features).flatten().unique(sorted=True)
