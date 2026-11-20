# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed per-estimator feature exchange without host materialization."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sdm.models.kumo.tabular.prepared import PreparedInputs


def exchange_member(
    prepared: PreparedInputs,
    table: Tensor,
    local_names: tuple[str, ...],
    owner_names: tuple[tuple[str, ...], ...],
    destination_names: list[tuple[str, ...]],
    row_bounds: list[tuple[int, int]],
) -> tuple[Tensor, Tensor]:
    """Redistribute one estimator's owned features and context means."""
    control = prepared.control
    assert control is not None
    control.check()
    rank, world = prepared.mesh.rank, prepared.mesh.world_size
    state = {}

    def allocate() -> None:
        if (
            tuple(local_names) != tuple(owner_names[rank])
            or table.dtype != torch.float32
        ):
            raise ValueError(
                "Fitted owner schema or preprocessing dtype changed"
            )
        if len(owner_names) != world or table.shape[-1] != len(local_names):
            raise ValueError("Invalid owner count or local feature width")
        flat_names = [name for names in owner_names for name in names]
        if len(flat_names) != len(set(flat_names)):
            raise ValueError("Duplicate feature ownership")
        batch = table.shape[:-2]
        rows = table.shape[-2]
        plans = []
        for names in owner_names:
            lookup = {name: index for index, name in enumerate(names)}
            parts = []
            for chosen, (left, right) in zip(
                destination_names, row_bounds, strict=True
            ):
                if not 0 <= left <= right <= rows:
                    raise ValueError("Invalid row partition")
                positions = tuple(
                    i for i, name in enumerate(chosen) if name in lookup
                )
                parts.append(
                    (positions, tuple(lookup[chosen[i]] for i in positions))
                )
            plans.append(parts)
        for destination, names in enumerate(destination_names):
            positions = sorted(
                i for owner in plans for i in owner[destination][0]
            )
            if positions != list(range(len(names))):
                raise ValueError(
                    "Destination features not covered exactly once"
                )
        # Preserve the original sequence of sign/cast/nanmean operations, now
        # on the processing device. CUDA rounding parity is a validation gate.
        mean = table.nanmean(dim=-2, keepdim=True)
        mean = mean.masked_fill(mean.isnan(), 0)
        sends, chunks = [], []
        for (_, columns), (left, right) in zip(
            plans[rank], row_bounds, strict=True
        ):
            indices = torch.tensor(
                columns, dtype=torch.int64, device=table.device
            )
            values = table[..., left:right, :].index_select(-1, indices)
            means = mean.index_select(-1, indices)
            chunk = torch.cat((values.reshape(-1), means.reshape(-1)))
            chunks.append(chunk)
            sends.append(chunk.numel())
        send = torch.cat(chunks).to(control.device)
        left, right = row_bounds[rank]
        local_rows = right - left
        receives = [
            prod(batch) * (local_rows + 1) * len(owner[rank][0])
            for owner in plans
        ]
        state.update(
            send=send,
            receive=torch.empty(
                sum(receives), dtype=torch.float32, device=control.device
            ),
            sends=sends,
            receives=receives,
            batch=batch,
            rows=local_rows,
            plans=plans,
            data=torch.empty(
                (*batch, local_rows, len(destination_names[rank])),
                dtype=torch.float32,
                device=control.device,
            ),
            mean=torch.empty(
                (*batch, 1, len(destination_names[rank])),
                dtype=torch.float32,
                device=control.device,
            ),
        )

    prepared.local_result(allocate)
    control.exchange(
        state["receive"], state["send"], state["receives"], state["sends"]
    )

    def consume() -> tuple[Tensor, Tensor]:
        offset = 0
        for source, count in enumerate(state["receives"]):
            positions = state["plans"][source][rank][0]
            width = len(positions)
            index = torch.tensor(
                positions, dtype=torch.int64, device=control.device
            )
            chunk = state["receive"][offset : offset + count]
            split = prod(state["batch"]) * state["rows"] * width
            state["data"].index_copy_(
                -1,
                index,
                chunk[:split].view(*state["batch"], state["rows"], width),
            )
            state["mean"].index_copy_(
                -1, index, chunk[split:].view(*state["batch"], 1, width)
            )
            offset += count
        return state["data"], state["mean"]

    return prepared.local_result(consume)
