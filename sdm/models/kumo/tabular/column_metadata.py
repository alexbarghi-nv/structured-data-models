# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from sdm import Stype

ColumnMetadata = list[tuple[tuple[str, ...], ...]]


def column_dictionary(
    columns: Mapping[Stype, Sequence[str]],
) -> tuple[str, ...]:
    """Canonical IDs for raw names and the default recipe's generated names."""
    return tuple(
        sorted(
            {
                *columns.get(Stype.numerical, ()),
                *columns.get(Stype.categorical, ()),
                *(
                    f"{name}__count"
                    for name in columns.get(Stype.categorical, ())
                ),
            }
        )
    )


def encode_column_metadata(
    metadata: ColumnMetadata, names: tuple[str, ...]
) -> Tensor:
    """Pack member lengths and ordered IDs, retaining duplicates."""
    ids = {name: index for index, name in enumerate(names)}
    packed = [len(metadata)]
    for member in metadata:
        packed.append(len(member))
        for columns in member:
            packed.append(len(columns))
            packed.extend(ids[name] for name in columns)
    return torch.tensor(packed, dtype=torch.int64, device="cpu")


def decode_column_metadata(
    packed: list[int], names: tuple[str, ...]
) -> ColumnMetadata:
    """Restore ordered column names and member field boundaries."""
    result = []
    offset = 1
    for _ in range(packed[0]):
        member = []
        fields = packed[offset]
        offset += 1
        for _ in range(fields):
            length = packed[offset]
            offset += 1
            member.append(
                tuple(
                    names[index] for index in packed[offset : offset + length]
                )
            )
            offset += length
        result.append(tuple(member))
    return result


def gather_column_metadata(
    encoded: Tensor,
    names: tuple[str, ...],
    *,
    group: ProcessGroup,
    device: torch.device,
) -> list[ColumnMetadata]:
    """Gather int64 metadata over the existing CPU/Gloo or CUDA/NCCL group.

    The caller has agreed on the dictionary and caught encoding failures before
    entry. Temporary storage is O(world_size * largest encoded rank payload).
    Host reads are explicit metadata-planning boundaries, not model hot paths.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return [decode_column_metadata(encoded.tolist(), names)]
    size = torch.tensor([encoded.numel()], dtype=torch.int64, device=device)
    sizes = [torch.empty_like(size) for _ in range(world)]
    dist.all_gather(sizes, size, group=group)
    lengths = torch.cat(sizes).cpu().tolist()
    width = max(lengths)
    payload = torch.zeros(width, dtype=torch.int64, device=device)
    payload[: encoded.numel()].copy_(encoded)
    received = [torch.empty_like(payload) for _ in range(world)]
    dist.all_gather(received, payload, group=group)
    host = torch.stack(received).cpu().tolist()
    return [
        decode_column_metadata(values[:length], names)
        for values, length in zip(host, lengths, strict=True)
    ]
