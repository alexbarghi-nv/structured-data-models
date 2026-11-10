# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch import Tensor

from sdm import EnsembleTable, Stype, TableTensor
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.processing.execution import _to_ensemble_table
from sdm.tensor.table import TableSchema

if TYPE_CHECKING:
    from sdm.models.kumo.tabular.prepared import PreparedInputs


def _fingerprint(*tensors: Tensor) -> str:
    """Compare exact host row identities and optional replicated targets."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(
            tensor.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _target_fingerprint(y: Tensor | TableTensor) -> str:
    table = y if isinstance(y, TableTensor) else TableTensor.from_tensor(y)
    digest = hashlib.sha256()
    digest.update(
        repr(
            tuple((str(stype), table.columns[stype]) for stype in Stype)
        ).encode()
    )
    digest.update(
        _fingerprint(table.numerical, table.categorical.code).encode()
    )
    for categories in table.categorical.categories:
        digest.update(str(categories.dtype).encode())
        digest.update(repr(categories.tolist()).encode())
    return digest.hexdigest()


def _local_inputs(
    prepared: PreparedInputs,
    shard: ColumnShard,
    y: Tensor | TableTensor | None,
    num_estimators: int | None,
    *,
    fitting: bool,
) -> EnsembleTable:
    def validate_local() -> tuple[EnsembleTable, dict[str, Any]]:
        table = shard.table
        if table.device.type != "cpu" or shard.row_ids.device.type != "cpu":
            raise ValueError("ColumnShard tables and row IDs must be on CPU")
        columns: dict[Stype, tuple[str, ...]] = dict.fromkeys(Stype, ())
        columns.update(
            {
                Stype(stype): tuple(names)
                for stype, names in shard.columns.items()
            }
        )
        if any(
            columns[stype]
            for stype in Stype
            if stype not in {Stype.numerical, Stype.categorical}
        ):
            raise ValueError(
                "ColumnShard supports numerical and categorical features"
            )
        global_columns = tuple(
            (name, stype) for stype in Stype for name in columns[stype]
        )
        names = tuple(name for name, _ in global_columns)
        if len(names) != len(set(names)):
            raise ValueError(
                "ColumnShard global schema contains duplicate column names"
            )
        if len(shard.column_ids) != table.size(-1):
            raise ValueError(
                "ColumnShard column IDs must match the local column count"
            )
        if any(
            index < 0 or index >= len(global_columns)
            for index in shard.column_ids
        ):
            raise ValueError(
                "ColumnShard column IDs lie outside the global schema"
            )
        local_columns = tuple(
            (name, stype) for stype in Stype for name in table.columns[stype]
        )
        if local_columns != tuple(
            global_columns[index] for index in shard.column_ids
        ):
            raise ValueError(
                "ColumnShard column IDs do not match the local names "
                "and semantic types"
            )
        if shard.row_ids.dtype != torch.int64 or tuple(
            shard.row_ids.shape
        ) != tuple(table.shape[:-1]):
            raise ValueError(
                "ColumnShard row IDs must be int64 with the table's row shape"
            )
        ordered = shard.row_ids.sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError(
                "ColumnShard row IDs must be unique within each cohort"
            )
        schema = TableSchema(columns=columns)
        if not fitting and (
            schema != prepared.raw_schema
            or shard.column_ids != prepared.column_ids
        ):
            raise ValueError(
                "Query ColumnShard schema and column ownership must match fit"
            )
        if not fitting and table.numerical.dtype != prepared.raw_dtype:
            raise ValueError(
                "Query ColumnShard numerical dtype must match fit"
            )
        target = None
        if fitting and y is not None:
            if y.device.type != "cpu" or tuple(y.shape[:-1]) != tuple(
                table.shape[:-1]
            ):
                raise ValueError(
                    "ColumnShard targets must be on CPU and follow "
                    "the local row cohort"
                )
            target = _target_fingerprint(y)
        ensemble = _to_ensemble_table(table, num_estimators)
        return ensemble, {
            "schema": schema,
            "ids": shard.column_ids,
            "rows": _fingerprint(shard.row_ids),
            "dtype": table.numerical.dtype,
            "members": (num_estimators, len(ensemble)),
            "targets": target,
        }

    raw, metadata = prepared.local_result(validate_local)
    owners: list[Any] = [None] * prepared.mesh.world_size
    dist.all_gather_object(owners, metadata, group=prepared.group)
    source = owners[0]
    for field, description in (
        ("schema", "global schema"),
        ("rows", "ordered row identities/cohort"),
        ("dtype", "numerical dtype"),
        ("members", "ensemble member count"),
    ):
        if any(owner[field] != source[field] for owner in owners):
            raise RuntimeError(
                f"ColumnShard {description} differs between workers"
            )
    width = sum(len(names) for names in source["schema"].columns.values())
    ownership = [index for owner in owners for index in owner["ids"]]
    if sorted(ownership) != list(range(width)):
        raise RuntimeError(
            "ColumnShard ownership must cover every global column exactly once"
        )
    if fitting:
        if source["targets"] is None:
            raise RuntimeError(
                "The source worker must provide ColumnShard targets"
            )
        if any(
            owner["targets"] is not None
            and owner["targets"] != source["targets"]
            for owner in owners
        ):
            raise RuntimeError("ColumnShard targets differ between workers")
        prepared.raw_schema = source["schema"]
        prepared.raw_columns = source["schema"].columns
        prepared.column_ids = shard.column_ids
        prepared.raw_dtype = shard.table.numerical.dtype
    return raw


def split_inputs(
    prepared: PreparedInputs,
    x: Tensor | TableTensor | ColumnShard | None,
    num_estimators: int | None,
    *,
    fitting: bool,
    y: Tensor | TableTensor | None = None,
) -> EnsembleTable:
    """Read owned column shards or scatter complete source-host columns."""
    shard_flags: list[Any] = [None] * prepared.mesh.world_size
    dist.all_gather_object(
        shard_flags, isinstance(x, ColumnShard), group=prepared.group
    )
    direct = any(shard_flags)
    if direct and not all(shard_flags):
        raise RuntimeError(
            "Every worker must provide a ColumnShard, including empty owners"
        )
    if not fitting and direct != prepared.prepartitioned:
        raise RuntimeError(
            "Predict must use the same source-input or ColumnShard form as fit"
        )
    if fitting:
        prepared.prepartitioned = direct
    if direct:
        return _local_inputs(
            prepared, cast(ColumnShard, x), y, num_estimators, fitting=fitting
        )

    def split_source() -> list[EnsembleTable]:
        assert isinstance(x, (Tensor, TableTensor))
        if x.device.type != "cpu":
            raise ValueError(
                "Raw distributed inputs must be on the source host"
            )
        table = x if isinstance(x, TableTensor) else TableTensor.from_tensor(x)
        if table.active_stypes - {Stype.numerical, Stype.categorical}:
            raise ValueError(
                "Sharded preprocessing supports numerical "
                "and categorical features"
            )
        if fitting:
            prepared.raw_schema = table.schema
            prepared.raw_columns = dict(table.columns)
        elif table.schema != prepared.raw_schema:
            raise ValueError(
                "Expected context and query features to share the same schema"
            )
        ensemble = _to_ensemble_table(table, num_estimators)
        parts = []
        for rank in range(prepared.mesh.world_size):
            names = []
            for stype in (Stype.numerical, Stype.categorical):
                columns = table.columns[stype]
                start, end = prepared.mesh.bounds(
                    len(columns), rank, prepared.mesh.world_size
                )
                names.extend(columns[start:end])
            parts.append(
                ensemble.replace_groups(
                    [
                        cast(TableTensor, group.select_columns(names).clone())
                        for group in ensemble._iter_groups()
                    ]
                )
            )
        return parts

    parts = prepared.source_result(split_source)
    return cast(EnsembleTable, prepared.scatter(parts))
