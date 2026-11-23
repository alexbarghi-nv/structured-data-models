# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch import Tensor

import sdm.processing as sp
from sdm import EnsembleTable, Recipe, Stype, TableTensor
from sdm.models.kumo.tabular.column_metadata import (
    column_dictionary,
    encode_column_metadata,
    gather_column_metadata,
)
from sdm.models.kumo.tabular.column_shard import split_inputs
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.models.kumo.tabular.partition import (
    PreparedContext,
    halo_indices,
)
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.models.kumo.tabular.sampling import (
    _FlipSign,
    _ShuffleColumns,
    sampling_recipe,
)
from sdm.models.kumo.tabular.typed_exchange import exchange_member
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)

if TYPE_CHECKING:
    from sdm.models.kumo.tabular.prepared import PreparedInputs


def _check_recipe(recipe: Recipe) -> int:
    if repr(recipe) != repr(default_recipe()):
        raise ValueError(
            "Sharded preprocessing requires KumoTabular.default_recipe()"
        )
    return next(
        module.max_columns
        for module in recipe.features.modules()
        if isinstance(module, sp.SelectColumns)
    )


def _deterministic_route(recipe: Recipe) -> sp.EnsembleProcessor:
    route = copy.deepcopy(
        cast(sp.EnsembleProcessor, next(iter(recipe.features.children())))
    )
    numerical = cast(
        sp.Sequential, cast(sp.StypeDispatch, route).processors["numerical"]
    )
    for key, child in tuple(numerical._modules.items()):
        if isinstance(child, sp.FlipSign):
            numerical._modules[key] = sp.EnsembleProcessor.as_processor(
                sp.Identity()
            )
    return route


def fit_sharded(
    prepared: PreparedInputs,
    x: Tensor | TableTensor | ColumnShard | None,
    y: Tensor | TableTensor | None,
    recipe: Recipe,
    num_estimators: int | None,
    generator: torch.Generator,
) -> list[PreparedContext]:
    """Fit complete owned columns, then exchange final model shards."""
    max_columns = prepared.local_result(lambda: _check_recipe(recipe))
    raw = split_inputs(prepared, x, num_estimators, fitting=True, y=y)

    prepared.num_estimators = num_estimators
    prepared.local_raw_stypes = tuple(raw[i].stypes for i in range(len(raw)))

    def fit_target() -> tuple[MemberContext, ...]:
        assert x is not None
        assert y is not None
        if y.device.type != "cpu":
            raise ValueError(
                "Raw distributed targets must be on the source host"
            )
        # Reuse the standard target/output execution, including ensemble
        # locations and target inversion, without processing any features.
        target_recipe = sampling_recipe(
            Recipe(target=recipe.target, output=recipe.output)
        )
        prepared.execution = RecipeExecution(target_recipe)
        source_table = x.table if isinstance(x, ColumnShard) else x
        dummy = torch.empty(
            (*source_table.shape[:-1], 0), dtype=torch.float32, device="cpu"
        )
        return prepared.execution.fit_transform(
            dummy,
            y,
            None,
            num_members=num_estimators,
            generator=generator,
        )

    targets = prepared.source_result(fit_target)
    # Source-input mode keeps the global schema on rank0. Share the dictionary
    # once per fit; ordered per-member metadata subsequently travels as IDs.
    names = (
        column_dictionary(prepared.raw_columns)
        if prepared.prepartitioned
        else prepared.broadcast(
            column_dictionary(prepared.raw_columns)
            if prepared.is_source
            else None
        )
    )
    processing_device = prepared.device

    def process_local() -> tuple[list[TableTensor], Tensor]:
        prepared.local_route = _deterministic_route(recipe).to(
            processing_device
        )
        on_device = raw.replace_groups(
            [
                cast(TableTensor, group.to(processing_device))
                for group in raw._iter_groups()
            ]
        )
        processed = prepared.local_route.fit_transform_ensemble(on_device)
        # Retain processed feature values on their processing device.
        local = [processed[i] for i in range(len(processed))]
        # Generated counts can reuse dropped raw numerical names. Fitted route
        # ownership determines the global numerical sign draws.
        numerical_route = cast(
            sp.StypeDispatch, prepared.local_route
        ).processors["numerical"]
        drop = next(
            module
            for module in numerical_route.modules()
            if isinstance(module, sp.DropConstantColumns)
        )
        categorical_route = cast(
            sp.StypeDispatch, prepared.local_route
        ).processors["categorical"]
        counter = next(
            module
            for module in categorical_route.modules()
            if isinstance(module, sp.EnsembleProcessorAdapter)
            and isinstance(module.processor, sp.AddCategoryCounts)
        )
        metadata = []
        for member, table in enumerate(local):
            raw_names = raw[member].columns[Stype.numerical]
            numeric_names = (
                tuple(raw_names[index] for index in drop._kept_indices[member])
                if raw_names
                else ()
            )
            count_names: tuple[str, ...] = ()
            if raw[member].categorical.size(-1):
                group, _ = counter._fitted_locations[member]
                counts = cast(sp.AddCategoryCounts, counter._processors[group])
                count_names = tuple(
                    f"{name}__count" for name in counts._columns
                )
            metadata.append(
                (
                    table.columns[Stype.numerical],
                    numeric_names,
                    count_names,
                    tuple(
                        name
                        for name, stype in prepared.local_raw_stypes[
                            member
                        ].items()
                        if stype == Stype.numerical
                    ),
                    tuple(
                        name
                        for name, stype in prepared.local_raw_stypes[
                            member
                        ].items()
                        if stype == Stype.categorical
                    ),
                )
            )
        # Encode inside local_result so unknown names fail on every rank before
        # entering the metadata tensor collectives.
        return local, encode_column_metadata(metadata, names)

    local, metadata = prepared.local_result(process_local)
    use_nccl = dist.get_backend(prepared.mesh.group) == "nccl"
    gathered = gather_column_metadata(
        metadata,
        names,
        group=prepared.mesh.group if use_nccl else prepared.group,
        device=prepared.device if use_nccl else torch.device("cpu"),
    )

    def draw_plan() -> tuple[list[Any], list[Any]]:
        numeric_raw = prepared.raw_columns[Stype.numerical]
        categorical_raw = prepared.raw_columns[Stype.categorical]
        categorical_order = (
            *(f"{name}__count" for name in categorical_raw),
            *categorical_raw,
        )
        columns_by_member = []
        numeric_by_member = []
        for member in range(len(local)):
            if any(
                name in categorical_raw
                for owner in gathered
                for name in owner[member][2]
            ):
                raise ValueError(
                    "Category count names collide with raw categorical columns"
                )
            retained = [
                name for owner in gathered for name in owner[member][0]
            ]
            if len(retained) != len(set(retained)):
                raise ValueError(
                    "Recipe generated duplicate processed column names "
                    "across owners"
                )
            numerical = {
                name for owner in gathered for name in owner[member][1]
            }
            categorical = set(retained) - numerical
            numeric_names = tuple(
                name for name in numeric_raw if name in numerical
            )
            columns = (
                *numeric_names,
                *(name for name in categorical_order if name in categorical),
            )
            if not columns:
                raise ValueError(
                    "Recipe must retain at least one numerical feature"
                )
            columns_by_member.append(columns)
            numeric_by_member.append(numeric_names)
        sign_tables = [
            TableTensor.from_tensor(
                torch.empty(
                    (*local[i].shape[:-2], 0, len(names)),
                    dtype=torch.float64,
                    device="cpu",
                ),
                columns=names,
            )
            for i, names in enumerate(numeric_by_member)
        ]
        if any(numeric_by_member):
            flip = _FlipSign()
            flip._fit_ensemble(
                EnsembleTable.from_tables(
                    sign_tables, member_table_ids=range(len(sign_tables))
                ),
                generator=generator,
            )
            signs = [sign.cpu() for sign in flip._signs]
        else:
            # The ordinary dispatcher skips FlipSign entirely when no
            # numerical route survives. Do not advance a CUDA RNG here.
            signs = [
                torch.empty(
                    (*table.shape[:-2], 1, 0),
                    dtype=torch.float64,
                    device="cpu",
                )
                for table in sign_tables
            ]
        shuffle_tables = [
            TableTensor.from_tensor(
                torch.empty(
                    (*local[i].shape[:-2], 0, len(names)),
                    dtype=torch.float64,
                    device="cpu",
                ),
                columns=names,
            )
            for i, names in enumerate(columns_by_member)
        ]
        shuffle = _ShuffleColumns(method="latin")
        shuffle._fit_ensemble(
            EnsembleTable.from_tables(
                shuffle_tables, member_table_ids=range(len(shuffle_tables))
            ),
            generator=generator,
        )
        plans = []
        target_parts = [[] for _ in range(prepared.mesh.world_size)]
        for member, columns in enumerate(columns_by_member):
            chosen = tuple(
                columns[i]
                for i in shuffle._host_permutations[member][:max_columns]
            )
            plans.append((numeric_by_member[member], signs[member], chosen))
            target = targets[member]
            for rank in range(prepared.mesh.world_size):
                row_rank = rank // prepared.mesh.feature_parts
                start, end = prepared.mesh.bounds(
                    target.y.size(-2), row_rank, prepared.mesh.row_parts
                )
                target_parts[rank].append(target.y[..., start:end, :].clone())
        return plans, target_parts

    plan_and_targets = prepared.source_result(draw_plan)
    plans = prepared.broadcast(
        plan_and_targets[0] if prepared.is_source else None
    )
    local_targets = prepared.scatter(
        plan_and_targets[1] if prepared.is_source else None
    )
    prepared.transport_owner_names = tuple(
        tuple(tuple(owner[member][0]) for owner in gathered)
        for member in range(len(local))
    )
    prepared.plans = plans
    prepared.local_signs = []
    signed = []
    for member, table in enumerate(local):
        numeric_names, signs, _chosen = plans[member]
        signs = signs.to(table.device)
        lookup = {name: index for index, name in enumerate(numeric_names)}
        local_sign = torch.ones(
            (*table.shape[:-2], 1, table.numerical.size(-1)),
            dtype=torch.float64,
            device=table.device,
        )
        for column, name in enumerate(table.columns[Stype.numerical]):
            if name in lookup:
                local_sign[..., column] = signs[..., lookup[name]]
        prepared.local_signs.append(local_sign)
        signed.append(
            table.replace_blocks(
                numerical=(table.numerical * local_sign).float()
            )
        )
    # Raw type lists travel in the same metadata payload. Reuse them for each
    # member instead of gathering Python type maps inside the exchange loop.
    input_types = [
        {
            name: stype
            for owner in gathered
            for names, stype in zip(
                owner[member][3:],
                (Stype.numerical, Stype.categorical),
                strict=True,
            )
            for name in names
        }
        for member in range(len(local))
    ]
    prepared.members = _exchange(
        prepared,
        signed,
        local_targets,
        fitting=True,
        input_types=input_types,
    )
    return prepared.members


def _exchange(
    prepared: PreparedInputs,
    local: list[TableTensor],
    targets: list[TableTensor] | None,
    *,
    fitting: bool,
    input_types: list[dict[str, Stype]] | None = None,
) -> list[Any]:
    mesh = prepared.mesh
    results = []
    for member, table in enumerate(local):
        chosen = prepared.plans[member][2]
        features = len(chosen)
        indices = halo_indices(features, mesh.rank, mesh, prepared.group_size)
        names = tuple(chosen[index] for index in indices.tolist())
        rows = table.size(-2)
        destination_names = [
            tuple(
                chosen[index]
                for index in halo_indices(
                    features, destination, mesh, prepared.group_size
                ).tolist()
            )
            for destination in range(mesh.world_size)
        ]
        row_bounds = [
            mesh.bounds(
                rows, destination // mesh.feature_parts, mesh.row_parts
            )
            if fitting
            else (0, rows)
            for destination in range(mesh.world_size)
        ]
        data, mean = exchange_member(
            prepared,
            table.numerical,
            tuple(table.columns[Stype.numerical]),
            prepared.transport_owner_names[member],
            destination_names,
            row_bounds,
        )
        output = TableTensor.from_tensor(data, columns=names)
        if not fitting:
            results.append(MemberQuery(output, None))
            continue
        assert targets is not None
        assert input_types is not None
        input_stypes = input_types[member]
        mask = torch.tensor(
            [input_stypes.get(name) == Stype.categorical for name in names],
            dtype=torch.bool,
            device="cpu",
        )
        results.append(
            PreparedContext(
                context=MemberContext(
                    x=output,
                    y=targets[member],
                    x_context_only=None,
                    related_tables=None,
                    input_stypes=input_stypes,
                ),
                categorical_mask=mask,
                context_rows=rows,
                feature_count=features,
                feature_indices=indices,
                mean=mean,
            )
        )
    return results


def transform_sharded(
    prepared: PreparedInputs,
    x: Tensor | TableTensor | ColumnShard | None,
) -> list[MemberQuery]:
    """Reuse local fitted column statistics and exchange query halos."""
    raw = split_inputs(prepared, x, prepared.num_estimators, fitting=False)
    processing_device = prepared.device

    def transform_local() -> list[TableTensor]:
        on_device = raw.replace_groups(
            [
                cast(TableTensor, group.to(processing_device))
                for group in raw._iter_groups()
            ]
        )
        processed = prepared.local_route.transform_ensemble(on_device)
        local = []
        for member in range(len(processed)):
            table = processed[member]
            local.append(
                table.replace_blocks(
                    numerical=(
                        table.numerical
                        * prepared.local_signs[member].to(table.device)
                    ).float()
                )
            )
        return local

    local = prepared.local_result(transform_local)
    return cast(
        list[MemberQuery], _exchange(prepared, local, None, fitting=False)
    )
