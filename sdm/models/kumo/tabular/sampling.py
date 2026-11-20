# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import cast

import torch

import sdm.processing as sp
from sdm import EnsembleTable, Recipe, TableTensor
from sdm.nn._buffer import BufferList


class _FlipSign(sp.FlipSign):
    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        signs = []
        for member in range(len(ensemble_table)):
            x = ensemble_table[member].numerical
            sign = torch.empty(
                (*x.shape[:-2], 1, x.size(-1)),
                dtype=x.dtype,
                device=x.device if generator is None else generator.device,
            ).bernoulli_(self.probability, generator=generator)
            signs.append(sign.mul_(-2).add_(1).to(x.device))
        self._signs = BufferList(signs)


class _ShuffleColumns(sp.ShuffleColumns):
    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = ensemble_table[0].device
        if generator is not None and generator.device != device:
            # Only column metadata is copied to the sampling device.
            tables = [
                cast(
                    TableTensor,
                    ensemble_table[i][..., :0, :].to(generator.device),
                )
                for i in range(len(ensemble_table))
            ]
            ensemble_table = EnsembleTable.from_tables(
                tables, member_table_ids=range(len(tables))
            )
        super()._fit_ensemble(ensemble_table, generator=generator)
        self.to(device)


class _ShuffleCategories(sp.ShuffleCategories):
    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = ensemble_table[0].device
        if generator is not None and generator.device != device:
            tables = [
                cast(
                    TableTensor,
                    ensemble_table[i][..., :0, :].to(generator.device),
                )
                for i in range(len(ensemble_table))
            ]
            ensemble_table = EnsembleTable.from_tables(
                tables, member_table_ids=range(len(tables))
            )
        super()._fit_ensemble(ensemble_table, generator=generator)
        self.to(device)


def sampling_recipe(recipe: Recipe) -> Recipe:
    """Keep stochastic draws on the caller's generator device."""
    recipe = copy.deepcopy(recipe)
    for pipeline in (recipe.features, recipe.target):
        for module in pipeline.modules():
            for name, child in tuple(module.named_children()):
                if type(child) is sp.FlipSign:
                    module.add_module(
                        name, _FlipSign(probability=child.probability)
                    )
                elif type(child) is sp.ShuffleColumns:
                    module.add_module(
                        name, _ShuffleColumns(method=child.method)
                    )
                elif type(child) is sp.ShuffleCategories:
                    module.add_module(
                        name, _ShuffleCategories(method=child.method)
                    )
    return recipe
