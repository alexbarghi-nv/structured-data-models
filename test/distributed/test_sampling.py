# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.models.kumo.tabular.sampling import sampling_recipe
from sdm.processing.execution import RecipeExecution


def test_sampling_recipe_preserves_options_context_state_and_generator():
    recipe = sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=sp.FlipSign(probability=0.25),
                categorical=sp.ShuffleCategories(method="random"),
            ),
            sp.ToNumerical(),
            sp.ShuffleColumns(method="random"),
        ],
        target=sp.FlipSign(probability=0.75),
    )
    original = repr(recipe)
    x = TableTensor(
        columns={"numerical": ("value",), "categorical": ("category",)},
        numerical=torch.arange(12, dtype=torch.float32).view(12, 1),
        categorical=CategoricalTensor(
            code=torch.arange(12).remainder(3).view(12, 1),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )
    y = torch.arange(12, dtype=torch.float32).view(12, 1)
    # Matching generators are necessary to compare stochastic recipe semantics.
    native_generator = torch.Generator().manual_seed(37)
    adapted_generator = torch.Generator().manual_seed(37)
    native = RecipeExecution(recipe)
    adapted = RecipeExecution(sampling_recipe(recipe))
    expected = native.fit_transform(
        x=x,
        y=y,
        related_tables=None,
        num_members=3,
        generator=native_generator,
    )
    actual = adapted.fit_transform(
        x=x,
        y=y,
        related_tables=None,
        num_members=3,
        generator=adapted_generator,
    )
    torch.testing.assert_close(
        native_generator.get_state(),
        adapted_generator.get_state(),
        rtol=0,
        atol=0,
    )
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left.x.numerical, right.x.numerical)
        torch.testing.assert_close(left.y.numerical, right.y.numerical)
        assert left.x.schema == right.x.schema
    query = x[:2]
    for left, right in zip(
        native.transform(query, None),
        adapted.transform(query, None),
        strict=True,
    ):
        torch.testing.assert_close(left.x.numerical, right.x.numerical)
    assert repr(recipe) == original
