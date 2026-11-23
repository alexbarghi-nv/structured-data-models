# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The distributed recipe adapter preserves native stochastic semantics."""

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.models.kumo.tabular.sampling import sampling_recipe
from sdm.processing.execution import RecipeExecution


@pytest.fixture
def recipe():
    """Nondefault options expose accidental loss of processor settings."""
    return sp.Recipe(
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


@pytest.fixture
def table():
    return TableTensor(
        columns={"numerical": ("value",), "categorical": ("category",)},
        numerical=torch.arange(12, dtype=torch.float32).view(12, 1),
        categorical=CategoricalTensor(
            code=torch.arange(12).remainder(3).view(12, 1),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )


def _fit(execution, table, generator):
    return execution.fit_transform(
        x=table,
        y=torch.arange(12, dtype=torch.float32).view(12, 1),
        related_tables=None,
        num_members=3,
        generator=generator,
    )


def test_sampling_recipe_preserves_options_context_state_and_generator(
    recipe, table
):
    """Adapted fit and replay match native results and random consumption."""
    original = repr(recipe)
    native = RecipeExecution(recipe)
    adapted = RecipeExecution(sampling_recipe(recipe))
    # Matching generators isolate adaptation from stochastic recipe draws.
    native_generator = torch.Generator().manual_seed(37)
    adapted_generator = torch.Generator().manual_seed(37)
    expected = _fit(native, table, native_generator)
    actual = _fit(adapted, table, adapted_generator)
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
    # Query transformation must reuse the state learned on the context.
    for left, right in zip(
        native.transform(table[:2], None),
        adapted.transform(table[:2], None),
        strict=True,
    ):
        torch.testing.assert_close(left.x.numerical, right.x.numerical)
    assert repr(recipe) == original
