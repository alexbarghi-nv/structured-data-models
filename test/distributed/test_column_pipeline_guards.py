# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local dispatch guards; numerical and schedule tests live separately."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sdm.cache import Cache
from sdm.distributed._summary_merge import OneTicketQueue
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.parallel import _ContextParallelKumoTabular


@pytest.fixture
def pipeline():
    """Build only the state needed to test dispatch before model execution."""
    model = _ContextParallelKumoTabular(
        cast(_KumoTabular, SimpleNamespace(row_embedding=SimpleNamespace())),
        cast(
            TableMesh,
            SimpleNamespace(feature_parts=1, features=lambda n: (0, n)),
        ),
    )
    assert model._fused_column_pipeline is False
    model._fused_pure_row = True
    model._fused_column_pipeline = True
    model._fused_replay_original = True
    model._fused_chunk_rows = 512
    model._fused_feature_tile = 64
    return model


def _embed(model, cache, dtype=torch.float32):
    return model._embed_rows(
        torch.zeros(1, 1, 4, dtype=dtype),
        torch.zeros(1, 1),
        torch.zeros(1, 4, dtype=torch.bool),
        1,
        cache,
    )


def _fail_original_replay(*args, **kwargs):
    raise RuntimeError("original replay")


def test_recording_uses_pipeline(pipeline, monkeypatch):
    """Valid inference dispatches to the opt-in recording pipeline."""
    monkeypatch.setattr(
        pipeline, "_embed_rows_fused_column_pipeline", lambda *a: "pipeline"
    )
    with torch.inference_mode():
        assert _embed(pipeline, Cache()) == "pipeline"


@pytest.mark.parametrize("invalid", ["autograd", "half", "tile32"])
def test_invalid_recording_rejected_before_mutation(pipeline, invalid):
    """Unsupported recording settings must fail before changing the cache."""
    cache = Cache()
    pipeline._fused_feature_tile = 32 if invalid == "tile32" else 64
    dtype = torch.float16 if invalid == "half" else torch.float32
    with (
        torch.inference_mode(invalid != "autograd"),
        pytest.raises(RuntimeError, match="configuration"),
    ):
        _embed(pipeline, cache, dtype)
    assert len(cache) == 0
    assert pipeline._fused_async_queue is None


def test_frozen_cache_uses_original_replay(pipeline, monkeypatch):
    """Prediction bypasses recording and consumes the existing cache."""
    monkeypatch.setattr(pipeline, "_embed_cells", _fail_original_replay)
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="original replay"),
    ):
        _embed(pipeline, Cache().freeze())


@pytest.fixture
def poisoned_pipeline(pipeline, monkeypatch):
    """Fail after admission so the queue still owns an outstanding ticket."""
    queue = OneTicketQueue()
    ticket = object()
    queue.ticket = ticket

    def fail(*args, **kwargs):
        pipeline._fused_async_queue = queue
        raise ValueError("column failure")

    monkeypatch.setattr(pipeline, "_embed_rows_fused_impl", fail)
    with pytest.raises(ValueError, match="column failure"):
        pipeline._embed_rows_fused_column_pipeline(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            1,
            Cache(),
        )
    assert pipeline._fused_async_queue is queue
    assert queue.ticket is ticket
    assert queue.poisoned is True
    return pipeline, queue, ticket


@pytest.mark.parametrize("fallback", ["disabled", "half", "replay"])
def test_failed_queue_cannot_be_bypassed(poisoned_pipeline, fallback):
    """Fallback routes must not release buffers that may still be in use."""
    model, queue, ticket = poisoned_pipeline
    model._fused_column_pipeline = fallback != "disabled"
    dtype = torch.float16 if fallback == "half" else torch.float32
    cache = Cache()
    if fallback == "replay":
        cache.freeze()
    with (
        torch.inference_mode(),
        pytest.raises(
            RuntimeError, match="Previous async summary did not close"
        ),
    ):
        _embed(model, cache, dtype)
    assert model._fused_async_queue is queue
    assert queue.ticket is ticket
