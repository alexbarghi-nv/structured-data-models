# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sdm.cache import Cache
from sdm.distributed._summary_merge import OneTicketQueue
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.parallel import _DistributedKumoTabular


def test_guards_and_replay_before_mutation(monkeypatch):
    model = _DistributedKumoTabular(
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
    x = torch.zeros(1, 1, 4)
    y = torch.zeros(1, 1)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    cache = Cache()
    monkeypatch.setattr(
        model, "_embed_rows_fused_column_pipeline", lambda *args: "pipeline"
    )
    with torch.inference_mode():
        assert model._embed_rows(x, y, mask, 1, cache) == "pipeline"
    with pytest.raises(RuntimeError, match="configuration"):
        model._embed_rows(x, y, mask, 1, cache)
    assert len(cache) == 0
    assert model._fused_async_queue is None
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="configuration"):
            model._embed_rows(x.half(), y, mask, 1, cache)
        model._fused_feature_tile = 32
        with pytest.raises(RuntimeError, match="configuration"):
            model._embed_rows(x, y, mask, 1, cache)
    model._fused_feature_tile = 64
    cache.freeze()
    monkeypatch.setattr(
        model,
        "_embed_cells",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("original replay")),
    )
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="original replay"),
    ):
        model._embed_rows(x, y, mask, 1, cache)


def test_failure_poison_retains_live_queue(monkeypatch):
    model = _DistributedKumoTabular(
        cast(_KumoTabular, SimpleNamespace(row_embedding=SimpleNamespace())),
        cast(TableMesh, SimpleNamespace(feature_parts=1)),
    )
    ticket = object()
    queue = OneTicketQueue()
    queue.ticket = ticket

    def fail(*args, **kwargs):
        model._fused_async_queue = queue
        raise ValueError("column failure")

    monkeypatch.setattr(model, "_embed_rows_fused_impl", fail)
    with pytest.raises(ValueError, match="column failure"):
        model._embed_rows_fused_column_pipeline(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            1,
            Cache(),
        )
    assert model._fused_async_queue is queue
    assert queue.ticket is ticket
    assert queue.poisoned is True
    x = torch.zeros(1, 1, 4)
    y = torch.zeros(1, 1)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    for flag, dtype, replay in (
        (False, torch.float32, False),
        (True, torch.float16, False),
        (True, torch.float32, True),
    ):
        model._fused_column_pipeline = flag
        cache = Cache()
        if replay:
            cache.freeze()
        with (
            torch.inference_mode(),
            pytest.raises(
                RuntimeError, match="Previous async summary did not close"
            ),
        ):
            model._embed_rows(x.to(dtype), y, mask, 1, cache)
        assert model._fused_async_queue is queue
        assert queue.ticket is ticket
