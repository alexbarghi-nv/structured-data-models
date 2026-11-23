# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-pipeline recording, cache replay, scheduling and failure ownership.

The numerical test compares two schedules on real CPU tensors and collectives:
finish all column summaries before applying columns, versus apply a completed
feature tile while the next tile is being merged. Both must produce the same
embedding and cache, and replay must leave that cache untouched.

The scheduling tests replace CUDA tickets with recording doubles on one CPU
rank. They check host-side order and retained state after errors. They do not
validate CUDA events, NCCL progress or actual communication/compute overlap.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

import sdm.models.kumo.tabular.parallel as parallel
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.parallel import _ContextParallelKumoTabular
from sdm.nn import InducedTransformerBlock
from test.distributed._runtime import distributed_session, spawn_workers


def _small_embedding_model() -> _KumoTabular:
    # Every rank must start with identical weights and input draws.
    torch.manual_seed(42)
    return _KumoTabular(
        num_classes=0,
        num_quantiles=1,
        cell_channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=3,
        group_size=2,
        num_frequencies=3,
        num_readout_tokens=2,
        icl_channels=16,
        num_icl_layers=1,
        num_icl_heads=2,
    )


def _column_schedule(core, mesh, *, pipelined):
    model = _ContextParallelKumoTabular(core, mesh)
    model._fused_pure_row = True
    model._fused_chunk_rows = 512
    model._fused_feature_tile = 64
    model._fused_replay_original = True
    model._fused_column_pipeline = pipelined
    return model


def _recording_cache(context):
    return Cache(
        cell_impute_mean=context.nanmean(dim=-2, keepdim=True),
        **{"distributed.feature_count": context.size(-1)},
    )


def _record_embedding(model, context, targets, mask, rows, cache):
    """Snapshot summaries before later column writes can change them."""
    summaries = []
    with ExitStack() as patches:
        for module in model.model.row_embedding.col_blocks:
            column = cast(InducedTransformerBlock, module)
            original = column.summarize

            def record(value, *args, _original=original, **kwargs):
                # Clone now: retaining a view would conceal accidental writes.
                summaries.append(value.detach().clone())
                return _original(value, *args, **kwargs)

            patches.enter_context(patch.object(column, "summarize", record))
        embedding = model._embed_rows(context, targets, mask, rows, cache)
    return embedding, summaries


def _assert_recorded_caches_match(actual, expected):
    """Check values and the K/V layout required by the original replay path."""
    assert set(actual) == set(expected)
    for name, entry in actual.items():
        reference = expected[name]
        if isinstance(entry, KVCacheEntry):
            for tensor, reference_tensor in zip(entry, reference):
                torch.testing.assert_close(
                    tensor, reference_tensor, atol=1e-6, rtol=1e-5
                )
                assert tensor.shape == reference_tensor.shape
                assert tensor.stride() == reference_tensor.stride()
                assert tensor.is_contiguous()
        elif isinstance(entry, torch.Tensor):
            torch.testing.assert_close(entry, reference, atol=1e-6, rtol=1e-5)
        else:
            assert entry == reference


def _assert_replay_preserves_cache(
    sequential, pipeline, caches, targets, mask, rows
):
    """Replay must agree without replacing or modifying fitted K/V."""
    sequential_cache, pipeline_cache = caches
    sequential_cache.freeze()
    pipeline_cache.freeze()
    entries = {
        k: v for k, v in pipeline_cache.items() if isinstance(v, KVCacheEntry)
    }
    saved_values = {
        k: tuple(t.clone() for t in entry) for k, entry in entries.items()
    }
    query = torch.randn(1, 3, 500)
    no_query_targets = targets[..., :0]
    expected = sequential._embed_rows(
        query, no_query_targets, mask, rows, sequential_cache
    )
    first = pipeline._embed_rows(
        query, no_query_targets, mask, rows, pipeline_cache
    )
    repeat = pipeline._embed_rows(
        query, no_query_targets, mask, rows, pipeline_cache
    )
    torch.testing.assert_close(first, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(repeat, first, atol=1e-6, rtol=1e-5)
    for name, entry in entries.items():
        # KVCacheEntry is immutable: preserving it also preserves its tensors.
        assert pipeline_cache[name] is entry
        for tensor, saved in zip(entry, saved_values[name]):
            assert torch.equal(tensor, saved)


def _pipeline_worker(rank, world, rendezvous, rows):
    with distributed_session(rank, world, rendezvous):
        core = _small_embedding_model()
        mesh = TableMesh(layout="mesh", mesh_shape=(world, 1))
        sequential = _column_schedule(core, mesh, pipelined=False)
        pipeline = _column_schedule(core, mesh, pipelined=True)
        # Strided inputs expose accidental writes and dense-input assumptions.
        context = torch.randn(1, rows, 1000)[..., ::2]
        targets = torch.randn(1, rows * 2)[..., ::2]
        original_context, original_targets = context.clone(), targets.clone()
        mask = torch.zeros(1, 500, dtype=torch.bool)
        start, end = mesh.rows(rows)
        caches = [_recording_cache(context) for _ in range(2)]
        with torch.inference_mode():
            recordings = [
                _record_embedding(
                    model,
                    context[..., start:end, :],
                    targets[..., start:end],
                    mask,
                    rows,
                    cache,
                )
                for model, cache in zip((sequential, pipeline), caches)
            ]
            expected, expected_reads = recordings[0]
            actual, actual_reads = recordings[1]
            assert len(expected_reads) == len(actual_reads)
            for before, after in zip(expected_reads, actual_reads):
                torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            _assert_recorded_caches_match(caches[1], caches[0])
            assert torch.equal(context, original_context)
            assert torch.equal(targets, original_targets)
            _assert_replay_preserves_cache(
                sequential, pipeline, caches, targets, mask, rows
            )


@pytest.mark.parametrize(
    ("world", "rows"),
    [(1, 7), (2, 7), (4, 7), (4, 2), (1, 1250)],
    ids=[
        "singleton",
        "uneven-two-rank",
        "uneven-four-rank",
        "empty-ranks",
        "row-chunk-tail",
    ],
)
def test_recording_and_replay_match_sequential_schedule(world, rows):
    """Pipelining preserves summary reads, outputs, cache layout and replay."""
    # 1250 rows crosses 512-row chunks; 2 rows on 4 ranks creates empty owners.
    spawn_workers(_pipeline_worker, world, rows)


class _RecordedTicket:
    """Immediate one-rank summary result, replacing a CUDA/NCCL ticket."""

    def __init__(self, out, lse, group, stream):
        self.value = (out, lse)
        self.consumed = False


class _RecordedQueue:
    """Record the scheduler's transitions while enforcing one live result."""

    def __init__(self, events):
        self.events = events
        self.ticket = None
        self.poisoned = False

    def submit(self, factory, *args):
        if self.ticket is not None:
            assert self.ticket.consumed
        self.ticket = factory(*args)
        self.events.append("submit")

    def drain(self):
        assert self.ticket is not None
        assert not self.ticket.consumed
        self.ticket.consumed = True
        self.events.append("drain")
        return self.ticket.value

    def close(self):
        self.events.append("close")
        self.ticket = None


@pytest.fixture
def scheduled_pipeline(monkeypatch, tmp_path):
    """Replace GPU operations with observable tickets on a CPU model."""
    with distributed_session(0, 1, str(tmp_path / "schedule")):
        events, queues = [], []

        def make_queue():
            queue = _RecordedQueue(events)
            queues.append(queue)
            return queue

        # Enter the asynchronous scheduler branch only. Math stays on CPU;
        # these doubles cannot demonstrate CUDA correctness or actual overlap.
        monkeypatch.setattr(
            torch.Tensor, "is_cuda", property(lambda self: True)
        )
        monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: None)
        monkeypatch.setattr(
            parallel, "chunk_memory_limit", lambda device: 2**30
        )
        monkeypatch.setattr(parallel, "OneTicketQueue", make_queue)
        monkeypatch.setattr(parallel, "CudaSummaryTicket", _RecordedTicket)
        core = _small_embedding_model()
        model = _column_schedule(
            core, TableMesh("mesh", mesh_shape=(1, 1)), pipelined=True
        )
        model._fused_async = True
        for column in core.row_embedding.col_blocks:
            column.register_forward_pre_hook(
                lambda module, args: events.append("column")
            )
        # 130 features create three tiles (64 + 64 + 2) in each of two layers.
        context, targets = torch.randn(1, 2, 130), torch.randn(1, 2)
        mask = torch.zeros(1, 130, dtype=torch.bool)

        def record():
            with torch.inference_mode():
                return model._embed_rows(
                    context, targets, mask, 2, _recording_cache(context)
                )

        yield SimpleNamespace(
            model=model,
            first_column=core.row_embedding.col_blocks[0],
            events=events,
            queues=queues,
            record=record,
        )


def test_next_tile_is_submitted_before_previous_column_compute(
    scheduled_pipeline,
):
    """The schedule opens a compute window while retaining only one ticket."""
    case = scheduled_pipeline
    case.record()
    # Consume tile0, submit tile1, then apply column0; repeat for tile1.
    # The last tile is applied after draining/closing the queue. Do this twice,
    # once for each embedding layer. This order enables overlap on a GPU.
    per_layer = [
        "submit",
        "drain",
        "submit",
        "column",
        "drain",
        "submit",
        "column",
        "drain",
        "close",
        "column",
    ]
    assert case.events == per_layer * 2
    assert case.model._fused_async_queue is None


@pytest.mark.parametrize(
    "stage", ["induced_key_value", "forward", "cache_assembly"]
)
def test_failure_keeps_queue_poisoned_and_owned(
    scheduled_pipeline, monkeypatch, stage
):
    """A failed recording retains its queue and poisons further use."""
    case = scheduled_pipeline

    def fail(*args, **kwargs):
        raise ValueError("injected column pipeline failure")

    if stage == "cache_assembly":
        original_cat = torch.cat

        def assemble(values, *args, **kwargs):
            # Fail only when joining the three K/V tiles, not other cat calls.
            if (
                kwargs.get("dim") == -4
                and len(values) == 3
                and all(v.ndim == 5 for v in values)
            ):
                fail()
            return original_cat(values, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", assemble)
    else:
        monkeypatch.setattr(case.first_column, stage, fail)
    with pytest.raises(ValueError, match="injected column pipeline failure"):
        case.record()
    queue = case.queues[-1]
    assert case.model._fused_async_queue is queue
    assert queue.poisoned
    if stage != "cache_assembly":
        assert case.events[:3] == ["submit", "drain", "submit"]
        assert queue.ticket is not None
