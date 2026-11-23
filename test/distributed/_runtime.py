# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process setup for CPU correctness and explicit CUDA tests."""

import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@contextmanager
def distributed_session(
    rank: int,
    world: int,
    rendezvous: str,
    *,
    backend: str = "gloo",
    timeout_seconds: int = 90,
) -> Iterator[torch.device]:
    """Give each worker one bounded process group and always tear it down.

    Gloo allows the algorithm tests to run on CPUs. NCCL is selected only by
    CUDA tests; neither this helper nor the CPU tests simulate GPU overlap.
    """
    torch.set_num_threads(1)
    device = (
        torch.device("cuda", rank)
        if backend == "nccl"
        else torch.device("cpu")
    )
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        yield device
    finally:
        dist.destroy_process_group()


def spawn_workers(worker: Callable[..., None], world: int, *args: Any) -> None:
    """Run worker(rank, world, rendezvous, *args) in isolated processes."""
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            worker,
            args=(world, str(Path(directory) / "rendezvous"), *args),
            nprocs=world,
            join=True,
        )


@contextmanager
def forbid_object_collectives() -> Iterator[None]:
    """Fail if a tensor-only code path accidentally falls back to pickle."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Object collective reached on typed path")

    with (
        patch.object(dist, "all_gather_object", forbidden),
        patch.object(dist, "broadcast_object_list", forbidden),
        patch.object(dist, "scatter_object_list", forbidden),
    ):
        yield
