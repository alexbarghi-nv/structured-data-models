# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from itertools import chain
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from sdm import TableTensor, TaskLike
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular
from sdm.models.kumo.tabular.parallel import _DistributedKumoTabular


class DistributedKumoTabular(KumoTabular):
    """KumoTabular inference with sharded activations and context K/V.

    Initialize ``torch.distributed`` first. Every rank supplies the same raw
    inputs, recipe and fit options; preprocessing remains the native replicated
    path. Call ``fit`` and ``predict`` collectively, in the same order. Weights
    are replicated and frozen, while context K/V is partitioned across ranks.

    Args:
        task: Tasks to initialize, as in :class:`KumoTabular`.
        size: Pretrained model size.
        pretrained: Load the original pretrained checkpoint.
        device: Local device for this rank's model parameters.
        layout: ``"hybrid"`` uses two feature owners. ``"mesh"`` partitions
            rows as well; four ranks default to (2,2), eight to (4,2).
        process_group: Parent communicator; defaults to the world group.
            The caller owns the distributed session and destroys it after
            all models finish. Child groups are shared within this session.
        mesh_shape: Optional (row_parts,feature_parts) factors for ``"mesh"``.
        query_chunk_size: Queries per attention score tile.
        key_chunk_size: Keys per attention score and K/V broadcast tile.
        batch_chunk_size: Flattened batch entries per attention tile.
        projection_batch_size: Batch entries per projection/output/MLP chunk.
            ``"auto"`` uses the original block's memory budget and agrees on
            the minimum limit across the participating ranks.
    """

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "medium", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
        *,
        layout: Literal["hybrid", "mesh"] = "hybrid",
        process_group: ProcessGroup | None = None,
        mesh_shape: tuple[int, int] | None = None,
        query_chunk_size: int = 128,
        key_chunk_size: int = 256,
        batch_chunk_size: int = 16,
        projection_batch_size: int | Literal["auto"] = "auto",
    ) -> None:
        mesh = TableMesh(layout, process_group, mesh_shape=mesh_shape)
        super().__init__(
            task=task, size=size, pretrained=pretrained, device=device
        )
        self.mesh = mesh
        self.requires_grad_(False)
        # Also synchronize random initialization and noncontiguous buffers;
        # inference must use the same parameters on every rank.
        for tensor in chain(self.parameters(), self.buffers()):
            contiguous = tensor.contiguous()
            dist.broadcast(contiguous, src=mesh.source, group=mesh.group)
            if not tensor.is_contiguous():
                tensor.copy_(contiguous)
        attention_options = {
            "query_chunk_size": query_chunk_size,
            "key_chunk_size": key_chunk_size,
            "batch_chunk_size": batch_chunk_size,
            "projection_batch_size": projection_batch_size,
        }
        for name, model in tuple(self.models.items()):
            self.models[name] = _DistributedKumoTabular(
                cast(_KumoTabular, model),
                mesh,
                attention_options=attention_options,
            )
        self.eval()

    @property
    def distributed_layout(self) -> dict[str, Any]:
        """Describe activation and context ownership in the parent group."""
        return {
            "mesh_shape": (self.mesh.row_parts, self.mesh.feature_parts),
            "embedding_row_parts": self.mesh.row_parts,
            "embedding_feature_parts": self.mesh.feature_parts,
            "icl_context_parts": self.mesh.world_size,
            "replicated_weights": True,
        }

    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        *args: Any,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit with identical recipe random choices on every rank."""
        if generator is None:
            generator = torch.Generator(device=x.device)
            seed = torch.empty((), dtype=torch.int64, device=x.device)
            seed.random_()
            generator.manual_seed(int(seed.item()))
        device = (
            next(self.parameters()).device
            if dist.get_backend(self.mesh.group) == "nccl"
            else "cpu"
        )
        state = generator.get_state().to(device)
        dist.broadcast(state, src=self.mesh.source, group=self.mesh.group)
        # Generator state is a CPU byte tensor; this synchronization is at the
        # fit boundary, before the original recipe draws random choices.
        generator.set_state(state.cpu())
        super().fit(x, y, *args, generator=generator, **kwargs)
