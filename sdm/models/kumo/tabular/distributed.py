# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, ClassVar, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from sdm import Recipe, RelatedTables, TableTensor, TaskLike
from sdm.cache import Cache
from sdm.distributed.layout import TableMesh
from sdm.models.base import Callback
from sdm.models.kumo.tabular.input import ColumnShard
from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular
from sdm.models.kumo.tabular.parallel import _DistributedKumoTabular
from sdm.models.kumo.tabular.prepared import PreparedInputs, Preprocessing
from sdm.models.kumo.tabular.sharded_offload import ShardedColumnOffload


class DistributedKumoTabular(KumoTabular):
    """KumoTabular inference with sharded activations and context K/V.

    Initialize ``torch.distributed`` first. Every rank uses the same recipe
    and fit options. The default path replicates raw inputs and preprocessing;
    optional preprocessing modes distribute source-host or preloaded columns.
    Call ``fit`` and ``predict`` collectively, in the same order. Weights are
    replicated and frozen, while context K/V is partitioned across ranks.
    CPU activation offload is a separate single-GPU :class:`KumoTabular` path.

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
        preprocessing: ``"replicated"`` keeps native preprocessing. Optional
            ``"rank0_cpu"`` fits a recipe once on the source host;
            ``"sharded_cpu"`` and ``"sharded_cuda"`` fit complete raw columns
            on their owners using the default recipe. Supply host inputs only
            on the source rank (``None`` on peers), or a ``ColumnShard`` on
            every rank for sharded modes. Only owned model rows and feature
            halos transfer to model devices. All learned state uses context
            rows only. Nonreplicated modes run estimators individually and
            offload fitted caches to CPU when there is more than one.
            Sharded modes use typed transport on the model communicator;
            rank0_cpu uses Gloo. Callbacks and related tables
            require the replicated path.
        query_chunk_size: Queries per attention score tile.
        key_chunk_size: Keys per attention score and K/V broadcast tile.
        batch_chunk_size: Flattened batch entries per attention tile.
        projection_batch_size: Batch entries per projection/output/MLP chunk.
            ``"auto"`` uses the original block's memory budget and agrees on
            the minimum limit across the participating ranks.
        sharded_column_offload: Experimental, default-off pure-row storage of
            one column K/V shard per rank on CPU for GPU reconstruction.
            Requires nonreplicated preprocessing and session-local caches.
    """

    supports_offload: ClassVar[bool] = False

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
        preprocessing: Preprocessing = "replicated",
        query_chunk_size: int = 128,
        key_chunk_size: int = 256,
        batch_chunk_size: int = 16,
        projection_batch_size: int | Literal["auto"] = "auto",
        sharded_column_offload: bool = False,
    ) -> None:
        mesh = TableMesh(layout, process_group, mesh_shape=mesh_shape)
        if sharded_column_offload and (
            mesh.feature_parts != 1 or preprocessing == "replicated"
        ):
            raise ValueError(
                "Sharded column offload requires pure-row "
                "distributed preprocessing"
            )
        super().__init__(
            task=task, size=size, pretrained=pretrained, device=device
        )
        self.mesh = mesh
        self.preprocessing = preprocessing
        self._prepared_inputs: PreparedInputs | None = None
        self._column_offload = (
            ShardedColumnOffload(mesh) if sharded_column_offload else None
        )
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
            "preprocessing": self.preprocessing,
            "source_input_rank": self.mesh.source,
        }

    def forward(self, *args: Any, **kwargs: Any) -> TableTensor:
        """Use fit/predict when preprocessing is distributed."""
        if self.preprocessing != "replicated":
            raise RuntimeError(
                "Distributed preprocessing uses fit() followed by predict()"
            )
        return super().forward(*args, **kwargs)

    def fit(
        self,
        x: Tensor | TableTensor | ColumnShard | None,
        y: Tensor | TableTensor | None,
        *args: Any,
        generator: torch.Generator | None = None,
        offload: Literal["cpu"] | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the original recipe and replace this rank's frozen KV cache."""
        self._validate_offload(offload, next(self.parameters()).device)
        if self.preprocessing != "replicated":
            self._fit_prepared(x, y, *args, generator=generator, **kwargs)
            return
        if isinstance(x, ColumnShard):
            raise ValueError(
                "ColumnShard requires sharded_cpu or sharded_cuda"
            )
        assert x is not None
        assert y is not None
        generator = self._synchronize_generator(x.device, generator)
        super().fit(x, y, *args, generator=generator, **kwargs)

    @torch.inference_mode()
    def _fit_prepared(
        self,
        x: Tensor | TableTensor | ColumnShard | None,
        y: Tensor | TableTensor | None,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        estimator_batch_size: int | None = 1,
        context_only_columns: Sequence[str] | None = None,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        if context_only_columns:
            raise ValueError(
                "DistributedKumoTabular does not support context-only columns"
            )
        if related_tables is not None or callbacks:
            raise ValueError(
                "Distributed preprocessing does not support related tables "
                "or callbacks"
            )
        self.clear()
        device = next(self.parameters()).device
        generator = self._synchronize_generator(
            device if generator is None else generator.device, generator
        )
        model = cast(_DistributedKumoTabular, next(iter(self.models.values())))
        prepared = PreparedInputs(
            self.mesh,
            self.preprocessing,
            device,
            model.model.row_embedding.cell_embedding.group_size,
        )
        members = prepared.fit(
            x,
            y,
            self.default_recipe() if recipe is None else recipe,
            num_estimators,
            generator,
        )
        # Preprocessing draws happen once on the source. Continue from that
        # exact state for model-side stochastic operations such as ECOC.
        generator = self._synchronize_generator(generator.device, generator)
        if generator.device != device:
            seed = torch.empty((), dtype=torch.int64, device=generator.device)
            seed.random_(generator=generator)
            generator = torch.Generator(device=device).manual_seed(
                int(seed.item())
            )
        cache = Cache(kwargs=kwargs, num_batches=len(members))
        for index, member in enumerate(members):
            context = member.context
            self._validate_context(
                x=context.x,
                y=context.y,
                x_context_only=None,
                related_tables=None,
            )
            context = context._replace(
                x=cast(TableTensor, context.x.to(device)),
                y=cast(TableTensor, context.y.to(device)),
            )
            mask = member.categorical_mask.to(device)
            batch_cache = Cache(
                classes=(
                    context.y.categorical.categories[0]
                    if context.y.categorical.size(-1)
                    else None
                ),
                categorical_mask=mask,
            )
            for module in self.models.values():
                cast(_DistributedKumoTabular, module).prepared = member
            try:
                self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=None,
                    x_context_only=None,
                    related_context_tables=None,
                    related_query_tables=None,
                    cache=batch_cache,
                    generator=generator,
                    categorical_mask=mask,
                    **kwargs,
                )
            finally:
                for module in self.models.values():
                    cast(_DistributedKumoTabular, module).prepared = None
            cache[index] = (
                self._column_offload.offload(batch_cache, index)
                if self._column_offload is not None
                else (
                    batch_cache.to("cpu")
                    if device.type == "cuda" and len(members) > 1
                    else batch_cache
                )
            )
        self._cache = cache.freeze()
        self._prepared_inputs = prepared

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor | ColumnShard | None,
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        """Reuse context state; new modes accept source-only host queries."""
        if self.preprocessing == "replicated":
            if isinstance(x, ColumnShard):
                raise ValueError(
                    "ColumnShard requires sharded_cpu or sharded_cuda"
                )
            assert x is not None
            return super().predict(x, related_tables, callbacks=callbacks)
        if self.training:
            raise RuntimeError("predict() requires model.eval()")
        if self._cache is None or self._prepared_inputs is None:
            raise RuntimeError("Call fit() before predict()")
        if related_tables is not None or callbacks:
            raise ValueError(
                "Distributed preprocessing does not support related tables "
                "or callbacks"
            )
        device = next(self.parameters()).device
        queries = self._prepared_inputs.transform(x)
        outputs = []
        for index, query in enumerate(queries):
            stored = cast(Cache, self._cache[index])
            cache = (
                self._column_offload.restore(stored, index, device)
                if self._column_offload is not None
                else stored.to(device)
            )
            output = self._forward(
                x_context=None,
                y_context=None,
                x_query=cast(TableTensor, query.x.to(device)),
                x_context_only=None,
                related_context_tables=None,
                related_query_tables=None,
                cache=cache,
                generator=None,
                categorical_mask=cast(Tensor, cache["categorical_mask"]),
                **cast(dict[str, Any], self._cache["kwargs"]),
            )
            outputs.append(cast(TableTensor, output.to(query.x.dtype)))
        return self._prepared_inputs.output(
            outputs, cast(Cache, self._cache[0])["classes"] is None
        )

    def clear(self) -> None:
        """Release fitted context and preprocessing state."""
        super().clear()
        self._prepared_inputs = None
        column_offload = getattr(self, "_column_offload", None)
        if column_offload is not None:
            column_offload.clear()

    def _synchronize_generator(
        self, device: torch.device, generator: torch.Generator | None
    ) -> torch.Generator:
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )
        state = generator.get_state()
        communication_device = (
            next(self.parameters()).device
            if dist.get_backend(self.mesh.group) == "nccl"
            else "cpu"
        )
        state = state.to(communication_device)
        dist.broadcast(state, src=self.mesh.source, group=self.mesh.group)
        generator.set_state(state.cpu())
        return generator
