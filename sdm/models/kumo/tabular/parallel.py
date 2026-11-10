# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor

from sdm._memory import chunk_memory_limit
from sdm.cache import Cache, KVCacheEntry
from sdm.distributed.attention import distributed_transformer_block
from sdm.distributed.layout import TableMesh
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.kumo.tabular.partition import PreparedContext
from sdm.nn import InducedTransformerBlock, TransformerBlock


class _DistributedKumoTabular(torch.nn.Module):
    def __init__(
        self,
        model: _KumoTabular,
        mesh: TableMesh,
        *,
        attention_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mesh = mesh
        self.prepared: PreparedContext | None = None
        self.attention_options = (
            {} if attention_options is None else attention_options
        )

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor,
        y: Tensor,
        categorical_mask: Tensor,
        *,
        cache: Cache | None = None,
        offload: Literal["cpu"] | None = None,
    ) -> Tensor:
        if offload is not None:
            raise ValueError("CPU offload requires single-GPU KumoTabular")
        if cache is None:
            raise RuntimeError("Use fit() followed by predict()")
        if cache.is_recording:
            if self.prepared is not None:
                prepared = self.prepared
                context_rows = prepared.context_rows
                row_start, _ = self.mesh.rows(context_rows)
                cache["distributed.context_rows"] = context_rows
                cache["distributed.feature_count"] = prepared.feature_count
                cache["distributed.feature_indices"] = (
                    prepared.feature_indices.to(x.device)
                )
                cache["cell_impute_mean"] = prepared.mean.to(x.device)
                y_local = y
            else:
                context_rows = x.size(-2)
                cache["distributed.context_rows"] = context_rows
                # Imputation uses the complete context before row partitioning.
                # Query values must not affect the fitted statistics.
                mean = x.nanmean(dim=-2, keepdim=True)
                cache["cell_impute_mean"] = mean.masked_fill(mean.isnan(), 0)
                row_start, row_end = self.mesh.rows(context_rows)
                x = x[..., row_start:row_end, :]
                y_local = y[..., row_start:row_end]
        else:
            context_rows = cast(int, cache["distributed.context_rows"])
            row_start = 0
            y_local = y
        rows = self._embed_rows(
            x, y_local, categorical_mask, context_rows, cache
        )
        rows = self.model.row_project(rows)
        if cache.is_recording:
            # Feature peers have the same row readouts. Divide them once more
            # so each ICL context position has exactly one owner globally.
            start, end = self.mesh.context(context_rows)
            rows = rows[
                ..., start - row_start : end - row_start, :
            ].contiguous()
            y_local = (
                y[..., start - row_start : end - row_start]
                if self.prepared is not None
                else y[..., start:end]
            )
        return self._icl(rows, y_local, context_rows, cache)

    def _embed_cells(
        self,
        x: Tensor,
        categorical_mask: Tensor,
        start: int,
        end: int,
        cache: Cache,
    ) -> Tensor:
        """Embed only owned positions, retaining global circular groups."""
        cell = self.model.row_embedding.cell_embedding
        features = cast(
            int, cache.get("distributed.feature_count", x.size(-1))
        )
        index = torch.arange(start, end, device=x.device).unsqueeze(-1)
        shifts = 2 ** torch.arange(cell.group_size, device=x.device) - 1
        index = (index + shifts) % features
        if "distributed.feature_indices" in cache:
            # Translate global circular groups into the received feature halo.
            index = torch.searchsorted(
                cast(Tensor, cache["distributed.feature_indices"]), index
            )
        impute_mean = cast(Tensor, cache["cell_impute_mean"])[..., index]
        x = x[..., index]
        mask = categorical_mask[..., None, index]
        frequency = torch.where(
            mask.unsqueeze(-1), cell.cat_freq.float(), cell.num_freq.float()
        )
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else cell.num_lin.weight.dtype
        )
        weight = torch.where(
            mask[..., None, None],
            cell.cat_lin.weight.to(dtype).view(
                *(1,) * mask.dim(), *cell.cat_lin.weight.shape
            ),
            cell.num_lin.weight.to(dtype).view(
                *(1,) * mask.dim(), *cell.num_lin.weight.shape
            ),
        )
        bias = (
            torch.where(
                mask[..., None],
                cell.cat_lin.bias.view(*(1,) * mask.dim(), -1),
                cell.num_lin.bias.view(*(1,) * mask.dim(), -1),
            )
            .sum(-2)
            .to(dtype)
        )
        out = x.new_empty((*x.shape[:-1], cell.channels), dtype=dtype)
        rows_per_chunk = max(x.size(-3), 1)
        if x.is_cuda:
            element_size = out.element_size()
            # Budget the float32 phases, Fourier features and projections;
            # the complete output and grouped inputs stay resident.
            bytes_per_cell = (
                cell.group_size * cell.num_frequencies * (4 + 2 * element_size)
                + 2 * cell.channels * element_size
            )
            fixed_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (frequency, weight, bias)
            )
            batch_size = math.prod(x.shape[:-3]) * (end - start)
            memory_limit = chunk_memory_limit(x.device) - fixed_bytes
            rows_per_chunk = max(
                1, memory_limit // max(batch_size * bytes_per_cell, 1)
            )
        for offset in range(0, x.size(-3), rows_per_chunk):
            cell._forward(
                x[..., offset : offset + rows_per_chunk, :, :],
                frequency,
                weight,
                bias,
                impute_mean=impute_mean,
                out=out[..., offset : offset + rows_per_chunk, :, :],
            )
        return out

    def _embed_rows(
        self,
        x: Tensor,
        y: Tensor,
        categorical_mask: Tensor,
        context_rows: int,
        cache: Cache,
    ) -> Tensor:
        embedding = self.model.row_embedding
        feature_count = cast(
            int, cache.get("distributed.feature_count", x.size(-1))
        )
        feature_start, feature_end = self.mesh.features(feature_count)
        cells = self._embed_cells(
            x, categorical_mask, feature_start, feature_end, cache
        )
        if y.numel():
            if embedding.y_emb is not None:
                target = embedding.y_emb(y)
            else:
                assert embedding.y_lin is not None
                target = embedding.y_lin(y.unsqueeze(-1))
            cells += target.unsqueeze(-2).to(cells.dtype)
            del target
        readout_count = embedding.readout_token.size(-2)
        local_readout_count = (
            readout_count if self.mesh.feature_rank == 0 else 0
        )
        if self.mesh.feature_rank == 0:
            readout = (
                embedding.readout_token.to(cells.dtype)
                .expand(*cells.shape[:-2], readout_count, embedding.channels)
                .clone()
            )
        else:
            readout = cells[..., :0, :]
        positions = torch.arange(
            readout_count + feature_start,
            readout_count + feature_end,
            device=x.device,
        )
        if self.mesh.feature_rank == 0:
            positions = torch.cat(
                [torch.arange(readout_count, device=x.device), positions]
            )
        for index, (column_module, row_module) in enumerate(
            zip(embedding.col_blocks, embedding.row_blocks)
        ):
            column = cast(InducedTransformerBlock, column_module)
            row = cast(TransformerBlock, row_module)
            column_key = f"row_embedding.col_block{index}"
            if feature_end > feature_start:
                column_input = cells.transpose(-3, -2)
                if cache.is_replaying:
                    cells = column(
                        query=column_input,
                        key_value=cast(KVCacheEntry, cache[column_key]),
                        batch_size_limit="auto",
                    ).transpose(-3, -2)
                else:
                    if self.mesh.row_parts == 1:
                        induced = column.inducing_block(
                            query=column.inducing_points,
                            key_value=column_input,
                            batch_size_limit="auto",
                        )
                    else:
                        induced = distributed_transformer_block(
                            column.inducing_block,
                            query=column.inducing_points,
                            key_value=column_input,
                            group=self.mesh.row_group,
                            global_key_length=context_rows,
                            replicated_query=True,
                            return_key_value=False,
                            **self.attention_options,
                        )
                    cells, entry = column.output_block(
                        query=column_input,
                        key_value=induced,
                        return_key_value=True,
                        batch_size_limit="auto",
                    )
                    cache[column_key] = entry
                    cells = cells.transpose(-3, -2)
                del column_input
            tokens = torch.cat([readout, cells], dim=-2)
            # Release the previous cell allocation before projecting attention.
            cells = tokens[..., :0, :]
            del readout
            final_layer = index == len(embedding.row_blocks) - 1
            result = distributed_transformer_block(
                row,
                query=tokens[..., :local_readout_count, :]
                if final_layer
                else tokens,
                key_value=tokens,
                group=self.mesh.feature_group,
                global_key_length=readout_count + feature_count,
                query_positions=positions[:local_readout_count]
                if final_layer
                else positions,
                key_positions=positions,
                return_key_value=False,
                **self.attention_options,
            )
            if final_layer:
                readout = result
                cells = result
            else:
                readout = result[..., :local_readout_count, :].clone()
                cells = result[..., local_readout_count:, :]
            del result, tokens
        if self.mesh.feature_rank != 0:
            readout = readout.new_empty(
                (*x.shape[:-1], readout_count, embedding.channels)
            )
        readout = readout.contiguous()
        dist.broadcast(
            readout,
            src=self.mesh.readout_source,
            group=self.mesh.feature_group,
        )
        return embedding.norm(readout).flatten(-2)

    def _icl(
        self, x: Tensor, y: Tensor, context_rows: int, cache: Cache
    ) -> Tensor:
        icl = self.model.icl_block
        if y.numel():
            if icl.y_emb is not None:
                target = icl.y_emb(y)
            else:
                assert icl.y_lin is not None
                target = icl.y_lin(y.unsqueeze(-1))
            x += target.to(x.dtype)
            del target
        for index, module in enumerate(icl.layers):
            layer = cast(TransformerBlock, module)
            key = f"icl_block.layer{index}"
            if cache.is_recording:
                # The final fit layer only records context K/V.
                x, entry = distributed_transformer_block(
                    layer,
                    query=x[..., :0, :] if index == len(icl.layers) - 1 else x,
                    key_value=x,
                    group=self.mesh.group,
                    global_key_length=context_rows,
                    return_key_value=True,
                    **self.attention_options,
                )
                heads = icl.kv_heads
                cache[key] = (
                    entry
                    if heads is None
                    else KVCacheEntry(
                        entry.key[..., :heads, :].clone(
                            memory_format=torch.contiguous_format
                        ),
                        entry.value[..., :heads, :].clone(
                            memory_format=torch.contiguous_format
                        ),
                    )
                )
                del entry
            else:
                x = distributed_transformer_block(
                    layer,
                    query=x,
                    key_value=cast(KVCacheEntry, cache[key]),
                    group=self.mesh.group,
                    global_key_length=context_rows,
                    replicated_query=True,
                    return_key_value=False,
                    **self.attention_options,
                )
        return icl.head(icl.norm(x))
