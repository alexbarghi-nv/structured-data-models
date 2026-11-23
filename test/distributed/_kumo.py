# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared small-model setup for context-parallel integration tests."""

import torch

from sdm import TableTensor
from sdm.models.kumo.tabular import ContextParallelKumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS

TINY_MODEL_KWARGS = {
    "cell_channels": 8,
    "num_embedding_layers": 2,
    "num_embedding_heads": 2,
    "num_inducing_points": 3,
    "group_size": 3,
    "num_frequencies": 2,
    "num_readout_tokens": 2,
    "icl_channels": 16,
    "num_icl_layers": 3,
    "num_icl_heads": 4,
    "num_icl_key_value_heads_for_query": 2,
}


def check_public_offload(rank: int, world: int, device: torch.device) -> None:
    """Cache sharding preserves first/repeat public predictions exactly."""
    original = MODEL_KWARGS["small"]
    MODEL_KWARGS["small"] = TINY_MODEL_KWARGS
    try:
        x = TableTensor.from_tensor(
            torch.arange(19 * 17).view(19, 17).float().sin()
        )
        query = TableTensor.from_tensor(
            torch.arange(51).view(3, 17).float().cos()
        )
        x.numerical[2, 0] = torch.nan
        for task in ("regression", "classification"):
            models = [
                ContextParallelKumoTabular(
                    task=task,
                    size="small",
                    pretrained=False,
                    device=device,
                    layout="mesh",
                    mesh_shape=(world, 1),
                    sharded_column_offload=enabled,
                )
                for enabled in (False, True)
            ]
            models[1].load_state_dict(models[0].state_dict())
            outputs = []
            try:
                y = torch.arange(19).view(-1, 1)
                y = y % 3 if task == "classification" else y.float().sin()
                for model in models:
                    model.fit(
                        x if rank == 0 else None,
                        y if rank == 0 else None,
                        num_estimators=2,
                        # Identical draws isolate cache storage from sampling.
                        generator=torch.Generator(device=device).manual_seed(
                            37
                        ),
                    )
                    first = model.predict(query if rank == 0 else None)
                    repeat = model.predict(query if rank == 0 else None)
                    torch.testing.assert_close(
                        first.numerical, repeat.numerical, atol=0, rtol=0
                    )
                    assert first.columns == repeat.columns
                    outputs.append(first.numerical.detach().cpu().clone())
                torch.testing.assert_close(
                    outputs[0], outputs[1], atol=1e-6, rtol=1e-5
                )
                if device.type == "cpu":
                    assert torch.equal(outputs[0], outputs[1])
            finally:
                for model in models:
                    model.clear()
    finally:
        MODEL_KWARGS["small"] = original
