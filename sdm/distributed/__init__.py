# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference primitives for models with distributed activation shards."""

from sdm.distributed.attention import (
    sharded_attention,
    distributed_transformer_block,
)

__all__ = ["sharded_attention", "distributed_transformer_block"]
