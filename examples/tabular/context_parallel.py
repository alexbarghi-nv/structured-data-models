# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Context-parallel fit and repeated prediction from source-host inputs.

Run on two, four or eight GPUs, for example::

    torchrun --standalone --nproc-per-node=4 \
        examples/tabular/context_parallel.py

Only rank zero loads the raw inputs. Each GPU preprocesses its owned columns;
row/feature activations and fitted context K/V are partitioned. All ranks call
fit and predict in the same order, even when only rank zero uses the result.
"""

import os

import torch
import torch.distributed as dist

import sdm


def main() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        model = sdm.models.ContextParallelKumoTabular(
            task="regression", size="small", device=device, layout="mesh"
        )
        context = targets = query = None
        if dist.get_rank() == 0:
            context = torch.arange(128 * 8).reshape(128, 8).float()
            query = torch.arange(16 * 8).reshape(16, 8).float().sin()
            targets = context[:, :1].cos()
            context = context.sin()
        model.fit(context, targets, num_estimators=2)
        for start in range(0, 16, 8):
            batch = query[start : start + 8] if query is not None else None
            predictions = model.predict(batch)
            if dist.get_rank() == 0:
                print(predictions.numerical.cpu())
        model.clear()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
