# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Context-parallel fit and repeated prediction with replicated preprocessing.

Run on two, four or eight GPUs, for example::

    torchrun --standalone --nproc-per-node=4 \
        examples/tabular/context_parallel.py

Every rank loads identical inputs. Weights and preprocessing are replicated;
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
        model = sdm.models.DistributedKumoTabular(
            task="regression", size="small", device=device, layout="mesh"
        )
        context = torch.arange(128 * 8, device=device).reshape(128, 8).float()
        query = torch.arange(16 * 8, device=device).reshape(16, 8).float()
        targets = context[:, :1].cos()
        model.fit(context.sin(), targets, num_estimators=2)
        for batch in query.sin().split(8):
            predictions = model.predict(batch)
            if dist.get_rank() == 0:
                print(predictions.numerical.cpu())
        model.clear()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
