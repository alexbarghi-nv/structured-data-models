# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Source-only loading and bounded distributed preprocessing.

Run, for example::

    torchrun --standalone --nproc-per-node=4 \\
        examples/tabular/distributed_preprocessing.py

Complete raw columns are distributed to preprocessing owners before fitting
Kumo's default recipe on CUDA. Typed tensors use the existing NCCL model
communicator. Only owned embedding rows, features and their circular halo
transfer to the model. Query statistics reuse the fitted context state.
"""

import os

import torch
import torch.distributed as dist

import sdm
from sdm.models.kumo.tabular import ContextParallelKumoTabular


def main() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        model = ContextParallelKumoTabular(
            task="classification",
            size="small",
            device=device,
            layout="mesh",
        )
        context = targets = query = None
        if dist.get_rank() == 0:
            # Replace these host tensors with source-only data loading.
            context = sdm.TableTensor.from_tensor(torch.randn(1024, 256))
            targets = torch.randint(2, (1024, 1))
            query = sdm.TableTensor.from_tensor(torch.randn(16, 256))
        model.fit(
            context,
            targets,
            num_estimators=8,
            generator=torch.Generator(device=device).manual_seed(37),
        )
        predictions = model.predict(query)
        if dist.get_rank() == 0:
            print(predictions.numerical.cpu())
        model.clear()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
