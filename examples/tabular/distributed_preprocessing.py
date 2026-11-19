# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Source-only loading and bounded distributed preprocessing.

Run, for example::

    torchrun --standalone --nproc-per-node=4 \\
        examples/tabular/distributed_preprocessing.py \
        --preprocessing sharded_cpu

``rank0_cpu`` fits the recipe once on the source host. ``sharded_cuda`` and
``sharded_cpu`` distribute complete raw columns to preprocessing owners before
fitting the default recipe. All modes send only owned embedding rows, features
and their circular halo to the model. Sharded modes use typed tensors on the
model communicator (NCCL on CUDA); the legacy rank0_cpu path uses Gloo for host
transport. Query statistics always reuse the fitted context state.
"""

import argparse
import os

import torch
import torch.distributed as dist

import sdm
from sdm.models.kumo.tabular import DistributedKumoTabular


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessing",
        choices=("rank0_cpu", "sharded_cuda", "sharded_cpu"),
        default="rank0_cpu",
    )
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        model = DistributedKumoTabular(
            task="classification",
            size="small",
            device=device,
            layout="mesh",
            preprocessing=args.preprocessing,
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
