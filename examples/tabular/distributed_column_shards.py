# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Load each worker's existing numerical shard files directly.

Run, for example::

    torchrun --standalone --nproc-per-node=4 \
        examples/tabular/distributed_column_shards.py /path/to/shards

The directory contains a schema.json manifest plus
context.rankN.npz and query.rankN.npz for every worker. Each NPZ contains only
that worker's raw feature columns in ``values``, their ``column_ids``, and the
actual record identities in ``row_ids``. Context files also contain identical
``targets`` or just the source context file contains targets. Empty owners use
values shaped (rows, 0). All numerical arrays use the same floating dtype.
Classifier targets are integer arrays shaped (rows, 1).

The manifest contains {"numerical": ["feature0", ...]}. IDs index these names.
Context and query files retain the same column ownership. Row IDs agree in
order across workers within each cohort; context and query cohorts can differ.
Files must already be partitioned: this example never opens a full data file.
Categorical inputs can use ColumnShard with a TableTensor carrying the local
column vocabularies instead of this numerical-only NPZ representation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import sdm
from sdm.models.kumo.tabular import ColumnShard, DistributedKumoTabular


def load_shard(
    path: Path,
    columns: dict[str, list[str]],
) -> tuple[ColumnShard, torch.Tensor | None]:
    with np.load(path, allow_pickle=False) as arrays:
        ids = tuple(int(index) for index in arrays["column_ids"])
        shard = ColumnShard(
            table=sdm.TableTensor.from_tensor(
                torch.from_numpy(arrays["values"]),
                columns=[columns["numerical"][index] for index in ids],
            ),
            column_ids=ids,
            columns=columns,
            row_ids=torch.from_numpy(arrays["row_ids"]),
        )
        targets = (
            torch.from_numpy(arrays["targets"])
            if "targets" in arrays
            else None
        )
    return shard, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    # Each worker opens only its own context and query data files.
    columns = json.loads((args.directory / "schema.json").read_text())
    context, targets = load_shard(
        args.directory / f"context.rank{rank}.npz", columns
    )
    query, _ = load_shard(args.directory / f"query.rank{rank}.npz", columns)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        model = DistributedKumoTabular(
            task="classification",
            size="small",
            device=device,
            layout="mesh",
            preprocessing="sharded_cuda",
        )
        model.fit(
            context,
            targets,
            num_estimators=8,
            generator=torch.Generator(device=device).manual_seed(37),
        )
        probabilities = model.predict(query)
        if rank == 0:
            print(probabilities.numerical.cpu())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
