# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor

from sdm import Stype, TableTensor


@dataclass(frozen=True)
class ColumnShard:
    """One worker's complete raw columns over an agreed row cohort.

    Use with ``preprocessing="sharded_cpu"`` or ``"sharded_cuda"``. Every
    worker passes a shard, including workers owning no columns. Shards stay
    on CPU until collective ownership checks finish. No worker needs the
    complete feature table.

    Args:
        table: Local host feature columns. All owners must use the same
            numerical dtype, including empty numerical blocks. Categorical
            vocabularies belong to their column owner and need not be copied
            to peers. Complete columns preserve exact context statistics.
        column_ids: Global IDs corresponding to local numerical columns,
            then local categorical columns, in their table order. The global
            IDs index ``columns`` in numerical-then-categorical order.
            Across workers, each global ID must appear exactly once.
        columns: Complete global feature names grouped by semantic type.
            Every worker must provide the identical schema. Numerical and
            categorical features are supported.
        row_ids: CPU int64 IDs shaped like ``table.shape[:-1]``. IDs must be
            unique within each row axis and match every peer exactly in
            order. They identify the actual source records, not positions
            assigned independently after loading. Query IDs describe the
            query cohort and need not equal the fitted context IDs.

    Pass targets on the source worker, or identical host targets on every
    worker that has them. Supplied targets must follow that worker's row IDs.
    Replicated targets must have identical values, dtype, names and category
    coding/vocabulary. Validation is collective at ``fit``/``predict`` so a
    bad shard fails on all workers before GPU loading or feature exchange.
    """

    table: TableTensor
    column_ids: tuple[int, ...]
    columns: Mapping[Stype, Sequence[str]] | Mapping[str, Sequence[str]]
    row_ids: Tensor
