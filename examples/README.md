# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples
- [**`timeseries/`**](timeseries/): Time series foundation model examples

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.

See [`tabular/context_parallel.py`](tabular/context_parallel.py) for collective context-parallel KumoTabular fit/predict with column-owned CUDA preprocessing.

See [`tabular/distributed_preprocessing.py`](tabular/distributed_preprocessing.py) for source-host preprocessing, or [`tabular/distributed_column_shards.py`](tabular/distributed_column_shards.py) for loading per-worker raw column files.
