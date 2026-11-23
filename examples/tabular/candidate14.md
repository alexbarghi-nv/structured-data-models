# Candidate14 distributed inference

This snapshot includes Seunghwa Kang’s [CPU offloading and chunked row-embedding PR #1319](https://github.com/NVIDIA/structured-data-models/pull/1319), through commit `48a2c39a8ad59da56164b3bbc964ec1bb40a01ab`, with context-parallel inference adapted to its token buffers and chunked attention interfaces.

Candidate14 adds typed preprocessing transport and builds on column-owned preprocessing, tiled column execution and sharded cache storage. The `sharded_cuda` mode uses the default recipe on each owner's model device and typed CUDA tensors over the existing NCCL communicator. Custom recipes use the replicated preprocessing path. CPU/Gloo tests exercise the internal tensor protocol; CUDA sharded execution does not create a companion Gloo group.

## Enabling the measured configuration

Initialize a process group, select the local CUDA device, and construct the model collectively. The measured configuration uses a pure-row mesh, the default recipe, FP32 inference, eight estimators and these experimental options:

```python
model = DistributedKumoTabular(
    task="classification",
    size="large",
    device=device,
    layout="mesh",
    mesh_shape=(dist.get_world_size(), 1),
    query_chunk_size=512,
    key_chunk_size=2048,
    batch_chunk_size=128,
    projection_batch_size="auto",
    preprocessing="sharded_cuda",
    sharded_column_offload=True,
)
for module in model.models.values():
    module._fused_pure_row = True
    module._fused_replay_original = True
    module._fused_chunk_rows = 512
    module._fused_feature_tile = 64
    module._fused_async = True
    module._fused_column_pipeline = True
```

The private fused flags and sharded cache offload are default-off; the default constructor alone does not reproduce the benchmark configuration. See `distributed_preprocessing.py` for collective fit/predict and process-group lifecycle, or `distributed_column_shards.py` for preloaded column-owner inputs. All ranks must use the same options and collective call order.

The tiled path uses a separate CUDA communication stream with explicit events and one outstanding summary-merge ticket. Row attention waits for all feature summaries. Fit records the cache; prediction uses the original embedding replay path. Sharded column offload copies only each rank’s portion of replicated column K/V to CPU and reconstructs it on GPU for replay. This is synchronous cache offload, distinct from PR #1319’s native CPU activation offload. Distributed `fit(offload="cpu")` remains unsupported, and fitted sharded caches are session-local.

## Validation scope

The benchmarked snapshot at commit `a8b30b31399268b2c4168c546477f5db256f1def` matched all 211 entries of candidate14’s manifest, SHA256 `98a7202e43766023951f6cccad6df83192d61b51eb5689f9a12de281caf09852`. This PR also includes a review cleanup that shares the row-pass implementation between fused schedules, clarifies ownership/types and removes duplicate code. The cleanup has separate CPU source-parity checks; GPU correctness and performance still need validation on the cleaned source. The timings below apply to the benchmarked snapshot. Measurement scripts, traces and result archives stay outside this PR.

The accepted DGX19 batch used 10,000 rows, 500 effective features, eight estimators and 512 query rows on eight V100 GPUs. Four balanced fresh-process pairs used equally populated compiled-kernel disk caches without model/KV prefit. Against candidate09, fit improved in all four pairs: median paired reduction 2.465%, with phase medians 25.615s to 24.953s. First and repeat prediction median paired reductions were 10.287% and 7.128%. These are maximum-rank phase timings, not end-to-end runtime or a native-single-GPU scaling result.

All 508 same-world prediction comparisons matched exactly. Historical native comparisons still fail strict equivalence: 43/1024 elements at eight GPUs, maximum absolute difference 8.613e-6, with zero class changes. Two GPUs showed 39/1024 and 6.378e-6. Tolerances remain `atol=1e-6, rtol=1e-5`; class agreement does not replace strict numerical validation.

The newer asynchronous cache-offload experiment is excluded. H100 performance and a fresh complete candidate14 1/2/4/8 scaling matrix are not established by these results.

See [the review guide](candidate14-review.md) for the execution path and invariants.
