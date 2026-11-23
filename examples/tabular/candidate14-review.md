# Reviewing candidate14

The public entry point is `DistributedKumoTabular`. Its default path preserves replicated preprocessing; `sharded_cuda` opts into column-owned preprocessing and typed transport. The measured pure-row configuration and private fused flags are listed in [candidate14.md](candidate14.md). Those options remain default-off.

## Read the execution path

| Stage                            | Main implementation                                           | Contract to review                                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fit lifecycle                    | `distributed.py`, `prepared.py`                               | All ranks use the same recipe, options and collective order. Preprocessing state is learned from context only.                                                                               |
| Input agreement                  | `column_shard.py`, `column_metadata.py`                       | Owners agree on schema, cohort IDs, target identity and a sorted column-name dictionary before exchanging features.                                                                          |
| Preprocessing and redistribution | `sharded_recipe.py`, `typed_exchange.py`, `tensor_control.py` | Values retain arithmetic order and preprocessing device. Per-estimator FP32 payloads use typed tensor collectives; descriptors and ordinary errors use the same communicator.                |
| Chunked embedding                | `parallel.py`, `_summary_merge.py`                            | Row chunks share one implementation. The tiled schedule projects/applies a completed feature tile while the next merge is queued. Row attention starts after every feature tile is complete. |
| Cache storage                    | `sharded_offload.py`                                          | Each rank copies its physical slice of replicated column K/V to CPU. Other cache entries follow their original ownership. Restore reconstructs complete column K/V before replay.            |
| Prediction and release           | `parallel.py`, `distributed.py`                               | Original embedding replay consumes the fitted cache. `clear()` releases preprocessing state and invalidates old session shards.                                                              |

## Concurrency and lifetime invariants

The communication stream waits for the producing stream's event before reading local summaries. Every rank submits feature tiles in the same MAX/SUM/SUM collective order. A single ticket owns its input references, temporary buffers, NCCL Work and completion event. The queue waits for host-observed completion before reusing its slot, bounding outstanding storage.

A consumed summary makes its projected K/V and column output eligible for computation. Cache K/V assembly concatenates along the feature axis. The resident token buffer remains alive across row passes; in-place column writes must not overlap a still-live summary input. Reordering work across the all-feature row-attention barrier changes the algorithm.

Failures after submission retain the live queue or transfer buffers and poison reuse. The caller must tear down the distributed session; recovery collectives and fallback through an outstanding queue are unsafe. Ordinary Python preprocessing errors are agreed before dependent payload collectives; a failed process or CUDA context still needs a timeout and external cleanup.

## Compatibility and scope

Seunghwa Kang's PR #1319 is included through `48a2c39a8ad59da56164b3bbc964ec1bb40a01ab`, with its commit history preserved. Native CPU activation offload uses its token buffers. Distributed column-cache sharding is a separate mechanism; distributed `fit(offload="cpu")` remains unsupported.

Sharded preprocessing supports the default recipe and uses typed NCCL control and payloads without a companion Gloo group. Custom recipes use the replicated preprocessing path. CPU/Gloo tests exercise the tensor protocol. Unknown metadata structures fail before payload admission; the codec intentionally supports a limited set of table/schema/tensor/container values rather than arbitrary pickled objects.

## Review cleanup and validation

The original fused implementation and tiled-column variant duplicated token-buffer setup, row processing and feature-summary construction. They now share `_embed_rows_fused_impl`; `_apply_column_tile` isolates the tile projection and column write. Their different merge/application schedules remain explicit. Type annotations, collective-error comments and scoped test patches make the ownership contracts inspectable without changing model arithmetic or tolerances.

Separate processes compared the cleanup with frozen commit `a8b30b31399268b2c4168c546477f5db256f1def`: 28 small-model CPU/Gloo cases on 1/2/4 ranks, 224 tensor comparisons and 1,401,488 elements matched exactly. Coverage includes two embedding layers, 500 features, row/feature tails, noncontiguous inputs, uneven/empty owners, K/V shapes/strides, fit recording and first/repeat original replay. Strict thresholds remain `atol=1e-6, rtol=1e-5`.

The source-parity study is CPU validation. CUDA scheduling, memory and clean performance must be checked on the exact cleaned source before adopting its historical timing claims. Existing native/distributed strict numerical differences remain documented separately from class agreement; the cleanup does not resolve those differences. Profiler and benchmark instrumentation remains outside production code.
