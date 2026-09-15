# Source changes to review

This branch starts at upstream `6746658f74200feb957f99ad4163297f3185739c`.
It carries the selected DFlash implementation and the compiler changes needed
to rebuild it. Historical optimization experiments are not included.

## AIR compiler

Commit `e32ebd119c3f62333e13c07e11768f845ee7de78` contains these fixes and six
focused regression tests:

| Area | Required behavior | Regression test under `mlir/test/` |
|---|---|---|
| Runtime DMA offsets | Preserve the static address bias when an offset also contains a runtime component, including KV transfers split into buffer descriptors. | `Conversion/AIRRtToNpu/runtime_offset_static_bias.mlir` |
| Shared send-buffer ownership | Acquire the buffer before the opaque kernel writes the next payload, while respecting the preceding DMA access. Acquiring only before the send is too late. | `Conversion/AIRToAIE/air_channel_to_locks_shared_buffer.mlir` |
| Single-slot L2 chain locks | Honor `air.no_chain_pingpong` without disabling the v2 producer/consumer chain locks. The earlier ping-pong pass is separately omitted for these buffers. | `Conversion/AIRToAIE/memtile_chain_lock_v2_single_slot.mlir` |
| Memory-tile activation replay | Keep a refed source's lock-driven reader ring separate from unrelated packet streams; diagnose exhausted channels instead of wrapping onto a replay channel. | `Conversion/AIRToAIE/memtile_refeed_packet_isolation.mlir` |
| Allocation placement | Honor `air.disable_alloc_sink` on an allocation or its enclosing `air.execute`, retaining pinned scratch and the intended projection ping-pong transformation. | `Transform/AIRDependencyScheduleOpt/disable_alloc_sink.mlir` |
| Mixed branch results | Trace the requested async-token result of `scf.if`/`scf.index_switch`; do not mistake scalar results for completion tokens. | `Transform/AIRDependencyCanonicalization/scf_index_switch_mixed_results.mlir` |

These are shared compiler changes, not claims that every other MLIR-AIR example
has been retested. The six focused tests and the selected DFlash builds are the
validation scope for this reproduction branch.

## Shared LLM sources

The changes outside this example are small but material:

- `programming_examples/fused_decode/fused_decode.py` can export layer-boundary
  activations through `DECODE_HIDDEN_TAPS`; the default remains disabled.
  The Qwen3-4B Q4NX inference wrapper allocates and reads the corresponding slots.
- `fused_decode/kernels/attn_kv.cc` provides the explicitly enabled
  `DFLASH_VECTOR_ATTN_NORM` path, retaining BF16 operands, FP32 products, and
  BF16 output rounding. The measured B1 baseline enables this path.
- `fused_decode/kernels/lut_based_ops.h` explicitly removes fractional bits
  before the integer exponential-table lookup. This prevents the core rounding
  mode from selecting the next integer's exponential while the fractional
  lookup still uses the preceding interval.

The B8 target and drafter also use the checked post-route transformations in
[`route_artifact.py`](../route_artifact.py): distinct shim arbiters and an
isolated main-output route. These remain part of the selected implementation;
the delivery does not claim that unmodified automatic routing is sufficient.

## Example and reproduction support

[`ARCHITECTURE.md`](../ARCHITECTURE.md) explains the selected B8 graph, normalized
activation cache, attention mapping, host/NPU division, and KV injection.
Target and drafter are separately compiled NPU programs with shared building
blocks. The greedy acceptance loop is host code in `speculative.py`.

The reviewer additions provide a pinned AIR build, model revisions and packed
weight identities, frozen benchmark inputs, the historical B1 baseline, and a
paired one-pass study. They do not change the selected B8 graph or arithmetic.
[`reviewer.md`](reviewer.md) gives the setup and benchmark commands.

For an immutable reproduction, check out the release tag. Record the produced
study's `study.json` and all its `runs/` files; the release alone cannot record
the destination machine's driver, power mode, or measured timings.
