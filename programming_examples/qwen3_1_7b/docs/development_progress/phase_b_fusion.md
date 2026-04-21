# Phase B — Fused decode ELFs (2026-04-20)

## Goal

Get every decode op onto NPU AND reduce per-process loaded ELF count to
escape the XRT load cap that Phase A surfaced. Fuse the per-leaf decode
kernels into multi-launch ELFs (mirroring what llama3 already ships).

## Result — architecture

All Qwen3 decode ops now run on NPU via 3 ELFs (down from 8 per-leaf):

| ELF | Internal launches | What it does |
|---|---|---|
| `rms_attn_gemvs_qknorm_rope.elf` | 8 | RMSNorm 1D + Q/K/V GEMV + Q-Norm + K-Norm + RoPE Q + RoPE K |
| `o_gemv_ffn_silu.elf` | 8 | O GEMV + Add + RMSNorm + Gate + Up + SiLU+Mul + Down + Add |
| `lm_head_gemv.elf` | 10 | unchanged — vocab projection 10×16384 partitions |

Per-token NPU launches: 169 → **57** (28 layers × 2 ELF calls + 1 LM head).
Total decode ELFs in cache: **3** (well under XRT cap).

## Result — correctness

End-to-end: `make clean && python3 qwen3_inference.py --decode npu --n-tokens 8`
generates `"The capital of France is Paris. The capital of France is also"`
— byte-identical to the host-baseline output.

Standalone gates each passed cos > 0.999:

| ELF | NPU run | Per-output cosine vs CPU |
|---|---|---|
| rms_attn_gemvs_qknorm_rope | 1.00 ms | normed/q/k/v/q_normed/k_normed/q_roped/k_roped ALL > 0.99993 |
| o_gemv_ffn_silu | 1.62 ms | proj/res1/normed2/gate/up/swiglu/down/out ALL > 0.99966 |

## Phase B+ host-side optimizations (added 2026-04-20, after fusion)

After Phase B's correctness was in, three llama3-style host-side optimizations
landed in `qwen3_decode.py`:

1. **Pre-transpose weights ONCE** per `LayerWeights` (cached as `_wq_t`,
   `_wk_t`, ... attributes). The previous code called
   `np.ascontiguousarray(...).T` on ~6 weights per layer per token. Each
   transpose copy is multi-MB and runs on the host CPU.
2. **Per-layer arg-list cache** (`_DECODE_ARG_CACHE`). The previous code
   allocated 17 (ELF1) + 15 (ELF2) numpy arrays per call, including a bunch
   of `np.zeros()` for intermediates. Now the list is built once per
   `(kernel, layer)` pair; only the dynamic slots (x_in, lut_q, lut_k,
   attn_out, x_res) get assigned per call.
3. **`preload_decode_weights()`** fires both fused ELFs once per layer with
   dummy inputs at startup. This populates all per-layer BO sets and writes
   weights to NPU BOs. After this, `static_input_indices` skips weight
   re-writes on every subsequent call.

### Wall-clock impact (decode at seq_len=512, 8 tokens)

| Stage | Before host opts | After host opts |
|---|---|---|
| Per-token decode wall | **0.90 s** | **0.09 s** |
| Tokens/s | 1.10 | **10.70** |
| Per-ELF-call cost (estimate) | ~14 ms | ~0.8 ms |
| End-to-end demo (8 tokens) | ~7 s | **~1.3 s** |

**~10× decode speedup** from removing host-side per-call overhead. Now
competitive with llama3-1B's 10.8 tok/s (which itself uses these same
optimizations).

The dominant savings:
- Weight transposes were running ~28 layers × 6 weights × 1.6 MB each per
  token = 270 MB/s of pure host-CPU copy. Eliminated.
- Building 32 numpy arrays per call × 56 calls/token = ~1800 allocations
  per token. Eliminated.
- First-call weight DMA cost, previously paid mid-decode on the first
  token's first layer call, now paid once at startup.

## Result — perf (initial fusion, before host opts)

| | Today (per-leaf, Phase A end) | Phase B (3 fused ELFs) |
|---|---|---|
| Decode ELFs in cache | 8 | 3 |
| ELF calls per token | 169 | 57 |
| NPU compute per token (sum of per-launch latencies) | ~10 ms | ~10 ms (same) |
| Per-token wall (warm) | 0.92 s | 0.90 s |
| `make run --n-tokens 8` end-to-end wall | ~7 s | ~7 s |

The projected ~3× speedup didn't materialize. **Per-call XRT/Python overhead
dominates, not launch count**. Each fused call now packs more work
(8 internal launches per call vs 1) so the per-call cost grew from
~5 ms to ~16 ms — net wall unchanged.

## Why fusion still mattered

1. **Architectural** — every decode op is now on NPU (Q/K Norm + RoPE +
   SiLU+Mul + adds). Phase A could only land Q/K Norm because adding more
   leaf ELFs busted the XRT load cap. Fusion sidesteps that cap entirely.
2. **Headroom for future cuts** — with 57 ELF calls/token instead of 169,
   any future per-call overhead reduction (e.g. moving the inference loop
   from Python to C++ via the XRT C++ API) gets 3× more leverage.
3. **Code clarity** — `run_decode_block` is now ~2 NPU calls per layer
   instead of 7, much easier to reason about.

## What did NOT work / is left

- Fusion alone doesn't speed up decode wall. The next perf lever is
  **per-call cost reduction**: lift the BO write/set_arg/start-wait sync
  loop out of Python, OR aggregate further (one ELF per token instead of
  two — would require the prefill kernel reuse pattern with KV cache
  paths integrated).
- Eltwise add at `n=1024` Peano-llc-crashed standalone but **works fine
  inside the multi-launch ELF**. So the standalone test was hitting a
  config-specific Peano regression, not a fundamental kernel bug.

## Files

**Created:**
- `qwen3_0_6b/multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py` — 8-launch
  builder (includes new `_build_qknorm_per_head_1d` helper for per-head
  RMSNorm with 1D-arg func, mirrors `_build_rms_1d`'s expand_shape pattern)
- `qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py` — 8-launch builder
  with **3-K matvec rename** (default `mv.o` for K=1024 Gate/Up; `mv_og.o`
  for K=2048 O via `og_matvec_*` rename; `mv_dg_qwen3.o` for K=3072 Down
  via existing `dg_matvec_*` rename). All three .o files compiled with
  the same DIM_M_OUTPUT=8 — only the exported symbol differs.

**Modified:**
- `_llm_shared/kernel_builder/external_kernels.py` — added
  `compile_mv_og(tile_m=8)` and `compile_mv_dg_qwen3(tile_m=8)`.
- `qwen3_0_6b/qwen3_decode.py::compile_decode_kernels` — drop per-leaf
  compiles, build the 3 fused ELFs.
- `qwen3_0_6b/qwen3_decode.py::run_decode_block` — 7 NPU calls + 5 host ops
  per layer reduced to 2 NPU calls + 1 host attention + 1 host KV append.
