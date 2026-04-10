# LLAMA-3.2-1B Decode — Progress

> **Development archive.** This documents the initial decode implementation.
> The decode pipeline has since been significantly optimized (92ms/token vs
> 351ms/token at this stage). See `docs/profile.md` for current performance.


## Current Status: Optimized Decode Pipeline (5% faster than IRON)

**Single-token autoregressive generation working end-to-end on NPU.**

```
Input:  "The capital of France is"
Output: "the capital of France is Paris"
Time:   ~351ms/token (steady state, token 5-99)
IRON:   ~370ms/token
```

### Architecture

- **9 unique kernels** compiled (3 multi-launch ELFs + 6 single kernels)
- **10 NPU calls per block** + CPU attention (×16 blocks per token)
- **Static weight BOs**: weights written once on first token, skipped on subsequent tokens
- **bo.map() zero-copy**: all BO reads/writes use memory-mapped views
- **Pre-transposed weights**: GEMV expects A[M,K], weights stored as (K,M), transposed once at init

### Per-Block Kernel Sequence (10 NPU calls)

| # | Kernel | Type | Herd | Notes |
|---|--------|------|------|-------|
| 1 | `rmsnorm` | xclbin | [1,1] | Pre-attention normalization |
| 2 | `qkv_gemv` | ELF (3 launches) | [8,1]×3 | Q+K+V GEMV merged |
| 3 | `rope_q` | xclbin | [1,1] | Single-position RoPE on Q |
| 4 | `rope_k` | xclbin | [1,1] | Single-position RoPE on K |
| — | CPU attention | — | — | KV cache update + GQA attention |
| 5 | `o_gemv_add` | ELF (2 launches) | [8,1]+[8,1] | O projection + residual add merged |
| 6 | `rmsnorm` | xclbin | [1,1] | Pre-FFN normalization |
| 7 | `gate_up_gemv` | ELF (2 launches) | [8,1]×2 | Gate+Up GEMV merged |
| 8 | `silu_mul` | xclbin | [8,1] | NPU SiLU×mul (8192 elements) |
| 9 | `gemv_down` | xclbin | [8,1] | Down GEMV (K=8192) |
| 10 | `add` | xclbin | [8,1] | Residual add |

### Performance (prompt_len=2048, 100 decode tokens)

| Metric | AIR | IRON | Notes |
|---|---|---|---|
| **Decode time (100 tokens)** | **35.6s** | **33.7s** | |
| **Tokens/second** | **2.81** | **2.94** | |
| **Time/token (avg)** | **356ms** | **370ms** | **AIR 4% faster** |
| **Steady-state (token 5-99)** | **~351ms** | **~370ms** | **AIR 5% faster** |
| **First token** | **~893ms** | **~370ms** | AIR BO allocation overhead |

### NPU Kernel Time Breakdown (Steady State, per token)

| Kernel | Calls | Total | Avg | Tiles |
|---|---|---|---|---|
| `qkv_gemv` (Q+K+V) | 16 | 15ms | 1.0ms | 8 cols × 3 launches |
| `o_gemv_add` (O+Add) | 16 | 10ms | 0.6ms | 8 cols + 8 cols |
| `gate_up_gemv` (Gate+Up) | 16 | 41ms | 2.5ms | 8 cols × 2 launches |
| `silu_mul` | 16 | ~5ms | 0.3ms | 8 cols |
| `gemv_down` | 16 | 34ms | 2.1ms | 8 cols |
| `rmsnorm` × 2 | 32 | 8ms | 0.3ms | 1 tile |
| `add` | 16 | 4ms | 0.3ms | 8 cols |
| `rope_q` | 16 | 4ms | 0.3ms | 1 tile |
| `rope_k` | 16 | 4ms | 0.2ms | 1 tile |
| **NPU total** | **160** | **~126ms** | | |
| **CPU (attn+reshape)** | | **~43ms** | | |
| **CPU LM Head** | 1 | **~50ms** | | |
| **Dispatch/Python overhead** | | **~132ms** | | |
| **Total per token** | | **~351ms** | | |

---

## Optimization History

### Phase 1: First Working Pipeline (~500ms/token)
- 15 NPU invocations/block + CPU attention + CPU SiLU×mul
- 8 kernels: 4 GEMV shapes + rmsnorm + add + rope_q + rope_k
- Python invoker overhead dominated (~480ms of ~500ms)

### Phase 2: Static Weight BOs + bo.map() (~340ms/token)
- `bo_key` per layer for weight BO isolation (128 BO sets, 8 XRT contexts)
- `static_input_indices` to skip weight writes after first token
- `bo.map()` zero-copy for all BO reads/writes
- Impact: ~500ms → ~340ms/token (160ms saved from eliminating redundant weight memcpy)

### Phase 3: Multi-Launch Merges + NPU SiLU (~351ms/token)
- Q+K+V GEMV merged (3→1 call, 3 launches)
- O GEMV + Add merged (2→1 call, 2 launches)
- Gate + Up GEMV merged (2→1 call, 2 launches)
- NPU SiLU×mul (moved from CPU, n=8192, [8,1] herd)
- 8-tile eltwise_add (was [1,1], now [8,1])
- Net: 15→10 calls/block, ~340ms → ~351ms (slightly higher due to extra NPU dispatch for SiLU, but enables future FFN full merge)

### Attempted: FFN Full Merge (5 launches) — BLOCKED
- Tried merging Gate+Up+SiLU+Down+Add into one 5-launch ELF
- **Blocked** by memref type mismatch: `linalg_fill_bf16` has signature `(bf16, memref<8xbf16, 2>)` for gate/up GEMV (tile_m=8) but needs `(bf16, memref<2xbf16, 2>)` for down GEMV (tile_m=2)
- These can't share the same private func declaration in one module
- Builder preserved at `multi_launch_builder/ffn_decode_full_multi.py` for future reference

---

## GEMV Kernel Performance

Detailed analysis: `docs/kernels/gemv.md`

### Optimal Configs (8-column, C++ harness profiling)

| Shape (M×K) | AIR (us) | IRON (us) | Gap |
|---|---|---|---|
| 2048 × 2048 (Q/O proj) | 233 | 214 | 1.1x |
| 512 × 2048 (K/V proj) | 81 | 98 | **0.8x (AIR faster)** |
| 8192 × 2048 (FFN gate/up) | 837 | 657 | 1.3x |
| 2048 × 8192 (FFN down) | 946 | 660 | 1.4x |

Best flags: `omit_pingpong=''` (ON for K=2048), `lock_fix=False`, `tile_sizes=[16,16]`.

The 1.3-1.4x gap for large shapes is architectural: AIR's DDR→L2→L1 data path adds L2 staging overhead vs IRON's direct DDR→L1 (ObjectFIFO). Closing this gap requires compiler-level BD pattern changes.

---

## Key Findings

1. **Broadcast DMA bug fixed** — multi-column GEMV (herd_m=8) works after mlir-air rebuild
2. **Weight transpose critical** — GEMV expects A[M,K], weights stored as (K,M). Must pre-transpose.
3. **Static weight BOs are the biggest optimization** — eliminates 1GB+ of redundant weight memcpy per token
4. **L2 staging overhead** — AIR's L2 path adds ~1.3-1.4x vs IRON's direct L3→L1
5. **`runtime_loop_tiling_sizes=[16,16]`** — largest single kernel perf impact (26% improvement)
6. **FFN full merge blocked** — different tile_m sizes produce incompatible linalg_fill_bf16 signatures

---

## Remaining Optimization Opportunities

| Priority | Action | Expected Impact |
|---|---|---|
| 1 | **NPU LM Head** — GEMV at 128256×2048, 8 cols | ~50ms/token (replace CPU matmul) |
| 2 | **FFN full merge** — resolve linalg_fill type mismatch | ~10ms/token (3 fewer dispatches × 16 layers) |
| 3 | **Multi-tile RMSNorm** — 8 cols instead of 1 | ~4ms/token (blocked by aiecc weight broadcast bug) |
| 4 | **NPU prefill for KV cache** — use NPU prefill pipeline | Faster init (currently 16s CPU) |
| 5 | **Unified inference script** — combine prefill + decode | User-facing convenience |

---

## Files

| File | Purpose |
|---|---|
| `llama3_decode.py` | Main decode pipeline (compile + prefill + decode loop) |
| `multi_launch_builder/rms_qkv_gemv_multi.py` | Q+K+V GEMV multi-launch builder |
| `multi_launch_builder/o_gemv_add_multi.py` | O GEMV + Add multi-launch builder |
| `multi_launch_builder/ffn_gemv_multi.py` | Gate+Up GEMV multi-launch builder |
| `multi_launch_builder/ffn_decode_full_multi.py` | FFN full merge attempt (blocked, kept for reference) |
| `ffn_swiglu/silu_and_mul.py` | SiLU×mul kernel builder |
| `docs/decode/DECODE_EXPLANATION.md` | Detailed code walkthrough and performance analysis |
| `docs/decode/DECODE_PLAN.md` | Original decode plan (historical) |
| `docs/decode/air_vs_iron_decode.md` | Architecture comparison with IRON |
| `docs/decode/iron_decode_reference.md` | IRON baseline numbers |
| `docs/decode/gemv_investigation.md` | Early GEMV investigation (historical) |
| `docs/kernels/gemv.md` | GEMV kernel analysis (configs, flags, comparison) |
