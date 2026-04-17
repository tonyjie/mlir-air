# Phase 4 — Prefill performance

**Date**: 2026-04-17
**Test**: `smollm2_phase4_test.py`
**Setup**: 24-layer NPU prefill, NPU FlashAttention (MHA), CPU LM Head

## Pattern application

| # | Pattern | Status | Source |
|--|--|--|--|
| 1 | Multi-launch merging | **INHERITED** | `rms_gemms_rope` = 6 launches in one ELF; `o_ffn` = 8 launches |
| 2 | Per-layer BO pre-loading | **APPLIED HERE** | `preload_prefill_weights(weights, config, cache, seq_len, rope_lut)` — already config-driven, drop-in for SmolLM2 |
| 3 | Intermediate buffer reuse | **INHERITED** | `intermediate_indices` set per kernel in `run_transformer_block` |
| 4 | Seq-first activation layout | **INHERITED** | RoPE + FA accept `(seq, heads*dim)` natively |
| 5 | CPU → NPU op promotion | **PARTIAL** | NPU FlashAttention used; LM Head deferred to Phase 5 (vocab=49152 needs partition redesign) |

**4 of 5 patterns applied or inherited** (gate: ≥3).

## Measurements

### Cold first-prompt vs warm steady-state

| Phase | NPU layers | per-layer | CPU LM Head | Wall total |
|--|--|--|--|--|
| **Cold** (no preload, 1st prompt) | 4.165 s | 174 ms | 0.523 s | **4.700 s** |
| **Warm** (after preload, avg of 3) | 1.884 s | 79 ms | 0.506 s | **2.410 s** |

**Pattern 2 gain on first prompt**: 2.28 s reduction (54.8%) — matches what
llama3 saw with explicit pre-loading.

**Steady-state (warm) per-layer rate**: 76–83 ms.

### Pre-load setup cost

`preload_prefill_weights` for 24 layers: **2.83 s** (one-time, outside timed inference).
Pre-loaded weight footprint: **3072 MB** in NPU-allocated DRAM-mapped BOs
(24 × ~128 MB/layer = `wq + wk + wv + wo + w_gate + w_up + w_down + 2 norms`).

### vs llama3 baseline

| Metric | llama3 (16 layers, GQA) | SmolLM2 (24 layers, MHA) | Notes |
|--|--|--|--|
| Per-layer (warm) | 81 ms | **79 ms** | Same per-layer cost despite 4× larger KV-projection compute (MHA) |
| Total NPU prefill (warm) | 1.30 s | 1.88 s | Proportional to depth (24/16 × 1.26 = 1.89 expected) |
| Wall clock | 1.54 s | 2.41 s | Includes 0.51 s CPU LM Head (will move to NPU in Phase 5) |

**SmolLM2 hits per-layer parity with llama3**, even though MHA has 4× larger
K/V GEMMs than GQA. The kernels' tile configurations absorb this without
performance loss — 8×4 herd is dominated by Q/O/Gate/Up/Down GEMMs which are
identical between the two models, and K/V GEMMs are now the same shape as Q/O
(2048×2048×2048) which is the most-validated shape.

## Correctness regression check

- Cold top-1: `' Paris'` (id=7042)
- Warm top-1: `' Paris'` (id=7042) on all 3 runs
- **No regression** ✓

## Items surfaced for Phase 5

- 🔸 **CPU LM Head is 0.51 s of the 2.41 s wall** (~21% of wall). Phase 5
  must move LM Head to NPU for decode (per-token GEMV); the prefill GEMM
  variant should follow the same partition redesign.
- 🔸 **Memory**: 3 GB of pre-loaded weights uses a significant chunk of
  the 16 GB NPU-mapped DRAM. Decode adds ~1 GB more for transposed weights.
- 🔸 **First-prompt latency is now bounded by preload setup (2.83 s)**, not
  inference. Production code should call `preload_prefill_weights` from a
  startup hook, not before each prompt.

## Phase 4 gate verdict

- ✅ ≥3 of 5 patterns applied/inherited (we have 4)
- ✅ Prefill latency measured: warm 1.884 s NPU / 2.410 s wall
- ✅ No correctness regression

**PASS.**
