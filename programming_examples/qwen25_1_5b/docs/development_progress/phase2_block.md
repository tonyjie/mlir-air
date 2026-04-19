# Phase 2 — Single-block correctness (Qwen2.5-1.5B)

**Date**: 2026-04-19
**Status**: PASS (with seq_len=512 caveat — see "Surfaced for Phase 3").

## Result

Single transformer block (layer 0), CPU attention fallback, NPU
rms_gemms_rope + Qwen2 host-side QKV bias add + NPU o_ffn,
seq_len=512 (5 real prompt tokens):

| Metric | Value | Gate |
|---|---|---|
| Whole-tensor cosine vs Qwen2.5 CPU ref | **0.9986** | > 0.99 ✓ |
| Per-position cosine min (real tokens) | **0.9977** | > 0.98 (head_dim=128) ✓ |
| MAE | 0.042 | informational |
| max abs err | 1.30 | informational |
| NaN in NPU | False | none ✓ |

**Phase 2 PASS.**

## Approach

QKV bias (Qwen2's new architectural feature) is added on the **host** after
the bias-free `rms_gemms_rope` ELF returns. Justified by RoPE's linearity:

    RoPE(q + bq) = RoPE(q) + RoPE(bq)

So we precompute `bq_roped = RoPE(broadcast(bq, seq_len))` per layer once,
and add it to the ELF's `q_roped` output. Same for `bk_roped`. V's bias is a
plain broadcast-add (V doesn't go through RoPE).

Implementation: `qwen25_bias.py` monkey-patches `llama3_prefill._run_cached`
to intercept `rms_gemms_rope` calls and inject the precomputed bias. Same
pattern as `_llm_shared/phase_helpers/headfirst_fa.py` — zero changes to
shared multi-launch builders.

## Tile config audit

For `build_rms_gemms_rope_module(emb_dim=1536, kv_dim=256, ...)` the only
tile config that builds AND produces correct output at seq_len=512:

    tile_n=64, herd_n=4   →  tile_n*herd_n = 256

This fits all three GEMMs:
- Q (N=emb_dim=1536):  1536 / 256 = 6 ✓
- K (N=kv_dim=256):     256 / 256 = 1 ✓
- V (N=kv_dim=256):     256 / 256 = 1 ✓

Default `tile_n=128, herd_n=4` (used by llama32_3b) gives `tile_n*herd_n =
512` which OVERSHOOTS K/V's narrow N=256 — silently produces corrupt K/V.

For o_ffn at hidden_dim=8960:
    gate_tile_n=64    →  8960 / (64*4) = 35 ✓
    swiglu_tile_n=2240 → 8960 / 2240   = 4  ✓  (~13 KB L1, fits)

## Surfaced for Phase 3 (BLOCKERS)

### 0. UPDATE 2026-04-19 — seq_len=2048 PASS via GQA-reindexed padding

Resolved the BD-allocator blocker. **Phase 2 also PASSES at seq_len=2048**:

| Metric | seq_len=512 (unpadded) | seq_len=2048 (GQA-reindexed pad) | Gate |
|---|---|---|---|
| Whole-tensor cosine | 0.9986 | **0.9988** | > 0.99 ✓ |
| Per-position cosine min | 0.9977 | **0.9981** | > 0.98 (head_dim=128) ✓ |
| MAE | 0.042 | 0.039 | informational |

Mechanism: pad emb_dim 1536→2048 and hidden_dim 8960→9216 host-side
(both BD-friendly multiples of 1024). The naive padding broke GQA
(LESSON 4); fixed via **`qwen25_pad.py` GQA-reindex**: pad Q heads
INSIDE each KV group, not at the end. Phantom Q heads at padded
positions {6,7} (in group 0) and {14,15} (in group 1) are zero;
real orig Q heads 0–5 stay at padded 0–5; real orig Q heads 6–11
move to padded 8–13. wq/bq/wo all reindexed via the same scheme.

CPU-only sanity test (orig vs padded transformer block) hits cosine
0.999998 — confirms the math is exact (within BF16 noise from
RMSNorm pre-scaling).

Compile costs at padded shapes: rms_gemms_rope 33s, o_ffn 50s. Both
use DEFAULT tile config (no halving needed) since padded dims are
BD-friendly.

Phase 3 is now UNBLOCKED at seq_len=2048.

### 1. seq_len=2048 hits BD allocator exhaustion (resolved by 0 above)

### 1. seq_len=2048 hits BD allocator exhaustion
The 6-launch `rms_gemms_rope` ELF stitches RMSNorm + Q/K/V GEMMs + RoPE Q/K
sharing the same shim-channel pool. At Qwen2.5's emb_dim=1536, each Q/V/RoPE
DMA splits into a 2-D pattern (size=512, stride=768) because 1536 doesn't fit
a single BD dim. Cumulative BD count exceeds the channel pool → "Allocator
exhausted available buffer descriptor IDs" at AIE lowering.

**Phase 3 fix options** (decide before Phase 3):
- (A) **Split the multi-launch ELF** into `rms_qkv_only` (4 launches: RMS +
  Q + K + V) and `rope_qk_only` (2 launches: RoPE Q + RoPE K). Two XRT calls
  per layer instead of one (~28 ms extra prefill on a 28-layer model).
  Predecessor builders already exist in
  `llama3/multi_launch_builder/superseded/{rms_attn_gemms_multi.py,
  rope_qk_multi.py}` — likely reusable with minor updates. **Recommended.**
- (B) Pad emb_dim 1536→2048 host-side (zero-pad weights). Wasteful (~33%
  more compute on Q/O), and K/V dim N=256 would still need a separate fix.
- (C) Investigate whether the 4-D shim BD support (or a different lowering
  knob) can pack the 1536-wide DMA more compactly.

### 2. seq_len < 512 silently produces garbage
GEMM tile constraint M=seq_len ≥ tile_m × herd_m = 64 × 8 = 512. Below this,
the kernel "runs" but corrupts outputs. Test confirmed cosine 0.02 at
seq_len=128 (M < 512) → cosine 0.9986 at seq_len=512.

This affects how the inference runner picks seq_len for short prompts —
must always pad to ≥512.

## Files produced

- `qwen25_phase2_test.py` — Phase 2 driver (uses Qwen2.5 weights + bias wrapper).
- `qwen25_bias.py` — bias precomputation + `_run_cached` monkey-patch.

## Verify-step warnings (informational)

`run_transformer_block(verify=True)` reports per-step compares against an
unbiased CPU reference. With our bias wrapper, NPU's v/q_roped/k_roped have
bias added; the verify ref doesn't → expected large deltas:
- v: max_err ≈ |bv|.max ≈ 3.5
- q_roped: max_err ≈ √2 · |bq|.max ≈ 42
- k_roped: max_err ≈ √2 · |bk|.max ≈ 316  (Qwen2's bk has very large
  magnitudes — min=-316, max=288, mean_abs=24.9)

The `output` step (final block output) is compared against an unbiased ref
that ALSO doesn't include bias-flow through residuals → output corr=0.9994
even though intermediate verifies look failed. The Qwen2.5-aware Phase 2
metric (final cosine vs `qwen25_reference.transformer_block`) is the
authoritative gate.
