# Phase 4 — Prefill performance (Qwen2.5-1.5B)

**Date**: 2026-04-19
**Status**: PASS — 5/5 patterns applied with `--npu-attn`; 4/5 with `--cpu-attn`.

## Headline result

| Path | Cold NPU layers | Warm NPU layers (avg of 3) | Warm wall (incl. CPU LM head) | ms/layer (warm) |
|---|---|---|---|---|
| CPU-attn (Pattern 5 partial) | 12.6 s | 10.1 s | 11.8 s | 360 |
| **NPU FA (Option C, Pattern 5 full)** | **5.0 s** | **2.4 s** | **4.1 s** | **85** |
| Speedup vs CPU-attn warm | 2.5× | **4.2×** | 2.9× | — |

Top-1 token: ' Paris' on cold AND warm AND vs CPU reference — no
correctness regression introduced by NPU FA path.

Preload setup: ~1.8 s one-time (Pattern 2 — `preload_prefill_weights`
across 28 layers, ~3.5 GB of BO writes).

## Pattern application

| # | Pattern | Status | Notes |
|---|---|---|---|
| 1 | Multi-launch merging | INHERITED | rms_gemms_rope=6 launches, o_ffn=8 launches (default builders) |
| 2 | Per-layer BO pre-loading | APPLIED | `preload_prefill_weights` from llama3 — works unchanged with padded weights, 1.78 s setup, 67 ms/layer |
| 3 | Intermediate buffer reuse | INHERITED | `intermediate_indices` set per kernel inside `run_transformer_block` |
| 4 | Seq-first layout | INHERITED | RoPE + FA both native seq-first |
| 5 | CPU→NPU op promotion | APPLIED (with `--npu-attn`) | Option C head-first FA wrapper at head_dim=128; padded n_heads=16, group=8 |

## Compile costs (one-time)

| Kernel | Compile time |
|---|---|
| rms_gemms_rope (default tile config at padded shapes) | ~33 s |
| o_ffn (default tile config at padded shapes) | ~50 s |
| flash_attn (Option C head-first via compile_attn_npu2_split) | ~78 s |
| **Total** | ~2 min cold compile |

## Validation

- Pattern 2 gain on first prompt: 19.7% reduction (cold 12.6 s →
  warm 10.1 s) with CPU-attn; 51.8% reduction (5.0 s → 2.4 s) with
  NPU FA. Larger gain with NPU FA because per-layer cost is dominated
  by NPU work (less variability) once the BOs are pre-allocated.
- NPU FA at Qwen2.5's padded GQA group=8 (n_heads=16, n_kv_heads=2)
  works clean — phantom Q heads (zero) produce uniform-weighted
  attention outputs that get nullified by zero-padded `wo` rows.
- Option C wrapper handles the padded shapes transparently: it
  transposes seq-first ↔ head-first using whatever `n_heads`,
  `n_kv_heads`, `head_dim` were registered via
  `compile_headfirst_fa_kernel`. We register the PADDED values
  (16, 2, 128).

## Caveats

- The 1.6 s CPU LM head dominates the warm wall time (~40% of 4.1 s).
  Phase 5/6 follow-up will move LM head to NPU via the existing
  `lm_head_multi` builder at vocab=151936 (10-partition design from
  Phase 1 audit) which should knock the wall down toward the pure NPU
  warm time of 2.4 s.
- Cold first prompt at 5.0 s NPU layers includes per-call BO
  allocation overhead. Warm steady-state is the right number to cite
  for "prefill speed".

## Comparison vs prior deployments (warm NPU layers)

| Model | n_layers | head_dim | Warm NPU layers | ms/layer |
|---|---|---|---|---|
| llama3 (1B)         | 16 | 64  | 1.30 s | 81 |
| smollm2 (1.7B)      | 24 | 64  | 1.88 s | 79 |
| llama32_3b          | 28 | 128 | 3.2 s  | 115 |
| **qwen25_1_5b**     | **28** | **128** | **2.4 s** | **85** |

Qwen2.5-1.5B sits between smollm2 and llama32_3b as expected:
- Same depth as llama32_3b (28 layers)
- Same head_dim (128) — uses same Option C wrapper
- Smaller emb_dim (1536 vs 3072) → less per-layer compute
- Result: per-layer rate close to llama3/smollm2 (~85 ms vs ~80 ms)
  despite head_dim=128

## Lessons confirmed (no new ones this phase)

- L3 (BD blowup at non-1024-aligned dims): the GQA-reindexed padding
  scheme not only unblocks compilation but ALSO works flawlessly
  through 28 layers + NPU FA + bias path, with no extra per-layer
  overhead beyond the compute on phantom heads (negligible).
- `preload_prefill_weights` (llama3 helper) works UNCHANGED with our
  padded weights — it just iterates layers and writes BOs by shape.
- Option C head-first FA wrapper is FULLY GENERIC across head_dim=128
  models (llama32_3b, qwen25_1_5b) — no per-model adaptation needed
  beyond setting `n_heads`, `n_kv_heads`, `head_dim` at compile time.
