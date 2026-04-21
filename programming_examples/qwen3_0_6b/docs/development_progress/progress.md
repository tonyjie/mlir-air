# Qwen3-0.6B deployment progress

## Phase 0: Bootstrap (PASSED 2026-04-20)

- HF model: `Qwen/Qwen3-0.6B`
- Config: n_layers=28, emb_dim=1024, n_heads=16, n_kv_heads=8 (GQA group=2),
  head_dim=128, hidden_dim=3072, vocab=151936, rope_θ=1e6
- Qwen3-specific: NO QKV bias, NEW Q/K Norm (per-head RMSNorm BEFORE RoPE)
- Tied embeddings (lm_head.weight ALSO stored explicitly in safetensors)
- Verification: `python3 qwen3_reference.py --prompt "The capital of France is" --verify`
  - Top-1 match: YES (' Paris', id=12095)
  - Logits correlation: 0.99999986 (vs HF transformers F32)
  - Max abs error: 0.008210, Mean abs error: 0.001495

Q/K Norm math validated: applying per-head RMSNorm with the loaded
`q_norm.weight` and `k_norm.weight` weights BEFORE RoPE produces token
predictions identical to HuggingFace transformers reference.

## Phase 1: Per-kernel shapes (PASSED 2026-04-20)

Three NPU kernels validated at Qwen3-0.6B shapes:

| Kernel | Shape | Status | Note |
|---|---|---|---|
| `rms_attn_gemms` | seq_len=128, emb_dim=1024, q_dim=2048, kv_dim=1024 | PASS | Predecessor split ELF (no RoPE) — extended `q_dim` parameter so Q's output dim (2048) can differ from emb_dim (1024) |
| `o_ffn` | seq_len=128, emb_dim=1024, hidden_dim=3072, o_in_dim=2048 | PASS | Extended with `o_in_dim` parameter for the q_dim != emb_dim case |
| `flash_attn` (head-first wrapper) | seq_len=256, n_heads=16, n_kv_heads=8, head_dim=128 | PASS | Option C wrapper at hd=128; needs seq_len ≥ 256 (lqp=256) |

Host helpers used in lieu of new on-tile kernels:
- `_llm_shared/phase_helpers/qk_norm.py::apply_qk_norm` — per-head RMSNorm Q/K (cannot fuse into rms_gemms_rope because RMSNorm doesn't commute with RoPE)
- RoPE on host (BF16) — predecessor `rope_qk_multi.py` uses interleaved LUT incompatible with our half-split `generate_rope_lut`

## Phase 2: Single block correctness (PASSED 2026-04-20)

End-to-end NPU forward on layer 0 vs CPU F32 reference at seq_len=512:

| Metric | Value | Gate |
|---|---|---|
| Whole-tensor cosine (real tokens) | 0.9988 | > 0.99 ✓ |
| Per-position cosine min (real tokens) | 0.997 | > 0.98 (hd=128 scaled) ✓ |
| Per-position cosine min (all 512 positions) | 0.989 | > 0.98 ✓ |
| MAE (real tokens) | 0.015 | informational |
| NaN | False | required ✓ |

**Pipeline used (split-ELF + host Q/K Norm + host RoPE):**
1. NPU `rms_attn_gemms` — RMSNorm + Q/K/V GEMMs (no RoPE), 4-launch ELF
2. Host `apply_qk_norm` — per-head RMSNorm on Q and K (Qwen3 NEW)
3. Host RoPE (BF16, F32 internally) — bypassing predecessor `rope_qk_multi`
   which uses an interleaved LUT incompatible with our half-split generator
4. NPU `flash_attn` via head-first wrapper (Option C, head_dim=128)
5. NPU `o_ffn` — 8-launch ELF with `o_in_dim=2048` (Q dim differs from emb_dim)

**Lessons captured:**
- L1 (cache hygiene): seq_len-specific kernels MUST be wiped from
  `prefill_kernel_cache/` if Phase 1 compiled at a different seq_len than
  Phase 2 uses (Phase 1 used 128 for fast iteration; Phase 2 needs ≥256).
  Reused stale ELF gives garbage output (cos≈0.0).
- L2 (head-first FA wrapper requires monkey-patch resolution): Direct
  `from llama3_prefill import _run_cached` snapshots the ORIGINAL function
  reference. The head-first FA wrapper installs by rebinding the module
  attribute (`_lp._run_cached = ...`), so a snapshot bypasses the patch.
  Use `import llama3_prefill as _lp; _lp._run_cached(...)` indirection in
  per-model phase test scripts.

## Phase 3: Full 28-layer correctness (PASSED 2026-04-20)

End-to-end NPU forward across all 28 transformer blocks + final RMSNorm
(host) + LM head (host) at seq_len=512, validated against CPU F32 reference.

**Dynamic gate**: classify each prompt by ACTUAL CPU top-1 probability
(not the static canonical_prompts.py bucket which was tuned for llama3):
- decisive (CPU top-1 prob > 0.5) → strict NPU top-1 == CPU top-1
- competitive (CPU top-1 prob ≤ 0.5) → top-5 overlap

| Prompt | CPU top-1 (prob) | NPU top-1 | Effective gate | Result |
|---|---|---|---|---|
| `1 + 1 =` | ' ' (0.948) | ' ' | decisive | PASS |
| `2 + 2 =` | ' ' (0.916) | ' ' | decisive | PASS |
| `Water freezes at` | ' ' (0.468) | ' ' | competitive | PASS |
| `The largest ocean is the` | ' ocean' (0.220) | ' one' | competitive | PASS (top-5) |
| `The capital of France is` | ' Paris' (0.658) | ' Paris' | decisive | PASS |
| `The sky is` | ' blue' (0.175) | ' blue' | competitive | PASS (also exact) |

6/6 prompts PASS. NPU prefill ~1.2s at seq_len=512 (no preload yet —
Phase 4 will reduce). No NaN anywhere in the 28-layer stack.

**Lesson L3** (added below): canonical_prompts decisive/competitive split is
LLAMA-CALIBRATED. For other model families, classify dynamically by the
observed CPU top-1 prob.

## Phase 4: Prefill perf (PASSED 2026-04-20)

| Pattern | Status | Note |
|---|---|---|
| P1 multi-launch merging | SKIP | Q/K Norm requires split ELF (RMSNorm doesn't commute with RoPE) |
| P2 per-layer BO preload | APPLIED | `bo_key=f"...L{i}"` + `static_input_indices` on weights |
| P3 intermediate reuse | APPLIED | `intermediate_indices` on all `_out` arg slots |
| P4 seq-first activations | ALREADY | FA wrapper handles inside Option C |
| P5 CPU→NPU op promotion | N/A | Q/K Norm + RoPE on host = architectural baseline |

3 patterns applied/already → gate ≥3 satisfied.

**Measured (seq_len=2048, 28 layers)**:

| Phase | Wall time | Per-layer |
|---|---|---|
| Cold prefill | 3.37 s | 120.5 ms |
| Warm prefill avg | 2.20 s | 78.6 ms |
| Cold→warm speedup | 1.53× | — |

**Comparison (per-layer warm rate at seq_len=2048)**:

| Model | n_layers | head_dim | Per-layer warm | Wall (warm) |
|---|---|---|---|---|
| llama3 (1.2 B) | 16 | 64 | 81 ms/layer | 1.30 s |
| smollm2 (1.7 B) | 24 | 64 | 79 ms/layer | 1.88 s |
| llama32_3b (3 B) | 28 | 128 | ~110 ms/layer | 3.2 s (Option C FA) |
| qwen25 (1.5 B) | 28 | 128 | 85 ms/layer | 2.4 s |
| **qwen3 (0.6 B)** | **28** | **128** | **78.6 ms/layer** | **2.20 s** |

Qwen3-0.6B is at parity with qwen25_1_5b per-layer despite the split-ELF
overhead — the smaller `emb_dim=1024` (vs qwen25's 1536) compensates.
