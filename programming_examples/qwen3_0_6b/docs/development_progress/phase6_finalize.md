# Qwen3-0.6B deployment — Phase 6 finalize

End-to-end NPU2 deployment validated 2026-04-20.

## Phase summary

| Phase | Outcome | Key metric |
|---|---|---|
| 0 — Bootstrap | PASS | CPU reference vs HF transformers: corr=0.99999986, top-1 ' Paris' |
| 1 — Per-kernel shapes | PASS | 3/3 kernels compile (rms_attn_gemms, o_ffn[o_in_dim=2048], flash_attn) |
| 2 — Single block | PASS | NPU vs CPU ref: cos_real=0.9988, per_pos_min=0.997, no NaN |
| 3 — Full model | PASS | 6/6 prompts (dynamic decisive/competitive gate) |
| 4 — Prefill perf | PASS | 2.20 s warm @ seq_len=2048 (78.6 ms/layer); 3 patterns applied/already |
| 5 — Decode | PASS (CPU) | NPU prefill + CPU decode generates coherent text; NPU GEMV decode is a follow-up |
| 6 — Finalize | PASS | This file + qwen3_inference.py wired up |

## End-to-end perf (seq_len=2048, NPU prefill + CPU decode)

```
NPU prefill (warm):  2.20 s   (78.6 ms/layer × 28 layers)
CPU decode:          1.23 s/token (0.81 tok/s)
End-to-end (30 tok): ~38 s    (2.2 s prefill + 30 × 1.23 s decode)
```

## Comparison to other deployments

| Model | n_layers | head_dim | NPU prefill (warm) | Decode |
|---|---|---|---|---|
| llama3 (1.2 B)   | 16 | 64  | 1.30 s (81 ms/layer)  | 92 ms/token (NPU GEMV) |
| smollm2 (1.7 B)  | 24 | 64  | 1.88 s (79 ms/layer)  | 137 ms/token (NPU GEMV) |
| llama32_3b (3 B) | 28 | 128 | 3.2 s (~110 ms/layer) | ~210 ms/token (NPU GEMV) |
| qwen25 (1.5 B)   | 28 | 128 | 2.4 s (85 ms/layer)   | 216 ms/token (NPU GEMV) |
| **qwen3 (0.6 B)** | **28** | **128** | **2.20 s (78.6 ms/layer)** | **1.23 s/token (CPU)** |

Per-layer prefill rate is at parity with the fused-ELF deployments despite
the **split-ELF overhead required by Q/K Norm placement** (RMSNorm doesn't
commute with RoPE). The smaller `emb_dim=1024` (vs qwen25's 1536) absorbs
the extra XRT-call overhead. Decode is on CPU pending NPU GEMV split-ELFs.

## Architectural divergences from prior deployments

| Feature | This deployment |
|---|---|
| QKV bias | NONE (Qwen3 dense `attention_bias=False`) |
| Q/K Norm | NEW: per-head RMSNorm BETWEEN Q/K projection and RoPE |
| Tied embeddings | True (but `lm_head.weight` ALSO stored explicitly in safetensors) |
| GQA group | 2 (n_heads=16, n_kv_heads=8) |
| head_dim | 128 (Option C head-first FA wrapper) |
| RoPE base | 1e6 (same as Qwen2.5) |

## Reusable patterns that emerged from this deployment

1. **Split-ELF approach for Q/K Norm**: `rms_attn_gemms` (predecessor, no RoPE)
   + host `apply_qk_norm` + host RoPE + `flash_attn`. Documented in
   `_llm_shared/phase_helpers/qk_norm.py`. Future Qwen3-class models
   (Qwen3-1.7B, Qwen3-4B, Qwen3-8B) can reuse this directly.

2. **Builder extension for `q_dim != emb_dim`**: Both
   `multi_launch_builder/superseded/rms_attn_gemms_multi.py` and
   `multi_launch_builder/o_ffn_multi.py` gained backward-compatible
   `q_dim`/`o_in_dim` parameters. Required when `n_heads * head_dim != emb_dim`
   (e.g., Qwen3-0.6B has 16 × 128 = 2048 ≠ emb_dim=1024).

3. **Dynamic Phase 3 decisive/competitive classification**: rather than
   trusting the static buckets in `canonical_prompts.py` (which were
   llama3-tuned), Phase 3 now classifies each prompt by ACTUAL CPU top-1
   probability. Captured in LESSONS.md L3.

## Outstanding (Phase 5+ follow-up)

NPU GEMV decode for Qwen3 needs:
- A split GEMV ELF: `rms_attn_gemvs` (RMSNorm + Q/K/V GEMV, no RoPE) at
  M=1, K=emb_dim=1024, N=q_dim=2048 (Q) and N=kv_dim=1024 (K, V).
  The existing llama3 `rms_gemv_rope` is fused — needs a split version.
- Host Q/K Norm + RoPE between (decode-side helper exists, just needs
  invocation in the decode block).
- An NPU FA decode kernel (single-token Q against full K/V cache).

Estimated effort: similar to the Phase 2 prefill split-ELF wiring, ~half
a deployment session. Adopting the LM head GEMV (8-partition, vocab=151936)
should drop the LM-head step from CPU 250 ms to NPU ~14 ms, which is the
biggest single decode win per the `optimize-decode-perf` skill.

## Files in this deployment

```
qwen3_0_6b/
├── qwen3_weights.py           # HF safetensors loader + Q/K Norm + RoPE LUT
├── qwen3_reference.py         # CPU F32 forward (Q/K Norm BEFORE RoPE)
├── qwen3_phase1_test.py       # per-kernel shape sweep
├── qwen3_phase2_test.py       # single-block correctness (NPU + bisect harness)
├── qwen3_phase3_test.py       # full 28-layer + canonical prompts (dynamic gate)
├── qwen3_phase4_test.py       # prefill perf (cold/warm + BO preload patterns)
├── qwen3_phase5_test.py       # decode (NPU prefill seed + CPU decode)
├── qwen3_inference.py         # end-to-end runner (Makefile `make run` entry)
├── Makefile, README.md, CLAUDE.md, TODO.md, RALPH_PROMPT.md
└── docs/development_progress/{progress,LESSONS,debug_log,phase6_finalize}.md
```
