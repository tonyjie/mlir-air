# Deployment: Qwen3-0.6B

## Phase status
- [x] 0: Bootstrap (PASSED 2026-04-20: corr=0.99999986, top-1 ' Paris')
- [x] 1: Per-kernel shapes (PASSED 2026-04-20: rms_attn_gemms, o_ffn[o_in_dim=2048], flash_attn at hd=128)
- [x] 2: Single block (PASSED 2026-04-20: cos_real=0.9988, per_pos_min=0.997, no NaN at seq_len=512)
- [x] 3: Full model (PASSED 2026-04-20: 6/6 prompts; Paris/blue exact match; ~1.2s NPU prefill at seq_len=512)
- [x] 4: Prefill perf (PASSED 2026-04-20: warm 2.20s @ seq_len=2048, 78.6 ms/layer; 3 patterns applied/already)
- [x] 5: Decode (PASSED 2026-04-20: full NPU decode via 3 fused ELFs. NPU prefill 2.09s @ seq_len=2048 + NPU decode 0.09s/token (10.7 tok/s) — at parity with llama3-1B. See `docs/development_progress/phase_b_fusion.md`.)
- [x] 6: Finalize (PASSED 2026-04-20: qwen3_inference.py + qwen3_verify_decode.py + qwen3_canonical_sweep.py)
- [x] 7: Evaluate (passed 2026-04-20: 6/6 canonical prompts via NPU prefill+LM head; 6/6 NPU decode tokens within CPU top-5 over multi-token verify)

## Active blockers

(none — all blockers resolved)

## Resolved blockers

**Q/K Norm integration** — RESOLVED via split-ELF approach:
1. NPU `rms_attn_gemms` (predecessor `superseded/rms_attn_gemms_multi.py`,
   extended with `q_dim` parameter for q_dim≠emb_dim case)
2. Host `_llm_shared/phase_helpers/qk_norm.py::apply_qk_norm`
3. Host RoPE (BF16; predecessor `rope_qk_multi.py` uses interleaved LUT
   incompatible with our half-split generator — RoPE-on-host is fast
   enough at this model size and validated to be correct)
4. NPU `flash_attn` via head-first wrapper (Option C)
5. NPU `o_ffn` (extended with `o_in_dim` parameter for q_dim≠emb_dim case)

**Shape adaptation for q_dim != emb_dim** — RESOLVED via backward-compatible
extensions to `rms_attn_gemms_multi.py` (`q_dim` kwarg) and
`o_ffn_multi.py` (`o_in_dim` kwarg). Default to `emb_dim` for llama-class
models; explicit value for Qwen3 (q_dim=2048, emb_dim=1024).

## Resolved follow-ups (Phase B + host-side opts, 2026-04-20)

The decode-perf "fuse the per-leaf launches" plan landed:
1. Fused `rms_attn_gemvs_qknorm_rope` (8 launches: RMSNorm + Q/K/V GEMV +
   Q-Norm + K-Norm + RoPE Q + RoPE K).
2. Fused `o_gemv_ffn_silu` (8 launches: O + add + RMSNorm + Gate + Up +
   SiLU+Mul + Down + add) with a 3-K matvec extern rename
   (`@matvec_*` for K=1024 → `mv.o`; `@og_matvec_*` for K=2048 →
   `mv_og.o`; `@dg_matvec_*` for K=3072 → `mv_dg_qwen3.o`, all built
   with DIM_M_OUTPUT=8). New `compile_mv_og` and `compile_mv_dg_qwen3`
   in `_llm_shared/kernel_builder/external_kernels.py`.
3. Host-side opts (pre-transposed weights cached on `LayerWeights`,
   per-layer arg-list cache `_DECODE_ARG_CACHE`, `preload_decode_weights`
   warmup) cut per-token wall from 0.90 s → 0.09 s (10× speedup, **at
   parity with llama3-1B at 10.7 tok/s**).

See `docs/development_progress/phase_b_fusion.md` for the detailed
fusion design + perf breakdown.

## Open follow-ups (lower priority)

- NPU attention for decode (CPU `decode_attention_cpu` works fine but
  scales O(current_pos); would need a static-shape attention kernel).
- 2D-herd matvec to use more cores per GEMV (would lift compute
  throughput; currently each GEMV uses 8 cores out of 32).
- Generic stitching DSL to replace the per-model multi-launch builder
  pattern (architectural cleanup, doesn't affect perf).

## Resolved config (pulled from HF)

n_layers: 28, emb_dim: 1024, n_heads: 16, n_kv_heads: 8,
head_dim: 128, hidden_dim: 3072, vocab_size: 151936, rope_theta: 1000000.0,
qk_norm: True, qkv_bias: False, tie_word_embeddings: True (lm_head also stored explicitly)
