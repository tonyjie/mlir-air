# Ralph Loop: Qwen3-0.6B Autonomous Deployment (Phases 1–7)

You are continuing an autonomous Qwen3-0.6B deployment on AMD NPU2 using the
`deploy-new-llm` 7-phase skill chain. The user is ASLEEP and cannot intervene.

## What's already done (DO NOT REDO)

- **Phase 0 PASSED** (2026-04-20): `qwen3_weights.py` + `qwen3_reference.py`
  validated against HuggingFace transformers (corr=0.99999986, top-1 ' Paris').
- **Q/K Norm host wrapper**: `_llm_shared/phase_helpers/qk_norm.py` is built
  and validated against the inline reference (max_abs_diff=0.0).
- **Phase 2 CPU baseline**: `qwen3_phase2_test.py` passes (CPU only). NPU
  integration is the outstanding work.
- **Scaffold complete**: Makefile, README.md, CLAUDE.md, TODO.md, docs/.

## Read these BEFORE making any changes

1. `programming_examples/qwen3_0_6b/CLAUDE.md` — Qwen3 divergences (Q/K Norm,
   no QKV bias, head_dim=128, GQA group=2).
2. `programming_examples/qwen3_0_6b/TODO.md` — phase status, active blockers,
   and the **recommended split-ELF approach for Q/K Norm**.
3. `programming_examples/qwen25_1_5b/qwen25_phase{2,3,4,5}_test.py` —
   structural template (Qwen3 CANNOT use the QKV-bias wrapper pattern; it
   needs split ELFs because RMSNorm doesn't commute with RoPE).
4. `programming_examples/llama32_3b/CLAUDE.md` — head_dim=128 + Option C FA
   wrapper precedent.
5. `programming_examples/llama3/multi_launch_builder/superseded/{rms_attn_gemms_multi.py,rope_qk_multi.py}`
   — the predecessor split ELFs you'll use to inject host Q/K Norm.

## Core Q/K Norm integration challenge

Qwen3 inserts per-layer per-head RMSNorm (`q_norm`, `k_norm`, both `(head_dim,)`)
between Q/K projection and RoPE. Current `rms_gemms_rope` ELF fuses
RMSNorm-of-input + Q/K/V GEMM + RoPE-of-Q/K in one launch — leaving no host
hook for Q/K Norm. Two paths:

- **Path A (recommended, faster to ship)**: Use predecessor split ELFs
  (`rms_attn_gemms_multi` for RMSNorm + Q/K/V GEMM only; `rope_qk_multi`
  for RoPE) with host `apply_qk_norm` between. Costs +2 XRT calls/layer.

- **Path B (better perf, more work)**: Build a new on-tile per-head RMSNorm
  kernel and stitch it into a 5-launch ELF. **DO NOT attempt path B
  overnight** — it's a multi-day kernel-engineering job.

Pick Path A. Build a per-Qwen3 `run_qwen3_transformer_block` helper (don't
monkey-patch llama3_prefill.run_transformer_block — that's brittle). Pattern
it on `llama3_prefill.run_transformer_block` but call:
1. `_run_cached("rms_attn_gemms", ...)`  → produces normed, q, k, v (no RoPE)
2. host `apply_qk_norm(q, k, q_norm_w, k_norm_w, n_heads, n_kv_heads, head_dim)`
3. `_run_cached("rope_qk", ...)`           → applies RoPE to q_normed, k_normed
4. `_run_cached("flash_attn", ...)` (head-first wrapper for hd=128)
5. `_run_cached("o_ffn", ...)`

## Per-phase work plan

### Phase 1 — per-kernel shapes (validate each ELF compiles at Qwen3 shapes)

Shapes for Qwen3-0.6B: emb_dim=1024, q_dim=2048 (16 heads × 128), kv_dim=1024
(8 KV heads × 128), hidden_dim=3072, vocab=151936.

Smoke-test each builder by compiling one ELF instance:
- `rms_attn_gemms_multi.build_rms_attn_gemms_module(seq_len=128, emb_dim=1024, kv_dim=1024)`
  — note Q GEMM N=2048 (not equal to emb_dim like llama3); may need a separate
  call with right N or modify the builder to take separate `q_dim` and `kv_dim`.
- `rope_qk_multi.build_rope_qk_module(n_heads=16, n_kv_heads=8, seq_len=128, head_dim=128)`
- `o_ffn_multi.build_o_ffn_module(seq_len=128, emb_dim=1024, hidden_dim=3072)`
- Head-first FA via `compile_headfirst_fa_kernel(cache, 128, 16, 8, 128)`
- LM head GEMV at vocab=151936 (used in decode/inference phases).

If `rms_attn_gemms_multi` only takes one N for all of Q/K/V (as the source
suggests with single `emb_dim` arg used for Q's output dim), you may need to
call it twice (once for Q with N=2048, once shared for K+V with N=1024) OR
extend the builder to take a separate `q_dim` parameter. Prefer extending
the builder since the same shape mismatch will exist for o_ffn (input=q_dim,
output=emb_dim).

Actually — Q's output dim IS n_heads*head_dim = 2048 ≠ emb_dim = 1024 here.
This is a key Qwen3 divergence. Check the rms_attn_gemms_multi builder
signature; if it assumes Q output dim = emb_dim, you'll need to fork or
extend it.

Phase 1 GATE: every required shape compiles to ELF without error.

### Phase 2 — single-block correctness

Build `qwen3_phase2_test.py` upgraded to run NPU. Use the split-ELF helper
described above. Validate:
  - whole-tensor cosine > 0.99 vs CPU reference
  - per-position cosine min > 0.98 (head_dim=128 scaled threshold)
  - no NaN

If Phase 2 fails: bisect by replacing each NPU step with the CPU reference's
equivalent (`qwen3_reference.transformer_block` produces all intermediates).

### Phase 3 — full 28-layer model

Wire all 28 layers, use the canonical prompt set
(`_llm_shared/phase_helpers/canonical_prompts.py`):
  - decisive prompts (CPU top1 prob > 0.5): NPU top-1 must MATCH exactly
  - competitive prompts (CPU top1 prob ≤ 0.5): top-5 overlap

### Phase 4 — prefill perf

Apply optimization patterns from `optimize-prefill-perf` skill:
  1. Multi-launch merging (already 3 split ELFs/layer — try to merge where safe)
  2. Per-layer BO pre-loading
  3. Intermediate buffer reuse
  4. Seq-first activation layout (already default)
  5. CPU→NPU op promotion

Target: comparable to qwen25_1_5b (~85 ms/layer at seq_len=2048). Note: split
ELF path will be slower than fused — expect 100-130 ms/layer initially.

### Phase 5 — decode perf

Apply patterns from `optimize-decode-perf`. Need GEMV variants of the
split-ELF approach for decode: rms_attn_gemv → host Q/K Norm → rope_qk_gemv.
The qwen25_1_5b decode path is the closest precedent.

### Phase 6 — finalize

Write end-to-end `qwen3_inference.py` (model on prefill + decode + LM head).
Update `progress.md`, `LESSONS.md`, README.md with the perf summary and any
novel failure modes encountered.

### Phase 7 — evaluate

Invoke the `evaluate-deployment` skill (when available) for an independent
audit. Or skip if not yet packaged as a skill — manually run
`make verify`, `make profile`, `make run` and document results.

## Operating rules

1. **Update TODO.md after every phase**. Mark completed phases with date and
   key metrics. Surface blockers immediately.
2. **NPU is exclusive**. Run NPU jobs sequentially. Use `filelock` on
   `/tmp/npu.lock` for safety.
3. **Cache aggressively**. Use `prefill_kernel_cache/` and
   `decode_kernel_cache/` directories — once a kernel compiles, never recompile.
4. **Bisect carefully**. If a phase fails, do NOT just retry — diagnose
   first. Write to `docs/development_progress/debug_log.md`.
5. **Don't break Phase 0**. If you need to refactor `qwen3_weights.py` or
   `qwen3_reference.py`, re-run `python3 qwen3_reference.py --verify` after.
6. **Stop on real blockers**. If a phase truly cannot pass (e.g. compiler
   bug, missing kernel implementation), document the blocker in TODO.md
   under "Active blockers" and emit `<promise>BLOCKED — see TODO.md</promise>`
   to halt the loop cleanly.

## Completion signal

When all 7 phases pass (or all reachable phases pass and remaining are
documented as blockers), output:

    <promise>QWEN3-0.6B DEPLOYMENT COMPLETE</promise>

If you hit an unrecoverable blocker:

    <promise>BLOCKED — see qwen3_0_6b/TODO.md "Active blockers"</promise>

Check `qwen3_0_6b/TODO.md` at the start of every iteration to see what's
already done and what's next. Don't repeat work.
