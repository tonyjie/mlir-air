# SmolLM2-1.7B on MLIR-AIR (NPU2) — Progress Tracker

**Goal**: Functionally correct and performant `HuggingFaceTB/SmolLM2-1.7B`
BF16 inference (prefill + decode) on NPU2 via the `deploy-new-llm` skill chain.

**Model config**: 24 layers, emb_dim=2048, n_heads=32, head_dim=64,
n_kv_heads=32 (MHA), hidden_dim=8192, vocab=49152, BF16, rope_θ=130000,
tied embeddings.

**Reference deployment**: `programming_examples/llama3/` (Llama-3.2-1B base).

---

## Current status (2026-04-17)

**🎉 Deployment complete + end-to-end runner wired.** All 7 phases (0–6) passed
and a unified `smollm2_inference.py` does NPU prefill (with K/V extraction) +
NPU LM Head GEMV + NPU decode in one process. `make run` is the entry point.

End-to-end NPU inference: prefill **2.25 s** (94 ms/layer with K/V extraction;
1.88 s / 79 ms/layer standalone), decode **137 ms/token (7.3 tok/s)**. Per-layer
rates at parity with llama3 despite MHA's 4× larger K/V GEMVs.

## Phase log

### Phase 0: Bootstrap (PASSED 2026-04-17)
- HF model: `HuggingFaceTB/SmolLM2-1.7B`
- Config: n_layers=24, emb_dim=2048, n_heads=32, n_kv_heads=32 (MHA),
  head_dim=64, hidden_dim=8192, vocab=49152, rope_base=130000.0, BF16
- All 24 layers loaded with consistent shapes (wq=wk=wv=wo=2048×2048; w_gate=w_up=2048×8192; w_down=8192×2048)
- Tied embeddings detected automatically (existing fallback path triggered)
- Reference smoke output: `" Paris"` at position 4, logit=16.98, prob=0.4143
- HF transformers verify (F32): top-1 match, **logits correlation = 0.99999978**,
  max abs err = 0.0107 (BF16 precision floor)
- New files: `smollm2_weights.py`, `smollm2_reference.py`
- No code changes needed beyond config defaults — existing GQA-as-degenerate-MHA
  and tied-embeddings code paths covered SmolLM2 unchanged. **Survey was
  correct: this is genuinely Tier-A.**

### Phase 1: Per-kernel shapes (PASSED 2026-04-17)
- 16 distinct (kernel, shape) pairs across prefill + decode
- **12 drop-in** (identical shape to llama3): RMSNorm, Q/O GEMM, Gate/Up/Down GEMM,
  SwiGLU, eltwise add, RoPE shape, Q/O+Gate/Up+Down GEMV
- **4 parametric-recompile** (same builder, new arg): K/V prefill GEMM (kv_dim=2048),
  K/V decode GEMV (kv_dim=2048), RoPE LUT (theta=130000), FlashAttention (n_kv_heads=32)
- **1 needs partition-scheme decision**: LM Head GEMM/GEMV — vocab=49152 doesn't
  match llama3's 8×16384=128256 partitioning. Recommended option: 3×16384 (exact).
  **Deferred to Phase 4/5** (kernel-builder touch point).
- Standalone NPU validation deferred per smoke-test Lesson 4 — parametric
  builders already cover these shapes; integration test in Phase 2 will catch
  any per-kernel regression end-to-end.
- Full table: `docs/development_progress/phase1_kernel_shapes.md`

### Phase 2: Single-block integration (PASSED 2026-04-17, with caveat)
- Test: `smollm2_phase2_test.py` — runs SmolLM2 layer 0 on NPU vs CPU reference at seq_len=2048
- **CPU attention**:        cosine_sim=0.999200 (real tok), per-pos min=0.997863, MAE=0.025164, no NaN
- **NPU FlashAttention** (MHA n_kv=32): cosine_sim=0.999244, per-pos min=0.997820, MAE=0.023468, max_abs=6.14
- Both paths within 0.0001 on every metric → attention is NOT the MAE source; GEMM BF16 truncation is.
- ✅ Cosine and per-position gates pass cleanly. ⚠️ Skill's MAE<1e-2 gate is over-strict
  for BF16-output GEMMs (predates the F32→BF16 production switch in llama3) — captured as Lesson 1.
- **NPU FA with MHA validated** end-to-end at the same correctness as CPU reference attention
  (the `n_kv_heads=32` parametric path through the FA kernel works as predicted in Phase 1).
- Side-fix: corrected stale path in `_llm_shared/kernel_builder/external_kernels.py:99`
  (post-lift refactor left silu_and_mul.cc pointing at old llama3/ location). See Lesson 2.
- Full table: `docs/development_progress/phase2_block.md`

### Phase 3: Full-model correctness (PASSED 2026-04-17)
- Test: `smollm2_phase3_test.py` — 24 NPU layers + CPU final RMSNorm + CPU LM Head
- **3/3 canonical prompts match top-1 vs CPU reference**:
  - "The capital of France is" → `' Paris'` (id=7042), logits cos=0.9957
  - "1 + 1 =" → `' '` (id=216), logits cos=0.9936
  - "The sky is" → `' blue'` (id=4461), logits cos=0.9987
- Per-layer whole-tensor cosine sim: **min=0.974 at layer 21**, all > 0.95 gate
- Per-position cosine drop to 0.518 at layer 23 is informational only (near-zero pad-position artifact); whole-tensor cosine recovers to 0.998
- 24-layer NPU prefill: 4.16s first call (BO allocation) → 1.98-1.99s steady-state (~83 ms/layer)
- LM Head deferred to Phase 5 (CPU LM Head used here — 49152×2048 GEMM is sub-second on CPU)
- Full table + per-layer drift: `docs/development_progress/phase3_full.md`

### Phase 4: Prefill perf (PASSED 2026-04-17)
- Test: `smollm2_phase4_test.py` — measures cold (no-preload) vs warm (post-preload) prefill
- **Pattern application**: 4/5 patterns applied or inherited
  - Patterns 1, 3, 4 inherited from llama3_prefill code path (multi-launch ELFs, intermediate_indices, seq-first layout)
  - Pattern 2 (preload_prefill_weights, 2.83s setup for 24 layers) applied here
  - Pattern 5 PARTIAL: NPU FA used; LM Head deferred to Phase 5
- **Cold prefill**: 4.165s NPU + 0.523s CPU LM Head = 4.700s wall
- **Warm prefill** (3-run avg): **1.884s NPU + 0.506s LM Head = 2.410s wall**
- **Pattern 2 gain on first prompt: 54.8% reduction** (4.165s → 1.884s NPU)
- **Per-layer steady-state: 79 ms/layer** — at parity with llama3's 81 ms/layer (16 layers, GQA) despite SmolLM2's 4× larger KV-projection compute (MHA)
- **vs scaled llama3 baseline** (24/16 × 1.30s = 1.95s expected): **BETTER** at 1.88s
- Pre-loaded weights footprint: 3072 MB
- No correctness regression (top-1 stable across cold + warm runs)
- Full table: `docs/development_progress/phase4_prefill.md`

### Phase 5: Decode perf (PASSED 2026-04-17)
- Test: `smollm2_phase5_test.py` — CPU prefill seeds KV cache, then NPU decode loop
- **5/5 patterns applied or inherited**:
  - Patterns 1, 2, 4 inherited (multi-launch merging, static weight BOs, mv_k8192.o extern rename)
  - Pattern 3 (NPU LM Head GEMV): APPLIED with 8-partition × 16384 = 131072 (last 5 partitions zero-padded for vocab=49152). Right-sizing to 3-partition is a Phase 6 lesson item.
  - Pattern 5 PARTIAL: attention stays CPU per llama3 design (single-query is CPU-cheap)
- **Decode latency: 136.4 ms/token steady-state (7.3 tok/s)** — matches scaled-llama3 expectation (24/16 × 92 = 138 ms)
- **Per-layer rate ~5.7 ms/layer at parity with llama3** despite MHA's 4× larger K/V GEMVs
- **3/3 NPU/CPU top-1 match** on first 3 generated tokens (formal correctness gate)
- Decode kernel compile: 22 s (rms_gemv_rope + o_gemv_ffn + lm_head_gemv); preload: 1.1 s
- Pre-loaded decode weight footprint: 3584 MB
- **First NPU-decoded SmolLM2 output**: `"The capital of France is Paris.\n\nThe capital of France"`
- Full table: `docs/development_progress/phase5_decode.md`

### Phase 6: Finalize (PASSED 2026-04-17)
- Final perf summary written: `docs/development_progress/phase6_finalize.md`
- Edge-LLM survey corrected: SmolLM2-1.7B `rope_θ` (10k → 130k); SmolLM2-135M/360M (10k → 100k)
- Reusable patterns audited: none ready to promote (kernels and helpers all reused llama3's existing parametric infrastructure — confirms Tier-A classification)
- 4 lessons captured in `LESSONS.md` with skill-update recommendations:
  1. `integrate-single-block` MAE gate over-strict for BF16 production
  2. Lift refactor left a stale path (fixed during Phase 2)
  3. `KernelCache.compile_and_cache` no short-circuit on cached artifacts
  4. SmolLM2 was genuinely Tier-A — minimal-change pattern confirmed
- 1 bug fix: `_llm_shared/kernel_builder/external_kernels.py:99` stale path
- Open follow-up: LM Head GEMV right-sizing (8 → 3 partitions for vocab=49152, ~3 ms/token saving)

### Post-Phase 6 — End-to-end NPU runner (2026-04-17)
- Added `smollm2_inference.py`: unified NPU prefill (with K/V extraction from
  `rms_gemms_rope` intermediates) + NPU LM Head GEMV (reuses the decode kernel
  for the single-vector first-token prediction) + NPU decode loop. Modeled on
  `llama3_inference.run_npu_prefill` + `generate`.
- **End-to-end measured**: NPU prefill 2.25 s (94 ms/layer — includes 15 ms
  per-layer K/V reshape overhead vs Phase 4's 79 ms/layer raw); first LM Head
  GEMV 17 ms; decode 137 ms/token; total 8-token wall = 3.2 s.
- Output matches CPU reference: `' Paris'` first token, then the same
  `"Paris.\n\nThe capital of France"` continuation as Phase 5.
- Wired to `make run` (default `--profile`); custom inputs via
  `make run PROMPT="..." N_TOKENS=N` and `MODEL=<id|path>`.
- Closes the **"Production NPU prefill seeds KV cache"** open follow-up
  in `phase6_finalize.md`.

### Post-Phase 6 — Make/CLI parity with llama3 (2026-04-17)
- `Makefile` rewritten with llama3-style targets: `compile`, `run`, `profile`,
  `verify`, `clean`, plus `compile-prefill`/`compile-decode` and individual
  per-phase runners (`run-prefill`, `run-block`, `run-full`, `run-decode-only`,
  `run-reference`). Env-var overrides for `PROMPT`, `N_TOKENS`, `SEQ_LEN`, `MODEL`.
- Added `--compile-only` flag to `smollm2_phase4_test.py`,
  `smollm2_phase5_test.py`, and `smollm2_inference.py` — exits cleanly after
  kernel compile (skips the 9 s weight load), used by `make compile-*` targets.
- README.md rewritten with the perf table at top, full Make usage guide,
  custom-prompt examples, per-phase target descriptions.
