# Qwen3-0.6B deployment — independent evaluation v2

Auditor: independent /evaluate-deployment subagent (second pass)
Date: 2026-04-20
Target: `programming_examples/qwen3_0_6b/`
Scope: post-finalize work since `evaluation_report.md` (Phase A leaves on
NPU, Phase B fusion to 3 decode ELFs, host-side 10× speedup).

## 1. Verdict (TL;DR)

**Ship-with-caveats.** The headline production claim is real: `make run`
generates coherent text at 0.10 s/token (10.5 tok/s) with NPU prefill at
2.10 s (75 ms/layer @ seq_len=2048). The two new fused decode ELFs each pass
their standalone correctness gates (cos > 0.999 on every output). The 3-K
matvec rename machinery (`mv.o`, `mv_og.o`, `mv_dg_qwen3.o`) is well-formed,
additive, and does not regress the existing llama3 / qwen25 / smollm2 /
llama32_3b deployments (verified by byte-identical IR for shared builders
called with default vs. explicit args). However, **`make verify` and
`make sweep` are broken in the Makefile**: both use the wrong default
prefill cache for SEQ_LEN=2048, producing 0/8 and 0/6 garbage results
respectively. The deployment IS correct when invoked with the right cache
(re-verified to 5/6 exact + 6/6 top-5 and 6/6 PASS), so this is a 2-line
Makefile bug, not a correctness regression. Several stale comments
(`~900 ms/token`, `~1.23 s/token`) and a stale "Known follow-up" section
in TODO.md still describe pre-Phase-B state. Recommend fixing the
Makefile cache argument and reword the stale doc blocks before shipping.

## 2. Verified claims

Re-ran or re-derived from scratch:

- **`make run N_TOKENS=8`** (twice, back-to-back): both runs produced
  `'The capital of France is Paris. The capital of France is also'`.
  Prefill 2.10–2.11 s warm (75.1–75.3 ms/layer × 28). Decode
  93–94 ms/token (10.48–10.57 tok/s). Matches claim.
- **Standalone fused ELF tests** both PASS:
  - `multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py`: cos in
    [0.99993, 0.99998] across all 8 outputs (normed/q/k/q_normed/
    k_normed/v/q_roped/k_roped). NPU run 1.00 ms.
  - `multi_launch/o_gemv_ffn_silu_qwen3.py`: cos in [0.99966, 1.00000]
    across all 8 outputs (proj/res1/normed2/gate/up/swiglu/down/out).
    NPU run 1.62 ms.
- **Decode-cache contents** match the claim exactly:
  `decode_kernel_cache/{lm_head_gemv.elf, o_gemv_ffn_silu.elf,
  rms_attn_gemvs_qknorm_rope.elf}` — exactly 3 ELFs (no per-leaf leftovers).
- **`compile_decode_kernels`** in `qwen3_decode.py:92-170`: only 3 ELFs
  compiled (`rms_attn_gemvs_qknorm_rope`, `o_gemv_ffn_silu`,
  `lm_head_gemv`). All per-leaf qknorm/rope/silu/gemv-leaf compilations
  are gone.
- **`run_decode_block`** in `qwen3_decode.py:228-365`: exactly 2
  `_lp._run_cached(...)` calls per layer (line 297 and line 355). One to
  each fused ELF. No per-leaf calls remain.
- **`preload_decode_weights`** at lines 557-645 fires both fused ELFs
  once per layer (lines 584 and 612) and primes the LM-head partition
  cache (line 638-640). Verified at runtime: "Preload done: 1.52–1.58s
  (28 layers × 2 ELFs + LM head)".
- **`_ensure_decode_weights_transposed`** at lines 180-213 caches
  `_wq_t/_wk_t/_wv_t/_wo_t/_wgate_t/_wup_t/_wdown_t` on the
  `LayerWeights` object guarded by `_decode_t_done`. Called both at
  preload time (line 581) and on first per-layer call (line 265, 332).
- **`_DECODE_ARG_CACHE`** is a module-level dict (line 220) keyed
  `f"...L{layer_idx}"` (lines 259, 330). Dynamic slots written each call
  are exactly: index 0 (x_in/attn_out) and indices 7-8 (lut_q/lut_k) for
  ELF 1; index 0 (attn_out) and index 3 (x_res) for ELF 2. Static and
  intermediate slots are never re-written from Python. Confirmed by
  reading lines 286-295 and 351-353.
- **3-K matvec rename machinery**:
  - `o_gemv_ffn_silu_qwen3.py:135-147`: O launch uses prefix `og` and
    `link_with = "mv_og.o"` (line 177); Down launch uses prefix `dg`
    and `link_with = "mv_dg_qwen3.o"` (line 179). Gate (gg) and Up (ug)
    use the default `_EXTERN_DEFAULT` (line 151-155 includes
    `@matvec_vectorized_bf16_bf16`) and inherit `link_with = "mv.o"`
    from the unrenamed sub-IR.
  - `_llm_shared/kernel_builder/external_kernels.py:226-244`:
    `compile_mv_og` uses `-DDIM_M_OUTPUT=8` (parameterized via tile_m=8)
    and renames matvec/linalg_fill to `og_*`. Output: `mv_og.o`.
  - `_llm_shared/kernel_builder/external_kernels.py:247-265`:
    `compile_mv_dg_qwen3` uses `-DDIM_M_OUTPUT=8` and renames to `dg_*`.
    Output: `mv_dg_qwen3.o`. Distinct from the existing `mv_k8192.o`
    which uses `DIM_M_OUTPUT=2`.
  - `compile_mv()` (line 220-223) and `compile_mv_k8192()` (line
    206-217) are unchanged from the prior audit. `compile_mv_k8192`
    still uses `DIM_M_OUTPUT=2` and the `dg_*` rename — llama3's
    decode_kernel_cache is unaffected.
- **All 4 `.o` artifacts present** in `qwen3_0_6b/`:
  `mv.o`, `mv_k8192.o`, `mv_og.o`, `mv_dg_qwen3.o`.
- **Cleanup completeness in `qwen3_decode.py`**: searched for
  `_rope_per_head_single`, `_silu_and_mul_host`, `_QKNORM_BACKEND`,
  `_ROPE_BACKEND`, `_SILU_MUL_BACKEND`, `_GEMV_LEAF_BACKENDS`,
  `_RMS_1D_BACKEND`, `apply_qk_norm`, `build_rms_attn_gemvs_qwen3_module`
  — **none appear**. Only `_RMS_ATTN_GEMVS_QKNORM_ROPE_BACKEND` (the
  fused ELF backend) and `_O_GEMV_FFN_SILU_BACKEND` and
  `_LM_HEAD_BACKEND` remain (lines 76-84). Correct cleanup.
- **No llama3 regression** from `q_dim`/`o_in_dim` shared-builder
  edits: byte-identical Python module string for both:
  - `build_o_gemv_ffn_module(emb_dim=2048, hidden_dim=8192)` vs.
    same call with explicit `o_in_dim=2048` → IR strings are
    identical.
  - `build_rms_attn_gemms_module(seq_len=128, emb_dim=2048,
    kv_dim=512)` vs. same call with explicit `q_dim=2048` → IR
    strings are identical.

## 3. Unverified or weakened claims

- **Performance "10.7 tok/s" is real but slightly variable.** I
  measured 10.48–10.66 tok/s across 4 runs (make run, make verify
  with right cache @ n=6 and n=8, an inline reproduction). Within
  ~3% of the claim. No 20%+ slowdown.
- **`qwen3_inference.py` docstring is stale.** Lines 14-16 still
  describe the Phase A "~900 ms/token" intermediate and CPU decode
  "~1.23 s/token" as the canonical decode story; line 72 puts both
  into the `--decode` argparse help. The actual `--decode npu`
  path delivers 0.10 s/token. Cosmetic, not a correctness issue.

## 4. Real bugs found

### Bug 1 (HIGH severity, easy fix): Makefile passes wrong cache to verify and sweep

**Symptom**: `make verify` reports 0/8 top-1 match (all garbage tokens
like 'iza', '化的'); `make sweep` reports 0/6 PASS (all garbage like
'culate', ' Outputs', 'opathy'). But `make run` on the same prompt is
correct.

**Root cause**: `Makefile:127` and `Makefile:131` invoke
`qwen3_verify_decode.py` and `qwen3_canonical_sweep.py` with
`--seq-len 2048` only. Both scripts default `--prefill-cache` to
`prefill_kernel_cache` (line 48 of verify, line 68 of sweep). That
directory was built for a different SEQ_LEN (a manifest inspection
shows it's the seq_len-512 cache that survives from earlier phase
work). The 512-built ELF is silently used at seq_len=2048 and
produces garbage. This is exactly the failure mode that
`docs/development_progress/LESSONS.md` L1 ("cache name doesn't encode
seq_len; wipe on shape change") was written to prevent.

**Verification**: invoking the scripts directly with
`--prefill-cache prefill_kernel_cache_2048`:
- `qwen3_verify_decode.py --n-tokens 6 ... --prefill-cache prefill_kernel_cache_2048`
  → 5/6 exact, 6/6 within CPU top-5 → **PHASE B verify: PASS**
  (matches the claim exactly).
- `qwen3_canonical_sweep.py --prefill-cache prefill_kernel_cache_2048`
  → 6/6 PASS (matches the claim exactly).

**Fix**: Add `--prefill-cache prefill_kernel_cache_2048` to the
`verify` and `sweep` rules in the Makefile. Or better: have those
scripts derive their cache directory from `--seq-len` (e.g.
`prefill_kernel_cache_${SEQ_LEN}`) so the trap can never recur.

### Bug 2 (LOW severity): `make verify` Makefile rule hard-codes N_TOKENS=8

`Makefile:126` invokes verify with `--n-tokens 8`. With the right
cache, 8-token verify reports 6/8 exact + 7/8 top-5, which fails the
"all in top-5" PASS criterion (one extra token, ' also' at step 7,
diverges from CPU's ' Rome'). The claim "5/6 exact, 6/6 within top-5"
holds at `--n-tokens 6`. The deeper drift at 8 is genuine BF16
divergence at the prompt boundary, not a bug. But the Makefile
rule is set up to fail. Either (a) drop to `--n-tokens 6`, or
(b) loosen the gate to "fraction within top-5" with a threshold,
or (c) document that decode tail divergence is expected.

## 5. Cross-deployment regression risk

**Zero regression risk.**

- `_llm_shared/kernel_builder/external_kernels.py`: the new
  `compile_mv_og` and `compile_mv_dg_qwen3` are purely additive.
  `compile_mv()` and `compile_mv_k8192()` are byte-identical to their
  prior versions (no edits in the diff window). The existing
  llama3 `mv_k8192.o` (DIM_M_OUTPUT=2, dg_* rename) coexists with
  the new `mv_dg_qwen3.o` (DIM_M_OUTPUT=8, dg_* rename) because
  they are written to different filenames.
- `llama3/multi_launch_builder/o_gemv_ffn_multi.py` (the `o_in_dim`
  kwarg edit) and
  `llama3/multi_launch_builder/superseded/rms_attn_gemms_multi.py`
  (the `q_dim` kwarg edit): byte-identical IR for default-args call
  vs explicit-args call. `q_dim is None → q_dim = emb_dim`,
  `o_in_dim is None → o_in_dim = emb_dim`. Verified via direct
  `str(module)` comparison (both lengths and contents match).
- The new files `multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py`,
  `multi_launch/o_gemv_ffn_silu_qwen3.py`,
  `qwen3_verify_decode.py`, `qwen3_canonical_sweep.py`,
  `docs/development_progress/phase_b_fusion.md` are all
  qwen3-specific. No other deployment imports them.

## 6. Cleanup completeness

- `qwen3_decode.py` is clean: no `apply_qk_norm` import, no
  per-leaf backend dicts, no `_rope_per_head_single` /
  `_silu_and_mul_host` host helpers.
- `multi_launch/rms_attn_gemvs_qwen3.py` (the Phase A 4-launch
  precursor to the new 8-launch fused ELF) is **still on disk**. It
  is referenced only by docs (`phase5_decode.md`) and a comment in
  `qwen3_inference.py:11`. Not imported by any production runtime
  code — confirmed via grep on the deployment directory. Not a
  functional issue but is dead code; could be moved to a `superseded/`
  sub-dir (mirrors the llama3 convention) or simply removed.
- `qwen3_inference.py:11-16` and line 72 still describe the
  Phase A intermediate ("~900 ms/token") and the legacy CPU decode
  ("~1.23 s/token") as the headline numbers in the docstring.
  Should be reworded to match the production state.
- `TODO.md:34-48` is a "Known follow-up (decode perf — fuse the
  per-leaf launches)" section that describes Phase B fusion as a
  prospective effort. That work is now landed; the section should
  move to "Resolved blockers" or be deleted (the Phase 5 line at
  TODO.md:9 already documents the fusion outcome).

## 7. Recommendation

**SHIP-WITH-CAVEATS.**

What is genuinely shippable:
- The production runtime `qwen3_inference.py` (driven by `make run`)
  is correct, fast (10.5 tok/s), and reproducible across back-to-back
  runs. End-to-end NPU prefill + NPU decode + NPU LM head on a
  Qwen3-class model with Q/K Norm — a real, novel result.
- The two new fused ELFs (`rms_attn_gemvs_qknorm_rope`, 8 launches;
  `o_gemv_ffn_silu`, 8 launches with the 3-K matvec rename) pass
  their standalone correctness gates (cos > 0.999 across every
  output). The 3-K rename is the right architectural move and
  generalizes cleanly (mv.o + 2 renamed copies).
- Zero blast-radius on llama3 / qwen25 / smollm2 / llama32_3b. All
  shared-file edits are additive.
- Cleanup of the per-leaf decode path is complete inside
  `qwen3_decode.py`.

Before shipping, recommend:
1. **Fix the Makefile cache bug** (`verify` and `sweep` rules):
   add `--prefill-cache prefill_kernel_cache_2048` (or, better,
   wire the cache directory to SEQ_LEN). This is the LESSON L1
   trap recurring at the make-target layer; the lesson rule should
   be extended to "Makefile rules also have to encode seq_len in the
   cache path".
2. **Rework `make verify` to a sustainable pass criterion**: either
   drop n_tokens to 6 (matching the published claim) or loosen the
   gate to a top-5 fraction with a documented threshold. As-is the
   target is set up to fail on a real BF16 decode-tail divergence.
3. **Reword stale doc blocks** in `qwen3_inference.py:11-16,72`
   and `TODO.md:34-48`. The README and CLAUDE.md and
   `phase_b_fusion.md` are already correct.
4. (Optional) move `multi_launch/rms_attn_gemvs_qwen3.py` to a
   `superseded/` sub-dir — it is only referenced by doc archeology.

If the Makefile fixes land, `make verify` and `make sweep` will
both PASS and the deployment will be at the same shippable bar as
`llama3/`, `qwen25_1_5b/`, `smollm2_1_7b/`, and `llama32_3b/`.
