# Qwen3-0.6B deployment — independent evaluation

Auditor: independent /evaluate-deployment subagent
Date: 2026-04-20
Target: `programming_examples/qwen3_0_6b/`

## 1. Verdict (TL;DR)

**Ship-with-caveats.** The deployment is genuinely correct end-to-end at the
phases it claims (Phase 0 HF parity, Phase 4 prefill perf, principled Phase 3
gate, no llama3 regression risk). The split-ELF + host Q/K Norm + host RoPE
pipeline is a defensible architectural choice, well-grounded in a real
incompatibility between the predecessor `rope_qk_multi.py` interleaved LUT and
the qwen3_weights half-split LUT. **The biggest single risk is that Phase 5
was redefined.** What other deployments call "Phase 5" (NPU GEMV decode using
the 5 `optimize-decode-perf` patterns) was replaced with CPU decode at
~1.23 s/token, and the deployment marks Phase 5 PASSED. That is a scope
reduction, not a failure of the work performed, but readers cannot rely on
"all 7 phases PASSED" to mean what it means in `qwen25_1_5b/` or `llama3/`.

## 2. Verified claims

I re-ran or re-derived each of the following from scratch (no trust in
progress.md / TODO.md):

- **Phase 0 HF parity**: re-ran `python3 qwen3_reference.py --prompt "The capital of France is" --verify`.
  - Logits correlation: **0.99999986** (matches claim exactly)
  - Top-1 prediction: ' Paris' (id=12095) — matches HF
  - Max abs error 0.008208, mean abs error 0.001495 — matches claim
- **Phase 3 dynamic gate is principled**: re-ran the CPU-only side of all 6
  prompts at seq_len=512, n_layers=28. Observed CPU top-1 probabilities
  match progress.md to 4 decimals:

  | Prompt | CPU top-1 | Prob (re-derived) | Bucket |
  |---|---|---|---|
  | `1 + 1 =` | ' ' | 0.9480 | decisive |
  | `2 + 2 =` | ' ' | 0.9162 | decisive |
  | `Water freezes at` | ' ' | 0.4682 | competitive |
  | `The largest ocean is the` | ' ocean' | 0.2200 | competitive |
  | `The capital of France is` | ' Paris' | 0.6577 | decisive |
  | `The sky is` | ' blue' | 0.1747 | competitive |

  Two of the four prompts the static `canonical_prompts.DECISIVE_PROMPTS`
  bucket lists are NOT actually decisive on Qwen3-0.6B — the dynamic
  reclassification is real, not a softening to make tests pass. Lesson L3
  is well-founded.
- **Phase 4 prefill perf**: re-ran `python3 qwen3_phase4_test.py --seq-len 2048 --cache-dir prefill_kernel_cache_2048 --iterations 1`.
  - Cold: 3.27 s (116.9 ms/layer)
  - Warm: 2.29 s (82.0 ms/layer)
  - Claimed: 2.20 s warm / 78.6 ms/layer. Observed value is +4 % over the
    claim — within run-to-run variance for a single iteration; not a
    fabrication.
- **Phase 2 actually runs NPU FA**: read `qwen3_phase2_test.py` and
  `_llm_shared/phase_helpers/headfirst_fa.py`. The wrapper rebinds
  `_lp._run_cached` and `_lp._attn_backend_kwargs`; the per-model test
  uses `import llama3_prefill as _lp; _lp._run_cached(...)` indirection
  exactly as Lesson L2 prescribes. The FA path is engaged only when the
  user does NOT pass `--cpu-attn`; the default path runs head-first NPU
  FA. (See Phase 2 helper at lines 226-240.)
- **Predecessor `rope_qk_multi` LUT genuinely is incompatible**: read both
  `superseded/rope_qk_multi.py::_build_lut` (interleaved
  `lut[:, 0::2]=cos, lut[:, 1::2]=sin`) and
  `qwen3_weights.py::generate_rope_lut` (half-split
  `lut[:, :half]=cos, lut[:, half:]=sin`). The two layouts genuinely
  cannot share a kernel without regenerating the LUT or rewriting the
  kernel. The host-RoPE fallback is honest, not a cover for a real bug.
- **Backward-compat of shared builder edits — VERIFIED IDENTICAL**:
  - `build_rms_attn_gemms_module(seq_len=128, emb_dim=2048, kv_dim=512)` produces
    a module that is **byte-identical** to the same call with explicit
    `q_dim=2048` (i.e., `q_dim=None` defaulting to `emb_dim` is preserved).
  - `build_o_ffn_module(seq_len=128, emb_dim=2048, hidden_dim=8192)` produces
    a module that is **byte-identical** to the same call with explicit
    `o_in_dim=2048`. No regression risk for llama3.
- **Make targets**: `make help` prints sensible help. `make run-reference`
  is wired to `qwen3_reference.py --verify` and was the source of the
  Phase 0 verification above.
- **Phase 0 weight loader exposes the right attributes**: `qk_norm=True`,
  `qkv_bias=False`, n_layers=28, head_dim=128, vocab=151936, rope_θ=1e6.

## 3. Unverified or weakened claims

- **Phase 5 PASS is a scope reduction.** `qwen3_phase5_test.py` itself
  documents the "relaxed gate":
  > Phase 5 GATE (relaxed for this deployment):
  >   - Decode produces sensible continuation tokens (no NaN, no all-zeros)
  >   - End-to-end NPU prefill + CPU decode wall-clock measured and reported

  Compare to `qwen25_phase5_test.py`, which exercises NPU GEMV decode with
  `run_decode_block`, `pre_transpose_decode_weights`,
  `preload_qwen25_lm_head`, etc. Qwen3 ships with **NONE of the 5
  `optimize-decode-perf` patterns applied** (P1 multi-launch, P2 static
  weight BOs, P3 NPU LM-head GEMV, P4 extern kernel rename, P5 CPU→NPU
  promotion). The TODO is honest about the follow-up; the
  `phase6_finalize.md` perf table also flags decode as CPU. But
  `[x] 5: Decode (PASSED)` in TODO.md is misleading next to the qwen25
  history, where Phase 5 means "NPU decode with patterns applied".
  Recommend reword: "Phase 5 deferred — CPU-decode interim".
- **Phase 2 cos_real=0.9988 / per_pos_min=0.997** — not re-run (would
  require a cold kernel-compile cycle). Phase 4 was re-run instead, which
  exercises the same compiled kernels at seq_len=2048; correctness wasn't
  regressed by the warm-prefill perf path.
- **Phase 4 "78.6 ms/layer"**: my single-iteration replication came in at
  82.0 ms/layer (+4 %). The claim's average of multiple iterations is
  plausibly the actual mean, but a single-iter audit cannot distinguish
  the claim from variance. Both are clearly in the qwen25-parity ballpark.
- **`make run` description is mildly misleading**: the help text says
  "qwen3_inference.py — NPU prefill (with KV extraction) + NPU decode" but
  the runner does NPU prefill + **CPU decode** (the file's own header is
  honest: "NPU prefill + CPU decode loop"). Bug in the help string only.

## 4. Real bugs found

- **`make help` describes `run` as NPU+NPU but it's NPU+CPU**
  (`Makefile:59`):
  ```
  @echo "  make run             qwen3_inference.py — NPU prefill (with KV extraction) + NPU decode"
  ```
  but `qwen3_inference.py:42` imports `cpu_decode_token` and the loop at
  `qwen3_inference.py:147` calls `cpu_decode_token(...)`. Cosmetic but
  contradicts the file's own honest header. Fix: replace "NPU decode"
  with "CPU decode (NPU GEMV decode is a Phase 5+ follow-up)".
- **No on-disk indication that decode is CPU in default `make run`
  output**: the runner prints "CPU decode (avg)" so a user inspecting the
  output learns the truth, but a reader of TODO.md plus `make help` could
  reasonably believe NPU decode was achieved.

No correctness bugs found in the code paths I read or the runs I executed.

## 5. Cross-deployment regression risk

- **Smoke-tested both shared edits** with the OLD llama3-style call
  signatures. Both produce **byte-identical** modules to the explicit
  pass-through case. The `q_dim is None → q_dim = emb_dim` and
  `o_in_dim is None → o_in_dim = emb_dim` defaults are sound.
- The new shared file `_llm_shared/phase_helpers/qk_norm.py` is purely
  additive — no other deployment imports it. Zero blast radius.
- Did not re-run llama3 `make run` end-to-end (would consume the NPU for
  several minutes); the byte-identical-IR check is stronger evidence than
  a runtime smoke would be, since identical IR ⇒ identical compiled ELF.

**Verdict: no regression risk.**

## 6. Lessons quality assessment

The three lessons in `docs/development_progress/LESSONS.md` are **real
root-cause-and-rule lessons**, not "we hit a bug" notes:

- **L1 (cache hygiene)** — concrete root cause (seq_len-128 ELF reused for
  seq_len-512 input → garbage), concrete rule (cache name doesn't encode
  seq_len; wipe on shape change). Generalizable.
- **L2 (head-first FA monkey-patch)** — concrete root cause (snapshotted
  function reference vs. module-attribute lookup), concrete rule
  (`import llama3_prefill as _lp; _lp._run_cached(...)`). The Phase 2/3/4/5
  test scripts ALL use this indirection — proof the lesson was internalized,
  not just written down. Cargo-cult risk is low here.
- **L3 (dynamic decisive/competitive)** — concrete root cause (canonical
  buckets calibrated on Llama-3.2-1B don't transfer), concrete rule
  (re-classify by observed CPU top-1 prob). I independently re-derived
  the prob table and confirmed two of four "decisive" prompts genuinely
  are not decisive on Qwen3-0.6B. The lesson is principled.

All three lessons are at the bar a staff engineer would expect. No
"we tried something and it worked" filler.

## 7. Recommendation

**SHIP-WITH-CAVEATS.**

What's good (no further action needed):
- Correctness pipeline is honest and reproducible. Phase 0 / 3 / 4 all
  re-derive cleanly.
- The split-ELF architectural decision (host Q/K Norm + host RoPE) is
  defensible — RMSNorm doesn't commute with RoPE for asymmetric weights,
  and the predecessor RoPE LUT layout genuinely doesn't match the qwen3
  generator. Documenting both as the rationale is appropriate.
- Shared builder extensions (`q_dim`, `o_in_dim`) are byte-compat with
  llama3. No regression risk.
- Lessons are root-cause + reusable rule, not anecdotes.

Caveats / before-shipping fixes recommended:
1. **Reword Phase 5 status everywhere** ("PASSED" → "deferred — CPU-decode
   interim") in TODO.md, phase6_finalize.md, and CLAUDE.md. The
   `phase5_test.py` already says "Phase 5 GATE (relaxed)"; let the
   summary docs match.
2. **Fix `Makefile` `make help` line** for `run`: change "NPU decode" to
   "CPU decode (NPU GEMV is a Phase 5+ follow-up)".
3. (Optional) Add a one-line caveat near the top of CLAUDE.md "Status"
   block clarifying that decode is CPU-only.

If the user does not need NPU decode for Qwen3-0.6B at this moment,
this deployment is shippable as the "NPU prefill + CPU decode" reference
for the Q/K-Norm / split-ELF pattern. Future Qwen3-class models
(Qwen3-1.7B / Qwen3-4B / Qwen3-8B) can directly reuse the qk_norm helper
and the builder extensions.

If NPU decode is required for parity with the other four deployments,
this deployment is **not** at parity; the documented ~half-deployment-
session of follow-up to add the rms_attn_gemvs split GEMV ELF + LM-head
GEMV would close that gap.
