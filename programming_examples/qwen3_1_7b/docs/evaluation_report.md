# Independent evaluation report — qwen3_1_7b

**Evaluator**: evaluate-deployment skill (subagent: general-purpose)
**Date**: 2026-04-21
**Verdict**: PASS-with-warnings

## Measured vs claimed (headline)

| Metric | Measured (this audit) | Claimed | Verdict |
|---|---|---|---|
| Phase 0 (HF logits corr) | 0.99999976 | 0.99999986 | OK |
| Phase 0 top-1 | ' Paris' | ' Paris' | OK |
| Phase 2 cosine_real | 0.998532 | 0.9985 | OK |
| Phase 2 per-pos min | 0.997222 | > 0.98 | OK |
| Phase 2 whole-tensor cosine | 0.999042 | (gate > 0.99) | OK |
| Phase 3 verify top-5 (canonical "France" prompt, N=8) | 8/8 | 8/8 | OK |
| Phase 3 verify top-1 (canonical) | 7/8 (BF16 reorder at step 5) | 7/8 | OK |
| Phase 3 canonical sweep | 6/6 | 6/6 | OK |
| Adversarial top-5 ("Light travels at", N=4) | 4/4 (all top-1 match) | (no prior claim) | OK |
| Adversarial top-5 ("DNA stands for", N=4) | 4/4 (all top-1 match) | (no prior claim) | OK |
| End-to-end first decoded token | ' Paris' | ' Paris' | OK |
| Reproducibility (2 runs byte-identical) | yes ("The capital of France is Paris. The capital of") | implicit | OK |
| Prefill warm @ seq_len=2048 | 2.80 s (100.0 ms/layer) | 2.81 s | OK |
| Decode | 6.73 tok/s (148.6 ms/token) | 6.73 tok/s | OK |

All measured numbers are within ~1% of claims; correctness gates re-derived independently.

## Per-category results

### Cat 1 — Static audit: WARN
- [WARN] No `README.md` and no per-deployment `CLAUDE.md` at the deployment
  root. `programming_examples/CLAUDE.md` (parent) explicitly states "See
  `<model>/CLAUDE.md` in each deployment dir for architecture, file map,
  design patterns, and tile configs" — this expectation is unmet here. The
  Makefile help-block partially substitutes for a README, but the parent
  CLAUDE.md contract is broken.
- [WARN] Makefile target `run-full` (Phase 3) references
  `qwen3_phase3_test.py` (Makefile:31, Makefile:113) but that file does
  NOT exist in the deployment dir. Calling `make run-full` will fail with
  `No such file or directory`. The actual full-model verify lives in
  `qwen3_verify_decode.py` / `qwen3_canonical_sweep.py` (wired via
  `make verify` / `make sweep`).
- [WARN] `qwen3_inference.py:45` imports `install_headfirst_fa_wrapper`
  but never calls it. The wrapper is monkey-patched at the Python level on
  the FA backend used by `qwen3_phase2_test.py:126`. For the production
  `make run` path, the prefill flow uses `npu_full_prefill` from
  `qwen3_phase4_test.py` which (per code reading) goes through head-first
  FA via the kernel cache; no top-level install is needed. Net: import is
  dead code, but the prefill output is correct (verified by Phase 3
  6/6 sweep + Cat 4 ' Paris' first token). Worth either removing the
  import or making the install call explicit for clarity.
- [PASS] No mocked imports, no stubbed `return True`/`return "PASS"`
  patterns, no `if False:` blocks, no commented-out asserts in non-doc
  files (grep over `*.py`).
- [PASS] Decode kernel cache contains the 3 expected fused ELFs:
  `rms_attn_gemvs_qknorm_rope.elf`, `o_gemv_ffn_silu.elf`,
  `lm_head_gemv.elf`. Prefill cache contains
  `rms_attn_gemms.elf`, `flash_attn.elf`, `o_ffn.elf`.

### Cat 2 — Weight + reference smoke: PASS
- `python3 qwen3_weights.py`: loaded HF safetensors, reported correct
  shapes for 28 layers (emb_dim=2048, head_dim=128, hidden_dim=6144,
  vocab=151936), Q/K Norm present, no QKV bias.
- `python3 qwen3_reference.py --prompt "The capital of France is" --verify`:
  CPU top-1 = ' Paris' (id=12095, p=0.5288). Matches HuggingFace
  transformers exactly. Logits correlation **0.99999976**. VERIFICATION
  PASSED.

### Cat 3 — Per-phase re-run: PASS
- Phase 2 (`python3 qwen3_phase2_test.py --no-preload --seq-len 512`):
  cosine_real = **0.998532**, per_pos_min = **0.997222**, MAE = 0.030,
  no NaN. Above the head_dim=128-scaled gate (whole-tensor > 0.99 AND
  per_pos_min > 0.98). Bisect numbers at intermediate stages all > 0.999.
- Phase 3 verify (`make verify N_TOKENS=8`): NPU prefill 2.81 s, first
  decoded token ' Paris'. **Top-1 exact 7/8, top-5 overlap 8/8**. Step 5
  drift (NPU ' the' vs CPU ' Italy') is within CPU's top-2, classified
  by the script as "BF16 reorder" — acceptable per the
  competitive/decisive prompt distinction.
- Phase 3 sweep (`make sweep`): **6/6 PASS**. NPU top-1 matches CPU top-1
  on all 6 canonical prompts (including the competitive "The sky is" →
  ' blue").
- **Adversarial Phase 3** (two prompts not in canonical set):
  - `"Light travels at"` → NPU/CPU agree on " a speed of " (4/4 top-1 match).
  - `"DNA stands for"` → NPU/CPU agree on " deoxyribon" (4/4 top-1 match).
  Both prompts produce non-trivial multi-token continuations. Skepticism
  heuristic about "trivial-token gates" is satisfied: while the canonical
  sweep has 3/6 prompts with ' ' (space, id=220) as top-1, the
  decisive-prompt subset and these two new adversarial prompts all
  exercise real semantic prediction.

### Cat 4 — End-to-end reproducibility: PASS
- Two consecutive `make run N_TOKENS=5` produced **byte-identical** output
  text: `'The capital of France is Paris. The capital of'`.
- Both runs reported NPU prefill 2.80 s warm (100.0 / 99.9 ms/layer),
  decode 6.73 tok/s. The two-run wall delta is within noise — kernels
  were already cached on disk, so both saw a "warm" path.
- Sanity: prefill 2.80 s for 28 layers @ seq=2048 (100 ms/layer) is well
  above the 0.5 s "kernels did fire" threshold; nowhere near the
  suspiciously-fast < 0.1 s region.
- Kernel caches present: `prefill_kernel_cache_2048/{rms_attn_gemms,
  flash_attn, o_ffn}.elf` and `decode_kernel_cache/{rms_attn_gemvs_qknorm_rope,
  o_gemv_ffn_silu, lm_head_gemv}.elf`.

### Cat 5 — Perf integrity: PASS
- Measured prefill 2.80 s vs claimed 2.81 s → 0.4% difference.
- Measured decode 6.73 tok/s vs claimed 6.73 tok/s → exact.
- No claim discrepancy > 20%; single-shot perf integrity confirmed.
  (Multi-trial variance is out of v1 scope.)

### Cat 6 — Cross-deployment regression: PASS
- Trigger fired: `git diff main..HEAD --name-only` shows changes in
  `programming_examples/_llm_shared/`, `programming_examples/llama3/*.py`,
  `programming_examples/llama3/multi_launch_builder/`, and
  `programming_examples/matrix_vector_multiplication/bf16/matvec.py`.
- Ran `cd programming_examples/llama3 && make verify N_TOKENS=4`:
  Logits (pos 5) corr 0.992235, NPU top-1 12366 (' Paris') == CPU top-1
  ' Paris', decode 7.44 tok/s. K/V cache per-layer correlations hold (Q/K
  > 0.97 down through layer 15, V cache slightly noisier 0.87-0.91 — this
  is pre-existing llama3 behavior, not introduced by this deployment).
  No regression observed in the reference llama3 deployment.

## Issues surfaced

1. **`Makefile:31,113`** references `qwen3_phase3_test.py` (`PHASE3 :=`
   variable + `run-full` target) but the file doesn't exist. `make run-full`
   will fail with FileNotFoundError. Either remove the dead target /
   variable, or rename to point at `qwen3_canonical_sweep.py` (which IS
   the Phase 3 gate that actually runs).
2. **Missing `README.md` and `CLAUDE.md`** at
   `/home/jiajli/apps/mlir-air/programming_examples/qwen3_1_7b/`. The
   parent `programming_examples/CLAUDE.md` documents these as expected.
   Other deployments (e.g. `llama3/CLAUDE.md`) have them.
3. **`qwen3_inference.py:45`** imports `install_headfirst_fa_wrapper`
   but never calls it. Either drop the import or make the call explicit
   for documentation. (Functional impact: none — head-first FA is wired
   in via the prefill kernel cache, verified by end-to-end correctness.)
4. **`Makefile:243` `args.profile or True`**: the `or True` short-circuits
   the `--profile` flag, always printing the perf block. Minor — but the
   flag is effectively dead. (`qwen3_inference.py:243`.)
5. Step 5 BF16 reorder (NPU ' the' vs CPU ' Italy') in the canonical
   "France" prompt is documented and within CPU top-2; not a defect, but
   worth flagging that "7/8 top-1 + 8/8 top-5" is the tight gate.

## What was NOT checked (v1 scope)
- Multi-trial perf variance — v2.
- Phase 4 (`qwen3_phase4_test.py`) and Phase 5 (`qwen3_phase5_test.py`)
  per-script re-runs — covered by Cat 4 end-to-end timing instead.
- BO corruption / first-vs-Nth-call drift over a longer decode horizon
  (>8 tokens) — only sampled N=8.
- Cross-deployment regression: only `llama3` was re-verified. The other
  three deployments (`smollm2_1_7b`, `llama32_3b`, `qwen25_1_5b`) were
  not touched in this audit.
- Cold-cache prefill timing — both `make run` invocations hit a
  pre-populated kernel cache.
