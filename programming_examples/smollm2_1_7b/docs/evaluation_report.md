# Independent evaluation report — smollm2_1_7b

**Evaluator**: evaluate-deployment skill (subagent: general-purpose, fresh context)
**Date**: 2026-04-18
**Verdict**: **PASS**

The deployment passes static audit, weight + reference smoke, all per-phase
correctness gates (including 2 adversarial prompts not in the canonical set),
and end-to-end byte-identical reproducibility across two `make run` invocations.
NPU prefill timings (2.25 s / 2.34 s) are well above the 0.1 s "real NPU"
floor, ruling out silent CPU fallback. All measured numbers are within
acceptable variance of the deployment's claims.

## Measured vs claimed (the headline)

| Metric | Measured (this run) | Claimed (deployment doc) | Verdict |
|---|---|---|---|
| Phase 0 — top-1 vs HF | ` Paris` (id=7042), corr=**0.99999978** | top-1 match, corr > 0.999 | ✓ |
| Phase 2 — whole cosine (real-tok) | **0.999244** | 0.999 (phase2_block.md) | ✓ |
| Phase 2 — per-pos min | **0.997820** | ≥ 0.997 | ✓ |
| Phase 2 — NaN | False | False | ✓ |
| Phase 3 — strict top-1 (canonical 6) | **6/6** | 3/3 in old log; 6/6 enriched set passes the gate | ✓ |
| Phase 3 — decisive top-1 (cpu_p>0.5) | **4/4** | all-pass | ✓ |
| Phase 3 — competitive top-5 overlap | **2/2** | all-pass | ✓ |
| Phase 3 — min logits cosine across prompts | 0.9607 (Pacific) | per-layer min 0.974 | ✓ |
| Adversarial top-5 overlap (`Light travels at`, `DNA stands for`) | **2/2** | (no claim — new adversarial check) | ✓ |
| End-to-end first decode token | `.` (after ` Paris` LM-Head seed) | `make run` text identical | ✓ |
| Reproducibility (2 sequential runs identical) | **YES** — `'The capital of France is Paris.\n\nThe'` byte-identical | (greedy decode → deterministic) | ✓ |
| NPU prefill wall (2.25 s / 2.34 s; 94 / 97 ms-per-layer) | matches | 2.25 s / 94 ms/layer (README, phase6) | ✓ |
| Decode tok/s (137 ms/tok ≈ 7.3 tok/s) | matches | 137 ms / 7.3 tok/s (README, phase6) | ✓ |

## Per-category results

### Category 1 — Static audit  [PASS]

- Scaffold present: `smollm2_weights.py`, `smollm2_reference.py`,
  `smollm2_inference.py`, `smollm2_phase{2,3,4,5}_test.py`, `Makefile`,
  `README.md`, `CLAUDE.md`, `TODO.md`, `docs/development_progress/`.
- Makefile targets (`compile`, `run`, `profile`, `verify`, `clean`,
  `run-block`, `run-full`, `run-prefill`, `run-decode-only`, `run-reference`)
  point at existing scripts.
- Grep for `TODO|FIXME|XXX|HACK|TEMPORARY|hardcoded|hard-coded` in `*.py`:
  **no matches**.
- Grep for `if False:` / `MagicMock` / `import mock`: **no matches**.
- README claims "uses NPU FlashAttention" — confirmed: `smollm2_phase2_test.py`
  default is `--npu-attn`, Phase 2 log reports `cpu_attn=False` and
  `attention = NPU FlashAttention`.
- README perf claims (94 ms/layer prefill, 7.3 tok/s decode) match Cat 4 measurements.
- Kernel caches present:
  - `prefill_kernel_cache/` → `flash_attn.elf`, `o_ffn.elf`,
    `rms_gemms_rope.elf`, `manifest.json` ✓
  - `decode_kernel_cache/` → `lm_head_gemv.elf`, `o_gemv_ffn.elf`,
    `rms_gemv_rope.elf`, `manifest.json` ✓

### Category 2 — Weight + reference smoke  [PASS]

`python3 smollm2_weights.py`:
- Config decoded correctly: 24 layers, emb_dim=2048, n_heads=32,
  **n_kv_heads=32 (MHA)**, head_dim=64, hidden=8192, vocab=49152, rope_θ=130000.
- `lm_head` correctly tied to `embed_table` (49152, 2048).
- All 24 per-layer tensor shapes assert clean (wq/wk/wv/wo all 2048×2048;
  w_gate/w_up 2048×8192; w_down 8192×2048).
- RoPE LUT shape (2048, 64), values plausible.

`python3 smollm2_reference.py --prompt "The capital of France is" --verify`:
- Top-1 = ` Paris` (id=7042, p=0.4143). Sensible.
- HF transformers verification: `Top-1 prediction match: YES`,
  **logits correlation 0.99999978** (> 0.999 gate).

### Category 3 — Per-phase re-run  [PASS]

**Phase 2** (`python3 smollm2_phase2_test.py --no-preload --npu-attn --seq-len 2048`):
- attention = NPU FlashAttention, NaN = False, real_tokens = 5.
- whole-tensor cosine (real-tok) = **0.999244**, MAE = 0.023468, max_abs = 6.14
- per-position min cosine = **0.997820**
- Gate (head_dim=64 scaled): whole > 0.99 ∧ per_pos_min > 0.99 ∧ no NaN — all met.
- **PASS**.

**Phase 3** (`python3 smollm2_phase3_test.py --seq-len 2048`, 6 canonical prompts):
- Strict top-1 match: **6/6** (`1+1=`, `2+2=`, `Water freezes at`, `The largest ocean is the`, `The capital of France is`, `The sky is`).
- Decisive (cpu_p > 0.5): 4/4 — all match (` `, ` `, ` `, ` Pacific`).
- Competitive: 2/2 — top-5 overlap (` Paris`, ` blue` recovered).
- Per-prompt logits cos: min=0.9607 (Pacific), max=0.9987 (sky); all > 0.95.
- NaN: False everywhere.
- **PASS**.

**Adversarial Phase 3** (`--prompts "Light travels at" "DNA stands for"`):
- `Light travels at`: NPU=` different` (p=0.238), CPU=` a` (p=0.245). Top-1 mismatch
  but **top-5 overlap = True** in both directions (each top-1 in the other's top-5).
  This is the documented "competitive" gate behavior — neither prediction is
  decisive (both p < 0.3), so the model legitimately sits between alternatives.
- `DNA stands for`: NPU=` de` (p=0.456), CPU=` de` (p=0.410). Top-1 match.
- Decisive: 0/0 (no decisive prompts in this set). Competitive top-5 overlap: **2/2**.
- **PASS**. The adversarial set successfully exercises out-of-canonical prompts;
  the gate held.

**Phase 4 / Phase 5** perf re-runs: SKIPPED per skill v1 scope (too slow).
The phase docs (`phase4_prefill.md`, `phase5_decode.md`) exist with timing
tables; cross-referenced perf numbers (94 ms/layer prefill, 137 ms/token decode)
are independently confirmed by the Cat 4 end-to-end runs below.

### Category 4 — End-to-end reproducibility  [PASS]

`make run N_TOKENS=5`, run twice sequentially (NPU exclusivity respected):

| Run | Generated text | Prefill | Decode (avg) | Wall (real) |
|---|---|---|---|---|
| 1 | `'The capital of France is Paris.\n\nThe'` | 2.25 s (94 ms/layer) | 137 ms/tok | 28.50 s |
| 2 | `'The capital of France is Paris.\n\nThe'` | 2.34 s (97 ms/layer) | 137 ms/tok | 28.64 s |

- **Generated text byte-identical** ✓ (greedy decode determinism upheld).
- **NPU prefill > 0.1 s on both runs** (2.25 s, 2.34 s) — well above the
  "silent CPU fallback" floor for 24 layers. First LM Head GEMV reported as
  17 ms in both runs — also a real NPU GEMV, not a CPU fallback.
- Both runs complete cleanly (no traceback).
- Kernel caches persisted across runs (compile = 0.0/0.1 s reported).
- First-token (` Paris`) matches the deployment's claim and the Phase 0 reference.
- Per-token decode times are stable across the loop (136–138 ms range), indicating
  the multi-launch ELF is being executed each token (no stale cached output).

Note on the "second run faster than first" heuristic from the skill: in this
deployment the kernel-cache hit happens **inside one process** (the multi-launch
ELFs are pre-built), so cold-vs-warm timing across two `make run` invocations
is dominated by re-loading 12.4 s of weight BOs both times. Run 2 is not
meaningfully faster than run 1 by wall clock — but per-token decode is stable
(137 ms both runs) and the cache hit is logged as `Compile / cache load: 0.0s`
on both invocations. This is consistent with cached, repeatedly-executed kernels.

## Issues surfaced

None at gate-blocking severity. Minor observations:

1. **Adversarial prompt `Light travels at`** — NPU and CPU diverge on top-1
   (` different` vs ` a`), but both predictions sit at p ≈ 0.24, so the
   reference itself is ambiguous. Top-5 overlap holds. This is informational,
   not a regression. (Captured by the new adversarial check; not previously
   in the deployment doc set.)
2. **Phase 3 doc claim "3/3 top-1 match"** in `phase6_finalize.md` is from
   the original 3-prompt canonical set; the in-tree script now runs 6 prompts
   and we verified 6/6. The doc is stale relative to the enriched test set,
   but in the safe direction (more rigorous now than claimed).
3. **`per-pos min` of 0.518 at layer 23** noted in `phase3_full.md` is
   correctly explained there as a single-padding-position artifact (whole-tensor
   cos at layer 23 = 0.998); we did not re-derive per-layer drift in this audit
   (`--diagnostic` not run), but the prediction-position cosine numbers we did
   measure (≥ 0.96 across 6 prompts) corroborate that the artifact is benign.

## What was NOT checked (v1 scope limitation)

- Phase 4 prefill warm/cold timing re-measurement (multi-trial variance).
- Phase 5 decode tok/s re-measurement at long generation (≥ 100 tokens).
- Per-layer cosine-drift sweep across all 24 layers (`--diagnostic`).
- Cross-deployment regression vs `llama3/`, `llama32_3b/`, `qwen25_1_5b/`.
- Multi-trial perf variance estimate (single-pair only here).
- `make verify` (CPU/NPU per-token top-1 cross-check) — slow (~25 s/token).
- The `_llm_shared/` and `llama3/` imports were not re-audited (treated as
  trusted shared infrastructure, not the deployment under test).
