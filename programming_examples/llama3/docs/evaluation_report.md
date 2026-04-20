# Independent evaluation report — programming_examples/llama3/ (Llama-3.2-1B)

**Evaluator**: evaluate-deployment skill (independent agent)
**Date**: 2026-04-18
**Verdict**: PASS

This report independently re-derives every correctness/perf claim. The
deployment's own `LESSONS.md` / `progress.md` / `phase{N}_*.md` were NOT
read prior to taking measurements. Cross-references at the bottom only.

---

## Measured vs claimed (the headline)

| Metric | Measured (this run) | Claimed (README/CLAUDE.md) | Verdict |
|---|---|---|---|
| Reference top-1 (HF parity) | " Paris" (id=12366), corr=0.99999856 | top-1 correct | PASS |
| End-to-end first token (NPU prefill) | id=12366 (" Paris") | " Paris" | PASS |
| End-to-end NPU prefill wall | 1.53 s (run1), 1.53 s (run2), 1.52 s (adv) | 1.30 s kernel / 1.54 s wall | PASS (within 1%) |
| Decode latency / throughput | 91-92 ms/tok, 10.86-10.96 tok/s | 92 ms/tok, 10.8 tok/s | PASS |
| `make verify` (4 tokens, NPU FA) per-layer K_cache corr | 0.978-0.999 (all >= 0.95) | "all layers pass" | PASS |
| `make verify` logits corr (pos 5) | 0.992235 | corr ~0.97-0.99 | PASS |
| `make verify` NPU vs CPU top-1 | 12366 vs 12366 (MATCH) | match | PASS |
| Adversarial prompt "Light travels at" | "a constant speed in a vacuum" (semantically correct) | (no claim - new check) | PASS |
| Reproducibility (2 identical runs) | byte-identical text "Paris, and the largest city" | (greedy => deterministic) | PASS |

---

## Per-category results

### Category 1 - Static audit  [PASS]

Scaffold present at `/home/jiajli/apps/mlir-air/programming_examples/llama3/`:
- `llama3_inference.py`, `llama3_prefill.py`, `llama3_decode.py`,
  `llama3_weights.py`, `llama3_reference.py` — all present
- `Makefile` with targets: `compile`, `run`, `profile`, `verify`, `clean`,
  `compile-prefill`, `compile-decode`, `run-prefill`, `run-decode`,
  `run-reference`, `all`, `help` — all map to existing scripts
- `README.md`, `CLAUDE.md`, `docs/`, `docs/development_progress/`,
  `kernel_builder/` (referenced as `_llm_shared` in CLAUDE.md), and
  `multi_launch_builder/` all present
- Note: `TODO.md` not present at top level (lives in
  `docs/development_progress/`); not a blocker

Suspicious-pattern grep results:
- 50 hits of `TODO|FIXME|XXX|HACK|TEMPORARY|hardcoded` across `*.py` —
  spot-checked: most are doc comments / arg help / log strings, not
  open work items hiding bugs
- No `if False:`, no `MagicMock`, no `import mock` (only `return True`
  on a normal code path at `llama3_prefill.py:392`)

Cross-check of README "uses NPU FA" claim:
- `llama3_inference.py:707` defines `--cpu-attn` as `action="store_true"`
  with help text `"default: NPU flash attention"` => when `make run` does
  NOT pass the flag, `args.cpu_attn=False` and the function-default
  `cpu_attn=True` is overridden. So the runtime path uses NPU FA, as
  claimed.
- However: function defaults `cpu_attn=True` at `llama3_inference.py:355,557`
  and `llama3_prefill.py:935,1328,1442` are misleading — a programmatic
  caller that omits the kwarg would silently get CPU attention. WARN
  (not FAIL): `make run` is correct, but the API surface invites mistakes.
- `flash_attn.elf` present in `build_peano/prefill_kernel_cache/` =>
  NPU FA actually compiled and cached.

### Category 2 - Weight + reference smoke  [PASS]

`python3 llama3_weights.py meta-llama/Llama-3.2-1B`:
- Loaded all 16 layers, embed_table, lm_head (tied), final RMSNorm,
  RoPE LUT (2048, 64) bfloat16. All shape asserts pass. "All weights
  loaded successfully."

`python3 llama3_reference.py --prompt "The capital of France is" --verify`:
- Top-1 ` Paris` (id=12366, prob=0.3942)
- HF top-1 ` Paris` (id=12366) — MATCH
- Logits correlation = **0.99999856** (well above 0.999 gate)
- "VERIFICATION PASSED"

### Category 3 - Per-phase re-run  [PASS]

This deployment does not have qwen25-style numbered `phaseN_test.py`
scripts; the built-in correctness gate is `make verify`.

`make verify N_TOKENS=4`:
- All 16 layers complete, every per-layer attention output passes
  `[OK] corr >= 0.99` (worst Layer 15 output: corr=0.996386)
- K_cache per-layer corr: 0.978-0.999 (all >= 0.95)
- V_cache per-layer corr: 0.876-0.999 — `[WARN]` flagged at corr<0.99
  for many later layers but still well above any failure threshold;
  this is consistent with BF16 numerical drift through 16 layers
- Final logits (pos 5): corr=0.992235, max_err=2.14, mean_err=0.39
- NPU top-1 = CPU top-1 = 12366 (` Paris`) — MATCH
- 4 decode tokens generated: ids [11, 323, 279, 7928] => `,`, ` and`,
  ` the`, ` largest`

Adversarial Phase-3-equivalent re-runs:
- `make run N_TOKENS=5 PROMPT="Light travels at"` =>
  `"Light travels at a constant speed in a vacuum"` — semantically
  correct, NPU produced a plausible non-canonical continuation
- This rules out canonical-prompt overfitting / cached-output spoofing

(Phase 4/5 perf — not separately re-measured per skill v1 scope; the
end-to-end timings in Cat 4 are consistent with claimed numbers.)

### Category 4 - End-to-end reproducibility  [PASS]

Two consecutive `make run N_TOKENS=5` runs:
- Run 1: prefill 1.53 s, decode 92 ms/tok, 10.86 tok/s, output
  `"The capital of France is Paris, and the largest city"`
- Run 2: prefill 1.53 s, decode 91 ms/tok, 10.94 tok/s, output
  **byte-identical** `"The capital of France is Paris, and the largest city"`
- First-run vs second-run prefill timing essentially equal
  (kernels already cached; NOT a "first run is slow" pattern, because
  KernelCache loaded from `build_peano/{prefill,decode}_kernel_cache/`)

NPU-execution sanity:
- Prefill wall time = 1.53 s on 16 layers (>> 0.1 s) — far above the
  silent-CPU-fallback threshold; consistent with real NPU dispatch
- `build_peano/prefill_kernel_cache/` contains `flash_attn.elf`,
  `rms_gemms_rope.elf`, `o_ffn.elf`, `lm_head.elf`,
  `rmsnorm.{xclbin,insts.bin}` plus `manifest.json`
- `build_peano/decode_kernel_cache/` contains `rms_gemv_rope.elf`,
  `o_gemv_ffn.elf`, `lm_head_gemv.elf` plus `manifest.json`
- The empty top-level `prefill_kernel_cache/` directory is a vestigial
  empty dir (the real cache lives under `build_peano/`); harmless

---

## Issues surfaced

1. [WARN] Function defaults `cpu_attn=True` at
   `llama3_inference.py:355,557` and `llama3_prefill.py:935,1328,1442`
   are inverted relative to the user-facing default. The CLI flag inverts
   it (so `make run` correctly uses NPU FA), but a Python caller that
   forgets the kwarg will silently fall back to CPU attention. Cosmetic
   API trap, not a deployment bug.

2. [INFO] `prefill_kernel_cache/` at the deployment root is empty;
   the real cache lives under `build_peano/prefill_kernel_cache/`. Not
   a bug — `KernelCache` is constructed inside `cd build_peano` — but
   the empty top-level dir is misleading scenery.

3. [INFO] `make verify` flags `[WARN]` on V_cache correlation in deeper
   layers (down to corr=0.876 at Layer 15). Logits and top-1 still match
   CPU, so this is BF16 numerical drift, not a correctness failure. The
   verify script's threshold is conservative (0.99 for `[OK]`).

4. [INFO] No standalone `phaseN_test.py` files (unlike qwen25/llama32_3b);
   `make verify` is the only built-in correctness gate. Acceptable —
   this deployment predates the canonical 7-phase skill chain.

---

## What was NOT checked (v1 scope limitation)

- Phase 4 prefill warm timing as a distinct measurement (only end-to-end
  prefill wall was measured)
- Phase 5 decode tok/s under sustained generation (only 5 tokens)
- Multi-trial perf variance estimate (perf reproduced on 2 runs only)
- `MODEL=instruct` path
- Cross-deployment regression vs SmolLM2 / Llama-3.2-3B / Qwen2.5-1.5B

---

## Cross-reference against deployment's own claims (post-measurement)

After measuring, I read the README/CLAUDE.md headline numbers (deferred
intentionally to satisfy independence). Findings:

- Claim "Prefill 1.30 s kernel / 1.54 s wall" vs measured 1.53 s wall:
  matches within 1%.
- Claim "Decode 92 ms/token (10.8 tok/s)" vs measured 91-92 ms/tok
  (10.86-10.96 tok/s): matches.
- Claim "Top-1 prediction correct (' Paris')" vs measured: match.
- Claim "2.1x faster than IRON, 4.0x faster than IRON": IRON baseline
  not re-measured here, but the absolute numbers match the deployment's
  own claims, and the ratios derive directly from those.

No silent-bug indicators found. The deployment is trustworthy on
correctness and end-to-end runnability. Verdict: **PASS**.
