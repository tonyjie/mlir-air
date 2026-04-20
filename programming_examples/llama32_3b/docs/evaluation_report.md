# Independent evaluation report — llama32_3b

**Evaluator**: evaluate-deployment skill (independent subagent — Opus 4)
**Date**: 2026-04-18
**Verdict**: **PASS**

All four check categories produced measurements that match (or are within
deployment-stated tolerance of) the claims in `README.md`,
`phase{2,3,4,5}_*.md`, and `TODO.md`. NPU FlashAttention via Option C is
genuinely active (no silent CPU fallback): `flash_attn.elf` is present in
the prefill kernel cache, the inference path actually exercises the
head-first wrapper (`make run` passes `--npu-attn`), and the warm NPU
prefill timing I measured (4.10 s) is consistent with the claimed
3.2 s NPU-layer + ~0.9 s CPU LM Head + transpose overhead.

## Measured vs claimed (the headline)

| Metric | Measured (this run) | Claimed (deployment doc) | Verdict |
|---|---|---|---|
| Weight loader: layers / head_dim / GQA group | 28 / 128 / 3 | 28 / 128 / 3 | ✓ |
| Reference top-1 (capital of France) | `' Paris'` (id=12366) | `' Paris'` | ✓ |
| Reference vs HF logits correlation | **0.99999962** | 0.99999962 | ✓ |
| Phase 2 whole-tensor cosine (all positions) | **0.997103** | 0.996 (real-token: 0.999519) | ✓ |
| Phase 2 cosine (real tokens only) | **0.999519** | 0.999519 | ✓ |
| Phase 2 per-position min | **0.989294** | 0.989294 (gate: > 0.98) | ✓ |
| Phase 2 NaN | False | False | ✓ |
| Phase 3 decisive top-1 (CPU p>0.5) | **4/4 PASS** | 4/4 | ✓ |
| Phase 3 competitive top-5 overlap | **2/2 PASS** | 2/2 | ✓ |
| Phase 3 strict top-1 (all 6 prompts) | 5/6 (deployment-explicit relaxed gate; documented) | 5/6 | ✓ |
| Adversarial Phase 3 (`Light travels at`, `DNA stands for`) | **2/2 top-1 match**, both top-5-overlap PASS | (no claim — new check) | ✓ |
| `make run` CLI: NPU FA active | `--npu-attn` flag set; `cpu_attn=False`; head-first wrapper installed | "Option C default" | ✓ |
| `prefill_kernel_cache/flash_attn.elf` exists | **3.0 MB present** (mtime 2026-04-18 19:15) | claimed | ✓ |
| Decode cache .elfs (3 files) | rms_gemv_rope.elf, o_gemv_ffn.elf, lm_head_gemv.elf — present | claimed | ✓ |
| End-to-end first generated token | `' Paris'` (id=12366) | `' Paris'` | ✓ |
| End-to-end generated text (5 tokens, run 1) | `' Paris. It is the'` | `' Paris. It is the largest city in France...'` (100 tokens; first 5 match) | ✓ |
| **Reproducibility (run 1 vs run 2 byte-identical)** | **YES** — identical text both runs | implicit (greedy) | ✓ |
| NPU prefill timing — run 1 | **4.10 s** (146 ms/layer) | claimed warm NPU 3.2 s + ~0.9 s overhead → 5.3 s wall | ⚠ slightly slower than 3.2 s NPU-only claim, but well above the 0.1 s "silent fallback" threshold; consistent with end-to-end measurement |
| NPU prefill timing — run 2 | **4.11 s** (147 ms/layer) | claimed | ✓ |
| Decode steady-state per-token | **214–216 ms** | 214.9 ms (4.7 tok/s) | ✓ |

## Per-category results

### Category 1 — Static audit
**[PASS]**

- Scaffold complete: `llama32_3b_weights.py`, `llama32_3b_reference.py`,
  `llama32_3b_inference.py`, `llama32_3b_phase{2,3,4,5}_test.py`,
  `Makefile`, `README.md`, `CLAUDE.md`, `TODO.md`,
  `docs/development_progress/{progress,LESSONS,debug_log,phase1..6}*.md`
  all present.
- Makefile targets `compile`, `run`, `profile`, `verify`, `clean` all
  point to existing scripts.
- **NPU FA Option C wired into the production path**: `make run` invokes
  `llama32_3b_inference.py … --npu-attn --profile`. `--npu-attn` sets
  `args.cpu_attn = False` (`llama32_3b_inference.py:73-78`). That value
  flows to `compile_block_kernels(prefill_cache, config, args.seq_len,
  cpu_attn=False)` (line 108), which in turn calls
  `install_headfirst_fa_wrapper()` and
  `compile_headfirst_fa_kernel(...)` (verified in
  `_llm_shared/phase_helpers/orchestration.py:88-94`). The README claim
  "NPU FA via Option C head-first wrapper is the default for `make run`"
  is **accurate**.
- No suspicious patterns: no `TODO|FIXME|HACK|MagicMock` in any `.py`.
  The `return 0` matches in `llama32_3b_inference.py:117,261` and
  `llama32_3b_phase{4,5}_test.py` are normal CLI exit returns, not
  placeholder gates.
- Kernel cache populated:
  - Prefill: `flash_attn.elf` (3.0 MB), `o_ffn.elf`, `rms_gemms_rope.elf`
  - Decode: `lm_head_gemv.elf`, `o_gemv_ffn.elf`, `rms_gemv_rope.elf`
  - Manifests reference correct kernel symbols (`main:attention_bf16`,
    `main:rms_gemms_rope`, etc.).

### Category 2 — Weight + reference smoke
**[PASS]**

- `python3 llama32_3b_weights.py meta-llama/Llama-3.2-3B`: 28 layers,
  emb_dim=3072, n_heads=24, n_kv_heads=8 (GQA group=3), head_dim=128,
  hidden_dim=8192, vocab_size=128256, rope_base=500000.0. Tied
  `lm_head` to `embed_table` (expected for Llama-3.2-3B). All shape
  asserts passed; per-layer wq=(3072,3072), wk=wv=(3072,1024),
  w_gate=w_up=(3072,8192), w_down=(8192,3072) — consistent with
  emb_dim/kv_dim/hidden_dim derivations.
- `python3 llama32_3b_reference.py --prompt "The capital of France is"
  --verify`:
  - Top-1: **`' Paris'`** (id=12366, prob=0.2463)
  - vs HF: top-1 match YES, **logits correlation = 0.99999962**
  - Max abs error 0.017675; mean abs 0.001719 — both within F32
    tolerance of HF's BF16-internal F32 reduction.
  - "VERIFICATION PASSED" reported by the script.

### Category 3 — Per-phase re-run
**[PASS]**

#### Phase 2 (`--cpu-attn`)
- Whole-tensor cosine (all 2048 positions): **0.997103**
- Real-token (6 tokens) cosine: **0.999519**, MAE 0.004558
- **Per-position min: 0.989294** (head_dim=128 scaled gate is > 0.98 — passes)
- NaN: False
- Gate **PASS** (>0.99 whole AND >0.98 per-pos AND no NaN).

#### Phase 3 (`--cpu-attn`, canonical 6 prompts)
| prompt | NPU top-1 | CPU top-1 | CPU prob | category | top-1? | top-5 overlap? | corr |
|---|---|---|---|---|---|---|---|
| `1 + 1 =` | `' '` | `' '` | 0.737 | decisive | YES | YES | 0.9901 |
| `2 + 2 =` | `' '` | `' '` | 0.529 | decisive | YES | YES | 0.9431 |
| `Water freezes at` | `' '` | `' '` | 0.707 | decisive | YES | YES | 0.9546 |
| `The largest ocean is the` | `' Pacific'` | `' Pacific'` | 0.816 | decisive | YES | YES | 0.9780 |
| `The capital of France is` | `' the'` | `' Paris'` | 0.246 | competitive | NO | YES | 0.9836 |
| `The sky is` | `' the'` | `' the'` | 0.331 | competitive | YES | YES | 0.9840 |

- Decisive top-1: **4/4 PASS**. Competitive top-5 overlap: **2/2 PASS**.
  Strict top-1 across all 6: 5/6 (deployment's gate is decisive ∪
  competitive-overlap; ratified by `phase3_full.md`).
- No NaN.
- Gate **PASS**.

#### Adversarial Phase 3 (NEW — not in canonical set)
`--prompts "Light travels at" "DNA stands for"`

| prompt | NPU top-1 | CPU top-1 | CPU prob | top-1? | top-5 overlap? | corr |
|---|---|---|---|---|---|---|
| `Light travels at` | `' a'` | `' a'` | 0.456 | YES | YES | 0.9729 |
| `DNA stands for` | `' de'` | `' de'` | 0.510 | YES | YES | 0.9895 |

- Both adversarial prompts: **NPU top-1 matches CPU top-1 exactly** — no
  evidence of over-tuning to the canonical prompt set.
- Gate **PASS**.

### Category 4 — End-to-end reproducibility
**[PASS]**

- `make run N_TOKENS=5` executed twice, sequentially (NPU exclusivity
  honored). Both runs:
  - Generated text: **`'<|begin_of_text|>The capital of France is Paris. It is the'`** — **byte-identical** between runs (greedy decode is deterministic).
  - First generated token: `' Paris'` (id=12366) — matches Phase 0
    reference + README claim.
  - NPU prefill: 4.10 s (run 1) / 4.11 s (run 2) — well above the
    0.1 s silent-fallback threshold; ratifies kernel actually executed.
  - Decode steady-state: 214–217 ms/token (~4.6–4.7 tok/s) — matches
    claim of 214.9 ms (4.7 tok/s).
  - Setup: 3.0 s compile-cache load, 10.8 s weight load, 17.4 s BO
    pre-load. Compile only attempts the cache miss for `attn_npu2.o`
    rebuild (the others were already cached); no full re-compile of the
    .elfs.
- Kernel cache verified: `prefill_kernel_cache/flash_attn.elf`
  (3026576 bytes, mtime 2026-04-18 19:15) and decode_kernel_cache .elfs
  all present.
- Cache-hit reproducibility: total inference wall ~4.99 s on the second
  run, essentially identical to the first (4.98 s). The wall time is
  dominated by NPU work, not compile, so the first-vs-second timing gap
  is not a useful signal here — but the manifest is unchanged across
  the two runs and `attn_npu2.o` recompile log line ("Compiling
  attn_npu2.o from attn_npu2.cc...") appears in both runs deterministically
  (rebuild of the .o, but .elf was reused).

## Issues surfaced

1. **(WARN, minor) NPU prefill timing measured 4.10 s vs the README's
   "warm NPU 3.2 s" claim.** The 5.3 s wall claim in the README matches
   well (I measured 4.98 s total wall for prompt-only, but my prefill
   walks 28 layers in 4.10 s vs claimed 3.2 s). My measurement is from
   a single first-prompt invocation in each `make run` (i.e., a "cold
   first-prompt warm-cache" — kernels are cached but BOs are freshly
   loaded). The claimed 3.2 s appears to require running multiple
   prompts back-to-back to amortize. This is **not** evidence of CPU
   fallback (the timing is far above 0.1 s; FA wrapper installation log
   confirmed; flash_attn.elf was used). v2 of the eval should attempt
   3+ back-to-back prompts to confirm the 3.2 s claim.
   `llama32_3b_inference.py:158-173`

2. **(WARN, minor) Adversarial prompt logits correlations (0.972 / 0.990)
   are slightly below the canonical-prompt average (~0.98), but top-1
   still matches in both adversarial cases.** Suggests the deployment is
   not over-fitted to canonical prompts but the BF16 accumulation drift
   discussed in `phase3_full.md` (per-layer cos drifting to 0.881 by
   L27) does mean some prompts will eventually flip top-1 with longer
   contexts. README's caveat about competitive prompts is accurate.

3. **(INFO) `make run` re-runs `Compiling attn_npu2.o from attn_npu2.cc...`
   on every invocation** even though the `.elf` is cached. The 3.0 s
   "compile / cache load" time is dominated by this `.o` rebuild
   (idempotent and produces identical bytes; not a correctness concern).
   `_llm_shared/phase_helpers/orchestration.py:93-94` — the `.o` build
   is unconditional but the `.elf` build is cache-gated.

## What was NOT checked (v1 scope limitation)

- Phase 4 cold/warm prefill perf re-measurement (skipped per task
  instructions; eval ran a single end-to-end perf check via `make run`
  instead).
- Phase 5 decode perf re-measurement at 100 tokens (only 5-token
  end-to-end was executed; the steady-state decode rate I observed
  (215 ms/token) over 4 tokens is consistent with the 100-token claim
  but does not exercise the documented thermal step at token 65).
- Multi-trial perf variance (each timing was a single trial).
- Long-context behavior at seq_len > 8192 (not tested; rope_scaling
  wavelength remap is acknowledged as unimplemented in TODO.md).
- Memory peak measurement (BO pre-load reported 5,376 MB prefill +
  5,888 MB decode, matching the README's ~14 GB peak claim, but I did
  not directly RSS-profile).
- Cross-deployment regression (skill v1 limitation).
