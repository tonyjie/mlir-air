# Phase-by-phase walkthrough: deploying a new LLM on NPU2 with the skill chain

This doc explains exactly what happens at each phase of `/deploy-new-llm`,
what code gets written, how it's tested, and what gates must pass before the
next phase starts. Companion to:

- [`skill_system_design_and_results.md`](skill_system_design_and_results.md) — high-level architecture + results across 4 deployments
- [`skill_system_overview.png`](skill_system_overview.png) — visual overview

Running example throughout: the **Qwen2.5-1.5B** deployment (`programming_examples/qwen25_1_5b/`) — the most recent and exercises the most complex skill paths (QKV bias, GQA-aware padding, K=8960 GEMV).

---

## TL;DR

Each phase has a **specific code artifact**, a **specific test command**, a
**specific numerical gate**, and a **specific failure-recovery recipe**.
Between phases the human approves and the next skill is invoked. The whole
chain is autonomous within a phase but human-gated between phases. Phase 7
then independently audits the whole thing without trusting the deployment's
own self-reports.

---

## Phase 0 — `bootstrap-model-config`

**Goal**: produce CPU artifacts the rest of the pipeline can trust.

**What you write** (2 files):

- `<model>_weights.py` — `LlamaConfig` dataclass + HF safetensors loader +
  `generate_rope_lut`. Handle Qwen2-style QKV bias if present (load
  `bq/bk/bv` 1-D vectors).
- `<model>_reference.py` — CPU F32 forward pass with `rms_norm`,
  `apply_rope`, `attention_reference`, `swiglu`, full `transformer_block`
  and `forward`.

**Test**:

```bash
python3 <model>_weights.py     # asserts every layer's shapes
python3 <model>_reference.py --prompt "The capital of France is" --verify
```

**Gate (must pass to advance)**:

1. All N layers load with expected shapes
2. CPU reference top-1 token matches HuggingFace transformers (loaded with
   `torch_dtype=float32`)
3. Logits correlation > 0.999 vs HF
4. No NaN

**Qwen2.5-1.5B example**: top-1 = `' Paris'`, corr = 0.99999992. PASS.

---

## Phase 1 — `validate-per-kernel-shapes`

**Goal**: prove every shape the model needs has a plan, BEFORE touching NPU.

**What you write**: `phase1_kernel_shapes.md` (a classification doc, no code).

**Steps** (in order — abort early on hard fails):

1. **Variant audit** — for FlashAttention specifically: identify which
   Python builder the lit test exercises and check production code uses the
   same one. (LESSON 3 — caught the head-first vs seq-first FA bug in
   llama32_3b.)
2. **BD-friendliness audit** (added 2026-04-19 from qwen25 lessons):
   compute and surface walls before they fail compile:
   - `emb_dim`, `hidden_dim`, `kv_dim` — flag if not multiples of 1024
     (Rule A from `_llm_shared/docs/aie2p_hardware_limits.md`)
   - `hidden_dim > 8160` → flag `down_k_split` need (Rule B)
   - GEMV B-input fires per launch — flag if > 127 (Rule C)
   - L2 budget for Down GEMV (Rule D)
3. **Tile-config safety check**: assert `N % (tile_n*herd_n) == 0` and
   `M ≥ tile_m*herd_m` for every GEMM. Would have caught qwen25's silent
   "cosine 0.02 garbage" Phase 2 detour.
4. **Enumerate shapes** for RMSNorm, Q/K/V/O/Gate/Up/Down GEMM/GEMV, RoPE,
   FA, SiLU+mul, eltwise add, LM head.
5. **Classify each**: `DROP-IN` (parametric, no code) /
   `RECOMPILE` (parametric, baked at shape) / `NEW` (genuinely novel work).

**Gate**: every shape has a class + plan. NEW items become Phase 2
prerequisites in `TODO.md`.

**Qwen2.5 example**: surfaced 3 NEW items (QKV bias, LM head 10-partition
for vocab=151936, `mv_k8960.o`) + 2 risks (non-1024-aligned dims → padding
needed; group=6 GQA at FA).

---

## Phase 2 — `integrate-single-block`

**Goal**: prove the FULL transformer block (1 layer) on NPU matches CPU
reference within numerical tolerance.

**What you write**: `<model>_phase2_test.py` (~150 LOC) — wires one block
end-to-end.

**Pipeline executed**:

```
RMSNorm → Q/K/V GEMM → RoPE → FlashAttention → O GEMM → +residual
       → RMSNorm  → Gate/Up GEMM → SiLU+mul → Down GEMM → +residual
```

**If Phase 1 surfaced padding need** (Step 0): apply GQA-aware reindexed
padding (`qwen25_pad.py` reference) BEFORE the NPU run. Always **CPU-only
sanity check** the padded vs orig forward FIRST (cosine should be ≥
0.999998) — catches reindex bugs in seconds without burning NPU time.

**Test** (real NPU execution):

```bash
python3 <model>_phase2_test.py --seq-len 2048
```

**Gate** (computed at real-token positions only, NOT padded positions):

- whole-tensor `cosine_sim > 0.99`
- per-position `cosine_sim` min > head_dim-scaled threshold:
  - `head_dim ≤ 64` → 0.99
  - `head_dim = 128` → 0.98
  - `head_dim ≥ 256` → 0.97
- no NaN

**If fails**: bisect by replacing each NPU step with CPU equivalent, find
the failing kernel, invoke `debug-bo-corruption` or
`superpowers:systematic-debugging`.

**Qwen2.5 example**: cosine 0.9988, per-pos min 0.9981. PASS @ seq_len=2048.

---

## Phase 3 — `validate-full-model-correctness`

**Goal**: prove all N layers + LM head produce correct top-1 token.

**What you write**: `<model>_phase3_test.py` — loops Phase 2 across all
layers + adds final RMSNorm + LM head, then evaluates 6 canonical prompts.

**Canonical prompts** (LESSON 2):

- **Decisive** (CPU top-1 prob > 0.5 — BF16 noise can't reorder):
  `"1 + 1 ="`, `"2 + 2 ="`, `"Water freezes at"`
- **Competitive** (CPU top-1 prob ≤ 0.5 — multiple plausible
  continuations): `"The largest ocean is the"`, `"The capital of France is"`,
  `"The sky is"`

**Test**:

```bash
python3 <model>_phase3_test.py
```

**Gate**:

- Decisive prompts: NPU top-1 = CPU top-1 EXACTLY (3/3 pass)
- Competitive prompts: top-5 overlap — `cpu_top1 ∈ npu_top5 AND npu_top1 ∈
  cpu_top5` (3/3 pass)
- No NaN

The decisive/competitive split is the LESSON 2 trick — without it, BF16
reordering of close-prob tokens looks like a regression when it's just noise.

**Per-layer cosine drift > 0.95 is INFORMATIONAL only** for `n_layers ≥ 24`
or `head_dim ≥ 128` — the noise budget naturally drives per-layer cos to
~0.88 by layer 28 even with no kernel bug.

**Qwen2.5 example**: 3/3 decisive + 3/3 competitive. Strict top-1 = 5/6
(the one miss is `'The sky is'` → `' '` vs CPU `' blue'`, both in each
other's top-5).

---

## Phase 4 — `optimize-prefill-perf`

**Goal**: apply 5 known prefill optimizations, measure cold vs warm.

**What you write**: `<model>_phase4_test.py` — measures cold prefill, then
runs `preload_prefill_weights`, then measures warm avg of N runs.

**5 patterns**:

| # | Pattern | How it lands |
|---|---|---|
| 1 | Multi-launch merging | INHERITED — `rms_gemms_rope` (6 launches), `o_ffn` (8 launches) |
| 2 | BO pre-loading | APPLIED — `preload_prefill_weights` writes per-layer BOs once |
| 3 | Intermediate buffer reuse | INHERITED — `intermediate_indices` set per kernel |
| 4 | Seq-first layout | INHERITED — RoPE + FA both native seq-first |
| 5 | CPU→NPU op promotion | APPLIED — switch to NPU FA (Option C wrapper for head_dim=128) |

**Test**:

```bash
python3 <model>_phase4_test.py    # NPU FA enabled by default
```

**Gate**:

- ≥ 3 of 5 patterns applied (or N/A with documented reason)
- Top-1 token preserved cold + warm (no regression)
- Prefill latency measured and recorded

**Qwen2.5 example**: 5/5 patterns. Warm prefill 2.4 s NPU layers
(85 ms/layer) — 4.2× speedup vs CPU-attn (10.1 s).

---

## Phase 5 — `optimize-decode-perf`

**Goal**: apply 5 decode optimizations, measure tok/s.

**What you write**: `<model>_phase5_test.py` (CPU prefill seeds KV → NPU
decode loop) + `<model>_decode_setup.py` if you need model-specific helpers
(e.g., `mv_kK.o` rename, custom LM-head partition count).

**5 patterns** (decode-specific):

| # | Pattern | Notes |
|---|---|---|
| 1 | Multi-launch merging | INHERITED — `rms_gemv_rope` (6) + `o_gemv_ffn` (8) |
| 2 | Static weight BOs | APPLIED — `pre_transpose_decode_weights` + per-layer BO preload |
| 3 | NPU LM Head GEMV | APPLIED — n-partition × 16384 (depends on vocab; qwen25 needed 10 partitions for 151936) |
| 4 | Extern kernel rename | APPLIED — `mv_kK.o` for Down GEMV (qwen25 needed `mv_k8960.o` + new `down_k_split` knob in matvec) |
| 5 | CPU→NPU op promotion | PARTIAL — attention stays on CPU per llama3 design |

**Special checks** (LESSON 5 / v2 BD-friendliness):

- If `hidden_dim > 8160`, set `down_k_split` to a divisor of `hidden_dim`
  where outer ≤ 255 (Qwen2.5 used `down_k_split=70` → splits 8960 as
  70 × 128)
- For LM head with M ≥ 8K, set `tile_m = m_input` to keep B-DMA fires
  under 127 per GEMV

**Test**:

```bash
python3 <model>_phase5_test.py --n-tokens 20 --cpu-verify
```

**Gate**:

- Decode latency measured (ms/token)
- ≥ 3 of 5 patterns applied
- NPU/CPU top-1 match rate ≥ 80%

**Qwen2.5 example**: 5/5 patterns. 216 ms/tok (4.6 tok/s), 7.7 ms/layer.
5/6 NPU/CPU match.

---

## Phase 6 — `finalize-deployment`

**Goal**: end-to-end runner + perf summary + clean README.

**What you write**:

- `<model>_inference.py` — entry for `make run`. Wires NPU prefill (with
  KV extraction) + first-token NPU LM Head + NPU decode loop.
- `phase6_finalize.md` — perf comparison table vs prior deployments +
  reusable-pattern audit
- `README.md` — quick-start, performance, file structure
- Update `TODO.md` (mark all phases ✓)

**Test** (the actual user-facing demo):

```bash
make run N_TOKENS=20
```

**Gate**: end-to-end produces coherent text. Perf table shows per-layer
rates in line with the family.

---

## Phase 7 — `evaluate-deployment` (independent audit)

**Goal**: independent verification of everything Phases 0–6 just claimed.

**Spawns a FRESH subagent** with explicit instructions:

- Treat the deployment as UNTRUSTED
- Do NOT read `LESSONS.md` / `progress.md` / `phase{N}_*.md` BEFORE
  measuring
- Re-derive every PASS/FAIL with a number from THIS run
- Use the skepticism heuristics (silent CPU fallback, trivial top-1,
  hardcoded PASS, etc.)

**6 check categories** (cost-ordered, abort early):

| # | Category | Cost | Catches |
|---|---|---|---|
| 1 | Static audit | seconds | missing files, suspicious TODOs, doc-vs-code mismatches |
| 2 | Weight + reference smoke | ~30s | model loads, CPU ref matches HF |
| 3 | Per-phase re-run + adversarial prompts | ~5–10 min | gates re-derived; NEW prompts not in canonical set |
| 4 | End-to-end reproducibility | ~30s × 2 | byte-identical output across runs; real NPU execution (not silent CPU fallback) |
| 5 | Perf integrity (multi-trial) | ~5 min | mean ± std vs claim; anti-fallback heuristics |
| 6 | Cross-deployment regression | ~10–30 min | re-runs OTHER deployments' Phase 2 if shared infra changed (caught the llama3 critical regression on 2026-04-20) |

**Output**: `<model>/docs/evaluation_report.md` with verdict
`PASS` / `PASS-with-warnings` / `FAIL`.

**Tag gate**: `git tag -a deployment-<model>-v1` only fires on PASS or
PASS-with-warnings. FAIL → mark deployment `needs-human-review` in TODO.md
and STOP; the human triages.

---

## Between-phase checks (the "stop and re-plan" gates)

After each phase the entry skill (`deploy-new-llm`) reports
PASS/FAIL/BLOCKED to the human and asks for "go" before advancing.
Specifically:

| Boundary | Question to answer before advancing |
|---|---|
| 0 → 1 | Does the CPU reference top-1 match HF? If not, the rest is hopeless. |
| 1 → 2 | Are all NEW work items planned with explicit code paths? Any unbisected risks? |
| 2 → 3 | Single-block cosine ≥ 0.99? If not, FA / GEMM / RoPE has a real bug — fix before scaling to N layers. |
| 3 → 4 | Decisive top-1 = 3/3? If not, drift is structural, not noise. |
| 4 → 5 | Did NPU FA actually fire? Did warm prefill timing change after preload? |
| 5 → 6 | Decode tok/s in the right ballpark vs sibling deployments? |
| 6 → 7 | End-to-end produces coherent text? |
| 7 → tag | Independent eval verdict PASS or PASS-with-warnings? |

The HUMAN says "go" after each step. The skill never advances autonomously
across the gate.

---

## File output convention (per deployment)

After all 7 phases, your `<model>/` directory looks like:

```
<model>/
├── <model>_weights.py          [Phase 0]
├── <model>_reference.py        [Phase 0]
├── <model>_phase2_test.py      [Phase 2]
├── <model>_phase3_test.py      [Phase 3]
├── <model>_phase4_test.py      [Phase 4]
├── <model>_phase5_test.py      [Phase 5]
├── <model>_inference.py        [Phase 6]
├── <model>_decode_setup.py     [optional, only if model needs custom decode helpers]
├── Makefile                    [scaffolded]
├── README.md, CLAUDE.md, TODO.md
└── docs/
    ├── evaluation_report.md    [Phase 7 — independent audit]
    └── development_progress/
        ├── progress.md         [updated at each phase]
        ├── LESSONS.md          [novel failures captured]
        ├── phase1_kernel_shapes.md
        ├── phase2_block.md
        ├── phase3_full.md
        ├── phase4_prefill.md
        ├── phase5_decode.md
        └── phase6_finalize.md
```

This structure is now consistent across all 4 deployments
(`llama3/`, `smollm2_1_7b/`, `llama32_3b/`, `qwen25_1_5b/`) after the
2026-04-20 alignment pass.

---

## Reusable infra each deployment can lean on

The deploy-new-llm skill chain is built around the assumption that the
following shared infrastructure exists and is back-compat-stable:

- `programming_examples/_llm_shared/kernel_builder/` — `KernelCache` +
  `compile_all_external_kernels` + `gemm_builder` + `rope_halfsplit.cc`
- `programming_examples/_llm_shared/phase_helpers/` — `metrics.py`,
  `canonical_prompts.py`, `decode_setup.py`, `headfirst_fa.py` (Option C
  wrapper for head_dim=128), `orchestration.py`, `prefill_runner.py`
- `programming_examples/_llm_shared/docs/aie2p_hardware_limits.md` — the
  AIE2P shim DMA / lowering walls (Rules A–D, BD-friendliness checklist)
- `programming_examples/llama3/multi_launch_builder/` — `rms_gemms_rope`,
  `o_ffn`, `lm_head` (prefill ELFs); `rms_gemv_rope`, `o_gemv_ffn`,
  `lm_head_gemv` (decode ELFs). Builders accept tile-config knobs +
  optional `down_k_split` for hidden_dim > 8160.

Each per-model deployment is "minimal scaffold + sys.path imports" —
~22 files in the model dir, the rest inherited from the shared infra.
This is the deliberate pattern from the `deploy-new-llm` skill (lifted
from the smollm2 `cp -r` retrospective).

---

## Convention summary across the 4 deployments

| Convention | Default |
|---|---|
| `cpu_attn` (function + CLI) | `False` (NPU FA) |
| `make run` | end-to-end NPU prefill + NPU decode, NPU FA enabled |
| `make verify` | decode tokens with NPU/CPU top-1 check |
| `make profile` | prefill perf measurement (cold + warm) |
| `make run-block` | Phase 2 single-block correctness |
| `make run-full` | Phase 3 full-model correctness |
| `make run-prefill` | Phase 4 prefill perf |
| `make run-decode-only` | Phase 5 decode perf |
| `make run-reference` | Phase 0 CPU reference vs HuggingFace |
| `N_TOKENS` default | 100 |
| Evaluation report path | `docs/evaluation_report.md` |

These are now consistent across `llama3`, `smollm2_1_7b`, `llama32_3b`,
`qwen25_1_5b`.

---

## What this lets you do as a manual auditor

For any new model deployed via this chain, you can:

1. **Read just the model dir** — it's small (~22 files) and the diff vs
   `llama3/` is the only model-specific code
2. **Re-run any phase independently** — `python3 <model>_phaseN_test.py`
   reproduces that phase's gate
3. **End-to-end smoke test** — `make run` produces coherent text in
   <10 seconds
4. **Spawn the auditor** — `/evaluate-deployment <model_dir>` runs an
   independent audit and produces `evaluation_report.md`
5. **Cross-check perf** — the per-layer ms/layer rate should fall on the
   family curve (head_dim=64 → ~80 ms/layer prefill, head_dim=128 →
   ~85–115 ms/layer)
