# Design Spec: LLM Mapping Skills for AMD NPU2

**Date:** 2026-04-17
**Author:** Brainstormed with Claude (Opus 4.7)
**Status:** Design — pending implementation plan

---

## 1. Motivation

The `programming_examples/llama3/` work successfully deployed LLAMA-3.2-1B BF16 inference end-to-end on AMD NPU2 (1.30s prefill kernel, 92ms/token decode — 2–4× faster than IRON baselines). The journey took 6 distinct phases (infrastructure → per-kernel correctness → end-to-end correctness → prefill perf → decode perf → reorganization) and accumulated rich knowledge in `docs/development_progress/`.

We want to **automate the deployment of additional decoder-only LLMs** on the same infrastructure, leveraging this experience without re-living every debug cycle. Goal: deploy a new LLM with high confidence in correctness and reasonable confidence in performance, with most mechanical work codified rather than rediscovered.

## 2. Scope

**In scope:** Decoder-only LLMs that fit the existing kernel inventory:
- Attention: GQA or MHA (MHA handled as degenerate GQA with `n_kv_heads == n_heads`)
- Normalization: RMSNorm
- FFN: SwiGLU
- Position encoding: RoPE (half-split convention)
- Single-stream generation (no MoE expert routing, no sliding-window attention, no MLA)

**Explicitly out of scope:**
- MoE models (gpt-oss family, Mixtral) — would require new expert-routing kernels
- Sliding-window attention (Mistral) — would require attention kernel changes
- Multi-modal models
- Encoder-decoder architectures
- Training/fine-tuning

If a target model has unsupported features, the entry skill **fails loudly** with a clear message rather than silently producing wrong output.

## 3. Architecture: Skills + Knowledge Base + Per-Model Workspace

Three orthogonal layers:

```
.claude/skills/                          ← Procedure (what to do, in what order)
   deploy-new-llm/                       ← Entry skill
   bootstrap-model-config/
   validate-per-kernel-shapes/
   integrate-single-block/
   validate-full-model-correctness/
   optimize-prefill-perf/
   optimize-decode-perf/
   finalize-deployment/
   debug-bo-corruption/                  ← Cross-cutting recipe skills
   debug-multi-launch-merge/
   merge-multi-launch-kernels/

programming_examples/
   llama3/                               ← Reference deployment + knowledge base
       docs/development_progress/        ← Authoritative knowledge (cited by skills)
   _llm_shared/                          ← Lifted reusable infrastructure (one-time setup)
       kernel_builder/                   ← Cache, stitching, gemm_builder, external_kernels
       multi_launch_builder_lib/         ← Generic multi-launch infra (if extracted)
   <new_model>/                          ← Per-model workspace (scaffolded by entry skill)
       <model>_inference.py
       <model>_weights.py
       <model>_reference.py
       multi_launch_builder/             ← Model-specific shape builders
       docs/development_progress/        ← Per-model decisions, progress, blockers
       TODO.md                           ← State contract between skills
```

**Skill location:** `.claude/skills/` (project-level, versioned with the repo). Skills are tightly coupled to mlir-air paths and infrastructure; they belong with the code.

**Per-model layout:** Entry skill bootstraps `programming_examples/<new_model>/` as a copy of `programming_examples/llama3/`, with import paths rewritten to point at `_llm_shared/`. Each model is a self-contained directory.

**One-time refactor (the lift):** First invocation of `deploy-new-llm` detects whether `programming_examples/_llm_shared/` exists. If not, it offers to **lift** `kernel_builder/` (and any genuinely-generic parts of `multi_launch_builder/`) out of `llama3/` into `_llm_shared/`, updating `llama3/` imports accordingly. After this, every new model imports shared infrastructure from `_llm_shared/`.

**Knowledge base location:** Stays in `programming_examples/llama3/docs/development_progress/` for now. Skills reference these docs by path. When a 2nd or 3rd model contributes lessons, we revisit whether to elevate to a shared location.

## 4. Skill Inventory

### 4.1 Entry skill

| Skill | Purpose |
|-------|---------|
| `deploy-new-llm` | Bootstrap: take HF model ID, validate architecture is in-scope, perform one-time lift if needed, scaffold `<new_model>/`, populate `TODO.md`, walk human through phases |

**Invocation:**
```
/deploy-new-llm TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name tinyllama
/deploy-new-llm HuggingFaceTB/SmolLM2-1.7B
```
Required: HF model ID. Optional: `--name <dirname>`, `--target npu2|npu1`, `--dtype bf16|fp16`.

If the user supplies an out-of-scope model (e.g., `openai/gpt-oss-7b`, which is MoE), the skill fails at the architecture compatibility check (step 2 of bootstrap) before scaffolding anything.

### 4.2 Per-phase skills

| # | Skill | Purpose |
|---|-------|---------|
| 0 | `bootstrap-model-config` | Adapt config dataclass + HF weight loader; produce `<model>_weights.py` and `<model>_reference.py` |
| 1 | `validate-per-kernel-shapes` | Enumerate unique shapes the model needs; verify each kernel against CPU reference |
| 2 | `integrate-single-block` | Assemble one transformer block via stitching; verify correlation > 0.99 vs reference |
| 3 | `validate-full-model-correctness` | Wire all N layers; verify top-1 prediction matches reference for canonical prompts |
| 4 | `optimize-prefill-perf` | Apply known prefill optimization patterns (multi-launch merging, weight pre-loading, BO reuse, seq-first layout); measure each step |
| 5 | `optimize-decode-perf` | Apply known decode optimization patterns (multi-launch merging, static weight BOs, NPU LM Head GEMV); measure each step |
| 6 | `finalize-deployment` | Final perf report, update knowledge base with new lessons, harvest reusable patterns |

### 4.3 Cross-cutting (recipe) skills

| Skill | When invoked |
|-------|--------------|
| `debug-bo-corruption` | NaN / shape mismatch / corrupted output despite passing kernel test |
| `debug-multi-launch-merge` | Multi-launch ELF compile failure (BD exhaustion, herd-load bug, etc.) |
| `merge-multi-launch-kernels` | Procedure for actually merging kernels (used by phase 4/5) |

Total: **11 skills** (1 entry + 7 per-phase + 3 cross-cutting). Each ~300–500 lines because content lives in cited docs, not in the skills.

## 5. Phase Verification Gates ("Correctness Guardrails")

Each phase has an objective pass criterion that Claude reports at the gate. Human cannot accidentally advance past a failed gate.

| Phase | Gate criterion | Verification | On failure |
|-------|----------------|--------------|------------|
| 0. Bootstrap | (a) `<model>_weights.py` loads all expected tensors with right shapes; (b) `<model>_reference.py` produces stable output for canonical prompt; (c) reference output for canonical prompt is lexically sensible | Smoke test script; numerical sanity (no NaN, sensible logit distribution) | Weight name mismatch — human resolves; this phase is hardest to fully automate because HF naming conventions vary |
| 1. Per-kernel shapes | Every unique shape passes `XRTRunner.run_test` with `rtol=1e-3, atol=1e-5` against CPU reference | Auto-loop: enumerate shapes from config → run each kernel test → collect pass/fail table | Invoke `debug-bo-corruption` recipe; if no recipe matches, fall through to `superpowers:systematic-debugging` per Section 7 |
| 2. Single block | Single transformer block on NPU vs reference: cosine similarity > 0.99, MAE < 1e-2 | Custom test harness running one block; comparison against reference | `superpowers:systematic-debugging`: bisect by replacing each kernel with CPU fallback to localize offending step |
| 3. Full-model correctness | (a) Top-1 token prediction matches reference for ≥3 canonical prompts; (b) per-layer activation correlation > 0.95 throughout the stack | Existing `--verify` pattern from llama3 inference, extended to per-layer | Bisect: per-layer correlation report identifies which layer drifted |
| 4. Prefill perf | (a) Prefill end-to-end latency measured and recorded; (b) the 5 known optimization patterns from LLAMA Phase 4 (see below) attempted, with ≥3 successfully applied; (c) no correctness regression (re-run phase 3 gate) | Profiler timing + regression re-test | Optimization pattern fails → log as known limitation, advance with reduced perf |
| 5. Decode perf | Same structure as phase 4 for decode | | |
| 6. Finalize | (a) `progress.md` complete; (b) `LESSONS.md` updated with new failure modes; (c) perf comparison table vs CPU and IRON (if exists) | Doc presence + format check | — |

**Two principles for the gates:**

**(A) Perf gates are "applied the playbook honestly", not "achieved a target number".** Insisting on absolute perf parity across model sizes would make every deployment fail or stall on perf tuning. Human can choose to invest more in perf after the gate.

**(B) Phase 3 per-layer correlation is the most expensive gate** (requires running CPU reference per-layer alongside NPU per-layer) but worth it — silent quality degradation is the failure mode most likely to make it to production unnoticed.

**The 5 known prefill optimization patterns** (referenced in Phase 4 gate, sourced from `programming_examples/llama3/docs/development_progress/perf_optimization.md` and the multi-launch docs):

1. **Multi-launch ELF merging** — fuse separate kernel launches into a single XRT invocation per logical block (10 → 3 invocations/layer in LLAMA)
2. **Per-layer Buffer Object pre-loading** — write weights to BOs once during setup; use `static_input_indices` to skip re-write on subsequent calls
3. **Intermediate buffer reuse** — use `intermediate_indices` to skip BO write for buffers the kernel overwrites
4. **Seq-first activation layout** — eliminate host-side transposes by accepting `(seq, heads×dim)` natively
5. **Move CPU-side ops to NPU** — promote small CPU eltwise/RMSNorm/attention to fused NPU kernels where they fit memory budget

The decode equivalents (Phase 5) are: multi-launch merging, static weight BOs, NPU LM Head GEMV via partitioning, extern kernel rename for shape-collision avoidance, and CPU→NPU promotion.

## 6. State Contract: `TODO.md`

Per-model `TODO.md` is the canonical state file between skills. Every per-phase skill reads it to know current state and writes back results. No hidden state — a deployment can be paused mid-phase and resumed in a new session.

```markdown
# Deployment: <model_name>

## Phase status
- [x] 0: Bootstrap (PASSED 2026-04-17, see progress.md#phase-0)
- [x] 1: Per-kernel shapes (12/12 PASSED)
- [ ] 2: Single block ← current, BLOCKED on integration

## Active blockers
- [debug-bo-corruption] phase 2 attn step shows NaN at layer 0

## Resolved config (pulled from HF)
n_layers: 22, emb_dim: 2048, n_heads: 32, n_kv_heads: 4,
head_dim: 64, hidden_dim: 5632, vocab_size: 32000, rope_theta: 10000.0
```

**Other state files** (auto-generated):
- `<model>/docs/development_progress/progress.md` — phase-by-phase log (matches LLAMA pattern)
- `<model>/docs/development_progress/LESSONS.md` — failure modes encountered (reused from LLAMA + new)
- `<model>/docs/development_progress/debug_log.md` — inner debug loop transcripts

## 7. Inner Debug-Loop Design ("Ralph-loop within a phase")

Within a phase, Claude runs a **bounded autonomous loop** — iterate-until-pass-or-escalate, not polling.

```
For each work-item in phase (each kernel shape, each block, each merge attempt):
  1. Attempt action  (compile / test / merge / measure)
  2. If passed → record success, advance
  3. If failed → match error against recipe library
     a. If recipe matches → apply auto-fix → retry once
        - Recovered → record `recovered_via=<recipe>`, advance
        - Still failing → escalate
     b. If no recipe matches → invoke `superpowers:systematic-debugging`,
        attempt one bisect-style diagnosis
        - Resolved → prompt human: "novel failure mode — capture as new recipe?"
        - Unresolved → escalate to human with full diagnosis report
  4. Escalation → update TODO.md "Active blockers", stop the phase
```

**Bounds:** at most 1 auto-retry per recipe per work-item, at most 1 systematic-debugging attempt per work-item. Prevents indefinite churn.

**Recipe library = the cross-cutting skills.** Each recipe skill (`debug-bo-corruption`, `debug-multi-launch-merge`, etc.) has:

```markdown
## Trigger pattern
Regex on stderr / specific error string / output symptom

## Hypothesis
What's likely wrong and why (cited from compiler_issues docs)

## Auto-fix attempts
Specific code/config change Claude can make autonomously

## Verification
How to confirm the fix worked
```

**Self-improvement:** Each novel failure either becomes a new recipe (codified for next time, scaffolded with `skill-creator`) or stays as an escalation (which is fine — humans handle the genuinely novel cases).

**Loop bounds across phases:** Phases are gated by human approval (Section 5). The system is autonomous *inside* a phase, gated *between* phases.

## 8. Knowledge Base & Skill ↔ Doc Cross-References

**Principle:** skills are short and procedural; docs are long and explanatory; skills cite docs by path.

| Category | Lives in | How skill uses it |
|----------|----------|-------------------|
| **Reference (how X works)** — per-kernel deep-dives, RoPE convention, GEMM tile strategy | `programming_examples/llama3/docs/development_progress/kernels/*.md`, `multi-launch/*.md` | Skill cites by path; Claude reads the doc only when needed |
| **Recipes (how to fix X)** — known failure modes + fixes | The cross-cutting skill itself | Triggered by error pattern match; skill IS the recipe |
| **History/decisions (why we did X)** — perf optimization journey, design choices | `progress.md`, `perf_optimization.md` | Skill cites at phase start as context; only read when phase needs to make a similar decision |

**Per-skill structure (`SKILL.md` template):**

```markdown
---
name: <skill-name>
description: <one-line trigger description>
---

## Purpose
Single-paragraph what-and-when.

## Knowledge base references
Read these BEFORE acting if they apply:
- `programming_examples/llama3/docs/development_progress/<relevant-doc>.md`

## Workflow
Step-by-step procedure.

## Verification
How to know the phase passed.

## Failure modes
Common failures and which cross-cutting skill to invoke.

## Update protocol
What to write back to TODO.md / progress.md when done.
```

**Promote auto-memory entries to skills.** The user's `MEMORY.md` has 3 entries directly relevant to this work:
- `project_multi_launch_segment.md` — air.segment wrapper requirement
- `project_bf16_dma_stride.md` — DMA stride limitation
- `feedback_quick_tests.md` — run 2-3 XRT tests after rebuild

These should be **explicitly cited from relevant skills**, since auto-memory is per-user but skills should work for any developer.

## 9. Pilot Models (validation strategy)

Test the skills on real models in increasing distance from LLAMA-3.2-1B:

| Order | Model | Why |
|-------|-------|-----|
| 1 (smoke test) | **Llama-3.2-1B-Instruct** | Trivially identical arch; verifies skills run end-to-end. ~30 min. |
| 2 (real pilot) | **TinyLlama-1.1B-Chat-v1.0** | Same `emb_dim=2048`, but differs in: `n_layers=22` (was 16), `n_kv_heads=4` (was 8 — different GQA ratio), `rope_theta=10000` (was 500000 — exercises RoPE LUT regen), `intermediate_size=5632` (was 8192 — different FFN GEMM shape), `vocab=32000` (was 128256). Pure Llama-arch, no structural kernel changes. ~2.2GB BF16. Well-known reference. |
| 3 (broader test) | **SmolLM2-1.7B** | LLAMA-arch, `hidden_size=2048` matches, but uses **full MHA** (`n_kv_heads=32`). Tests that GQA kernel handles the degenerate case. `n_layers=24`, `vocab=49152`. |

**Rejected as pilots:**
- LLAMA-3.2-3B: ~6GB BF16, too tight on NPU2 memory without Q4 (which is currently shelved)
- Qwen2.5-1.5B: Has bias on Q/K/V projections — would need structural change to `rms_gemms_rope` multi-launch builder (out of scope for skill validation)
- Mistral-7B: Sliding-window attention (out of scope) and 7B too large
- gpt-oss family: MoE (explicitly out of scope per Section 2)

Success criterion for the pilot phase: TinyLlama-1.1B passes phases 0–6 with documented results in `<model>/docs/development_progress/`. Any escalations should result in either new recipes or a documented limitation.

## 10. Non-Goals

- **Not** a generic LLM deployment framework — scope is decoder-only LLAMA-like only.
- **Not** a fully autonomous deployment — human approval at every phase boundary.
- **Not** a perf-parity guarantee — gates verify "applied the playbook", not absolute speed.
- **Not** a refactor of `llama3/` beyond the one-time lift of `kernel_builder/` to `_llm_shared/`.
- **Not** an automated regression test suite — could evolve toward this after 2nd or 3rd deployment proves which abstractions are load-bearing.

## 11. Open Questions / Risks

1. **HF weight name remapping** is the most fragile part of phase 0. Different model families use different names (`model.layers.0.self_attn.q_proj.weight` vs `transformer.h.0.attn.c_attn.weight` etc.). The skill needs a robust adapter layer or explicit per-family mapping.
2. **The one-time lift** could break llama3 if import paths are incorrectly rewritten. Mitigation: skill runs `make verify` on llama3 after the lift to confirm no regression.
3. **Phase 4 perf optimization patterns** are only loosely codified — the LLAMA journey involved judgment calls (which kernels to merge, which fusion order). The skill should attempt the patterns mechanically but defer judgment-heavy choices to the human.
4. **MoE roadmap (post-deployment)**: if the user later wants gpt-oss, the kernel work (expert routing, top-k gating) needs to happen *before* the skill can handle it. This is intentionally out of scope but worth explicitly tracking as future work.

## 12. Implementation Order (preview for `writing-plans`)

Rough sequence the implementation plan will detail:

1. **Pre-work**: Lift `kernel_builder/` to `_llm_shared/` (one-time, manual, verify llama3 still works)
2. **Author skills in dependency order**: (using `skill-creator` for each)
   - `merge-multi-launch-kernels` (used by perf phases)
   - `debug-bo-corruption`, `debug-multi-launch-merge` (recipes)
   - `bootstrap-model-config` → `validate-per-kernel-shapes` → `integrate-single-block` → `validate-full-model-correctness` → `optimize-prefill-perf` → `optimize-decode-perf` → `finalize-deployment`
   - `deploy-new-llm` (entry, last because it depends on all the above)
3. **Smoke test**: Run `/deploy-new-llm` on Llama-3.2-1B-Instruct
4. **Real pilot**: Run on TinyLlama-1.1B; iterate on skills as friction surfaces
5. **Broader test**: Run on SmolLM2-1.7B; update knowledge base
6. **Document the framework** in `programming_examples/_llm_shared/README.md`
