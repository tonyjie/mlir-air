---
name: kernel-validation
description: Phase 1 of LLM deployment — for every leaf kernel × shape the model needs, run the standalone NPU test harness and verify cosine ≥ threshold vs CPU F32 reference. Hard gate before integration. Invoked by deploy-new-llm after Phase 0.
---

## Purpose

Take the per-kernel decomposition Phase 0 produced (in
`<model>_reference.py`), enumerate every (kernel, shape) the model
needs, and verify each one standalone on real NPU2 against the CPU
oracle. Phase 1 isolates per-kernel correctness from integration bugs
and produces a verified shape catalog Phase 2+ can rely on.

The core loop is small:

1. Derive the kernel × shape list from the model's config.
2. For each (kernel, shape) → run the standalone harness → record cosine + perf.
3. Update the kernel registry's "Used by" columns with the new model.

## Phase 1 PASS criteria (HARD GATES)

Every (kernel, shape) the model needs must satisfy all four. Each
catches a different bug class:

1. **Standalone test exists**: a `make run` command in one of the
   harness dirs under `programming_examples/_llm_shared/kernel_builder/`
   (catalogued in `kernel_registry/supported_kernels.md`'s "How to test"
   for each kernel). If a needed kernel has no harness, build one
   (mirror `rmsnorm_multitile/` or `rope_halfsplit/` as templates).
   Catches: silently-skipped kernels.

2. **CPU numerical correctness**: cosine ≥ the kernel-specific
   `min_correlation` threshold documented in
   `kernel_registry/supported_kernels.md` for that kernel, against an
   F32 numpy reference. This is the **real correctness gate** — not
   theoretical compile-time rules.
   Catches: silent-corruption tile configs (qwen25_1_5b LESSON 2 hit
   3 such bugs because Phase 1 trusted Phase 2's whole-block cosine
   instead of doing per-leaf checks).

   **Also record `max_abs` and `max_rel` error in the catalog row**
   (informational — not gated, since absolute thresholds depend on
   input distribution). The 7 standalone harnesses already compute
   and print these. Recording them gives future deployments a baseline
   for cheap regression checks (e.g., "new deployment's max_abs at
   this same shape should be ≤ 1.5× the recorded value").

3. **Tile utilization documented**: each (kernel, shape) records its
   herd config. Targets:
   - Compute-bound (GEMM, FA): full 8×4 = 32 tiles
   - Row-parallel (RMSNorm, RoPE, GEMV-decode, SiLU+Mul, Eltwise): 8×1 = 8 tiles
   When achieved < target, justify in the catalog's Notes column
   (e.g., "M=1 decode RMSNorm uses 1 tile because batch=1 has no
   row-parallelism"). Catches: silent under-utilization.

4. **Catalog written**: every verified (kernel, shape) goes into
   `programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
   matching `llama3.2_1b.md`'s row format. Catches: deployments that
   pass without leaving a reusable record.

Failure on ANY criterion blocks Phase 2.

## Knowledge base references

PRIMARY (read before starting):

- `programming_examples/_llm_shared/docs/kernel_registry/supported_kernels.md`
  — every supported leaf kernel: builder location, **Tunable parameters**
  (knobs + hard constraints + tradeoffs), tested shapes across
  deployments, exact `make run` / `make profile` commands, tolerance
  defaults.
- `programming_examples/_llm_shared/docs/kernel_registry/llama3.2_1b.md`
  — template for the per-model shape catalog you'll produce. Shows the
  exact (kernel, shape, tile, cosine, perf, status) row format your
  `<model>.md` must match.

HEURISTIC catalog (read AFTER a test fails — not a gate):

- `programming_examples/_llm_shared/docs/aie2p_hardware_limits.md` —
  4 BD-friendliness rules (A/B/C/D) observed in past deployments.
  These are debug starting points when a test fails, NOT a-priori
  filters. Some may be context-dependent; trust the cosine test, not
  the rule. See "Failure modes" below for how to use them.

## Workflow

### Step 1: Derive the model's shape table

Read the HF `config.json` (or the model's `<model>_weights.py:Config`
dataclass after Phase 0). Use
[`kernel_registry/llama3.2_1b.md`](../../programming_examples/_llm_shared/docs/kernel_registry/llama3.2_1b.md)
as the **template** — it shows, for a generic decoder-only LLM, the
exact (kernel call site → shape) mapping using standard transformer
identities (Q proj output dim = `n_heads * head_dim`, etc.).

Create
`programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
with the shape table populated and the cosine + perf + status columns
**empty** (Step 2 fills them in). Match `llama3.2_1b.md`'s column
schema exactly.

(If the model has unusual ops — Q/K Norm, post-norm, ops between
currently-fused launches — flag in `<model>/TODO.md` as a Phase 2
prerequisite. The actual integration decision lives in
`single-block-validation` Step 0a.)

### Step 2: Per-kernel verification

For each row in your new `<model>.md`:

**a. Pick harness + initial tile config.** `supported_kernels.md` for
each kernel has the "How to test" command and a Tunable parameters
table (knobs + hard constraints + tradeoffs). If your shape exists in
that kernel's "Tested shapes" table → reuse the tile config. Else →
mirror the nearest-shape entry; verify the hard constraints hold for
your shape; adjust if not (e.g., GEMM `N % (tile_n × herd_n) != 0` →
pick smaller `tile_n`).

**b. Run correctness + profile.**

```bash
cd programming_examples/_llm_shared/kernel_builder/<harness_dir>
flock -x -w 1800 /tmp/mlir-air-npu.lock make run       # cosine vs CPU
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile   # 5w + 20iter timing
```

NPU is shared on this machine — every NPU command must be
`flock`-wrapped (see project CLAUDE.md). Compile-only steps don't
need the lock.

If cosine < threshold → see "Failure modes" below. Bound: 1 retry per
recipe per shape; if still failing, escalate to TODO.md "Active blockers".

**c. Record results in `<model>.md`.** Fill the row: cosine, profile
(ms / GFLOPS), tile config used, tiles in flight, status, and Notes
(especially when tiles-in-flight is below target — justify why).

### Step 3: Extend the kernel registry

For each (kernel, shape) verified in Step 2 that's **new** (not already
in `supported_kernels.md`'s "Tested shapes" table for that kernel),
append a row with: shape + tile config + tiles-in-flight count, "Used
by" listing your new model, cosine + profile from Step 2, status ✅,
and a link to your `<model>.md` for full context.

This grows the registry organically: each verified deployment extends
the menu of known-good shapes future deployments can copy from.

## Failure modes

When `make run` fails (cosine < threshold, compile error, hang), match
the symptom to a likely cause. The BD-friendliness rules in
`aie2p_hardware_limits.md` are starting points for debug, not gates.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `'aiex.npu.push_queue' op Repeat count exceeds [0:255]` | GEMV K too large (Rule B); auto-split outer dim ≥ 256 | Set `k_split` so `K = k_split × inner` and `k_split ≤ 255` |
| `Allocator exhausted available buffer descriptor IDs` | Non-1024-aligned dim ballooning BD pool (Rule A) | Pad dim to 1024-aligned (qwen25_pad.py for GQA-aware reindexed padding) OR use kernel-first split-ELF path |
| `L2 capacity exceeded` (matvec.py builder assert) | GEMV staged buffer > 512 KiB (Rule D) | Reduce `tile_m` (e.g., 8 → 2 for K=8192) or `herd_m` |
| Output all-zero / cosine = NaN | Bare-herd kernel without launch+segment wrapper | Wrap via `_wrap_ir_in_launch` (see `rmsnorm_multitile/` for template) |
| Cosine = 0.02 or other small | GEMM `N % (tile_n × herd_n) != 0` silent corruption (qwen25 LESSON 2 — builder does NOT assert) | Pick `tile_n` so divisibility holds at this N |
| FA all-NaN at runtime | Compile-flag mismatch on `attn_npu2.cc` macros (LESSON 3); OR lit-test verified a different builder than production uses | `-Dlqp` must be per-tile (lqp/num_q_tiles); see `debug-fa-runtime-failure` skill |
| Compile hangs > 10 min | Compiler scaling issue at large multi-launch | Cap and document; don't retry |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 1 PASS:

- Mark Phase 1 in `<model>/TODO.md`, append "(N/N kernels PASSED)"
- Append summary to `<model>/docs/development_progress/progress.md`
  (per-kernel cosine min/max, total time)
- `kernel_registry/<model>.md` is the durable artifact — no separate
  `phase1_kernel_shapes.md` needed
- `kernel_registry/supported_kernels.md` "Used by" columns reflect this
  model on every kernel × shape it exercises
