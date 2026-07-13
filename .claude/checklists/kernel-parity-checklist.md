# Kernel Parity Checklist (v1)

The definition of "done" for adding a kernel to `programming_examples/kernel_registry`.
A kernel is **complete and trustworthy** only when every gate below passes. The
`kernel-verifier` and `kernel-planner` (goal-check) agents check these items
**mechanically** — each is phrased so it can be confirmed by "file exists",
"grep finds it", or "table row has a real number, not a `—` placeholder".

The standard is **parity with an already-merged, trusted kernel** (GEMM #1674,
GEMV #1678, RMSNorm #1679, FlashAttention #1680). A new kernel — however simple —
is held to the same bar. "It's a simple kernel" is never a reason to skip a gate;
EltwiseAdd was waved through precisely because it looked simple.

Source of truth for the methodology: `kernel_registry/README.md` ("Methodology
notes") and `kernel_registry/kernel_adding_todolist.md` ("通用注意点"). This file
operationalizes those into hard gates.

---

## Gate A — Provenance (kernel-researcher) — HUMAN CHECKPOINT

This gate ends with a human review before anything else runs. Provenance errors
are the cheapest to catch here and the most expensive to catch later (the RoPE
half-split-vs-interleaved trap; the GEMV fused-vs-pure trap).

- **A1** The standalone harness == the variant llama actually uses. If they
  differ, the difference is stated explicitly (e.g. "fused cascade adds a
  residual epilogue; core matvec is bit-identical"). Evidence: builder call
  site `file:line`.
- **A2** llama's actual config (tile / herd / dtype) is verified and cited with
  `file:line`.
- **A3** The corresponding GPU/HF **reference computation and its test
  threshold** are located by name + `file:line` (e.g. PyTorch `test_addmv`,
  `rtol=1.6e-2`). If the op has no dedicated test (e.g. a plain `a+b`), cite the
  framework's nearest general bf16 reference convention and say so explicitly.

---

## Gate B — Implementation understanding (`01_implementation.md`)

- **B1** dtype layering written out: input/output **storage** dtype vs
  **accumulator** dtype.
- **B2** kernel compute structure documented with `.cc` source `file:line`
  (how MAC / reduction / non-linearity is actually organized).
- **B3** codegen path count stated (external `.o` vs direct codegen).
- **B4** a comparison table against the most-similar already-registered kernel
  (GEMV did "GEMV vs GEMM at a glance").
- **B5** llama-usage table: where this kernel is used, at what shapes, `file:line`.
- **B6** "code changes" section: if any constant was exposed as a tunable
  parameter (see D2), the harness diff is recorded here (file + line count),
  mirroring GEMV's "+11 lines `--perf-iters`" note.

---

## Gate C — Precision (`02_precision.md`)

- **C1** reference is a **full-output FP32** computation (bf16 upcast to f32 →
  compute → cast back to output dtype). Reference code cited `file:line`.
- **C2** the gate is an **element-wise** check (`np.isclose` / `assert_close`)
  over the full output. **Correlation / cosine-similarity as the correctness
  gate is FORBIDDEN** — it is blind to systematic per-element scale error
  (this masked the RMSNorm bf16-accumulation bug). Cosine may appear only as a
  supplementary sanity note, never as the gate.
- **C3** for any kernel with a reduction, the accumulator is **f32**. A bf16
  accumulator is a real finding and must be fixed (RMSNorm 6.9e-2 → 4.2e-3).
- **C4** the tolerances (rtol/atol) are justified against the Gate-A3 GPU
  standard, with the source recorded.
- **C5** precision numbers cover **every tested shape**, and the report states
  whether accuracy is tile/herd-independent (with the data that shows it).
- **C6** for a stateful kernel (e.g. FlashAttention writes scratch into its
  output), precision and performance are measured in **separate runs** — a
  timing loop re-runs without re-syncing inputs and corrupts the precision
  number if taken from the same run.

---

## Gate D — Performance (`03_performance.md`)

- **D1** the config space is stated: the compute fabric is used to its full
  width (e.g. `herd_x` / `herd_m` = 8) with a one-line reason.
- **D2** **sweep the dimensions that matter** — this requires a genuine,
  per-kernel understanding of which knobs affect performance, not a blanket
  full-grid sweep and not skipping the sweep with an unbacked assertion.
  - A knob argued to have no performance effect may be exempted from a full
    sweep **only if** 1-2 data points are shown to justify the claim.
  - **A meaningful knob that is currently hardcoded must be exposed as a
    tunable parameter and then swept** (this is the authorized harness change of
    type (a): expose-an-existing-constant only — never change compute logic;
    keep the original default so lit tests don't break; record the diff in B6).
  - The sweep CSV lives in `details/internal_<KERNEL>/data/` and its row count
    matches the number of legal configurations swept.
- **D3** each shape's **best config is selected from the sweep** (not asserted),
  with all knob values recorded (herd / tile / etc.).
- **D4** a memory-bound kernel reports **bandwidth / latency** (not headline
  GFLOPS); a compute-bound kernel reports GFLOPS. The classification is
  justified with arithmetic intensity.
- **D5** llama's actual config vs the sweep-best is compared (how much is left
  on the table; whether to suggest a builder default change).

---

## Gate E — Documentation completeness

- **E1** internal triplet + README exist and are non-empty:
  `details/internal_<KERNEL>/{01_implementation.md, 02_precision.md,
  03_performance.md, README.md}`. (No `internal_notes.md` — that was a historical
  artifact; 01/02/03 + README suffice.)
- **E2** the public `details/<KERNEL>.md` "tested shapes" table has a **real
  measured number in every row — zero `—` placeholders**.
- **E3** "How to reproduce" gives both the **full sweep** command and a
  **single best-tile** command, copy-pasteable.
- **E4** `supported_kernels.md` has a new index row, and its numbers match the
  detail page exactly.
- **E5** header / copyright / scope boilerplate matches the existing kernels.

---

## Gate F — Truthfulness (kernel-verifier, cross-cutting over B–E)

- **F1** the headline numbers were **actually produced on NPU2 this round**.
  The verifier independently re-runs at least one best-tile performance point
  and one precision point, quoting the actual output line. (Analogous to the
  SPMW rule "`simulator unavailable` in stdout means the run did not happen.")
- **F2** every `file:line` citation in the reports actually resolves (verifier
  spot-greps a sample).
- **F3** any claim made **without** supporting data (e.g. "tile_n is not a
  tuning target") is downgraded to a finding unless backed by sweep data.

---

## Gate G — Cross-document consistency (kernel-doc-reviewer)

This gate exists because the registry's value is that the *whole* set of documents
agrees with itself. Adding one kernel touches several shared documents, and a stale
cross-reference or an un-updated scope sentence silently erodes trust. This is a
pure-markdown gate (no NPU) — cheap, so there is no excuse to skip it.

**Scope is the documents this kernel touches** — not a full-registry re-audit:
the new detail page, the shared index/README/tracker, and any *other* kernel's
numbers that the new page cites.

- **G1** `README.md` is updated to reflect the new kernel: the scope sentence
  (the "currently covers ..." line) lists it, and the roadmap table moves it from
  "not yet" to done (or removes its row).
- **G2** `kernel_adding_todolist.md` status row for this kernel is updated (status,
  PR number if any, "下一个建议" pointer).
- **G3** every cross-kernel number the new detail page cites (e.g. "cleaner than
  GEMM ~9.3e-3, RMSNorm ~4.2e-3") matches that kernel's **current** detail page —
  no stale borrowed figures.
- **G4** the new page's metric conventions are parallel to same-class kernels
  (memory-bound kernels all report bandwidth/latency; rtol/atol wording and tier
  labels are consistent with peers).
- **G5** if this kernel surfaced a general methodology lesson (e.g. the RMSNorm
  f32-accumulation rule), `README.md`'s "Methodology notes" is updated to reflect
  it. If there is no such lesson, state that explicitly — do not silently skip.
- **G6** no dangling links and no self-contradictory scope text across the
  documents this kernel touched (the new page, index, README, tracker).

---

## How the gates map to agents

| Gate | Owner | Checked by |
|------|-------|-----------|
| A    | kernel-researcher | **human checkpoint** + kernel-planner |
| B    | kernel-runner | kernel-verifier (F2) |
| C    | kernel-runner | kernel-verifier (F1 precision) |
| D    | kernel-runner | kernel-verifier (F1 perf, F3) |
| E    | kernel-runner | kernel-verifier + kernel-planner |
| F    | kernel-verifier | kernel-planner (goal-check) |
| G    | kernel-doc-reviewer | kernel-planner (goal-check) |

A kernel is declared done **only** when the kernel-planner's goal-check
confirms every gate against primary evidence — not against another agent's
PASS claim.
