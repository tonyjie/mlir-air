# Deploy Parity Checklist (v1)

The definition of "done" for deploying a new decoder-only LLM onto NPU2 via
`/deploy-new-llm`. A deployment is **complete and trustworthy** only when every
gate below passes. The `deploy-verifier` and `deploy-planner` (goal-check) agents
check these items **mechanically** — each is phrased so it can be confirmed by
"file exists", "grep finds it", "the run printed PASS", or "the execution-location
audit found no un-logged CPU fallback".

The standard is **parity with an already-merged, trusted deployment**
(`llama32_1b` is the reference exemplar; the 9 models merged via #1698–#1700 are
the working set). A new model — however similar to an existing sibling — is held
to the same bar. "It's just a re-parameterization" is never a reason to skip a
gate.

## Why this checklist exists (the failure it prevents)

The single-agent skill chain shipped models that were **correct but not
optimized**: kernels that clearly should run on NPU (GEMM/GEMV/RMSNorm/RoPE/FA)
silently fell back to CPU, and mergeable ELF groups were left unmerged. Every
correctness gate (per-layer cosine, token-set) still passed, because **those
gates are blind to *where* a kernel runs and to *how many* ELFs are dispatched**.
This checklist adds the two invariants those gates miss — **NPU-execution
(Gate H)** and **merge-completeness (Gate I)** — and makes an independent verifier
re-check them rather than trusting the runner's PASS.

Two distinct meanings of "fallback" — keep them separate, they were conflated in
the old skill:
- **Verification-method fallback** (allowed): no standalone harness for a kernel
  → verify it via `make diagnosis` per-layer cosine instead of a harness. This is
  about *how you check*, not *where the kernel runs*.
- **Execution-location fallback** (gated): a kernel runs on CPU instead of NPU.
  Allowed ONLY under Gate H's recorded exceptions. A silent one is a FAIL.

---

## Gate 0 — CPU/HF reference baseline (phase-0-build-cpu-reference)

- **0.1** `<model>_weights.py` exists and loads the HF checkpoint; config
  (n_layers, emb_dim, n_heads, n_kv_heads, head_dim, hidden_dim, vocab,
  rope_base, eps) matches the HF `config.json` (cite the values).
- **0.2** `<model>_cpu_helpers.py` provides the NumPy helpers production
  prefill/decode import (rms_norm, attention_reference, etc.).
- **0.3** the HF bf16 reference loads and runs via the shared `llms/verify/`
  HfRunner — `make` reference path prints a token/logit for a canonical prompt.
- **0.4** the reference is **bf16** (`torch_dtype=torch.bfloat16`), same dtype as
  NPU — no FP32 oracle. Cite `hf_runner.py` line.

## Gate 1 — Kernel validation on NPU (phase-1-kernel-validation)

- **1.1** for **every leaf kernel × shape** the model needs, correctness is
  verified **on real NPU2** against the registry's GPU/vLLM-aligned standard.
  Primary gate where a standalone harness exists: the harness's full-output
  element-wise `np.isclose` (rtol/atol) vs an FP32 reference — the PASS `make run`
  prints.
- **1.2** verification-method fallback (allowed): a kernel with **no** standalone
  harness is verified via `make diagnosis` per-layer cosine vs HF bf16. Record
  which kernels used this path and why (no harness), so it is auditable.
- **1.3** each verified (kernel, shape) is recorded as a row in
  `kernel_registry/supported_kernels.md` + `details/<Kernel>_bf16.md`
  (Used by = `<model>`), with a real precision number (not `—`/TBD).
- **1.4** **every** registry-eligible kernel type the model uses
  (GEMM, GEMV, RMSNorm, RoPE, FlashAttention, SwiGLU/silu_and_mul, EltwiseAdd)
  has a validated NPU implementation for the model's shapes. A kernel type that
  is left CPU-only at this phase must be listed in the model's
  `docs/TODO.md` "NPU-execution exceptions" section with a Gate-H reason —
  not silently skipped. (This is where the old chain broke.)

## Gate 2 — Single-block integration (phase-2-single-block-validation)

- **2.1** the Phase-1 kernels are wired into ONE transformer block on NPU;
  layer-0 whole-tensor `ffn_out` cosine (NPU vs HF bf16) **≥ 0.99** (gate).
- **2.2** no NaN anywhere in NPU block output.
- **2.3** host-side glue ops are limited to **allowed** operations (Gate H.2):
  bias add, per-head QK-norm reshape, RoPE LUT prep. Any GEMM/GEMV/attention/
  norm running on host at this stage is a Gate-H violation, not "integration
  glue".

## Gate 3 — Full-model validation (phase-3-full-model-validation)

- **3.1** all N layers wired; `make verify` **token-set top-5 inclusion vs HF
  bf16** PASSES (exit 0) on the canonical prompt set (2 prompts × 32 tokens).
- **3.2** `make diagnosis` per-layer cosine table shows no layer collapse
  (informational lens, used to localize drift — not the gate).
- **3.3** KV-cache / layer-indexed weight loading correct (no per-layer
  shape/index bug — caught by 3.1 holding across all layers).

## Gate 4 — Prefill optimization (phase-4-prefill-optimization)

- **4.1** `make verify` token-set gate still PASSES after every optimization
  step (correctness preserved).
- **4.2** prefill kernel time **strictly less** than the Phase-3 baseline
  (record both numbers).
- **4.3** Gate H holds: every NPU-eligible prefill kernel runs on NPU
  (re-audited after optimization — an optimization step must not introduce a CPU
  fallback to "make it pass").
- **4.4** Gate I holds for prefill: mergeable prefill ELF groups are merged.

## Gate 5 — Decode optimization (phase-5-decode-optimization)

- **5.1** `make verify` token-set gate still PASSES.
- **5.2** decode TPOT/TPS improved vs Phase-4-entry baseline (record both).
- **5.3** Gate H holds for decode (GEMV/RMSNorm/RoPE/etc. on NPU).
- **5.4** Gate I holds for decode: mergeable decode ELF groups merged (with the
  N-way extern rename correctly applied — see `opt-merge-multi-launch-kernels`).

## Gate 6 — Finalize (phase-6-finalize-and-learn)

- **6.1** clean `<model>_inference.py` integrating Phase-4 prefill + Phase-5
  decode; `verify_adapter.py` hooks the shared `llms/verify/`; Makefile has
  run / verify / verify-full / diagnosis / profile.
- **6.2** `make verify` (production path) PASSES — the production-readiness gate.
- **6.3** README + ARCHITECTURE record measured TTFT/TPS (from `make run`, not
  copied from a sibling) and the model's architecture deltas.
- **6.4** the kernel cache is per-model and shared across verify/run/profile/
  diagnosis (`build_peano/{prefill,decode}_kernel_cache`), anchored absolute so
  it never cross-contaminates another model.

## Gate 7 — Independent evaluation (phase-7-independent-evaluator / deploy-verifier)

- **7.1** a fresh evaluator (no inherited context) audits `make verify`:
  confirms the token-set gate runs the **production** NPU path vs HF bf16, not a
  shortcut or a cached/mocked result (anti-reward-hacking).
- **7.2** the evaluator **re-runs** `make verify` itself and quotes the PASS line.
- **7.3** the evaluator runs the Gate-H execution-location audit and the Gate-I
  merge audit independently (see below), and quotes the evidence.
- **7.4** `docs/evaluation_report.md` written — a human can read the full
  deployment state (correctness + NPU-execution + merge completeness) in 2 min.

---

## Gate H — NPU-execution invariant (cross-cutting, owned by deploy-verifier)

**The invariant:** every kernel with a registry-validated NPU implementation for
the model's shapes runs **on NPU**, not CPU. This is checked by auditing the
actual execution path, because the correctness gates are blind to it.

- **H.1** **audit, don't trust.** The verifier greps the model's prefill/decode
  drivers for each leaf kernel's dispatch and confirms it goes through
  `cache.load_and_run(...)` (NPU) — not a NumPy/`*_reference`/host implementation.
  Report per-kernel: `NPU` or `CPU`, with the `file:line` of the dispatch.
- **H.2** **allowed host ops** (NOT violations): elementwise bias add, per-head
  QK-norm reshape+broadcast when done as a thin wrapper, RoPE LUT construction,
  final-token LM-head selection, tokenization/embedding lookup. These are light
  glue, not compute kernels.
- **H.3** **forbidden without a logged reason:** GEMM, GEMV, RMSNorm (as a
  standalone reduction), RoPE apply, FlashAttention / attention scores+softmax,
  SwiGLU/silu_and_mul, EltwiseAdd running on CPU. Each such kernel MUST appear in
  `docs/TODO.md` under "NPU-execution exceptions" with ONE of:
  - **(a) CPU measured faster** — cite the two numbers (NPU vs CPU) proving it.
  - **(b) NPU path genuinely broken** — cite the specific compile error / hang /
    limitation and the debug attempts (link the relevant `debug-*` skill run).
  An exception without one of these is a **FAIL, class = silent-cpu-fallback**.
- **H.4** the exceptions list is **finite and shrinking**: a reason of type (b)
  is a known-limitation ticket, not a permanent excuse. If a shared solution
  exists (e.g. head-first FA for head_dim=128 via `shared/infra/fa_headfirst.py`,
  standalone GEMV for large-K decode, low-precision direct GEMM for large-N
  gate/up), the exception is INVALID — the solution must be applied.

## Gate I — Merge-completeness invariant (cross-cutting, owned by deploy-verifier)

**The invariant:** ELF groups that *can* be merged into one multi-launch ELF
*are* merged, so dispatch count is minimized. Left-unmerged mergeable groups are
a FAIL unless a compile blocker is recorded.

- **I.1** the verifier counts ELF dispatches per layer (prefill / decode) and
  compares to the reference exemplar's lean shape (llama32_1b: 3 prefill / 2
  decode ELFs per layer). A model dispatching one-kernel-per-ELF where a sibling
  merges them is a finding.
- **I.2** each kernel group that the exemplar merges but this model does not MUST
  have a recorded compile blocker in `docs/TODO.md` "merge exceptions" — a
  specific `aie.dma_bd` / BD-exhaustion / channel-routing error (link
  `debug-multi-launch-merge`), not "didn't get to it".
- **I.3** the merge ceiling is not a compiler wall (proved: 15 llama launches
  fuse into 1 ELF). So "too many launches to merge" is INVALID as a reason;
  only a concrete per-group compile error counts.
- **I.4** known real blockers that ARE valid merge exceptions (from the batch
  deploy): decode `o_gemv_ffn` fused cascade needs K%512==0 AND K≤~2048; large-N
  (9728/11008) gate/up fused-cast overflows `aie.dma_bd` stride. When hit, record
  the shape + error, don't silently unmerge everything.

---

## Gate T — Truthfulness (cross-cutting, deploy-verifier, over all gates)

- **T.1** headline correctness (`make verify` PASS) and performance (TTFT/TPS)
  numbers were **produced on NPU2 this round** — the verifier re-runs `make
  verify` and at least one `make run`/`profile`, quoting the actual output lines.
  A number that only appears in a report but can't be reproduced is a FAIL.
- **T.2** the Gate-H per-kernel NPU/CPU verdict is derived from the verifier's
  **own** grep of the drivers, never from the runner's status or the report.
- **T.3** the Gate-I dispatch count is derived from the verifier's own inspection
  of the compiled ELF set / kernel-cache manifest, not asserted.
- **T.4** any claim without supporting evidence is downgraded to a finding.

---

## How the gates map to agents

- **deploy-planner** — decomposes the deployment into the phase task chain
  (Gates 0–6), and owns the final goal-check: re-derives whether every gate
  (including H, I, T) is met by checking primary evidence, not trusting PASSes.
- **deploy-runner** — executes each phase (build refs, validate kernels, wire
  blocks, optimize), produces the reports and status files.
- **deploy-verifier** — the trust gate. Independently re-runs `make verify`, runs
  the Gate-H execution-location audit and Gate-I merge audit, and refuses to
  inherit the runner's PASS. Owns Gates H, I, T; audits 0–6.
- **orchestrator** (`/deploy-new-llm`) — routes by filename + PASS/FAIL only;
  never reads report content; keeps its own context minimal so a long deployment
  never approaches the context limit.

Source of truth for the deployment methodology: the `phase-*`, `opt-*`, and
`debug-*` skills under `.claude/skills/` — the agents consult them as the
knowledge base for *how* to execute and debug each phase. This checklist defines
*what* "done" means; the skills define *how* to get there.
