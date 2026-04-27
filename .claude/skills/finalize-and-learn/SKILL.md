---
name: finalize-and-learn
description: Phase 6 of LLM deployment — integrate Phase 4 prefill + Phase 5 decode into a clean `<model>_inference.py` with `make run` / `make verify` / `make profile` targets. The `make verify` numerical check vs CPU reference is the production-readiness gate. Capture lessons learned. Invoked after Phase 5 PASS.
---

## Purpose

Phase 4-5 produced an optimized prefill kernel and an optimized decode
kernel — but they may live in separate scripts with optimization-time
warmup hacks scattered through the main flow. Phase 6 integrates them
into a single clean `<model>_inference.py` that:

1. Has a clean **setup → prefill → decode** structure with no warmup
   hacks in the profiled scope (preprocess / weight pre-load happens
   ONCE in `setup()` BEFORE the timed region).
2. Exposes a `Makefile` with three targets that downstream users
   (and the Phase 7 evaluator) rely on:
   - `make run` — runs inference, prints **TTFT** (prefill ms) +
     **TPS** (tokens/sec)
   - `make verify` — strict numerical check against CPU reference
     (the production-readiness gate)
   - `make profile` — per-phase + per-key-kernel breakdown
3. Captures any new experience from this deployment in
   `LESSONS.md` so future deployments and skill updates can learn.

**Why `make verify` matters**: prior deployments shipped with only
top-1 spot checks — silent KV-cache bugs at decode time slipped
through Phase 3 (which only verifies prefill). Phase 6's `make verify`
adds **multi-token greedy match** vs CPU, catching decode-side issues
that Phase 3 cannot.

## Phase 6 PASS criteria (HARD GATES)

1. **`<model>_inference.py` exists** and has the clean structure:
   `setup()` (one-time preprocess, weight pre-load, BO allocation)
   called ONCE before the profiled `prefill() + decode_loop()` region.
   No warmup hacks, cache prime calls, or timing resets in the main
   flow.
2. **`make run` works**: invokes inference at default `--n-tokens 100`,
   prints TTFT (prefill kernel ms) + TPS (tokens/sec).
3. **`make verify` PASSES** the numerical gate (production-readiness):
   - Phase 3 gates re-run still hold (per-layer cosine ≥ 0.85, final
     logits cosine ≥ 0.95, no cliff, top-1 strict for decisive prompts)
   - **PLUS multi-token greedy match**: NPU greedy generates N tokens,
     CPU greedy generates N tokens (same prompt, same temperature=0
     argmax), every token ID is identical for `i in [0, N)`. Catches
     decode-side KV cache bugs Phase 3 cannot see.
4. **`make profile` works**: outputs per-phase total (setup / prefill
   total / per-token decode avg / LM head) + key-kernel ms (FA, Down
   GEMM/GEMV, LM Head — the bottleneck candidates).
5. **`LESSONS.md` updated** with any new experiences from this
   deployment (downgraded importance — informational; not gating the
   technical artifacts above).

If `make verify` fails, the deployment is NOT production-ready
regardless of how good Phase 4/5 perf numbers look.

## Knowledge base references

PRIMARY:

- `programming_examples/llama3/llama3_inference.py` — reference clean
  inference structure (setup / prefill / decode_loop / `--verify`
  mode); copy from
- `programming_examples/llama3/Makefile` — reference 3-target Makefile
- `programming_examples/<model>/docs/development_progress/{phase4_prefill,phase5_decode}.md`
  — Phase 4/5 outputs: which integration path was used + the
  optimized prefill/decode runners to integrate

SECONDARY:

- `programming_examples/llama3/docs/development_progress/LESSONS.md`
  — reference LESSONS format (per-deployment lessons file)
- `programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
  — Phase 1 catalog (Phase 6 confirms registry "Used by" reflects this model)

## Workflow

### Step 1: Integrate prefill + decode into `<model>_inference.py`

Copy `programming_examples/llama3/llama3_inference.py` as starting
point. The structure should be:

```python
def setup(weights, config):
    """ONE-TIME preprocess: pre-load weight BOs, allocate caches,
    install head-first FA wrapper if head_dim ≥ 128, etc. Everything
    that should NOT be inside the profiled scope."""
    ...

def run_npu_prefill(input_ids, ...):
    """Phase 4 prefill — clean of warmup hacks."""
    ...

def run_npu_decode_loop(prefill_state, n_tokens):
    """Phase 5 decode loop — clean."""
    ...

def inference(prompt, n_tokens, verify=False):
    setup(...)                 # once, BEFORE timed region
    t0 = time.time()
    prefill_out = run_npu_prefill(...)
    ttft = time.time() - t0
    decoded, decode_times = run_npu_decode_loop(prefill_out, n_tokens)
    tps = n_tokens / sum(decode_times)
    print(f"TTFT: {ttft*1000:.1f} ms  |  TPS: {tps:.1f} tok/s")
    if verify:
        ...  # see Step 3
```

Audit Phase 4/5 scripts for **warmup hacks** that crept into the main
flow — pre-warm cache calls, dummy runs, timing resets — and move
them into `setup()` (or delete if no longer needed). The profiled
scope must be ONLY `prefill + decode_loop`.

### Step 2: Wire the Makefile

Three targets, mirroring `programming_examples/llama3/Makefile`:

```makefile
run:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        python3 <model>_inference.py --n-tokens 100

verify:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        python3 <model>_inference.py --verify --n-tokens 5

profile:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        python3 <model>_inference.py --profile --n-tokens 20
```

`--verify` uses fewer tokens (5 is enough to catch greedy mismatch
without long CPU baseline cost). `--profile` uses moderate token count
for stable per-token decode timing.

### Step 3: Run `make verify` — the production-readiness gate

`--verify` mode runs:

1. **Phase 3 re-check** (per-layer cosine + final logits + top-1)
2. **Multi-token greedy match**: at temperature=0 (greedy argmax),
   NPU generates N tokens; in parallel, CPU reference generates N
   tokens with `<model>_reference.py`. For each token position
   `i ∈ [0, N)`: assert `npu_token_ids[i] == cpu_token_ids[i]`.

If any token diverges: STOP at first divergence, print which token,
print npu_top_5 vs cpu_top_5 at that position. The most common cause
is a KV cache update bug at decode time (Phase 3 only verified
prefill, so cache-write bugs only manifest at decode token i ≥ 1).

### Step 4: Run `make run`, capture TTFT + TPS

Record final numbers in `<model>/docs/development_progress/phase6_finalize.md`:

| Metric | Value | vs reference llama3 |
|---|---|---|
| TTFT (prefill kernel ms) | X | Y× / Y% |
| TPS (tokens/sec) | A | B× / B% |
| Decode ms/token | T | — |

### Step 5: Run `make profile`, capture breakdown

`--profile` mode prints:

```
=== Setup ===
  Weight load + BO alloc: ... ms (one-time, NOT in profiled scope)

=== Prefill (1 forward pass) ===
  Total: ... ms
  - rms_gemms_rope.elf:  ... ms ×16 layers
  - flash_attn.elf:       ... ms ×16
  - o_ffn.elf:            ... ms ×16
  - lm_head_gemv.elf:     ... ms

=== Decode (avg over N tokens) ===
  Total per token:        ... ms
  - rms_gemv_rope.elf:    ... ms ×16
  - CPU attention:        ... ms ×16   (or NPU FA)
  - o_gemv_ffn.elf:       ... ms ×16
  - lm_head_gemv.elf:     ... ms

=== Key kernels (the bottleneck candidates) ===
  FA:        ... ms (prefill only)
  Down GEMM: ... ms (prefill)
  Down GEMV: ... ms (decode per layer)
  LM Head:   ... ms
```

### Step 6: Update LESSONS.md + flag promotion candidates

Append to `<model>/docs/development_progress/LESSONS.md` for any new
experience (debug techniques used, surprising failures, configs that
mattered). Format mirrors `llama3/docs/.../LESSONS.md`.

Then audit for promotion candidates (don't promote speculatively — only
if 2+ uses):

1. Did this deployment **add a new C++ kernel**? Phase 1 should have
   added a standalone harness in `_llm_shared/kernel_builder/`. Confirm.
2. Did Phase 4/5 build **new model-specific multi-launch ELFs** under
   `<model>/multi_launch/` (kernel-first path)? Cross-reference
   against other deployments — if a 2nd kernel-first deployment uses
   the same pattern, promote to `_llm_shared/`.
3. Did this deployment hit a **new BD-friendliness rule** or compiler
   quirk worth adding to `_llm_shared/docs/aie2p_hardware_limits.md`?
4. Did this deployment surface a **new skill-chain change** worth
   making? Edit the relevant `.claude/skills/<phase>/SKILL.md` directly
   (the git history is the change trail — no separate central tracker
   needed). Surface in `<model>/TODO.md` as a follow-up if not done
   inline.

### Step 7: Confirm kernel registry reflects this deployment

Sanity check that Phase 1's Step 4 actually completed:

- `_llm_shared/docs/kernel_registry/<model>.md` exists with all rows
  filled (cosine, max_abs/max_rel, profile, status)
- `_llm_shared/docs/kernel_registry/supported_kernels.md` "Used by"
  columns mention this model on every kernel × shape it exercises

If gaps, fix here before Phase 7.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| `make verify` Phase 3 re-check passes but multi-token greedy diverges at token i ≥ 1 | KV cache update bug at decode time (Phase 3 only verifies prefill) | Print K/V cache values after token i-1 vs CPU; usually a layout / write-offset bug in decode kernel |
| `make verify` greedy diverges at token 0 | Same as Phase 3 failure (LM Head precision, final norm, etc.) | Re-run Phase 3 gate; root cause is in prefill, not decode |
| TTFT regressed vs Phase 4 baseline | Integration introduced overhead (warmup hack creep, redundant setup in main flow) | Compare Phase 4 standalone profile to current `make profile` setup section |
| TPS regressed vs Phase 5 baseline | Same as above for decode | Compare Phase 5 standalone profile |
| `make run` works but `make profile` shows huge "Setup" time inside profiled scope | `setup()` called inside the timed region instead of once before | Refactor — `inference()` must call `setup()` BEFORE `t0 = time.time()` |
| Multi-token diverges only at long contexts (i > 50) | KV cache overflow / position embedding wrap-around | Test at smaller N first, narrow range |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 6 PASS:

- `<model>/docs/development_progress/phase6_finalize.md`: TTFT + TPS
  + profile breakdown + LESSONS summary
- `<model>/TODO.md`: mark Phase 6 PASSED
- `<model>/CLAUDE.md`: write or update with final summary (model
  config, key file map, perf headline)
- **Hand off to Phase 7**: deploy-new-llm orchestrator now spawns
  `independent-evaluator` to re-derive every claim from scratch.
  Phase 6 is "deployment is internally complete"; Phase 7 is "deployment
  is independently audited".
