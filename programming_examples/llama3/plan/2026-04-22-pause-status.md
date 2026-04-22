# Pause Status — Multi-Turn Chat for Llama-3.2-1B / NPU2

**Date paused:** 2026-04-22
**Branch:** `heads/llama3-v0` (the working branch — distinct from the `tags/llama3-v0` snapshot)
**Goal:** Support multi-turn interactive chat on Llama-3.2-1B/NPU2 with the existing 2048-token context window.
**Approach committed to:** Chunked prefill at `chunk_size = C = 64` so per-turn prefill cost ≈ proportional to new-message length, with KV cache persistent across turns.

This document captures the state at pause so the work can be resumed cold. Companion docs:

- `2026-04-21-multi-turn-chat-roadmap.md` — full plan with phase decomposition, kernel inventory, design decisions
- `2026-04-21-phase-a-status.md` — details on what each per-kernel test validates + open issues

---

## 1. The intended end state (what "done" looks like)

Multi-turn REPL where:
- User types `>>> What is the capital of France?` → assistant replies `Paris.`
- User types `>>> What is its population?` → assistant correctly resolves "its" via prior turn's KV cache
- Per-turn prefill cost is proportional to the new message length, **not** to the full transcript length
- Total transcript fits in ≤ 2048 tokens; capacity-exceeded raises a clean error

This is built on top of the existing single-turn deployment (`llama3_inference.py`, `llama3_prefill.py`, `llama3_decode.py`).

---

## 2. The strategy: chunked prefill at C = 64

Phase decomposition (from the roadmap):

- **Phase A** — validate each unique standalone kernel needed for chunked prefill at `M = 64`
- **Phase B** — fuse validated kernels into 3 multi-launch ELFs (`rms_gemms_rope_chunk`, `flash_attn_chunk`, `o_ffn_chunk`) mirroring today's prefill structure
- **Phase C** — build a stateful `ChatEngine` Python class that drives the chunked-prefill loop with persistent KV cache + transcript across turns
- **Phase D** — REPL CLI (`llama3_chat.py` + `make chat`)

Phase A enumerated 10 unique standalone kernels needed at the chunked shape (full table in `2026-04-21-phase-a-status.md`).

---

## 3. What's working today (Phase A — 9 of 10 kernels green)

All standalone kernel tests live under `programming_examples/llama3/standalone_kernels/K<id>_<name>/` with their own `kernel.py` + `Makefile`. Each runs via `make run` (correctness) and, for K2–K5, `make profile` (latency + GFLOPS + correlation).

| ID | Kernel | Shape (BF16) | Where it'll be used in chunked prefill | Status |
|----|---|---|---|---|
| K1 | `rmsnorm_chunk` | (64, 2048), w (2048,) → (64, 2048) | rms_gemms_rope_chunk #1 (attn norm), o_ffn_chunk #3 (ffn norm) | ✅ PASS |
| K2 | `gemm 64×2048→2048` | A(64,2048) × B(2048,2048) → (64,2048) | Q-proj, O-proj | ✅ PASS, profiled (753 GFLOPS) |
| K3 | `gemm 64×2048→512` | A(64,2048) × B(2048,512) → (64,512) | K-proj, V-proj | ✅ PASS, profiled (602 GFLOPS) |
| K4 | `gemm 64×2048→8192` | A(64,2048) × B(2048,8192) → (64,8192) | Gate-proj, Up-proj | ✅ PASS (single invoc) |
| K5 | `gemm 64×8192→2048` | A(64,8192) × B(8192,2048) → (64,2048) | Down-proj | ✅ PASS (single invoc) |
| K6 | `rope_chunk_q` | (64, 2048) + lut → (64, 2048) | rms_gemms_rope_chunk #5 | ✅ PASS |
| K7 | `rope_chunk_k` | (64, 512) + lut → (64, 512) | rms_gemms_rope_chunk #6 | ✅ PASS |
| K8 | **`flash_attn_chunk`** | q(64,2048), k(2048,512), v(2048,512) → out(64,2048) | flash_attn_chunk (single launch) | ❌ **BLOCKED** (see ISSUE-2) |
| K9 | `silu_and_mul_chunk` | gate, up (64,8192) → (64,8192), flat 1D n=524288 | o_ffn_chunk #6 | ✅ PASS |
| K10 | `eltwise_add_chunk` | a, b (64,2048) → (64,2048), flat 1D n=131072 | o_ffn_chunk #2 (residual1), #8 (residual2) | ✅ PASS |

**The 9 green kernels prove the chunked shape is viable for everything except FA.**

### What we proved beyond correctness

For the GEMM family (K2/K3) we have measured baseline performance (numbers in `phase-a-status.md`). Correlation against CPU F32 reference is **0.9999** on every tested kernel — the production `_build_gemm_module` path with `GEMM_TRANSFORM_IR` overlay (F32 accumulator + cast hoisting) gives full precision at our chunk shape.

### Reproducible infrastructure

- Each kernel: `make -C programming_examples/llama3/standalone_kernels/K<id>_<name>/ run` to validate, `make profile` (where supported) to measure.
- Shared profiling helper at `standalone_kernels/_profile.py` — reports avg/min/max latency, GFLOPS, herd utilization, **correlation vs CPU reference, max abs err, mean abs err**.
- Plan + status docs at `programming_examples/llama3/plan/`.

---

## 4. What we're blocked on

### BLOCKER 1 — K8 flash_attn_chunk (Phase A)

**One-liner:** The existing FlashAttention kernel can't be coerced into a rectangular `lq < lk` shape; it fails AIE backend lowering with a "BD chain with unassigned IDs" error on the K-cache DMA descriptor.

**Why it matters:** Chunked prefill at C=64 needs FA to compute `Q(64×2048) × K(2048×512)`-style attention against the full-context KV cache. Without K8, Phase B can't fuse the per-layer block, and Phase C / D have nothing to drive.

**What we know (from `2026-04-21-phase-a-status.md` "K8 — Detailed Blocker Analysis"):**
- Surface assertion `lq == lk` was overly restrictive; relaxed to `lq <= lk` (committed). Got past the assertion.
- Built-in `apply_causal_mask` works correctly for the FIRST-CHUNK case (current_pos=0) but is wrong for mid-stream chunks. Mid-stream needs an explicit mask buffer input — design sketched in the roadmap, not yet built.
- Compile fails *after* the assertion, in the AIE backend's BD allocation. Reproduces with `causal=False` too — so it's NOT a causal-flag issue. It's the rectangular `lq < lk` shape changing the K-cache DMA descriptor pattern beyond what the AIE backend's per-channel BD-slot budget can handle.
- Hypothesized cause: with `lqp=64 num_q_tiles=1` (vs production's `lqp=256 num_q_tiles=4`) the per-core dataflow degenerates — a single Q tile must consume all 32 K tiles in one BD chain, exceeding the budget.

**Estimated effort to unblock:** real kernel work (1–3 days). Likely needs (a) re-tiled K-streaming for shorter Q, (b) cascade-pipeline restructuring for `num_q_tiles=1`, AND (c) explicit mask-buffer input so mid-stream chunks work.

**Files:**
- WIP at `programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/{kernel.py, Makefile}`
- Failing IR preserved at `K8_flash_attn_chunk/build_peano/air_project/aiecc_failure_*.mlir`
- Source we're trying to adapt: `programming_examples/flash_attention/kernel_fusion_based/{attn_npu2_seqfirst.py, attn_npu2.cc}`

### BLOCKER 2 — GEMM under-utilization (impacts Milestone 1 perf)

**One-liner:** Our K2–K5 GEMMs use 4 of 32 AIE cores (12.5% util) because the production `GEMM_TRANSFORM_IR` is hand-tuned for `tile_m=64`. With `tile_m=64` and our `M=64`, we can fit only `herd_m=1` in the M dimension.

**Why it matters:** A chunked prefill that's slower than today's bulk prefill defeats Milestone 1's perf goal. Each chunk has 7 GEMM invocations × 16 layers = 112 GEMM calls. At 12.5% utilization those calls dominate per-chunk cost.

**What we measured:**

| Kernel | Latency (min) | GFLOPS | Cores |
|---|---|---|---|
| K2 (Q/O proj) | 0.71 ms | 753 | 4/32 |
| K3 (K/V proj) | 0.22 ms | 602 | 4/32 |
| K4 (Gate/Up) | profile loop hangs (ISSUE-3 below) | — | 4/32 |
| K5 (Down) | profile loop hangs (ISSUE-3 below) | — | 4/32 |

**What we discovered along the way:**
- The matmul/bf16 example's standalone test "passes" in CI via stochastic sampling + loose tolerance (`max_mismatch_percentage=5`). Our llama3-v0 has a tightened version that reveals the bare matmul produces ~5–15% per-element BF16 noise at K=2048.
- The production `_build_gemm_module` overlays `GEMM_TRANSFORM_IR` which:
  1. Promotes the `vector.contract` accumulator to F32 (fixes precision)
  2. Hoists `extf`/`truncf` cast pairs outside the K loop (makes the F32 cast viable for perf)
  3. Tiles outer matmul with `[2, 2, 1]` then unroll factor 2 — **this is what assumes `tile_m ≥ 16`** (the linalg.generic's M outer dim must be ≥ 2 when divided by `mmul_m=8`)

**Three viable fix paths (in order of effort):**

1. **`tile_m=16, herd_m=4, herd_n=4`** at C=64 → 16/32 cores (50% util, 4× better than now). Just a parameter change; transform IR's `tile_using_for [2,2,1]` works since outer M = 16/8 = 2. Quickest experiment.
2. **chunk_size=512 with tile_m=64, herd_m=8, herd_n=4** → full 32/32 cores. No kernel change, but C=512 wastes compute on short user messages (~30 tokens typical). Forces a chunk-size design tradeoff.
3. **Restructure `GEMM_TRANSFORM_IR` for any `tile_m`** so we can use `tile_m=8, herd_m=8` at C=64 → full 32/32 with no chunk-size compromise. Real MLIR-transform work — has to handle the `tile_using_for [2,2,1]` + `unroll factor=2` cleanly when the inner linalg.generic is small.

**Estimated effort:**
- Path 1: ~30 min experiment to confirm.
- Path 2: ~1 hour to swap CHUNK constant + re-test K2–K5.
- Path 3: 1–3 days of MLIR-transform work.

### BLOCKER 3 (P1, deferred) — K4/K5 profile-loop hang

**One-liner:** Single-invocation correctness PASSes for K4 and K5; the multi-invocation profile loop produces wrong outputs (corr 0.56 / 0.79) and 60+ second per-call latency. Scales with output buffer size — K2/K3 (smaller outputs) profile cleanly under the same harness.

**Why it matters:** Blocks getting baseline numbers for Gate/Up and Down GEMMs. Doesn't block correctness.

**Hypothesis:** BO state isn't being reset between invocations correctly for large-output kernels, OR the kernel's first-invocation post-state needs explicit FROM_DEVICE sync between iterations. Profile harness reuses the same loaded ELF + same BOs across iterations following the `weighted_rms_norm.py` reference pattern — that pattern works for small-output kernels.

**Estimated effort:** few hours of targeted debugging.

---

## 5. Open design decisions

These need a decision before resuming Phase B/C/D meaningfully:

1. **Chunk size.** C=64 is the committed plan, but BLOCKER 2 makes that hard. If we can't unblock GEMM utilization at C=64 (Path 3 above), do we accept Path 2 (C=512) and live with the wasted-compute-for-short-messages tradeoff? Or wait for Path 3?
2. **K8 design.** Three options outlined in `phase-a-status.md`: (A) dedicated K8 design pass, (B) build Phase B for non-FA pieces in parallel, (C) fallback Approach C (use existing FA at lq=2048 with chunk Q at offset — defeats perf goal).
3. **Whether to push for Milestone 1's stated perf goal** (per-chunk prefill ≪ today's 1.30s) at all in this iteration, or accept a slower-but-functional first cut and treat perf as M2.

---

## 6. Suggested resume points

In rough order of value:

### Quick wins (hours)

1. **Try Path 1 for BLOCKER 2** (`tile_m=16, herd_m=4` at C=64). Edit K2's `TILE_M`, `HERD_M`. Re-run correctness + profile. Expect ~50% utilization. If it works, immediately apply to K3/K4/K5 and re-measure.
2. **Debug BLOCKER 3** (K4/K5 profile-loop hang). Try adding `FROM_DEVICE` sync between iterations, or re-zero output BO before each invocation. Should expose true K4/K5 latency.

### Medium (days)

3. **Try Path 2 for BLOCKER 2** (C=512 with full herd) as a comparison data point. Tells us how much chunk_size affects perf at full utilization.
4. **K8 design pass.** Treat as a focused sub-project: write a design doc covering BD-budget analysis + K-streaming restructure + mask-buffer integration. Then build incrementally with parity tests.

### Big (week+)

5. **Restructure `GEMM_TRANSFORM_IR`** so it works at any `tile_m`. Real MLIR-transform work but unblocks chunked prefill at any chunk size.
6. **Phase B** — fuse the validated kernels into 3 multi-launch ELFs (depends on BLOCKER 1 being resolved for ML2; ML1 and ML3 can proceed without it).
7. **Phases C + D** — Python `ChatEngine` + REPL.

---

## 7. Where everything lives

```
programming_examples/llama3/
├── plan/                                  # planning + status docs
│   ├── 2026-04-21-multi-turn-chat-roadmap.md   ← full plan
│   ├── 2026-04-21-phase-a-status.md            ← Phase A details + open issues
│   └── 2026-04-22-pause-status.md              ← THIS FILE
├── standalone_kernels/                    # Phase A per-kernel tests
│   ├── _profile.py                              ← shared profile helper
│   ├── K1_rmsnorm_chunk/{kernel.py, Makefile}   ✅
│   ├── K2_gemm_64x2048_to_2048/                 ✅ + profile
│   ├── K3_gemm_64x2048_to_512/                  ✅ + profile
│   ├── K4_gemm_64x2048_to_8192/                 ✅ correctness; profile hangs (BLOCKER 3)
│   ├── K5_gemm_64x8192_to_2048/                 ✅ correctness; profile hangs (BLOCKER 3)
│   ├── K6_rope_chunk_q/                         ✅
│   ├── K7_rope_chunk_k/                         ✅
│   ├── K8_flash_attn_chunk/                     ❌ BLOCKER 1
│   ├── K9_silu_and_mul_chunk/                   ✅
│   └── K10_eltwise_add_chunk/                   ✅
├── kernel_builder/                        # production kernel infrastructure (unchanged)
│   ├── gemm_builder.py                          ← _build_gemm_module + GEMM_TRANSFORM_IR
│   ├── external_kernels.py                      ← compile_rope, compile_silu_and_mul, ...
│   ├── rope_halfsplit.cc                        ← half-split RoPE kernel (HF Llama convention)
│   └── ffn_swiglu/silu_and_mul.{py,cc}          ← SwiGLU activation
├── multi_launch_builder/                  # Phase B target (rms_gemms_rope_chunk, etc. — not built yet)
└── llama3_{inference,prefill,decode,weights,reference}.py   # existing single-shot deployment
```

Modified file outside `llama3/`:

- `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` — assertion relaxed from `lq == lk` to `lq <= lk` (BLOCKER 1 work-in-progress)

---

## 8. Resume command sheet

```bash
# Verify Phase A is still green:
for k in K1_rmsnorm_chunk K2_gemm_64x2048_to_2048 K3_gemm_64x2048_to_512 \
         K4_gemm_64x2048_to_8192 K5_gemm_64x8192_to_2048 \
         K6_rope_chunk_q K7_rope_chunk_k \
         K9_silu_and_mul_chunk K10_eltwise_add_chunk; do
  make -C programming_examples/llama3/standalone_kernels/$k clean
  make -C programming_examples/llama3/standalone_kernels/$k run
done

# Confirm K8 still fails the same way (intermediate IR preserved):
make -C programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk clean
make -C programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk run

# Re-measure GEMM baseline:
make -C programming_examples/llama3/standalone_kernels/K2_gemm_64x2048_to_2048 profile
make -C programming_examples/llama3/standalone_kernels/K3_gemm_64x2048_to_512 profile
```
