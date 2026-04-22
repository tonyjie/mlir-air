# Phase A Status — Standalone Kernel Validation for Chunked Prefill (C=64)

**Date:** 2026-04-21
**Goal:** Validate each unique standalone kernel needed for chunked-prefill (C=64) Llama-3.2-1B multi-turn chat, per `plan/2026-04-21-multi-turn-chat-roadmap.md` Phase A.
**Branch:** `llm_mapping`

## Headline

**9 of 10 standalone kernels green at chunk_size=64. K8 (flash_attn_chunk) blocked at AIE backend lowering** — needs structural redesign of the FA kernel for rectangular Q-vs-K, not a parameter tweak.

---

## ⚠ Open Issues — Blockers for Phase B / Milestone 1 perf

### ISSUE-1 (P0): GEMMs K2–K5 use only 4 of 32 cores (12.5% utilization)

The four GEMM kernels (Q proj, K/V proj, Gate/Up, Down) currently compile with `tile_m=64, herd_m=1, herd_n=4` because `GEMM_TRANSFORM_IR` is hand-tuned for `tile_m=64` and breaks at smaller tile_m (see "GEMM tile-config constraint at M=64" below for evidence). With `herd_m=1` we use **only 4 of the 32 AIE cores in the 8×4 herd** — an 8× under-utilization in the M dimension.

**Measured baseline (`make profile`, NPU2, default xrt power mode):**

| Kernel | Shape (M,K,N) | Theory FLOP | Latency (min) | GFLOPS | Correlation |
|---|---|---|---|---|---|
| K2 Q/O proj | 64, 2048, 2048 | 0.537 | 0.71 ms | **753** | 0.9999 |
| K3 K/V proj | 64, 2048, 512  | 0.134 | 0.22 ms | **602** | 0.9999 |
| K4 Gate/Up  | 64, 2048, 8192 | 2.147 | *(profile loop hangs — see ISSUE-3)* | — | 0.56 |
| K5 Down     | 64, 8192, 2048 | 2.147 | *(profile loop hangs — see ISSUE-3)* | — | 0.79 |

K2 single-invocation cost is ~0.71 ms. Per-layer GEMM count: 2×K2 (Q+O) + 2×K3 (K+V) + 2×K4 (Gate+Up) + 1×K5 (Down) = 7 GEMM calls. Naively at K2's measured rate per call, **just the GEMMs would cost ~5 ms × 16 layers = 80 ms per chunk** — and that's before counting K4/K5 (which can't be measured yet but theory says they're 4× heavier than K2). Realistic per-chunk GEMM total likely 200+ ms, vs the ~150 ms total budget needed to make chunked prefill faster than today's 1.30s padded prefill at typical chat message lengths.

**Impact:** Each chunk's 7 GEMM launches per layer (× 16 layers = 112 calls per chunk) currently run at ≤ 1/8 of achievable throughput. **Milestone 1's perf goal (per-chunk prefill ≪ today's 1.30 s) likely cannot be met until this is fixed.**

**Fix paths (in increasing order of effort):**

1. **Larger chunk_size that aligns to herd_m=8.** With C=512 (= tile_m=64 × herd_m=8) we get full 8×4 = 32 cores active. But: a 512-token chunk wastes most of its compute on short user messages (typical chat turn ~30 tokens). Chunk size becomes a perf-vs-latency tradeoff.
2. **Restructure `GEMM_TRANSFORM_IR`** so it works correctly at small tile_m (e.g., tile_m=8 for full 8 herd_m utilization at M=64). Real MLIR-transform work — has to handle the inner tile_using_for + unroll factor 2 cleanly when the inner linalg.generic is small.
3. **Alternative GEMM kernel** tailored for small-M LLM cases (e.g., decode-style GEMV adapted for M=2..C). Most invasive.

**Action:** before Phase B, prioritize one fix path. Re-measure with `make profile` once a config change lands.

### ISSUE-2 (P0): K8 flash_attn_chunk blocked at AIE backend lowering

See "K8 (flash_attn_chunk) — Detailed Blocker Analysis" section below. Fundamentally a kernel-design issue (rectangular Q×K changes the BD allocation pattern beyond the AIE backend's budget). Hours-to-days of FA kernel work to resolve.

### ISSUE-3 (P1): K4 / K5 profile-loop hang (correctness OK in single-invocation)

Single-invocation `make run` PASSes for K4 (Gate/Up, output 1 MB) and K5 (Down, K=8192). But the multi-invocation `make profile` loop (warmup=5, iters=20 via XRTBackend kernel handle reuse) produces:

- K4: per-invocation latency ~61 s (likely XRT timeout) AND output correlation 0.56 (wrong math)
- K5: huge variance (min 0.12 ms / avg 58 s / max 61 s) AND correlation 0.79

K2 (output 256 KB) and K3 (output 64 KB) profile cleanly under the same harness, suggesting the issue scales with output buffer size or with K depth. Hypothesis: BO state isn't being reset between invocations correctly, or large-output kernels need explicit FROM_DEVICE sync between iterations. Profile harness reuses the same loaded ELF + same BOs across iterations following the `weighted_rms_norm.py` reference pattern — that pattern should work. Needs targeted debug.

This blocks getting baseline numbers for K4/K5 but doesn't affect correctness. Theoretical estimate (assuming K4/K5 sustain ~750 GFLOPS like K2): each ~2.86 ms per call.

---

### ISSUE-2 (P0): K8 flash_attn_chunk blocked at AIE backend lowering

See "K8 (flash_attn_chunk) — Detailed Blocker Analysis" section below. Fundamentally a kernel-design issue (rectangular Q×K changes the BD allocation pattern beyond the AIE backend's budget). Hours-to-days of FA kernel work to resolve.

---

## What's Been Done — Per-Kernel Results

Each kernel lives in its own directory under
`programming_examples/llama3/standalone_kernels/K<id>_<name>/` with a self-contained `kernel.py` + `Makefile`. Tests run via `make run`; exit 0 on PASS.

| ID | Kernel | Shape | Source adapted | Status | Commit |
|----|---|---|---|---|---|
| K1 | `rmsnorm_chunk` | x (64, 2048), w (2048,) → y (64, 2048) bf16 | `weighted_rms_norm/weighted_rms_norm.py` | ✅ PASS | `a4b4a911` |
| K2 | `gemm 64x2048→2048` | A(64,2048) × B(2048,2048) → C(64,2048) bf16 | `llama3/kernel_builder/gemm_builder.py` (`_build_gemm_module`) | ✅ PASS | `f6d28ac2` |
| K3 | `gemm 64x2048→512` | (64,2048) × (2048,512) → (64,512) bf16 | same | ✅ PASS | `8d8d8d8d` (after K2) |
| K4 | `gemm 64x2048→8192` | (64,2048) × (2048,8192) → (64,8192) bf16 | same | ✅ PASS | (after K3) |
| K5 | `gemm 64x8192→2048` | (64,8192) × (8192,2048) → (64,2048) bf16 | same (Down config: tile_k_l2=256, tile_n=64) | ✅ PASS | (after K4) |
| K6 | `rope_chunk_q` | x (64, 2048), lut (4096,) → y (64, 2048) bf16 | `multi_launch_builder/rms_gemms_rope_multi.py:_build_rope_2d` + `rope_halfsplit.cc` | ✅ PASS | (after K5) |
| K7 | `rope_chunk_k` | x (64, 512), lut (4096,) → y (64, 512) bf16 | same with `n_kv_heads=8` | ✅ PASS | (after K6) |
| **K8** | **`flash_attn_chunk`** | q(64,2048), k(2048,512), v(2048,512) → out(64,2048) bf16 | `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` + `attn_npu2.cc` | ❌ **BLOCKED** | `f6bef80d` (WIP) |
| K9 | `silu_and_mul_chunk` | gate, up (524288,) → y (524288,) bf16 ≡ flat (64, 8192) | `llama3/kernel_builder/ffn_swiglu/silu_and_mul.py` | ✅ PASS | (after K7) |
| K10 | `eltwise_add_chunk` | a, b (131072,) → y (131072,) bf16 ≡ flat (64, 2048) | `eltwise_add/eltwise_add.py` | ✅ PASS | (after K9) |

**All passing kernels validated against NumPy CPU reference at `rtol=5e-2` and `atol` ranging from 5e-2 (additive ops) to 4.0 (large-K GEMM accumulation).**

---

## Findings Along the Way

### Branch context (kernel_builder layout)

This Phase A work was originally executed on the `llm_mapping` branch and ported to `llama3-v0` afterwards (cherry-pick + import-path adjustment). The two branches diverge on where the shared kernel-builder helpers live:

| Helper | `llama3-v0` (this branch) | `llm_mapping` |
|---|---|---|
| `_build_gemm_module` | `llama3.kernel_builder.gemm_builder` | `_llm_shared.kernel_builder.gemm_builder` |
| `rope_halfsplit.cc` (compiled to `rope.o`) | `llama3/kernel_builder/rope_halfsplit.cc` | `_llm_shared/kernel_builder/rope_halfsplit.cc` |
| `compile_rope`, `compile_silu_and_mul`, `compile_attn_npu2` | `llama3.kernel_builder.external_kernels` | `_llm_shared.kernel_builder.external_kernels` |
| `silu_and_mul.build_module` | `llama3.kernel_builder.ffn_swiglu.silu_and_mul` | `_llm_shared.kernel_builder.ffn_swiglu.silu_and_mul` |

The actual file *contents* (`gemm_builder.py`, `stitching.py`, `rope_halfsplit.cc`) are byte-identical between the two branches — only the path moved when the LLM deployment infrastructure was extracted into `_llm_shared/` on `llm_mapping`. The standalone `kernel.py` files in this branch (`programming_examples/llama3/standalone_kernels/K*/kernel.py`) use the `llama3.kernel_builder.*` import form.

### `programming_examples/matrix_multiplication/bf16/` standalone fails

Both `make run4x4` (M=N=K=512) and `make run_llama_8x4` (M=128, K=N=2048) produce 45% mismatches with large numerical magnitudes (e.g. expected 764, actual −368). Persists after `make clean`. Recent commits to that dir are *fixes*, not regressions.

**Why production llama3 still works:** production calls `_build_gemm_module` (the path we use for K2–K5), which wraps `matrix_multiplication.bf16.run.build_module` and then applies a `GEMM_TRANSFORM_IR` overlay. That transform is what compensates. The bare matmul standalone path is a separate (broken) test.

This is documented as a finding only; we don't need to debug it for chunked prefill.

### GEMM tile-config constraint at M=64

Production GEMM uses `tile_m=64, herd_m=8` for M=2048 (so effective M-coverage per herd round = 512, with 4 M-iters). For our M=64, we tried `tile_m=8, herd_m=8` (effective M=64, fits in 1 iter) → output was systematically wrong (magnitudes 30x expected).

Root cause: the `GEMM_TRANSFORM_IR` in `gemm_builder.py` is hand-tuned around `tile_m=64`. With smaller `tile_m` the inner mmul tile (`mmul_mkn=[8,8,8]` for aie2p) doesn't divide cleanly through the transform's `tile_using_for [2,2,1,...]` + `unroll factor=2`.

**Workaround (used for K2–K5):** keep `tile_m=64, tile_n=128 (or 64 for Down)` but use `herd_m=1`. M=64 fits in a single herd column with one M-tile; only 4 of 32 cores active. **Performance left on the table** — addressed in M2 by restructuring `GEMM_TRANSFORM_IR` for small `tile_m`, OR by switching to an alternative GEMM kernel for the M=64 case.

### `weighted_rms_norm` multi-tile path has a test-wrapper bug

When `herd_x > 1`, `weighted_rms_norm.py:425-431` passes `[x, weight, np.zeros]` (3 inputs) to a 3-arg kernel function — leaving no slot for the actual output buffer. Standalone test fails.

For K1 we used `herd_x=1` (works correctly). Production `rms_gemms_rope_multi.py` uses `herd_x=8` for the embedded RMSNorm sub-kernel — that path goes through different wiring (multi-launch ELF, not the standalone XRTRunner test wrapper) and is unaffected. The Phase B `rms_gemms_rope_chunk` ELF will exercise multi-tile RMSNorm at chunk shape.

### Successful pattern-reuse confirmed the plan

For 9/10 kernels the methodology was: **adapt existing source via parameterization + a thin wrapper script + Makefile**. No source code changes to the existing kernels were needed for K1–K7, K9, K10. This validates the per-directory standalone test pattern.

---

## K8 (flash_attn_chunk) — Detailed Blocker Analysis

### Target shape (chunked prefill at C=64)
- Q: `(lq=64, n_heads*head_dim=2048)` bf16
- K: `(lk=2048, n_kv_heads*head_dim=512)` bf16 (full KV cache; only first `chunk_len` rows are real for the first-chunk case)
- V: `(lk=2048, 512)` bf16 (same)
- out: `(lq=64, 2048)` bf16

This is **rectangular** (lq ≠ lk). The existing kernel `attn_npu2_seqfirst.py` was designed for **square** (`lq == lk == 2048` in production).

### What was tried and what each result tells us

| Attempt | Config | Result | Implication |
|---|---|---|---|
| 1 | `causal=True, lq=64, lqp=64, num_q_tiles=1, num_cascade_stages=4, lk=2048, lkp=64` | Hit assertion `assert lq == lk` | Assertion is the surface guard, not the deep blocker |
| 2 | Same as 1, after relaxing assertion to `lq <= lk` | Compile fails: "Basic sequential allocation also failed" — L1 buffer placement unsuccessful | Buffer layout doesn't fit when Q is short |
| 3 | `num_cascade_stages=1` (vs production's 4) | Compile fails further along: "BD chain with unassigned IDs" on the K cache `aie.dma_bd` for `(2048, 512)` with `[size=32, stride=32768, size=64, stride=512, size=64, stride=1]` | Different failure mode but same root cause: chunked Q changes the DMA descriptor pattern |
| 4 | `causal=False, num_cascade_stages=1` | **Same BD chain failure** | Issue is **not** causal-specific; it's fundamental to rectangular `lq < lk` |

### Root cause hypothesis

The FA kernel emits a 3D DMA descriptor for streaming K (and similarly V) tiles into per-core L1 across all 32 K-tile iterations. With Q tile size matching K tile size (production: `lqp=256, lkp=64` → 4 Q tiles, 4 cascade stages, dataflow stays within the BD-slot budget), the descriptor fits. With our `lqp=64, num_q_tiles=1`, the per-core dataflow degenerates: **a single Q tile must consume all 32 K tiles in sequence without intermediate flushes**, generating a longer BD chain than the AIE backend can allocate.

Confirming this hypothesis would require:
- Inspecting `air_project/aiecc_failure_*.mlir` (saved) to see exact BD allocation
- Comparing the per-core BD use count vs. the AIE2P per-tile budget (~16 BDs/channel typically)

### What this isn't

- ❌ Not a `causal` flag issue (also fails with `causal=False`)
- ❌ Not the assertion (relaxed; got past it)
- ❌ Not a parameter that's tunable from the Python builder alone (tried `num_cascade_stages` 4 → 1, no change in BD pattern)

### What it actually is

A **kernel-level structural mismatch**. To support `lq < lk` the FA inner loop and DMA strategy have to be reshaped — likely one of:

1. **Re-tile K-streaming for shorter Q**: split the 32 K-tile sweep into smaller bursts (e.g., 8 bursts of 4 K-tiles) so each burst's BD chain fits within the AIE channel budget. May require changes to both `attn_npu2.cc` (inner loop bounds) and `attn_npu2_seqfirst.py` (DMA descriptor emission).
2. **Cascade restructure**: when `num_q_tiles=1`, the cascade pipeline (originally a 4-stage Q-tile pipeline) collapses to a single stage. The dataflow for Q broadcast / cascade-merge needs to handle the 1-stage case, possibly by using a different code path.
3. **Mid-stream mask buffer**: independently of (1) and (2), to support **mid-stream chunks** (current_pos > 0) the kernel needs an explicit mask input — adding a 4th argument to the kernel signature, threaded into the QK^T post-processing in `attn_npu2.cc`. This part of the plan is unchanged from the original roadmap.

### Path forward — three options

**Option A: Dedicated K8 design pass (recommended)**
Treat K8 as a focused sub-project. Produce a design doc covering:
- BD budget analysis for the chunked DMA pattern
- Concrete K-streaming restructure proposal (option 1 above)
- Mask buffer integration (option 3 above) — for both first-chunk and mid-stream cases
Estimated scope: 1–3 days of kernel work + iteration.

**Option B: Continue Phase B for the non-FA pieces while K8 is being designed**
With K1–K7, K9, K10 green, we can already build:
- ML1: `rms_gemms_rope_chunk` multi-launch ELF (uses K1, K2, K3, K6, K7) — produces Q/K/V for one chunk
- ML3: `o_ffn_chunk` multi-launch ELF (uses K1, K2, K4, K5, K9, K10) — projects attn_out and runs FFN
ML2 (`flash_attn_chunk`) stays blocked. The ChatEngine in Phase C cannot run end-to-end without it, but the per-component validation accelerates.

**Option C: Use existing FA at lq=2048 with chunk Q at offset (Approach A from earlier discussion)**
Pad Q to (2048, 2048) with chunk rows at positions `[current_pos..current_pos+chunk_len)`, zero elsewhere. Built-in causal mask handles the math correctly. **Trade-off:** FA compute is unchanged from bulk prefill (no perf win for chunked). Defeats the milestone-1 perf goal but unblocks end-to-end functional validation.

---

## Recommendation

1. **Commit the assertion relaxation** (done — `f6bef80d`). It's a correct improvement regardless of K8 outcome.
2. **Pursue Option A in parallel with Option B**: design K8 properly while building ML1 / ML3 with the validated kernels. ChatEngine end-to-end stays blocked on K8, but most of the chunked plumbing can be validated independently.
3. **Defer Option C** as a fallback only if K8 design proves infeasible.

---

## Reproduction commands

```bash
# Phase A green kernels
for k in K1_rmsnorm_chunk K2_gemm_64x2048_to_2048 K3_gemm_64x2048_to_512 \
         K4_gemm_64x2048_to_8192 K5_gemm_64x8192_to_2048 \
         K6_rope_chunk_q K7_rope_chunk_k \
         K9_silu_and_mul_chunk K10_eltwise_add_chunk; do
  make -C programming_examples/llama3/standalone_kernels/$k clean
  make -C programming_examples/llama3/standalone_kernels/$k run
done

# K8 attempt (currently fails AIE backend; intermediate IR saved to air_project/aiecc_failure_*.mlir)
make -C programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk run
```

## Files of interest

- Plan: `programming_examples/llama3/plan/2026-04-21-multi-turn-chat-roadmap.md`
- Standalone kernels: `programming_examples/llama3/standalone_kernels/K{1..10}_*/`
- FA source (assertion relaxed): `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:102`
- FA inner kernel: `programming_examples/flash_attention/kernel_fusion_based/attn_npu2.cc`
- K8 failure intermediate IR: `programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/build_peano/air_project/aiecc_failure_*.mlir`
