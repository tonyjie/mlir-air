# SmolVLA Backbone Port — Progress

Target: lerobot/smolvla_base 16-layer SmolLM2-360M backbone on NPU2.
Spec: docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md

## Milestone A1: COMPLETE

All phases below are done; `make verify` PASSes end-to-end on real NPU2
hardware (see Task 3.3/3.4). See `README.md` for how to run and
`ARCHITECTURE.md` for the technical design. A2 (action expert) and A3
(vision encoder) are future work, not started.

## Phase status
- [x] Phase 0: CPU reference + oracle hooks
- [x] Phase 1: kernel validation (7 existing shapes + non-causal attn)
- [x] Phase 2: single-block validation
- [x] Phase 3: full-backbone + end-to-end action-chunk gate

## Task 3.3 — end-to-end hybrid inference (DONE)
Hybrid pipeline: CPU vision+prefix (lerobot venv) -> NPU 16-layer backbone
prefill (worktree python, subprocess bridge) -> inject per-layer post-RoPE K/V
into the CPU expert past_key_values -> unchanged 10-step denoise -> (1,50,6)
action chunk. Execution model = BRIDGED (the two venvs are disjoint).

- Step-1 exported K/V vs CPU cache (`make export-kv`): K cos min 0.998 / mean
  0.999; V cos min 0.995 / mean 0.998; position_ids match lerobot exactly.
- E2E action-chunk gate (`make verify`): median per-position cosine **0.9971**,
  raw MSE **9.13e-4**, normalized MSE **0.0078**, max_abs **0.054** -> **PASS**
  at cos_min=0.99, nmse_max=0.04 (`verify_adapter.py`'s `regression_gate`;
  nmse is magnitude-invariant, NOT a raw MSE threshold). Deterministic (fixed
  zero noise). Re-confirmed live on real NPU2 hardware for Task 3.4.

## Task 3.4 — finalize (DONE)
`README.md` + `ARCHITECTURE.md` written; Makefile confirmed coherent
(help/oracle/export-kv/run/verify/diagnosis/clean — no changes needed, a
SmolVLA VLA has no autoregressive `chat`/`profile`-over-tokens analog).
`make verify` re-run end-to-end on real NPU2 hardware, confirmed PASS with the
numbers above. A1 milestone complete.

Gotcha found + documented (README/ARCHITECTURE): this Makefile's targets
already self-lock (`$(NPU_LOCK)` inside each recipe), unlike some sibling
Makefiles that expect the caller to wrap `make run` in an external `flock`.
Wrapping `make verify` in an *additional* outer `flock
/tmp/mlir-air-npu.lock` self-deadlocks (nested flock on the same path from a
parent/child process pair) and times out after 30 min as `make: ***
[Makefile:54: verify] Error 1` — run `make verify` directly, no outer flock.

## Tested (kernel, shape) — filled in Phase 1

7 registry kernels validated on real NPU2 at SmolVLA backbone shapes (SmolLM2-360M
backbone: emb=960, kv_dim=320, hidden=2560, head_dim=64; seq 241 padded to 256).
All PASS the harness element-wise `np.isclose` gate vs an FP32 reference. GEMM uses
**bf16-out** (drain, tile_m=32) — mirrors every llama/qwen sibling's projections,
which feed further matmuls. Recorded as rows in the kernel_registry detail pages +
`supported_kernels.md`.

| Kernel | Shape (M×K×N or M×N or N) | M | PASS | mean_rel_L1 | Tile config |
|---|---|---|---|---|---|
| RMSNorm | 256×960 | 256 | ✅ | 4.2e-3 | herd_x=8 |
| GEMM q/o_proj (bf16-out) | 256×960×960 | 256 | ✅ | 9.5e-3 | drain tile_m=32 / tk2=320 / TILE_N=80 / HERD_N=4 |
| GEMM k/v_proj (bf16-out) | 256×960×320 | 256 | ✅ | 9.4e-3 | drain tile_m=32 / tk2=320 / TILE_N=80 / HERD_N=4 |
| GEMM gate/up (bf16-out) | 256×960×2560 | 256 | ✅ | 9.4e-3 | drain tile_m=32 / tk2=320 / TILE_N=128 / HERD_N=4 |
| GEMM down (bf16-out) | 256×2560×960 | 256 | ✅ | 9.4e-3 | drain tile_m=32 / tk2=320 / TILE_N=80 / HERD_N=4 |
| RoPE (half-split) | 256×64 | 256 | ✅ | 2.8e-3 | herd_x=8, herd_y=1 (θ via host LUT) |
| SiLU-and-Mul | 256×2560 (N=655360) | 256 | ✅ | 1.0e-2 | herd_x=8, herd_y=1, tile_n=4096 |
| EltwiseAdd | 256×960 (N=245760) | 256 | ✅ | 1.9e-3 | herd_x=8, herd_y=1, **tile_n=1920** |

Non-obvious findings (also recorded in the registry detail pages):
- **GEMM K=960 + tile_k_l2=160 silently corrupts** (mean_rel_L1≈0.77, no compile
  error). tile_k_l2 ∈ {64,192,320,960} are clean; used 320.
- **EltwiseAdd stock tile_n=2048 silently produces zeros** at N=245760 (passes the
  divisibility assert but mis-DMAs the 15 odd inner iterations). Used tile_n=1920.
- RoPE θ: harness hardcodes θ=500000 for its LUT; deployment uses θ=10000 via a
  host-provided LUT. The kernel only applies the rotation, so the 256×64 datapath
  test is valid regardless of θ.

## Phase 4: prefill optimization

Applied the shared opt skillset to the A1 backbone after the correctness-first
port landed. Both routes evaluated on real NPU2 hardware; results measured
with `bench_backbone.py` (median of 5, one-time compile excluded) and gated
with `make verify`.

| Route | Outcome | Result |
|---|---|---|
| Route 1 — batch attention GEMMs per GQA group | **APPLIED** (commit `f404998e`) | 31→11 dispatches/layer; NPU backbone 400ms→226ms (~1.8x); NPU/CPU ratio 1.16x→0.71x (NPU now faster than CPU-numpy); `make verify` unaffected (cosine 0.9971 / nmse 0.0078, PASS) |
| Route 2 — buffer-object reuse | **already in place / N/A for attention** | `rms_gemms_rope` and `o_ffn` already use `static_input_indices` (per-layer weight+LUT BOs) + `intermediate_indices`, inherited from the llama/qwen pattern. Attention GEMMs (qkt/pv) take only per-layer activations (Q/K/V/probs) as input — no static weights exist to pre-load; marking activations `static` would corrupt across calls. Structural conclusion, not a skipped optimization. |
| Route 3 — fuse attention into the per-layer fused ELF | **future work** | The ~208 total remaining dispatches (16 layers × 11 dispatches + per-layer `rms_gemms_rope`/`o_ffn`) are still the main latency cost. Folding attention's 11 dispatches/layer into the existing fused ELF (alongside `rms_gemms_rope`/`o_ffn`) via `opt-merge-multi-launch-kernels` is the next headroom. |

Before/after backbone latency (16 layers, `cpu_attn=False`, NPU2):

| stage | before opt | after opt |
|---|---|---|
| NPU backbone | ~400 ms | ~226 ms |
| CPU backbone (numpy fp32) | ~344 ms | ~321-353 ms (run-to-run) |
| ratio NPU/CPU | 1.16x (slower) | 0.71x (NPU faster) |

Non-obvious finding from Route 1: the external-mm.o codegen path produces
wrong results when the M-direction outer `air.launch` loop iterates more than
once (found via `validate_attn_gemms.py`, mean_rel_L1 ~0.67 at M=768 with
tile_m=32/herd_m=8 — a pre-existing bug, not specific to this change). Worked
around by scaling `tile_m` (96) to keep the batched M=768 within a single
launch iteration; verified correct (mean_rel_L1 ~9.6e-3, within the bf16-out
GEMM tier).

## A3-5 Step 4-5: NPU vision spliced into the hybrid (DONE)

Step 4 put the connector (modality projection) on NPU as one more ELF
(`gemm_connector`, 64x12288x960, drain tile_m16/tn80, herd 4x4); Step 5 wired
the NPU vision tower into `run_hybrid_forward` behind `npu_vision=True` via a
new all-cameras-in-one-call bridge (`run_npu_vision.py`).

| gate | result |
|---|---|
| connector ELF in isolation vs `vision_oracle['connector']` | cosine 0.999988 |
| NPU encoder + connector, end to end | cosine 0.996624 |
| `make verify` (NPU backbone) | cosine 0.997144 / nmse 0.007783 **PASS** |
| `make verify-npu-vision` (NPU vision + backbone) | cosine 0.993975 / nmse 0.012201 **PASS** |

End-to-end `predict_action_chunk` (`make bench-e2e`, median of 5): pure CPU
967 ms; hybrid NPU backbone 2158 ms as measured / 1562 ms compute-only; hybrid
NPU vision+backbone 2589 ms as measured / 1425 ms compute-only. The vision stage
itself is a 1.5x NPU win (148 ms/image warm vs 222 ms CPU) but the pipeline as a
whole is still slower than CPU — the NPU backbone (229 ms) is 2.5x slower than
lerobot's torch CPU backbone (91 ms), each NPU stage is a fresh subprocess so it
always pays cold-start, and the 307 ms CPU action expert is untouched. Full
numbers, per-stage split and the reusable BLAS-thread-contention finding are in
`docs/TODO.md`, section "A3-5 Step 4-5".
