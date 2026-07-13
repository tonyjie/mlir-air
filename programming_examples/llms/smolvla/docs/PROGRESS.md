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
