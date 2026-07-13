# SmolVLA Backbone Port — Progress

Target: lerobot/smolvla_base 16-layer SmolLM2-360M backbone on NPU2.
Spec: docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md

## Phase status
- [ ] Phase 0: CPU reference + oracle hooks
- [ ] Phase 1: kernel validation (7 existing shapes + non-causal attn)
- [ ] Phase 2: single-block validation
- [ ] Phase 3: full-backbone + end-to-end action-chunk gate

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
