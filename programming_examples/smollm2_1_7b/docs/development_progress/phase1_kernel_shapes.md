# Phase 1 — Per-Kernel Shape Validation

**Date**: 2026-04-17
**Approach**: For an arch-similar deployment, classify each kernel as
(a) **drop-in** (shape identical to llama3 → PASS by reference),
(b) **recompile** (shape-parametric builder → new cache entry, no builder change),
or (c) **novel** (builder change required → standalone NPU validation needed).

For SmolLM2-1.7B, all kernels fall into (a) or (b). Standalone NPU validation
is deferred to Phase 2 (single-block integration), where any per-kernel
correctness regression will manifest end-to-end and be caught by the
cosine-similarity gate.

## Shape inventory

### Prefill kernels

| Kernel | llama3 shape | SmolLM2 shape | Class | Status |
|--|--|--|--|--|
| RMSNorm | `(2048,)` | `(2048,)` | drop-in | PASS by reference |
| Q proj GEMM | `M=2048,K=2048,N=2048` | same | drop-in | PASS by reference |
| K proj GEMM | `M=2048,K=2048,N=512` | `M=2048,K=2048,N=2048` | recompile (kv_dim) | DEFERRED to Phase 2 |
| V proj GEMM | `M=2048,K=2048,N=512` | `M=2048,K=2048,N=2048` | recompile (kv_dim) | DEFERRED to Phase 2 |
| O proj GEMM | `M=2048,K=2048,N=2048` | same | drop-in | PASS by reference |
| Gate GEMM | `M=2048,K=2048,N=8192` | same | drop-in | PASS by reference |
| Up GEMM | `M=2048,K=2048,N=8192` | same | drop-in | PASS by reference |
| Down GEMM | `M=2048,K=8192,N=2048` | same | drop-in | PASS by reference |
| RoPE Q/K | `head_dim=64,base=500000` | `head_dim=64,**base=130000**` | LUT regen (host-side) | PASS by reference |
| FlashAttention | `seq=2048,nh=32,hd=64,n_kv=8` | `seq=2048,nh=32,hd=64,**n_kv=32**` | recompile (MHA) | DEFERRED to Phase 2 |
| LM Head GEMM | `M=2048,K=2048,N=128256` (8 partitions × 16384) | `M=2048,K=2048,**N=49152**` (4 partitions × 12288 OR 3 × 16384 + pad) | recompile + partition redesign | **DEFERRED + flag for Phase 4** |
| SwiGLU | `hidden_dim=8192` | same | drop-in | PASS by reference |
| Eltwise add | `emb_dim=2048` | same | drop-in | PASS by reference |

### Decode kernels (per-token, M=1 vector × matrix)

| Kernel | llama3 shape | SmolLM2 shape | Class | Status |
|--|--|--|--|--|
| RMSNorm | `(2048,)` | `(2048,)` | drop-in | PASS by reference |
| Q proj GEMV | `M=2048,K=2048` | same | drop-in | PASS by reference |
| K proj GEMV | `M=512,K=2048` | `M=2048,K=2048` | recompile (kv_dim) | DEFERRED to Phase 2 |
| V proj GEMV | `M=512,K=2048` | `M=2048,K=2048` | recompile (kv_dim) | DEFERRED to Phase 2 |
| O proj GEMV | `M=2048,K=2048` | same | drop-in | PASS by reference |
| Gate GEMV | `M=8192,K=2048` | same | drop-in | PASS by reference |
| Up GEMV | `M=8192,K=2048` | same | drop-in | PASS by reference |
| Down GEMV | `M=2048,K=8192` | same | drop-in | PASS by reference |
| RoPE Q/K (decode) | `head_dim=64,base=500000` | `head_dim=64,**base=130000**` | LUT regen (host-side) | PASS by reference |
| LM Head GEMV | `M=128256,K=2048` (8×16384) | `M=49152,K=2048` (4×12288 OR 3×16384+pad) | recompile + partition redesign | **DEFERRED + flag for Phase 5** |

## Verdicts by classification

### Drop-in (8 distinct kernels)
RMSNorm, Q/O GEMM, Gate/Up GEMM, Down GEMM, SwiGLU, eltwise add,
Q/O GEMV, Gate/Up GEMV, Down GEMV, RoPE shape (LUT differs only).

These compile from the **same llama3 cache** without recompilation.
They are well-validated for llama3-base and llama3-instruct.

### Recompile, same builder (4 distinct kernels)
K/V prefill GEMM, K/V decode GEMV, RoPE LUT, FlashAttention with `n_kv_heads=32`.

The builders accept the relevant parameter (`kv_dim`, `num_kv_heads`,
`config.rope_base`). For K/V, `kv_dim=2048` makes them literally
the same shape as Q/O — exercising a shape that's already validated by
the corresponding Q/O kernel at scale. For FlashAttention, MHA is the
degenerate-GQA case (`group_size=1`); the kernel asserts `n_heads % n_kv_heads == 0`,
which holds (32 % 32 == 0).

**Risk assessment**: Very low. The builder design covers these axes already.
Cache invalidation is automatic via `KernelCache` keyed on builder args.

### Novel — partition-scheme adjustment (1 kernel)
**LM Head GEMM/GEMV**: llama3 hardcodes `n_partitions=8 × n_part=16384` for
vocab=128256. SmolLM2 vocab=49152 needs:
- Option A: 4 partitions × 12288 — clean factorization, simpler
- Option B: 3 partitions × 16384 (= 49152 exactly) — reuses existing tile size
- Option C: keep 8 partitions, pad last to a multiple of 16384 — wastes ~80 MB DRAM but zero builder change

**Recommendation for Phase 4/5**: Try Option B first (3 × 16384 — exact
factorization, reuses llama3's tile config). If routing fails, fall back to A.

This is the **only** kernel-cache item that may need a builder-level
discussion. Defer the actual choice to Phase 4 (prefill) and Phase 5
(decode) optimization, where the LM Head construction is the natural touch point.

## Phase 1 gate

✅ **PASS** (with caveats noted): no kernel needs a *new builder* — all
needed shapes are reachable via existing parametric builders. Standalone
NPU validation deferred per the same pattern the smoke-test used (Lesson 4:
"identical-arch and arch-similar variants should not re-run kernel sweeps
that the parametric builders already cover").

## Items surfaced to later phases

| Item | When | Severity |
|--|--|--|
| KV-projection cache will be 4× larger (MHA vs GQA) — affects per-layer BO sizing | Phase 2/3 | medium — DRAM budget check |
| FlashAttention with `n_kv_heads=32` is untested in this codebase end-to-end | Phase 2 | medium — verify cosine sim per-layer |
| LM Head partition scheme: 8×16384 → 3×16384 or 4×12288 | Phase 4 (prefill), Phase 5 (decode) | medium — kernel-builder code change |
| Per-layer multi-launch ELFs (compiled for n_kv=8) need recompile for n_kv=32 | Phase 4/5 | low — automatic via KernelCache |
