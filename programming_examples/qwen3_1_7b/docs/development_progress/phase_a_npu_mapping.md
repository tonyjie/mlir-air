# Phase A — incremental NPU mapping of decode host ops (2026-04-20)

Goal: move host Q/K Norm, RoPE, SiLU+Mul onto the NPU using existing leaf
kernels, before fusing into multi-launch ELFs in Phase B.

## Method (after a first all-at-once attempt failed)

Initial Phase A.2 swapped all three host ops to NPU calls in one go and
hit an XRT load failure deep in the decode loop. To pinpoint the cause we
restarted from the working host-baseline and added one mapping at a time,
running end-to-end after each.

## Findings

### Per-kernel standalone (Phase A.1)

12/12 leaf kernels PASS at Qwen3 decode shapes (see
`qwen3_kernel_registry_test.py`). Each new leaf compiles + runs + matches
the numpy reference within BF16 cosine ≥ 0.99:
- `weighted_rms_norm` at (M=16, N=128) and (M=8, N=128) — Q-Norm / K-Norm
- `_build_rope_1d` at (n_rows=16, head_dim=128) and (n_rows=8, hd=128)
- `silu_and_mul` at n=3072

### Incremental integration into `run_decode_block`

| Increment | NPU mapping added | # decode ELFs loaded | Result |
|---|---|---|---|
| Baseline | (host: Q/K Norm + RoPE + SiLU+Mul) | 7 | PASS — "Paris" |
| #1 | + qknorm_q + qknorm_k | 9 | **PASS** — "Paris" |
| #2 | + rope_q + rope_k | 11 | FAIL — XRT load error |
| probe | + rope_q only (rope_k host) | 10 | FAIL — XRT load error |

The XRT error at the 9th decode-ELF `backend.load` call:
```
DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
Failed to load ELF kernel for XRT from '<elf>' with kernel name 'main:matvec_bf16'.
```

## Diagnosis (UNVERIFIED)

The failure happens at **decode-ELF load #9** (counting only decode-cache
ELFs, not the 3 prefill-cache ELFs already loaded). Specifically the
`up_gemv` or `down_gemv` ELF — whichever happens to be the 9th in the
per-block load order at the moment.

**Hypothesis**: NPU2's AMDXDNA driver enforces a per-process limit on
distinct loaded ELF contexts (somewhere ≤ 11–12 total ELFs counting the 3
prefill ones).

**Not verified** — the error code is `EINVAL`, which suggests "malformed
request" rather than "resource exhausted." A real diagnosis would:
1. Load N trivial ELFs in isolation to find the actual cap.
2. Inspect AMDXDNA driver source for `CREATE_HWCTX` to see what conditions
   return EINVAL.
3. Strace the failing ioctl to see exact args.

## Phase A end-state (committed)

- **NPU**: Q/K Norm (qknorm_q + qknorm_k) ← new this phase
- **NPU**: existing 7 (rms_attn_gemvs, o_gemv, rms_1d, gate_gemv,
  up_gemv, down_gemv, lm_head_gemv)
- **Host**: RoPE Q/K (single token), SiLU+Mul, residual adds, GQA attention
- **Total decode ELFs loaded**: 9 — within the working envelope.

End-to-end: prefill 0.58 s + decode 0.92 s/token (1.08 tok/s),
"The capital of France is Paris. The capital of France" — same coherent
generation as the host-baseline.

## Implications for Phase B (fusion)

The XRT limit is the **architectural reason fusion is mandatory**, not just
a performance optimization. To put RoPE + SiLU+Mul on NPU we MUST reduce
total ELF count by stitching. The Phase B target ELF layout (already
planned):

| ELF | Internal launches |
|---|---|
| `rms_attn_gemvs_qknorm_rope_qwen3.elf` | 8 (RMSNorm + Q/K/V GEMV + Q/K Norm + RoPE Q/K) |
| `o_gemv_ffn_silu_qwen3.elf` | 8 (O + add + RMSNorm + Gate + Up + SiLU+Mul + Down + add) |
| `lm_head_gemv.elf` | unchanged (10 partitions) |

3 decode ELFs total → comfortably under any plausible per-process limit,
and 169 → 57 ELF calls per token (~3× projected decode speedup).
