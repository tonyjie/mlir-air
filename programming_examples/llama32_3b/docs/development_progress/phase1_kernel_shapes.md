# Phase 1 — Per-Kernel Shape Validation

**Date**: 2026-04-17
**Approach**: Following the smollm2_1_7b precedent (Lesson 4 — for arch-similar
deployments the parametric builders cover all needed shapes; defer standalone
NPU validation to Phase 2's cosine gate). Classify each kernel as
**(a) drop-in** (shape identical to llama3 → PASS by reference),
**(b) recompile** (shape-parametric builder → new cache entry, no builder change),
or **(c) novel** (builder change, compile-flag retuning, or L1-budget redesign required).

For Llama-3.2-3B vs llama3 (Llama-3.2-1B), the **non-trivial divergences** are
`emb_dim 2048→3072`, `head_dim 64→128`, `n_layers 16→28`, and the `(n_heads,
n_kv_heads) = (24, 8)` GQA tuple (group_size=3 vs llama3's 4). All shape-side
work is reachable via existing parametric builders; the **one** real risk is
the FlashAttention L1 budget at `head_dim=128`, which requires picking
`lkp != head_dim` and recompiling `attn_npu2.o` with the right `-Dlqp`/`-Dlkp`
defines.

---

## Kernel-source compatibility audit

Inspected each .cc kernel for `head_dim` hardcoding:

| Kernel | head_dim handling | Needs source change for hd=128? |
|---|---|---|
| `_llm_shared/kernel_builder/rope_halfsplit.cc` | `dims` is **runtime** parameter (extern C `rope(...)` takes `int32_t dims`); kernel computes `half = dims/2` at runtime | **NO** — runtime parametric |
| `flash_attention/kernel_fusion_based/attn_npu2.cc` | `dk`/`dv` are `#define` defaults (line 31-37); softmax constexpr `constexpr_sqrt_dk` already branches on `dk_full == 128` returning `sqrt(128)=11.313` | **NO** — recompile with `-Ddk=128 -Ddv=128 -Ddv_full=128` |
| `_llm_shared/kernel_builder/ffn_swiglu/silu_and_mul.cc` | Operates on `hidden_dim`, not head_dim | **NO** — drop-in (hidden_dim=8192, same as llama3) |
| `matrix_vector_multiplication/bf16/mv.cc` | Operates on (M, K), not head_dim | **NO** |

**Verification of head_dim=128 in the build pipeline**: there's an existing
lit test
`flash_attention/kernel_fusion_based/run_npu2_makefile_peano_llama3_8b.lit`
that runs FA with `DK=128 DV=128 NUM_HEADS=32 NUM_KV_HEADS=8` and CHECKs PASS.
This proves the kernel + builder support head_dim=128 end-to-end.

The `compile_attn_npu2(head_dim=...)` helper in
`_llm_shared/kernel_builder/external_kernels.py:119` already accepts
`head_dim` and passes `-Dlqp={head_dim} -Dlkp={head_dim} -Ddk={head_dim}
-Ddk_full={head_dim} -Ddv={head_dim} -Ddv_full={head_dim}`. **This default of
`lqp=lkp=head_dim` is wrong for head_dim=128 — see "Risk surfaced" below.**

---

## Shape inventory

### Prefill kernels (seq_len=2048)

| Kernel | llama3 shape | Llama-3.2-3B shape | Class | Notes |
|---|---|---|---|---|
| RMSNorm | `(emb=2048,)` | `(emb=3072,)` | recompile | 3072 % 8 (broadcast tiles) == 0 ✓ |
| Q proj GEMM | `M=2048, K=2048, N=2048` | `M=2048, K=3072, N=3072` | recompile | K=3072 / tile_k_l2=64 = 48 ✓; N=3072 / 64 / herd_n=4 = 12 ✓ |
| K proj GEMM | `M=2048, K=2048, N=512` | `M=2048, K=3072, N=1024` | recompile | n_kv_heads=8 × head_dim=128 = 1024 |
| V proj GEMM | `M=2048, K=2048, N=512` | `M=2048, K=3072, N=1024` | recompile | same |
| O proj GEMM | `M=2048, K=2048, N=2048` | `M=2048, K=3072, N=3072` | recompile | same shape as Q proj |
| Gate GEMM | `M=2048, K=2048, N=8192` | `M=2048, K=3072, N=8192` | recompile | hidden_dim=8192 unchanged |
| Up GEMM | `M=2048, K=2048, N=8192` | `M=2048, K=3072, N=8192` | recompile | same |
| Down GEMM | `M=2048, K=8192, N=2048` | `M=2048, K=8192, N=3072` | recompile | K=8192 same; N=3072 |
| RoPE Q/K | `head_dim=64, base=500000` | `head_dim=128, base=500000` | recompile | LUT regen + new dims; rope_halfsplit.cc accepts `dims` at runtime |
| FlashAttention | `lq=lk=2048, lkp=64, lqp=256, dk=dv=64, n_heads=32, n_kv=8` | `lq=lk=2048, lkp=64, lqp=256, **dk=dv=128**, n_heads=24, n_kv=8` | **novel** | dk_chunks = 128/64 = 2 (NOT shared buffers); see L1 budget below |
| LM Head GEMM | `M=2048, K=2048, N=128256 (8×16384)` | `M=2048, K=3072, N=128256 (8×16384)` | recompile | partition scheme drop-in (same vocab); K dim recompile |
| SwiGLU | `hidden_dim=8192` | `hidden_dim=8192` | drop-in | identical |
| Eltwise add | `emb_dim=2048` | `emb_dim=3072` | recompile | n.b. used inside `o_ffn_multi` builder |

### Decode kernels (per-token, M=1)

| Kernel | llama3 shape | Llama-3.2-3B shape | Class |
|---|---|---|---|
| RMSNorm | `(2048,)` | `(3072,)` | recompile |
| Q proj GEMV | `M=2048, K=2048` | `M=3072, K=3072` | recompile |
| K proj GEMV | `M=512, K=2048` | `M=1024, K=3072` | recompile |
| V proj GEMV | `M=512, K=2048` | `M=1024, K=3072` | recompile |
| O proj GEMV | `M=2048, K=2048` | `M=3072, K=3072` | recompile |
| Gate GEMV | `M=8192, K=2048` | `M=8192, K=3072` | recompile |
| Up GEMV | `M=8192, K=2048` | `M=8192, K=3072` | recompile |
| Down GEMV | `M=2048, K=8192` | `M=3072, K=8192` | recompile (uses `mv_k8192.o` extern-rename) |
| RoPE Q/K (decode) | `head_dim=64` | `head_dim=128` | recompile (LUT) |
| LM Head GEMV | `M=128256, K=2048 (8×16384)` | `M=128256, K=3072 (8×16384)` | recompile (partition drop-in, K recompile) |

---

## Verdicts by classification

### Drop-in (3 kernels)
SwiGLU (`hidden_dim=8192`), LM-Head **partition scheme** (vocab=128256
unchanged from llama3 → 8 partitions × 16384), and the `mv_k8192.o`
extern-rename mechanism. These reuse the same builder args and the same
`.o` artifact as llama3.

### Recompile, same builder (most kernels)
All matmul shapes (Q/K/V/O/Gate/Up/Down GEMM and GEMV at K=3072 or N=3072)
fit existing tile constraints (`tile_m=64, tile_k_l2=64, tile_k_l1=32,
tile_n=64, herd_m=8, herd_n=4`):
- 3072 = 1024×3 — divisible by tile_k_l2=64 (48 chunks), tile_k_l1=32 (96
  chunks), and tile_n × herd_n = 256 (12 chunks).
- 1024 (K/V projection N) — divisible by 256 (4 chunks).

RMSNorm at `emb_dim=3072`: 3072 / herd_x=8 = 384, divisible by the
broadcast tile width.

RoPE: `head_dim=128` → 64 cos/sin pairs, 4× more loop iterations than
head_dim=64 in `rope_halfsplit.cc` (still vectorized at N=16). LUT
regenerated from `rope_base=500000`.

LM-Head: vocab=128256 unchanged → same 8×16384 partition scheme as llama3
(this is the key win over smollm2 which had vocab=49152 needing a new
partition scheme).

**Cache invalidation is automatic via `KernelCache` keyed on builder args.**

### Novel — FlashAttention L1 budget at head_dim=128 (1 kernel)

**The one item that needs explicit attention.** `compile_attn_npu2(head_dim=N)`
defaults to `-Dlqp=N -Dlkp=N -Ddk=N -Ddv=N`, coupling tile sizes to head_dim.
At head_dim=128 this gives:

```
Per-core L1 with lkp=lqp=128, dk=dv=128 (shared-buffers ON, lkp==dk):
  Q tile : [tile_size_q=lqp/4=32, dk=128] =  8 KB
  K tile : [lkp=128, dk=128]              = 32 KB  (shares with Q)
  V tile : [lkp=128, dv=128]              = 32 KB
  Gp     : [tile_size_q=32, dv=128]       =  8 KB
  Misc   : up, sp, r, masks               =  2 KB
  TOTAL  : 32 (Q∪K) + 32 (V) + 8 (Gp) + 2 = 74 KB  > 64 KB L1 ✗
```

The working configuration for head_dim=128 — proven by the existing
`run_npu2_makefile_peano_llama3_8b.lit` test — is to **decouple** lkp from
dk:

```
lkp=64, lqp=256, dk=dv=128 (shared-buffers OFF, lkp != dk → dk_chunks=2):
  Q tile : [tile_size_q=64, dk=128]       = 16 KB
  K tile : [lkp=64, dk_tile=lkp=64]       =  8 KB  (1 of 2 dk chunks at a time)
  V tile : [lkp=64, dv_tile=lkp=64]       =  8 KB
  Gp     : [tile_size_q=64, dv=128]       = 16 KB
  Misc   : up, sp, r, masks               =  2 KB
  TOTAL  : 16 + 8 + 8 + 16 + 2            = 50 KB  ✓ fits 64 KB
```

**Action for Phase 2**: do NOT call `compile_attn_npu2(head_dim=128)`.
Instead, call `_compile_kernel` directly from the per-model script with
explicit flags `-Dlqp=256 -Dlkp=64 -Ddk=128 -Ddk_full=128 -Ddv=128
-Ddv_full=128` (plus the fixed flags from `compile_attn_npu2`). Python
builder side: `build_attn(lq=2048, lk=2048, lkp=64, lqp=256, dk=128,
dv=128, num_heads=24, num_kv_heads=8, causal=True)` with
`enable_shared_buffers=False` and `omit_while_true_loop=True`. This adds
the host round-trip cost (8 per layer at lq/lqp=8) to attention but is the
only way to satisfy the 64-KB L1 budget.

Note the causal-masking precondition `lqp/num_q_tiles == lkp` (line 103-106
in `attn_npu2_seqfirst.py`) — `256/4 == 64` ✓.

A perf optimization to consider in Phase 4: try `lkp=128, lqp=128,
num_q_tiles=8` (tile_size_q = 128/8 = 16) to enable shared buffers. With
tile_size_q=16:
- Q ∪ K: max(16*128, 128*128) = 32 KB
- V: 32 KB
- Gp: [16, 128] = 4 KB
- TOTAL: 32 + 32 + 4 + 2 = 70 KB — still over.

Or `lkp=128, lqp=128, num_q_tiles=4`, tile_size_q=32:
- Q ∪ K: max(32*128, 128*128) = 32 KB
- V: 32 KB; Gp: [32, 128] = 8 KB
- TOTAL: 72 KB — still over.

Conclusion: **lkp=64 with dk_chunks=2 is the only L1-feasible config** for
Llama-3.2-3B FA at head_dim=128 with the current memory layout. The host
round-trip overhead becomes a Phase 4 perf concern.

---

## Phase 1 gate

✅ **PASS** (with one item flagged for Phase 2): no kernel needs a *new
builder* — all needed shapes are reachable via existing parametric
builders. The single non-trivial item (FA at head_dim=128) is a
compile-flag/Python-arg change, not new code authoring.

Standalone NPU validation deferred per smollm2 Lesson 4: "identical-arch
and arch-similar variants should not re-run kernel sweeps that the
parametric builders already cover".

## Items surfaced to later phases

| Item | When | Severity |
|---|---|---|
| **FA L1 budget**: must use `lkp=64, lqp=256, dk=dv=128, dk_chunks=2` (not the `compile_attn_npu2(head_dim=128)` default of `lkp=lqp=128`). Per-model script must call `_compile_kernel` directly with the right flags, OR add a new `compile_attn_npu2_split(lqp, lkp, dk, dv)` API to `external_kernels.py`. | **Phase 2** | **medium-high** — must resolve before single-block runs |
| **DRAM budget**: 28 layers × ~36 MB BF16 weights/layer = ~1 GB weights; plus per-layer multi-launch ELF intermediates and BO replicas. Llama-3.2-3B's BF16 weights are ~6 GB on disk; runtime working set is closer to 11 GB on the 16 GB NPU2 system DRAM. Check that BO pre-loading doesn't OOM; may need a layer-streaming variant for Phase 4. | Phase 4 | medium — DRAM headroom audit |
| **GQA group_size=3** (24 heads / 8 kv-heads). The FA builder asserts `num_heads % num_kv_heads == 0` (24 % 8 == 0 ✓), but `head//group_size` indexing in attention will see kv_idx values 0..7 with 3 Q-heads per kv-head (vs llama3's 4 Q-heads per kv-head). No code change expected; verify in Phase 2 cosine gate. | Phase 2 | low |
| **Per-layer BO arrays sized to 28** (vs llama3's 16). The orchestration in `llama3_inference._preload_decode_weights` sizes its BO list from `config.n_layers`, so this is already parametric. Verify in Phase 5. | Phase 3/5 | low — should be automatic |
| **rope_scaling=llama3 long-context**: deferred per Path A roadmap. For `seq_len <= 8192` the standard LUT (already implemented) is correct. If/when we test at `seq_len > 8192`, generate the wavelength-remapped LUT with `factor=32, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192`. | Phase 6 (deferred) | low (deferred) |

## Decision points for Phase 2

1. **`compile_attn_npu2_split` vs inline `_compile_kernel`**: The cleanest
   long-term fix is to add a `compile_attn_npu2_split(lqp, lkp, dk, dv)`
   API in `_llm_shared/kernel_builder/external_kernels.py`, with
   `compile_attn_npu2(head_dim)` as a thin wrapper. **Recommend** the
   split-arg API — it's the obvious refactor and Llama-3-8B (head_dim=128
   too) would also benefit.

2. **omit_while_true_loop**: must be `True` for our non-shared-buffer FA
   config (no NPU-internal loop). Phase 4 perf will likely be limited by
   the 8 host round-trips per layer for FA — quantify before optimizing.
