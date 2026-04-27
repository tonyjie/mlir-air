# Supported Kernels Registry — LLM Deployment on NPU2

**Version**: v1 (in `kernel_registry/`) — 2026-04-26
**Per-model shape catalogs**: [`llama3.2_1b.md`](llama3.2_1b.md) (and future models)
**Sibling docs**: [`../kernels/`](../kernels/) (per-kernel design notes), [`../aie2p_hardware_limits.md`](../aie2p_hardware_limits.md)

---

## What this doc is

A **model-agnostic** index of every leaf kernel that the LLM-deployment
skill chain can compose. For each kernel:

1. **Builder location** + signature.
2. **All known-tested shapes** across deployments — a single shape table
   per kernel showing which model uses what config and the measured
   cosine + perf.
3. **How to test** — copy-pasteable commands, with documented
   tolerances (rtol / atol / min_correlation).
4. **Constraints** — BD-friendliness rules, silent-corruption traps.

For the question "what shapes does *llama3-1B* use?", see
[`llama3.2_1b.md`](llama3.2_1b.md). For each future model, a sibling
file documents its specific shape recipe.

**Status legend** (for the "Status" column in shape tables):
- ✅ — passed on real NPU at this exact shape, with the documented threshold
- ⚠️ — used in a deployment whose end-to-end verification is still pending
- ❌ — broken or missing

---

## NPU2 compute-tile usage at a glance

NPU2 (Strix) has **32 physical compute tiles** arranged as 8 columns × 4
rows. Each kernel allocates them differently depending on the math
structure and what limits parallelism:

| Kernel | At llama3-1B prefill shape | Tiles in flight | Why |
|---|---|---|---|
| **GEMM** (Q/K/V/O/Gate/Up/Down prefill) | M×K×N, large | **32 tiles** = 1 segment × 8×4 herd | Single matmul split across the full chip (M dim across rows, N dim across cols) |
| **FA** (FlashAttention) | LQ=LK=2048, 32/8 heads | **32 tiles** = 2 segments × 4×4 herd | Each segment processes one of 2 attention heads concurrently; 16 launch iterations cover all 32 heads |
| **GEMV** (decode Q/K/V/O/Gate/Up/Down) | M=output_dim, K=input_dim, batch=1 | **8 tiles** = 1 segment × 8×1 herd | Decode batch=1 → only the M dim has parallelism; row-parallel across 8 columns |
| **RMSNorm** | seq × emb_dim | **8 tiles** = 1 segment × 8×1 herd | Row-parallel: one row per tile (inner reduction is vectorized within each tile) |
| **RoPE** | rope_rows × head_dim | **8 tiles** = 1 segment × 8×1 herd | Row-parallel; each tile rotates `rope_rows / 8` rows |
| **SiLU + Mul** (within FFN block) | seq × hidden_dim | **8 tiles** = 1 segment × 8×1 herd | Element-wise op; throughput-bound on DMA, not on compute |

So **GEMM and FA both fully utilize the 32-tile fabric** at large shapes;
pointwise/row-parallel ops use only 8 tiles because either batch=1 or the
per-element cost is too low to justify wider parallelism.

---

## 1. RMSNorm (weighted)

> Per-row weighted RMS normalization: `out[b,n] = w[n] * x[b,n] / RMS(x[b,n])`

- **Builder**: `programming_examples/weighted_rms_norm/weighted_rms_norm.py:build_module(M, N, dtype, vector_size=16, herd_x=1)`
- **Multi-tile standalone wrapper**: `_llm_shared/kernel_builder/rmsnorm_multitile/run.py`
  (wraps the bare `air.herd` that `build_module` emits at `herd_x ≥ 2`
  in `air.launch + air.segment` via `_wrap_ir_in_launch`)
- **C++ kernel**: none (pure MLIR, vectorized inner loop)
- **Tolerances**: `rtol=5e-2, atol=5e-1, min_correlation=0.99`
- **Design notes**: [`../kernels/rmsnorm.md`](../kernels/rmsnorm.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `vector_size` | 16 | `N % vector_size == 0`; AIE2P BF16 vector width is 16 | 16 is the only sensible value for BF16 |
| `herd_x` | 1 (decode), 8 (prefill) | `M % herd_x == 0` | herd_x>1 emits a **bare herd** (needs `_wrap_ir_in_launch` for standalone testing — see `rmsnorm_multitile/`) |

For new shapes, start by mirroring the closest-shape llama3 entry below, then tune.

### Tested shapes

| (M, N) | herd | Used by | Cosine | max_abs / max_rel | Profile (5w + 20iter) | Status |
|---|---|---|---|---|---|---|
| 2048 × 2048 (2D, prefill) | 8×1 (8 tiles, via `_wrap_ir_in_launch`) | llama3-1B prefill final-norm + per-block norms | **0.999942** | 0.148 / 0.090 | kernel 0.899 ms / wall 2.020 ms | ✅ via `_llm_shared/kernel_builder/rmsnorm_multitile/` |
| 256 × 2048 (2D, single-tile smoke) | 1×1 (1 tile) | smoke test | XRTRunner PASS | — / — | — | ✅ |
| **M=1, N=2048** (1D variant `_build_rms_1d`, decode) | 1×1 | llama3-1B decode (in `rms_gemv_rope.elf`) | **0.999991** | 0.133 / 0.081 | kernel 0.058 ms (5w+20iter) | ✅ via `_llm_shared/kernel_builder/rmsnorm_decode_1d/` |

### How to test

```bash
cd programming_examples/_llm_shared/kernel_builder/rmsnorm_multitile

# Correctness ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock make run

# Profile (5 warmup + 20 iter) ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

---

## 2. GEMM (BF16 matrix multiply)

> Generic `C = A @ B` for Q/K/V/O/Gate/Up/Down weight projections.

- **Builder**: `_llm_shared/kernel_builder/gemm_builder.py:_build_gemm_module(m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n)`
- **Cosine sweep wrapper**: `_llm_shared/kernel_builder/gemm_verify/run.py`
- **C++ kernel**: none (direct codegen via Peano)
- **Tolerances**: `rtol=8e-2, atol=2.0, min_correlation=0.999` (block cosine — replaces upstream spot-check rtol=4e-2 which is too tight for K=2048 BF16)
- **Constraints**: silent-corruption trap if `N % (tile_n × herd_n) != 0` (qwen25 LESSON 2 — builder does NOT assert)
- **Design notes**: [`../kernels/gemm.md`](../kernels/gemm.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `tile_m` | 64 | `M % (tile_m × herd_m) == 0` | larger → fewer M iterations, more L1 per tile |
| `tile_k_l2` | 64 or 256 | `K % tile_k_l2 == 0` | larger → fewer DMA reloads of A, more L2 used |
| `tile_k_l1` | 32 | `tile_k_l2 % tile_k_l1 == 0` | inner-accumulation chunk |
| `tile_n` | 64 (small N) or 128 (large N) | **`N % (tile_n × herd_n) == 0`** ⚠️ **silent corruption** if violated (qwen25 LESSON 2; not asserted) | larger → fewer N iterations, more L1 per tile |
| `herd_m` | 8 | `M ≥ tile_m × herd_m` | row-parallelism (typically 8 = full chip rows) |
| `herd_n` | 4 | `N ≥ tile_n × herd_n` | column-parallelism (typically 4 = full chip cols → 8×4=32-tile herd) |

**L1 budget**: `tile_m × tile_k_l1 + tile_k_l1 × tile_n + tile_m × tile_n` BF16 elements per tile must fit ≤ 64 KB.

For new shapes: start with the nearest llama3 entry's tile config; **always
verify `N % (tile_n × herd_n) == 0`** before running (silent failure mode).

### Tested shapes

| (M, K, N) | tile_m, tile_k_l2, tile_n | herd | Used by | Cosine | max_abs / max_rel | Profile (10w + 20iter, C++ harness) | Status |
|---|---|---|---|---|---|---|---|
| 2048 × 2048 × 2048 | 64, 256, 64 | 8×4 (32 tiles) | llama3-1B Q/O proj | **0.999910** | — / — | 2.79 ms / **6167 GFLOPS** | ✅ |
| 2048 × 2048 × 512 | 64, 64, 128 | 8×4 (32 tiles) | llama3-1B K/V proj | **0.999910** | — / — | 0.75 ms / ~5700 GFLOPS | ✅ |
| 2048 × 2048 × 8192 | 64, 64, 128 | 8×4 (32 tiles) | llama3-1B Gate/Up proj | **0.999910** | — / — | 10.86 ms / ~6300 GFLOPS | ✅ |
| 2048 × 8192 × 2048 | 64, 256, 64 | 8×4 (32 tiles) | llama3-1B Down proj | **0.999778** | — / — | 10.10 ms / ~6800 GFLOPS | ✅ |

> ⚠️ **Dead code**: `programming_examples/llama3/llama3_prefill.py:688-718`
> contains a byte-for-byte duplicate of `_build_gemm_module`, unused at
> runtime. Flagged for cleanup, not removed.

### How to test

```bash
cd programming_examples/_llm_shared/kernel_builder/gemm_verify

# Correctness — pick a shape (qo / kv / gateup / down) ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SHAPE=qo

# Profile via the existing C++ test harness (10 warmup + 20 iter):
cd programming_examples/matrix_multiplication/bf16
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make profile M=2048 K=2048 N=2048 \
    TILE_M=64 TILE_K_L2=256 TILE_K_L1=32 TILE_N=64 \
    MAX_HERD_M=8 MAX_HERD_N=4 AIE_TARGET=aie2p
```

---

## 3. GEMV (BF16 matrix-vector multiply)

> Decode hot path: `y[m] = A[m,k] @ x[k]`. Used for every projection at decode time (M=1 token, but the kernel processes the full output dim).

- **Builder**: `programming_examples/matrix_vector_multiplication/bf16/matvec.py:build_module(m, k, dtype, dtype_out, tile_m=8, herd_m=8, m_input=4, k_split=None, ...)`
- **C++ kernel**: `matrix_vector_multiplication/bf16/mv.cc` → `mv.o` (or renamed variants `mv_k8192.o`, `mv_og.o`, `mv_dg_qwen3.o` for multi-launch ELF symbol uniqueness)
- **Tolerances**: `rtol=0.04, atol=1e-3` (XRTRunner spot-check; correlation also reported)
- **Constraints**:
  - **Rule B** (K-DMA repeat ≤ 255 → max practical K ≈ 8160). Use `k_split` above that.
  - **Rule D** (L2 cap): `K × herd_m × tile_m × 2B ≤ 524288`. Builder asserts.
- **Design notes**: [`../kernels/gemv.md`](../kernels/gemv.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `tile_m` | 8 (most) / 2 (Down K=8192) | `M % (tile_m × herd_m) == 0`; **Rule D**: `K × herd_m × tile_m × 2 ≤ 524288` (asserted by builder) | larger → fewer launches, but L2 cap is the wall |
| `m_input` | 4 (most) / 2 (Down) | divides `tile_m`; affects inner kernel-call count | `tile_m == m_input` → inner_loop=1, simpler Rule C arithmetic |
| `herd_m` | 8 | typically 8 = full chip rows | column-parallelism |
| `k_split` | None (most) / 70 (qwen25 K=8960) | `K % k_split == 0`; **NOT a CLI flag**, only a `build_module` kwarg | **required** when K > 8160 (Rule B); pre-splits K-DMA so lowering doesn't auto-split into > 255 repeats |

**Rule C (combined channel reads)**: when multiple GEMVs in a multi-launch
ELF share an L2→L1 channel, their `launch_count × (tile_m / m_input)`
values **add up** and must stay ≤ 255 per channel. Standalone test
exempt; multi-launch users must check.

### Tested shapes

| (M, K) | tile_m | m_input | herd | Used by | Cosine | max_abs / max_rel | Profile (10w + 20iter, C++) | Status |
|---|---|---|---|---|---|---|---|---|
| 2048 × 2048 | 8 | 4 | 8×1 (8 tiles) | llama3-1B Q/O proj (decode) | XRTRunner PASS | — / — | 320.6 µs / 26.17 GFLOPS | ✅ |
| 512 × 2048 | 8 | 4 | 8×1 (8 tiles) | llama3-1B K/V proj (decode) | XRTRunner PASS | — / — | 117.5 µs / 17.85 GFLOPS | ✅ |
| 8192 × 2048 | 8 | 4 | 8×1 (8 tiles) | llama3-1B Gate/Up proj (decode), LM Head per partition | XRTRunner PASS | — / — | 1154.6 µs / 29.06 GFLOPS | ✅ |
| 2048 × 8192 | **2** | 2 | 8×1 (8 tiles) | llama3-1B Down proj (decode); uses `mv_k8192.o` rename | XRTRunner PASS | — / — | 1038.0 µs / **32.32 GFLOPS** | ✅ |

> Down requires `tile_m=2` (Rule D): default `tile_m=8` would request 1 MB
> L2 buffer (8192 × 8 × 8 × 2B), exceeding the 512 KiB cap. Builder asserts
> and errors out cleanly.

### How to test

```bash
cd programming_examples/matrix_vector_multiplication/bf16

# Correctness — all 4 shapes ✅
for c in \
  "M=2048 K=2048 TILE_M=8 M_INPUT=4 HERD_M=8" \
  "M=512  K=2048 TILE_M=8 M_INPUT=4 HERD_M=8" \
  "M=8192 K=2048 TILE_M=8 M_INPUT=4 HERD_M=8" \
  "M=2048 K=8192 TILE_M=2 M_INPUT=2 HERD_M=8"; do
  flock -x -w 1800 /tmp/mlir-air-npu.lock make run $c
done

# Profile — same parameters with `make profile` (10 warmup + 20 iter via C++ test exe)
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make profile M=2048 K=2048 TILE_M=8 M_INPUT=4 HERD_M=8
```

---

## 4. RoPE (half-split rotary positional encoding)

> Per-row half-split rotation matching HuggingFace Llama: `out[i] = x[i]*cos[i] - x[i+half]*sin[i]`, `out[i+half] = x[i]*sin[i] + x[i+half]*cos[i]`. LUT layout `[cos[0:half], sin[0:half]]`.

- **Builder + harness**: `_llm_shared/kernel_builder/rope_halfsplit/run.py:build_module(outer_rows, outer_cols, embed_dim, dtype, herd_x)`
- **C++ kernel**: `_llm_shared/kernel_builder/rope_halfsplit.cc` → `rope.o`
- **Tolerances**: `rtol=4e-2, atol=5e-2, min_correlation=0.99`
- **Design notes**: [`../kernels/rope.md`](../kernels/rope.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `embed_dim` (= head_dim) | 64 (1B), 128 (3B/qwen) | `embed_dim % 16 == 0` (vector width); also `dims/2 % 16 == 0` for the C++ kernel inner loop | hardcoded at compile time of rope.o — recompile if changed |
| `herd_x` | 8 (prefill 2D) / 1 (decode 1D) | `rope_rows % (herd_x × herd_y) == 0`; herd_y fixed at 1 | row-parallelism |
| `outer_rows`, `outer_cols` (2D) / `n_rows` (1D) | derived from (seq_len, n_heads, head_dim) | `total = outer_rows × outer_cols % embed_dim == 0` | data shape, not really a knob |

**LUT layout** is fixed by the C++ kernel: `[cos[0:half], sin[0:half]]`
concatenated per row. To switch to interleaved layout, a different `.cc`
kernel is required (none exists today).

### Tested shapes

| Shape | head_dim | rope_rows | herd | Used by | Cosine | max_abs / max_rel | Profile (5w + 20iter) | Status |
|---|---|---|---|---|---|---|---|---|
| outer 2048×2048 (2D) | 64 | 65536 | 8×1 (8 tiles) | llama3-1B Q-RoPE prefill | **0.999963** | — / — | kernel **0.634 ms** / wall 1.985 ms | ✅ |
| outer 2048×512 (2D) | 64 | 16384 | 8×1 (8 tiles) | llama3-1B K-RoPE prefill | **0.999994** | — / — | (¼ Q shape) | ✅ |
| **n_rows=32, hd=64 (1D variant `_build_rope_1d`)** | 64 | 32 | 1×1 (1 tile, herd_x=1) | llama3-1B Q-RoPE **decode** | **0.999995** | 0.0156 / 0.0078 | kernel **0.061 ms** (5w+20iter) | ✅ via `_llm_shared/kernel_builder/rope_decode_1d/` |
| **n_rows=8, hd=64 (1D variant)** | 64 | 8 | 1×1 (1 tile) | llama3-1B K-RoPE **decode** | **0.999995** | — / — | (smaller than Q) | ✅ |

### How to test

```bash
cd programming_examples/_llm_shared/kernel_builder/rope_halfsplit

# Q-RoPE shape (default) — correctness ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock make run

# Q-RoPE shape — profile (5 warmup + 20 iter) ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile

# K-RoPE shape (smaller outer_cols) ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 run.py --outer-cols 512
```

---

## 5. FlashAttention head-first kernel (the original NPU2 FA)

> Causal attention with cascade-stage merge. The original NPU2 FA
> implementation; predates the seq-first variant in §6.

- **Builder**: `programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py:build_module(lk, lkp, lq, lqp, dk, dv, num_heads, num_kv_heads, num_q_tiles=4, num_cascade_stages=4, causal=True)`
- **C++ kernel**: `flash_attention/kernel_fusion_based/attn_npu2.cc` → `attn_npu2.o`
- **Tolerances**: cosine reported in test output (no hard threshold by default)
- **Design notes**: [`../kernels/flash_attention.md`](../kernels/flash_attention.md), [`flash_attention/kernel_fusion_based/CLAUDE.md`](../../../flash_attention/kernel_fusion_based/CLAUDE.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `lqp` (Q chunk per launch iter) | 256 | `lqp % num_q_tiles == 0`; `lqp ≥ tile_size_q × num_q_tiles` | larger → fewer launches, more L1 per tile |
| `lkp` (K/V chunk per inner iter) | 64 | `lk % (lkp × num_cascade_stages) == 0`; `dk % lkp == 0` (dk_chunks = dk/lkp); typically `lkp = head_dim` for shared-buffers fast path | **`lkp == dk` enables shared buffers + while-true loop → ~4× perf**; otherwise host round-trip per Q chunk |
| `dk`, `dv` | 64 (head_dim=64) / 128 (head_dim=128) | `dk_chunks = dk / lkp`; per-tile L1 must fit `tile_size_q × dk` BF16 | `dk=128` may exceed L1 budget at `lkp=128` — use `lkp=64` (dk_chunks=2) and the seq-first/head-first split |
| `num_q_tiles` | 4 | `lqp % num_q_tiles == 0`; `tile_size_q = lqp / num_q_tiles` | partitions Q across herd rows |
| `num_cascade_stages` | 4 | `lk % (lkp × num_cascade_stages) == 0` | partitions K/V across herd cols; merge via cascade channel |
| `num_heads_per_unroll` | 2 (fixed in builder) | `num_heads % num_heads_per_unroll == 0` | segment dimension; gives 2 segments × herd_size = 32 tiles total |
| `causal` | True (LLM) | bool | adds causal-mask conditional in inner loop |

**Compile-flag trap** (LESSON 3, llama32_3b 2026-04-18): the `.o` macros
`-Dlqp` etc. must equal **per-tile** sizes (`lqp / num_q_tiles`),
not the launch-level `lqp`. Mixing conventions silently produces
all-NaN. The seq-first wrapper `compile_attn_npu2(head_dim)` passes
`num_q_tiles=1` to keep `-Dlqp == head_dim` for back-compat — **do
not change**.

### Tested shapes

| (LQ, LK, head_dim) | (LQP, LKP) | n_heads / n_kv_heads | tiles in flight | Used by | Cosine | max_abs / max_rel | Profile (10w + 20iter, C++) | Status |
|---|---|---|---|---|---|---|---|---|
| 2048, 2048, 64 | 256, 64 | 32 / 8 | **32** (2 segments × 4×4 herd) | llama3-1B FA path (also lit-tested at default 512×512) | (lit smoke 512×512 only — production correctness via §6 seq-first) | — / — | **15.26 ms / 2251 GFLOPS** (no turbo) | ✅ |
| 2048, 2048, 128 | 256, 64 | 32 / 8 | 32 | llama32_3b, qwen25_1_5b (via Option C wrapper) | (deployment-level) | — / — | not measured this session | ⚠️ |

> **How 32 tiles are organized**: `air.launch (16 iters)` → `air.segment sizes=[2,1]` → `air.herd sizes=[4,4]` = **2 segments × 16 herd-tiles = 32 tiles in flight**, processing 2 heads concurrently per launch. 16 launch iterations cover all 32 heads.

### How to test

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Llama3-1B head-first shape ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make run LQ=2048 LK=2048 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8

# Profile (10 warmup + 20 iter via C++ test exe) ✅
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make profile LQ=2048 LK=2048 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8

# head_dim=128 shape (llama32_3b / qwen25_1_5b production):
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  make run LQ=2048 LK=2048 LKP=64 LQP=256 DK=128 DV=128 NUM_HEADS=24 NUM_KV_HEADS=8
```

> ⚠️ Lit tests in this dir (`run_npu2_makefile_peano_llama3_8b.lit`,
> `qwen25_7b.lit`, `gptoss_20b.lit`) accept the Makefile defaults
> `LQ=LK=512` and only override head shape — they are **smoke tests
> of the head_dim path**, NOT full sequence length validation.

---

## 6. FlashAttention seq-first variant (head_dim ≤ 64)

> Variant of §5 that takes seq-first activation layout
> `[lq, num_heads, dk]` directly — matches a llama3-style prefill
> pipeline (RMSNorm → Q/K/V GEMM → seq-first FA), avoiding the host
> transpose Option C needs. Same `attn_npu2.cc` C++ kernel as §5,
> different macro flags.

- **Builder**: `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:build_module(seq_len_q, seq_len_k, num_heads, num_kv_heads, head_dim_q, head_dim_v, ...)`
- **External kernel compile**: `external_kernels.compile_attn_npu2(head_dim=64)` (passes `num_q_tiles=1` for seq-first compatibility — do not change)
- **Tolerances**: `rtol=0.04, atol=0.15, min_correlation=0.99` (per `attn_npu2_seqfirst.py:1308-1311`)

### Tunable parameters

Same as §5 — both share `attn_npu2.cc` C++ kernel and most builder
parameters. Differences from §5:

| Knob | Seq-first usage | Note |
|---|---|---|
| Activation layout | seq-first `[lq, num_heads, dk]` | matches RMSNorm/GEMM/RoPE upstream — no host transpose |
| `num_q_tiles` (in compile_attn_npu2) | **1** (back-compat) | preserves `-Dlqp == head_dim` macro convention |
| `dk_chunks > 1` path | **broken at runtime** (LESSON 3) | head_dim=128 deployments must route through head-first kernel (§5) via Option C wrapper (§A) |

### Tested shapes

| (LQ, LK, head_dim) | (LQP, LKP) | n_heads / n_kv_heads | tiles in flight | Used by | Cosine | max_abs / max_rel | Profile | Status |
|---|---|---|---|---|---|---|---|---|
| 2048, 2048, 64 | 256, 64 | 32 / 8 | 32 (same structure as §5) | llama3-1B prefill (production), smollm2_1_7b | **0.996872** | — / — | (Python `--profile` not implemented; for FA timing use §5's `make profile` — same `attn_npu2.o` underneath) | ✅ |

### How to test

```bash
cd programming_examples/flash_attention/kernel_fusion_based
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 attn_npu2_seqfirst.py \
    --lk 2048 --lq 2048 --lkp 64 --lqp 256 \
    --dk 64 --dv 64 --num-heads 32 --num-kv-heads 8
```

> head_dim=128 deployments (llama32_3b, qwen25_1_5b) cannot use seq-first
> because the `dk_chunks > 1` shim-DMA path hangs at runtime
> (llama32_3b LESSON 3). They route to the head-first kernel (§5) via the
> Option C orchestration wrapper (§A below).

---

## 7. SiLU + Mul (SwiGLU activation)

> `out[i] = silu(gate[i]) * up[i]`. The standalone test in
> `_llm_shared/kernel_builder/ffn_swiglu/` bundles Gate GEMM + Up GEMM +
> this SwiGLU + Down GEMM into ONE 4-launch ELF — testing the block also
> exercises GEMM at all 3 production shapes (a strong end-to-end
> correctness signal — see §2).

- **Builder (kernel)**: `_llm_shared/kernel_builder/ffn_swiglu/silu_and_mul.py:build_module_2d(rows, cols, tile_n, dtype, herd_x=8, herd_y=1)`
- **Builder (block)**: `_llm_shared/kernel_builder/ffn_swiglu/run.py:build_ffn_module(seq_len, emb_dim, hidden_dim, ...)`
- **C++ kernel**: `_llm_shared/kernel_builder/ffn_swiglu/silu_and_mul.cc` → `silu_and_mul.o`
- **Tolerances**: `rtol=0.04, atol=4.0, min_correlation=0.999`
- **Design notes**: [`../kernels/silu_and_mul.md`](../kernels/silu_and_mul.md), [`../kernels/ffn_swiglu.md`](../kernels/ffn_swiglu.md)

### Tunable parameters (1D `build_module` and 2D `build_module_2d`)

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `n` (1D) or `(rows, cols)` (2D) | 2048 × 8192 (2D) | `n % tile_n == 0`; for 2D, `rows × cols % tile_n == 0` after collapse | data shape |
| `tile_n` | 4096 (in FFN block) | divides `n` (or `rows × cols`) | larger → fewer launches |
| `herd_x` | 8 (in FFN block 2D variant) | `n / tile_n % herd_x == 0` | row-parallelism |
| `herd_y` | 1 | (defaults vary; 1D builder defaults to 2) | usually 1 for the 2D variant used in FFN |

External kernel `silu_and_mul.o` is fixed (no compile flags).

### Tested shapes (via FFN block)

| seq × emb × hidden | herd | Used by | Cosine | max_abs / max_rel | Profile (1w + 5iter) | Status |
|---|---|---|---|---|---|---|
| 2048 × 2048 × 8192 | 8×4=32 tiles (GEMMs) + 8×1=8 tiles (SwiGLU) | llama3-1B FFN block | **0.999575** | — / — | kernel **35.4 ms** / wall **47.1 ms** (vs 4 separate kernels' 109 / 149 ms — **3.1× speedup**) | ✅ |

### How to test

```bash
cd programming_examples/_llm_shared/kernel_builder/ffn_swiglu/build_peano

# K2 below: ffn_swiglu/run.py needs both PYTHONPATH entries (sys.path bug):
PYTHONPATH="/home/jiajli/apps/mlir-air/programming_examples:\
/home/jiajli/apps/mlir-air/programming_examples/llama3:$PYTHONPATH" \
  flock -x -w 1800 /tmp/mlir-air-npu.lock make -f ../Makefile run

PYTHONPATH="..." \
  flock -x -w 1800 /tmp/mlir-air-npu.lock make -f ../Makefile profile
```

---

## 8. Eltwise Add (residual)

> Used by llama3-1B (and other deployments) for the **two residual adds
> per transformer layer** — post-attn `proj + x_residual` and post-FFN
> `proj + x_residual`. Both prefill and decode FFN blocks instantiate it.

- **Builder**: `programming_examples/eltwise_add/eltwise_add.py:build_module(n, tile_n, dtype, vector_size=16, herd_x=1, herd_y=2)`
- **Custom wrappers** (in deployment-local code):
  - `llama3/multi_launch_builder/o_ffn_multi.py:_build_add_2d_to_2d` (prefill, post-attn)
  - `llama3/multi_launch_builder/o_ffn_multi.py:_build_add_2d_to_1d` (prefill, FFN-out)
  - `llama3/multi_launch_builder/o_gemv_ffn_multi.py` calls `eltwise_add.build_module` directly + wraps via `_wrap_ir_in_launch` (decode)
- **Design notes**: [`../kernels/eltwise_add.md`](../kernels/eltwise_add.md)

### Tunable parameters

| Knob | Llama3 used | Hard constraint | Tradeoff / note |
|---|---|---|---|
| `n` | 2048 (decode 1D) / 4194304 (synthetic) | `n % tile_n == 0`; `n % (herd_x × herd_y) == 0` | data shape |
| `tile_n` | 256 (decode) / 2048 (default) | divides `n` and `n / (herd_x × herd_y)` | per-tile DMA chunk |
| `vector_size` | 16 | `tile_n % vector_size == 0`; AIE2P BF16 width | usually 16 |
| `herd_x` | 8 | `n / tile_n % herd_x == 0` | row-parallelism |
| `herd_y` | 1 (decode) / 2 (default) | `n / (tile_n × herd_x) % herd_y == 0` | column-parallelism |

⚠️ **Peano `llc` crash** at certain `(n, herd_x, herd_y)` combos —
known to crash at `n=1024, herd_x=8, herd_y=1` (qwen3). Production
uses `n=2048, herd_x=8, herd_y=1` — works.

⚠️ **Standalone bare-herd at herd_x=8 needs `_wrap_ir_in_launch`**
(same gap as RMSNorm — closed by `eltwise_add_multitile/`).

### Tested shapes

| n | tile_n | herd | Used by | Cosine | max_abs / max_rel | Profile (5w + 20iter) | Status |
|---|---|---|---|---|---|---|---|
| n=2048, tile_n=256 (1D) | 256 | 8×1 (8 tiles) | llama3-1B decode FFN block ×2 (post-attn + FFN residuals) | **0.999996** | 0.0078 / 0.0078 | kernel **0.111 ms** | ✅ via `_llm_shared/kernel_builder/eltwise_add_multitile/` |
| rows=2048, cols=2048, 2D→2D (`_build_add_2d_to_2d`) | — | 8×1 | llama3-1B prefill post-attn residual | **0.999996** | 0.0078 / 0.0078 | kernel **0.509 ms** | ✅ via `_llm_shared/kernel_builder/eltwise_add_prefill_2d/` (MODE=2d_to_2d) |
| rows=2048, cols=2048, 2D→1D (`_build_add_2d_to_1d`) | — | 8×1 | llama3-1B prefill FFN residual | **0.999996** | 0.0078 / 0.0078 | kernel **0.494 ms** | ✅ via `_llm_shared/kernel_builder/eltwise_add_prefill_2d/` (MODE=2d_to_1d) |
| 4,194,304 (synthetic) | (default) | 8×1 | smoke test | not run this session | — / — | — | n/a |

> Eltwise Add was previously believed unused in production. Wrong: llama3
> uses it 4× per layer (2 prefill residuals + 2 decode residuals). The
> 1D 8-tile path is now standalone-verified; 2D prefill variants share
> the same underlying kernel and are validated transitively via FFN block.

> Known issue: `n=1024 + herd_x=8 + herd_y=1` triggers a Peano `llc`
> compiler crash (qwen3 found this). llama3 uses `n=2048` with the
> same herd config — no crash at that shape.

### How to test

```bash
cd programming_examples/eltwise_add/build_peano
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 ../eltwise_add.py --n 4194304 --dtype bf16 --herd-x 8 --herd-y 1
```

---

## A. FlashAttention Option C orchestration (head_dim=128 deployments)

`_llm_shared/phase_helpers/headfirst_fa.py` —
`install_headfirst_fa_wrapper()` + `compile_headfirst_fa_kernel()`.
**Not a kernel** — orchestration layer that monkey-patches
`llama3_prefill._run_cached` to do host transposes between seq-first
(rest of pipeline) and head-first (calls §5 kernel under the hood).

**Why it exists**: seq-first FA's `dk_chunks > 1` shim-DMA path hangs at
runtime at head_dim=128 (llama32_3b LESSON 3); routing through the
head-first kernel via host transposes is the workaround.

**Used by**: llama32_3b, qwen25_1_5b in production. Phase 3 + Phase 5
end-to-end correctness ✅ for both. No standalone test for the wrapper
itself.

---

## B. RoPE-1D (head_dim=128 deployments, qwen3 only)

Lives in `qwen3_0_6b/rope_1d.py` and `qwen3_1_7b/rope_1d.py`; not yet
promoted to `_llm_shared/`. Tested via
`qwen3_kernel_registry_test.py:217-252`.

**Promotion criterion**: a 3rd non-qwen3 deployment validates the same
pattern.

---

## C. Q/K Norm (qwen3 only — reuses RMSNorm builder)

Reuses `weighted_rms_norm.py:build_module` at small shape (M=n_heads
e.g. 16, N=head_dim 128). Applied after Q/K projections in Qwen3.
Tested via `qwen3_kernel_registry_test.py`.

---

# Open infrastructure gaps

Items resolved this session:

| ID | Issue | Resolution |
|---|---|---|
| ~~G1~~ | RMSNorm `herd_x>1` standalone returned all-zero (bare herd needs `launch`/`segment` wrapper) | ✅ `_llm_shared/kernel_builder/rmsnorm_multitile/` wrapper |
| ~~G2~~ | GEMM standalone `rtol=4e-2` spot-check too tight for K=2048 BF16 | ✅ `_llm_shared/kernel_builder/gemm_verify/` cosine wrapper |
| ~~G3~~ | RMSNorm decode 1D path (`_build_rms_1d`) not standalone-tested | ✅ `_llm_shared/kernel_builder/rmsnorm_decode_1d/` wrapper (cosine 0.999991, kernel 0.058 ms) |
| ~~G4~~ | RoPE decode 1D path (`_build_rope_1d`) not standalone-tested | ✅ `_llm_shared/kernel_builder/rope_decode_1d/` wrapper (cosine 0.999995, kernel 0.061 ms) |
| ~~G5~~ | Eltwise Add (production user, 4× per layer) not standalone-tested; previously believed "no production user" | ✅ `_llm_shared/kernel_builder/eltwise_add_multitile/` wrapper (cosine 0.999996, kernel 0.111 ms) |

Remaining open:

| ID | Issue | Effort |
|---|---|---|
| K2 | `ffn_swiglu/run.py` sys.path goes up 2 levels instead of 3, breaks `make run` without external `PYTHONPATH` | ~3 lines |

---

# Glossary

### Llama config terms

- **`seq_len`** — number of tokens in the prefill context.
- **`emb_dim`** — embedding / hidden state width.
- **`hidden_dim`** — FFN intermediate width.
- **`head_dim`** — per-attention-head channel width.
- **`n_heads`** — number of Q (query) attention heads.
- **`n_kv_heads`** — number of K/V heads (grouped-query attention).
- **`kv_dim`** — K/V projection width = `n_kv_heads * head_dim`.
- **`vocab_size`** — output dimension of LM Head.

### NPU2 hardware terms

- **AIE2P** — the compute tile architecture used by NPU2 (Strix). Each
  tile has BF16 vector units (16 lanes), 64 KB local memory (L1).
- **Herd** — a 2D rectangular array of compute tiles inside a segment.
  e.g. "herd 8×4" uses 32 tiles.
- **Segment** — a partition of the chip that hosts one or more herds.
  Segments can be replicated within a launch (`segment sizes=[N, 1]`).
- **MemTile (L2)** — shared memory between cores, 256 KB per tile,
  total budget per segment ~512 KiB.
- **Shim BD** — DMA descriptor for moving data between DDR and L2 / L1.

### BD-friendliness rules (informal)

Hardware constraints on the BD (Buffer Descriptor — DMA controller's
instruction format). Full detail in
[`../aie2p_hardware_limits.md`](../aie2p_hardware_limits.md).

- **Rule A — DMA inner dim ≤ 1024**: row width must fit in one DMA
  "stride 1" burst. `emb_dim` / `hidden_dim` should be a multiple of
  1024 to avoid awkward multi-step DMA.
- **Rule B — GEMV K-DMA repeat ≤ 255 → max practical K ≈ 8160**: above
  this, use `k_split` to pre-split the K dim.
- **Rule C — combined GEMV reads ≤ 255 per channel**: when multiple
  GEMVs share a channel in one multi-launch ELF, reads accumulate.
- **Rule D — L2 buffer ≤ 512 KiB**: for BF16 GEMV,
  `K × herd_m × tile_m ≤ 131072`. Builder asserts.

---

## Versioning

**v1 (kernel_registry/, 2026-04-26)**: split off from the previous
combined doc. This file is now **model-agnostic** — it lists every
kernel and every tested shape across all models. For a model-specific
view ("which shapes does llama3-1B use?"), see
[`llama3.2_1b.md`](llama3.2_1b.md).
