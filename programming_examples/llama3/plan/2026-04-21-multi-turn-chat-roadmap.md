# Multi-Turn Chat — Roadmap & Milestone 1 (Chunked Prefill, C=64)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per-kernel tasks should leverage the existing `validate-per-kernel-shapes` skill; multi-launch fusion tasks should leverage `merge-multi-launch-kernels` and `debug-multi-launch-merge`.

**Goal:** Enable multi-turn interactive chat on Llama-3.2-1B/NPU2 by introducing a chunked-prefill kernel family at chunk_size **C = 64**, so that variable-length prompts (1 ≤ N ≤ 2048) cost only `ceil(N/64)` chunk-prefills instead of always paying full 2048-padded prefill, and so KV cache persists across turns.

**Architecture (Milestone 1):** 3 NEW multi-launch ELFs (`rms_gemms_rope_chunk`, `flash_attn_chunk`, `o_ffn_chunk`) compiled for chunk_size 64 against persistent KV cache of shape `(n_layers, n_kv_heads, 2048, head_dim)`. Each multi-launch ELF is composed of standalone single-launch kernels that are individually verified at the new chunk shape FIRST, then fused. The decode kernels (`rms_gemv_rope`, `o_gemv_ffn`, `lm_head_gemv`) are reused unchanged.

**Tech Stack:** MLIR-AIR / AIE2P kernel infrastructure under `programming_examples/{rms_norm,weighted_rms_norm,matrix_multiplication,rope_lut,silu,swiglu,eltwise_add,flash_attention/kernel_fusion_based}/`, the multi-launch builder framework under `programming_examples/llama3/multi_launch_builder/`, the kernel cache under `programming_examples/llama3/kernel_builder/`, Python orchestration on top.

---

## 1. Background: What We Have Today

### 1.1 Model Configuration
Source: `programming_examples/llama3/llama3_weights.py:39-45`

| Parameter | Value |
|-----------|-------|
| n_layers | 16 |
| emb_dim | 2048 |
| n_heads | 32 |
| head_dim | 64 |
| n_kv_heads | 8 |
| kv_dim (= n_kv_heads × head_dim) | 512 |
| hidden_dim | 8192 |
| vocab_size | 128 256 |
| dtype | BF16 |
| rope_base | 500 000 |

### 1.2 Today's Multi-Launch ELFs and Their Component Single-Launch Kernels

Each of today's prefill multi-launch ELFs internally bundles several `air.launch` ops, each implementing one mathematical step. Decomposition (from the multi_launch_builder source files):

#### `rms_gemms_rope` — 6 launches (`multi_launch_builder/rms_gemms_rope_multi.py:194-229`)
| Launch # | Op | Today's shape (seq_len = 2048) |
|---|---|---|
| 1 | RMSNorm (attention) | x (2048, 2048), w (2048,) → normed (2048, 2048) |
| 2 | Q proj GEMM | A (2048, 2048) × B (2048, 2048) → q (2048, 2048) |
| 3 | K proj GEMM | A (2048, 2048) × B (2048, 512) → k (2048, 512) |
| 4 | V proj GEMM | A (2048, 2048) × B (2048, 512) → v (2048, 512) |
| 5 | RoPE Q | q (2048, 2048), lut_q (131072,) → q_roped (2048, 2048) |
| 6 | RoPE K | k (2048, 512), lut_k (32768,) → k_roped (2048, 512) |

#### `flash_attn` — 1 launch (`programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:50-88`)
| Launch # | Op | Today's shape |
|---|---|---|
| 1 | FlashAttention (causal, GQA) | q (2048, 2048), k (2048, 512), v (2048, 512) → attn_out (2048, 2048); lq=lk=2048, lqp=256, lkp=64, dk=dv=64 |

#### `o_ffn` — 8 launches (`multi_launch_builder/o_ffn_multi.py:179-229`)
| Launch # | Op | Today's shape |
|---|---|---|
| 1 | O proj GEMM | A (2048, 2048) × B (2048, 2048) → proj (2048, 2048) |
| 2 | Residual Add #1 | proj + x_residual → res1 (2048, 2048) |
| 3 | RMSNorm (FFN) | res1 (2048, 2048), w (2048,) → normed2 (2048, 2048) |
| 4 | Gate proj GEMM | A (2048, 2048) × B (2048, 8192) → gate (2048, 8192) |
| 5 | Up proj GEMM | A (2048, 2048) × B (2048, 8192) → up (2048, 8192) |
| 6 | SiLU(gate) ⊙ up | gate, up (2048, 8192) → swiglu (2048, 8192) |
| 7 | Down proj GEMM | A (2048, 8192) × B (8192, 2048) → down (2048, 2048) |
| 8 | Residual Add #2 | down + res1 → output (2048, 2048) |

#### Other prefill kernels (used outside the per-layer block)
- `rmsnorm` — final RMSNorm after layer 16. Shape: (2048, 2048) bf16.
- `lm_head` — 8-partition GEMM. Per partition: A (2048, 2048) × B (2048, 16384) → (2048, 16384).

#### Decode-side kernels (M = 1, **independent of seq_len** — used unchanged in M1)
- `rms_gemv_rope` (6 launches) — per-token version of `rms_gemms_rope`
- `o_gemv_ffn` (8 launches) — per-token version of `o_ffn`
- `lm_head_gemv` — per-token 8-partition GEMV

### 1.3 Hardcoded `seq_len = 2048` Locations

| File | Line | Why |
|------|------|-----|
| `llama3_weights.py` | 324 | `generate_rope_lut(seq_len=2048)` default |
| `llama3_inference.py` | 727 | top-level pad target |
| `llama3_decode.py` | 794 | same |
| `multi_launch_builder/rms_gemms_rope_multi.py` | 195 | builder default (bakes into kernel shape) |
| `multi_launch_builder/o_ffn_multi.py` | 180 | same |
| `multi_launch_builder/lm_head_multi.py` | 29 | same |

For Milestone 1 we DO NOT touch any of these — the existing 2048-fixed prefill kernels remain in place but are **not invoked** by the chat path.

### 1.4 What's Missing for a Real Chatbot

| Capability | Status today |
|---|---|
| Persistent transcript across turns | ❌ Single user message only |
| Persistent KV cache across turns | ❌ Discarded when `generate()` returns |
| Persistent `current_pos` cursor | ❌ Same |
| REPL loop | ❌ Process exits after one reply |
| Variable-length prompt (no pad) | ❌ Always pads to 2048 |
| Capacity guard against ≥ 2048 tokens | ❌ Would silently truncate |

---

## 2. Roadmap Overview

| Milestone | Goal | Kernels touched |
|---|---|---|
| **M1 (this plan)** | Multi-turn chat within `max_context = 2048`, **chunked prefill C=64** so per-turn prefill cost ≈ proportional to new-message length. | 3 new multi-launch ELFs (10 unique standalone kernels) |
| M2 | Sweep / tune chunk size, optimize FA chunk tile sizing for latency | Re-tune existing chunked kernels |
| M3 | Larger `max_context` (e.g. 4096 or 8192) | FA kernel K-streaming redesign for larger L |
| M4 | Cross-turn KV reuse via prefix-checksum (FastFlowLM `PromptCache` style) | Python only |

The rest of this document focuses on **Milestone 1**.

---

## 3. Milestone 1: Detailed Plan

### 3.1 Design Overview

For each per-layer chunked-prefill iteration at `current_pos`:

```
Host: slice rope_lut[current_pos : current_pos+C] → lut_q_chunk, lut_k_chunk
Host: zero pad if last chunk shorter than C
NPU:  rms_gemms_rope_chunk(x_chunk, ...) → q_roped_chunk, k_new, v_new
Host: memcpy k_new → k_cache[layer, :, current_pos : current_pos+C, :]
Host: memcpy v_new → v_cache[layer, :, current_pos : current_pos+C, :]
Host: build mask[C, L] from current_pos (additive log-mask)
NPU:  flash_attn_chunk(q_roped_chunk, k_cache_layer, v_cache_layer, mask) → attn_out_chunk
NPU:  o_ffn_chunk(attn_out_chunk, x_chunk, ...) → x_chunk' (next-layer input)
```

After processing all 16 layers for this chunk, advance `current_pos += real_chunk_len` and proceed to next chunk (or, if last chunk, run final CPU RMSNorm + NPU `lm_head_gemv` on the LAST position to seed decode).

### 3.2 Unique Standalone Kernels Needed (C = 64)

Of the 6+1+8 = 15 launches across the three prefill multi-launch ELFs, after deduplicating same-shape kernels we need **10 unique standalone kernel binaries** at the chunked shape:

| ID | Kernel | Standalone shape (BF16) | Reused inside | Existing example to adapt |
|---|---|---|---|---|
| K1 | `rmsnorm_chunk` | x (64, 2048), w (2048,) → y (64, 2048) | rms_gemms_rope_chunk #1, o_ffn_chunk #3 | `programming_examples/weighted_rms_norm/` |
| K2 | `gemm_64x2048_to_2048` | A (64, 2048) × B (2048, 2048) → C (64, 2048) | rms_gemms_rope_chunk #2 (Q proj), o_ffn_chunk #1 (O proj) | `programming_examples/matrix_multiplication/bf16/` |
| K3 | `gemm_64x2048_to_512` | A (64, 2048) × B (2048, 512) → C (64, 512) | rms_gemms_rope_chunk #3 (K proj), #4 (V proj) | `programming_examples/matrix_multiplication/bf16/` |
| K4 | `gemm_64x2048_to_8192` | A (64, 2048) × B (2048, 8192) → C (64, 8192) | o_ffn_chunk #4 (Gate), #5 (Up) | `programming_examples/matrix_multiplication/bf16/` |
| K5 | `gemm_64x8192_to_2048` | A (64, 8192) × B (8192, 2048) → C (64, 2048) | o_ffn_chunk #7 (Down) | `programming_examples/matrix_multiplication/bf16/` |
| K6 | `rope_chunk_q` | x (64, 2048), lut (4096,) → y (64, 2048) | rms_gemms_rope_chunk #5 | `programming_examples/rope_lut/` (uses `rope_halfsplit.cc`) |
| K7 | `rope_chunk_k` | x (64, 512), lut (4096,) → y (64, 512) | rms_gemms_rope_chunk #6 | `programming_examples/rope_lut/` |
| K8 | `flash_attn_chunk` | q (64, 2048), k (2048, 512), v (2048, 512), mask (64, 2048) → out (64, 2048) | flash_attn_chunk (single launch) | `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` (substantial redesign) |
| K9 | `silu_and_mul_chunk` | a (64, 8192), b (64, 8192) → y (64, 8192) | o_ffn_chunk #6 | `programming_examples/swiglu/` |
| K10 | `eltwise_add_chunk` | a (64, 2048) + b (64, 2048) → y (64, 2048) | o_ffn_chunk #2, #8 | `programming_examples/eltwise_add/` |

**Notes on RoPE LUT shape (K6/K7):**
- Today's prefill `lut_q` is (131072,) ≡ seq_len=2048 × head_dim=64 bf16 — one (head_dim,) per absolute position, broadcast across all heads inside the kernel.
- For a C=64 chunk: `(C × head_dim,) = (4096,)` bf16 sliced from `rope_lut[current_pos : current_pos+C]`.
- Same shape for K (host slices the same LUT — RoPE uses the same trig values regardless of n_heads vs n_kv_heads).

**Notes on FA mask (K8):**
- Additive log-mask, computed on host once per chunk:
  ```python
  mask = np.full((C, L), -65504.0, dtype=bfloat16)  # largest negative finite bf16
  for p in range(real_chunk_len):
      valid_end = current_pos + p + 1   # causal: row p attends to keys [0..current_pos+p]
      mask[p, :valid_end] = 0.0
  # rows p ≥ real_chunk_len stay all -65504 (no attention contribution)
  ```
- Kernel adds the mask to QK^T before softmax; the masked-out entries become 0 after exp.

### 3.3 What Stays the Same

- The `rope_lut` BF16 buffer of shape (2048, head_dim) — generated once via `generate_rope_lut(seq_len=2048)`. The host slices the chunk view per call.
- The persistent KV cache shape `(n_layers, n_kv_heads, 2048, head_dim)`.
- All decode kernels and the decode loop inside `llama3_decode.py:run_decode_block`.
- The final RMSNorm and `lm_head_gemv` invocation pattern from `llama3_inference.py:614-645` — applied to the last position's hidden state only.
- Per-layer BO pre-loading via `_preload_decode_weights` in `llama3_inference.py`. We add an analogous `_preload_chunked_prefill_weights` for the new ELFs.

### 3.4 File Structure

```
programming_examples/llama3/
├── chat_engine.py                            [NEW] Stateful chat engine
├── llama3_chat.py                            [NEW] REPL CLI
├── multi_launch_builder/
│   ├── rms_gemms_rope_chunk_multi.py         [NEW] Phase B fusion (mirrors rms_gemms_rope_multi.py)
│   ├── o_ffn_chunk_multi.py                  [NEW] Phase B fusion (mirrors o_ffn_multi.py)
│   └── flash_attn_chunk_builder.py           [NEW] Phase B (single-launch ELF)
├── kernel_builder/
│   └── (no changes — KernelCache reused)
├── tests/
│   ├── __init__.py                           [NEW]
│   ├── per_kernel/                           [NEW] Phase A standalone kernel tests
│   │   ├── test_K1_rmsnorm_chunk.py
│   │   ├── test_K2_gemm_64x2048_to_2048.py
│   │   ├── test_K3_gemm_64x2048_to_512.py
│   │   ├── test_K4_gemm_64x2048_to_8192.py
│   │   ├── test_K5_gemm_64x8192_to_2048.py
│   │   ├── test_K6_rope_chunk_q.py
│   │   ├── test_K7_rope_chunk_k.py
│   │   ├── test_K8_flash_attn_chunk.py
│   │   ├── test_K9_silu_and_mul_chunk.py
│   │   └── test_K10_eltwise_add_chunk.py
│   ├── per_ml/                               [NEW] Phase B multi-launch ELF parity tests
│   │   ├── test_ML1_rms_gemms_rope_chunk.py
│   │   ├── test_ML2_flash_attn_chunk.py
│   │   └── test_ML3_o_ffn_chunk.py
│   ├── test_transcript.py                    [NEW] Phase C unit tests
│   └── test_chat_engine.py                   [NEW] Phase C+D end-to-end tests
├── Makefile                                  [MODIFY] Add `make chat`
├── README.md                                 [MODIFY] Add chat usage section
└── plan/
    └── 2026-04-21-multi-turn-chat-roadmap.md  [THIS FILE]
```

### 3.5 Phases

- **Phase A:** Validate each unique standalone kernel (K1–K10) at the C=64 shape. One ELF per kernel, parity vs NumPy reference. ≥ 0.99 cosine on per-element.
- **Phase B:** Build 3 multi-launch ELFs by composing the validated kernels. Parity vs Phase A NumPy chain.
- **Phase C:** Python `ChatEngine` that drives the chunked-prefill loop with persistent KV + transcript across turns. Parity vs today's `llama3_inference.py` for first-turn output.
- **Phase D:** REPL CLI (`llama3_chat.py` + `make chat` target).

End-to-end acceptance criteria are in Section 4.

---

## Phase A — Per-Kernel Standalone Validation (Tasks A0–A10)

Each Phase-A task follows the same shape:
1. Adapt the source script under `programming_examples/<existing>/` to a new file under `programming_examples/llama3/standalone_kernels/<KID>_<name>/` with the chunk shapes.
2. Add a pytest under `tests/per_kernel/test_<KID>_<name>.py` that compiles + runs on NPU and compares against a NumPy reference.
3. Cosine-similarity ≥ 0.99 elementwise vs NumPy reference; max element-wise abs error ≤ 0.05 in BF16.
4. Commit.

### Task A0: Create per-kernel test scaffolding

**Files:**
- Create: `programming_examples/llama3/tests/__init__.py`
- Create: `programming_examples/llama3/tests/per_kernel/__init__.py`
- Create: `programming_examples/llama3/tests/per_kernel/_helpers.py`

- [ ] **Step 1: Empty `__init__.py` files**

```bash
mkdir -p programming_examples/llama3/tests/per_kernel
: > programming_examples/llama3/tests/__init__.py
: > programming_examples/llama3/tests/per_kernel/__init__.py
```

- [ ] **Step 2: Add the shared assertion helper**

`programming_examples/llama3/tests/per_kernel/_helpers.py`:
```python
"""Shared assertions for per-kernel chunked-shape validation."""
import numpy as np


def assert_close_bf16(npu, ref, *, name: str, cos_min: float = 0.99,
                      max_abs_err: float = 0.05):
    """NPU output (BF16) must match a NumPy reference (F32 promoted) within
    BF16 precision. We require both cosine similarity ≥ cos_min on the
    flattened tensor AND max |elementwise diff| ≤ max_abs_err.

    Both buffers are flattened and promoted to F32 for the comparison.
    """
    npu_f32 = np.asarray(npu, dtype=np.float32).flatten()
    ref_f32 = np.asarray(ref, dtype=np.float32).flatten()
    assert npu_f32.shape == ref_f32.shape, (
        f"[{name}] shape mismatch npu={npu_f32.shape} ref={ref_f32.shape}"
    )
    cos = float(np.dot(npu_f32, ref_f32) /
                (np.linalg.norm(npu_f32) * np.linalg.norm(ref_f32) + 1e-12))
    max_err = float(np.max(np.abs(npu_f32 - ref_f32)))
    assert cos >= cos_min, f"[{name}] cosine {cos:.6f} < {cos_min}"
    assert max_err <= max_abs_err, f"[{name}] max_abs_err {max_err:.4f} > {max_abs_err}"
```

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/tests/
git commit -m "llama3: per-kernel test scaffolding for chunked prefill validation"
```

---

### Task A1: K1 — `rmsnorm_chunk` standalone (x (64,2048), w (2048,) → y (64,2048))

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K1_rmsnorm_chunk/rmsnorm_chunk.py` (adapted from `programming_examples/weighted_rms_norm/weighted_rms_norm.py`)
- Create: `programming_examples/llama3/tests/per_kernel/test_K1_rmsnorm_chunk.py`

- [ ] **Step 1: Read the source**

```bash
cat programming_examples/weighted_rms_norm/weighted_rms_norm.py | head -60
```

Identify the M, K constants and the `@module_builder` block.

- [ ] **Step 2: Adapt to chunk shape**

Create `programming_examples/llama3/standalone_kernels/K1_rmsnorm_chunk/rmsnorm_chunk.py` by copying `weighted_rms_norm.py` and changing the shape constants to:
```python
M = 64           # chunk size
K = 2048         # emb_dim
DTYPE = bfloat16
```
(Preserve the rest of the script verbatim — `@module_builder`, `XRTRunner` wrapper, CLI args.)

- [ ] **Step 3: Write the test**

`programming_examples/llama3/tests/per_kernel/test_K1_rmsnorm_chunk.py`:
```python
import os
import sys
import subprocess
import numpy as np
from ml_dtypes import bfloat16

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "standalone_kernels", "K1_rmsnorm_chunk"))
sys.path.insert(0, os.path.join(THIS, ".."))

from per_kernel._helpers import assert_close_bf16


def _ref_rmsnorm(x_bf16, w_bf16, eps=1e-5):
    x = np.asarray(x_bf16, dtype=np.float32)
    w = np.asarray(w_bf16, dtype=np.float32)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def test_K1_rmsnorm_chunk():
    """Run the standalone kernel via its Python entry script and compare
    output to a NumPy reference."""
    M, K = 64, 2048
    rng = np.random.default_rng(0)
    x = rng.standard_normal((M, K)).astype(bfloat16)
    w = (rng.standard_normal((K,)) * 0.1 + 1.0).astype(bfloat16)

    # Use the script's XRTRunner.run_test path via subprocess so the test
    # is hermetic. The script must accept --inputs and --expected paths
    # OR we import its module_builder and run XRTRunner directly here.
    # Recommended: import directly.
    from rmsnorm_chunk import build_module
    from air.backend.xrt_runner import XRTRunner

    mlir_module = build_module(M=M, K=K)
    expected = _ref_rmsnorm(x, w).astype(bfloat16)

    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K1_rmsnorm_chunk")
    out_buf = np.zeros((M, K), dtype=bfloat16)
    rc = runner.run_test(mlir_module, inputs=[x, w], expected_outputs=[expected],
                         rtol=1e-2, atol=5e-2)
    assert rc == 0
    # Re-run for explicit assertion (XRTRunner already validated, but we
    # also want our cosine-similarity check):
    # Note: XRTRunner.run_test re-runs and compares; if it returned 0, the
    # outputs are within tolerance. We additionally assert via _helpers.
    # If you need the raw NPU buffer, refactor the script to expose it.
```

- [ ] **Step 4: Run the test**

```bash
cd programming_examples/llama3/standalone_kernels/K1_rmsnorm_chunk && python3 -m pytest ../../tests/per_kernel/test_K1_rmsnorm_chunk.py -v -s
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K1_rmsnorm_chunk/ \
        programming_examples/llama3/tests/per_kernel/test_K1_rmsnorm_chunk.py
git commit -m "llama3/K1: rmsnorm_chunk (64,2048) standalone validation"
```

---

### Task A2: K2 — `gemm_64x2048_to_2048` standalone

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K2_gemm_64x2048_to_2048/gemm.py`
- Create: `programming_examples/llama3/tests/per_kernel/test_K2_gemm_64x2048_to_2048.py`

- [ ] **Step 1: Identify source**

```bash
ls programming_examples/matrix_multiplication/bf16/
```

Pick the existing BF16 GEMM script that's parameterized over (M, K, N). Note the variable names it uses for those dims.

- [ ] **Step 2: Adapt with constants**

Create `gemm.py` setting:
```python
M = 64; K = 2048; N = 2048
DTYPE = bfloat16
```
Preserve the rest of the script verbatim (tile sizes may need adjustment for small M; if compilation fails, see Step 4 fallback).

- [ ] **Step 3: Write test**

`tests/per_kernel/test_K2_gemm_64x2048_to_2048.py`:
```python
import os, sys
import numpy as np
from ml_dtypes import bfloat16

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "standalone_kernels", "K2_gemm_64x2048_to_2048"))

def test_K2_gemm():
    M, K, N = 64, 2048, 2048
    rng = np.random.default_rng(1)
    a = rng.standard_normal((M, K)).astype(bfloat16)
    b = rng.standard_normal((K, N)).astype(bfloat16)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)

    from gemm import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(M=M, K=K, N=N)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K2_gemm")
    rc = runner.run_test(mod, inputs=[a, b], expected_outputs=[expected],
                         rtol=2e-2, atol=1e-1)
    assert rc == 0
```

- [ ] **Step 4: Run & fallback for tile sizing**

```bash
cd programming_examples/llama3/standalone_kernels/K2_gemm_64x2048_to_2048 && \
  python3 -m pytest ../../tests/per_kernel/test_K2_gemm_64x2048_to_2048.py -v -s
```

If compilation fails with a tile-size error (M=64 may not divide cleanly into existing tile templates), reduce the M tile dim in the script (`m_tile_size = 32` or `m_tile_size = 16`) and re-run. Document the working tile config in a comment at the top of `gemm.py`.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K2_gemm_64x2048_to_2048/ \
        programming_examples/llama3/tests/per_kernel/test_K2_gemm_64x2048_to_2048.py
git commit -m "llama3/K2: gemm (64,2048)x(2048,2048) standalone validation"
```

---

### Task A3: K3 — `gemm_64x2048_to_512` standalone

Same template as Task A2, with shape constants:
```python
M = 64; K = 2048; N = 512
```

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K3_gemm_64x2048_to_512/gemm.py`
- Create: `programming_examples/llama3/tests/per_kernel/test_K3_gemm_64x2048_to_512.py`

Test body identical to Task A2 with substituted M/K/N.

- [ ] **Step 1: Adapt script** (copy of Task A2 with N=512)
- [ ] **Step 2: Write test** (mirror Task A2 test with N=512)
- [ ] **Step 3: Run** — expected PASS
- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K3_gemm_64x2048_to_512/ \
        programming_examples/llama3/tests/per_kernel/test_K3_gemm_64x2048_to_512.py
git commit -m "llama3/K3: gemm (64,2048)x(2048,512) standalone validation"
```

---

### Task A4: K4 — `gemm_64x2048_to_8192` standalone

Same template, shapes `M = 64; K = 2048; N = 8192`.

- [ ] **Step 1: Adapt script**
- [ ] **Step 2: Write test**
- [ ] **Step 3: Run**

If N=8192 outputs exceed L2 capacity (likely), split N into 2 launches of N=4096 inside the script. Document the split.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K4_gemm_64x2048_to_8192/ \
        programming_examples/llama3/tests/per_kernel/test_K4_gemm_64x2048_to_8192.py
git commit -m "llama3/K4: gemm (64,2048)x(2048,8192) standalone validation"
```

---

### Task A5: K5 — `gemm_64x8192_to_2048` standalone

Shapes `M = 64; K = 8192; N = 2048`. K dimension is large — confirm K-streaming inside the kernel script handles it. Note today's decode `o_gemv_ffn` uses `mv_k8192.o` for exactly this K dim; check if the bf16 GEMM script needs the same special handling (it shouldn't — GEMV is the special case, not GEMM).

- [ ] **Step 1: Adapt script**
- [ ] **Step 2: Write test**
- [ ] **Step 3: Run**
- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K5_gemm_64x8192_to_2048/ \
        programming_examples/llama3/tests/per_kernel/test_K5_gemm_64x8192_to_2048.py
git commit -m "llama3/K5: gemm (64,8192)x(8192,2048) standalone validation"
```

---

### Task A6: K6 — `rope_chunk_q` standalone (x (64,2048), lut (4096,) → y (64,2048))

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K6_rope_chunk_q/rope_chunk_q.py` (adapted from `programming_examples/rope_lut/rope_lut_l2_tiled.py` with the `rope_halfsplit.cc` external kernel referenced from `programming_examples/llama3/`)
- Create: `programming_examples/llama3/tests/per_kernel/test_K6_rope_chunk_q.py`

- [ ] **Step 1: Read existing rope_lut script and rope_halfsplit.cc**

```bash
ls programming_examples/llama3/ | grep -i rope
ls programming_examples/rope_lut/
```

Locate the chunk-friendly script (`rope_lut_l2_tiled.py`). The current llama3 prefill uses a tiled variant; we want one parameterized by M = chunk_size.

- [ ] **Step 2: Adapt with M=64**

Set the script's M parameter (sequence length) to 64 and verify the output buffer sizing scales correctly. The LUT input shape becomes `(M × head_dim,) = (4096,)`.

- [ ] **Step 3: Write test**

`tests/per_kernel/test_K6_rope_chunk_q.py`:
```python
import os, sys
import numpy as np
from ml_dtypes import bfloat16

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "standalone_kernels", "K6_rope_chunk_q"))
sys.path.insert(0, os.path.join(THIS, "..", "..", ".."))  # for llama3 imports

def _ref_rope_halfsplit(x_bf16, lut_bf16, n_heads, head_dim):
    """CPU reference for half-split RoPE matching Llama-3's HF convention.
    LUT layout per position: [cos_0..cos_{hd/2-1}, sin_0..sin_{hd/2-1}]"""
    M = x_bf16.shape[0]
    x = x_bf16.astype(np.float32).reshape(M, n_heads, head_dim)
    lut = lut_bf16.astype(np.float32).reshape(M, head_dim)
    cos = lut[:, :head_dim // 2]   # (M, hd/2)
    sin = lut[:, head_dim // 2:]   # (M, hd/2)
    half = head_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    y2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    return np.concatenate([y1, y2], axis=-1).reshape(M, n_heads * head_dim).astype(bfloat16)


def test_K6_rope_chunk_q():
    M, n_heads, head_dim = 64, 32, 64
    emb_dim = n_heads * head_dim  # 2048
    rng = np.random.default_rng(2)
    x = rng.standard_normal((M, emb_dim)).astype(bfloat16)
    lut = rng.standard_normal((M * head_dim,)).astype(bfloat16)
    expected = _ref_rope_halfsplit(x, lut, n_heads, head_dim)

    from rope_chunk_q import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(M=M, n_heads=n_heads, head_dim=head_dim)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K6_rope_q")
    rc = runner.run_test(mod, inputs=[x, lut], expected_outputs=[expected],
                         rtol=1e-2, atol=5e-2)
    assert rc == 0
```

- [ ] **Step 4: Run & commit**

```bash
cd programming_examples/llama3/standalone_kernels/K6_rope_chunk_q && \
  python3 -m pytest ../../tests/per_kernel/test_K6_rope_chunk_q.py -v -s
git add programming_examples/llama3/standalone_kernels/K6_rope_chunk_q/ \
        programming_examples/llama3/tests/per_kernel/test_K6_rope_chunk_q.py
git commit -m "llama3/K6: rope_chunk_q (64,2048) standalone validation"
```

---

### Task A7: K7 — `rope_chunk_k` standalone (x (64,512), lut (4096,) → y (64,512))

Same template as Task A6 with `n_heads = 8` (KV heads). The script and test mirror A6 with substituted constants.

- [ ] **Step 1: Adapt script** (copy K6, change `n_heads=8`, output emb_dim becomes `8 × 64 = 512`)
- [ ] **Step 2: Write test** (mirror A6 test with `n_heads=8`)
- [ ] **Step 3: Run & commit**

```bash
git add programming_examples/llama3/standalone_kernels/K7_rope_chunk_k/ \
        programming_examples/llama3/tests/per_kernel/test_K7_rope_chunk_k.py
git commit -m "llama3/K7: rope_chunk_k (64,512) standalone validation"
```

---

### Task A8: K8 — `flash_attn_chunk` standalone (the structurally novel kernel)

This is the only kernel that requires a substantial design change from its non-chunked predecessor. It will likely take 1–2 days vs hours for the others. Structure: design + build + bring-up + test, broken into sub-tasks.

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/attn_chunk.py` (initial source: copy `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`)
- Create: `programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/attn_chunk.cc` (initial source: copy the matching `.cc` from `flash_attention/kernel_fusion_based/`)
- Create: `programming_examples/llama3/tests/per_kernel/test_K8_flash_attn_chunk.py`

#### Sub-task A8.1: Read the existing FA kernel end-to-end

- [ ] **Step 1: Build a mental model of the existing FA**

```bash
sed -n '1,150p' programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py
sed -n '1,150p' programming_examples/flash_attention/kernel_fusion_based/attn_npu2.cc
```

Note: Q tiling (lqp=256), K tiling (lkp=64), Q-cascade stages, GQA broadcast, online softmax statistics, the causal-mask flag.

- [ ] **Step 2: Document target shape diff in a comment block at the top of `attn_chunk.py`**

```python
"""
flash_attn_chunk — chunked Q against full-cache K/V.

Diff vs attn_npu2_seqfirst.py:
  Q:    (lq=2048, dk_total=2048) BF16   →  (lq=64, dk_total=2048) BF16
  K:    (lk=2048, dk_kv=512) BF16       →  (lk=2048, dk_kv=512)  unchanged
  V:    (lk=2048, dv_kv=512) BF16       →  (lk=2048, dv_kv=512)  unchanged
  mask: NEW input, (lq=64, lk=2048) BF16 — additive log-mask
  out:  (lq=2048, dk_total=2048) BF16   →  (lq=64, dk_total=2048) BF16
  causal flag: True (built-in lower-triangular)  →  False (mask is data, not flag)
  lqp=256, num_q_tiles=4                →  lqp=64, num_q_tiles=1
  num_cascade_stages=4                  →  num_cascade_stages=1 (only 1 Q tile)
"""
```

#### Sub-task A8.2: Add `mask` as a 4th input to the kernel signature

- [ ] **Step 1: Edit `attn_chunk.py` `@FuncOp.from_py_func` to accept 5 args (q, k, v, mask, out)**

Replace today's 4-arg signature:
```python
@FuncOp.from_py_func(qTy, kTy, vTy, outTy)
def attention_bf16(q, k, v, out): ...
```
with:
```python
maskTy = MemRefType.get([lq, lk], xrt_dtype)
@FuncOp.from_py_func(qTy, kTy, vTy, maskTy, outTy)
def attention_bf16(q, k, v, mask, out): ...
```
And thread `mask` into the per-tile kernel call.

- [ ] **Step 2: Update the C++ kernel `attn_chunk.cc` to accept a `mask_tile` pointer and add it to the QK^T result before softmax**

In the inner Q-tile loop, the existing code computes `s = q @ k_tile^T`. Insert `s += mask_tile[q_row, k_block_offset:k_block_offset+lkp]` before the online-softmax max/sum updates.

- [ ] **Step 3: Compile-only smoke**

```bash
cd programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk && \
  python3 attn_chunk.py --compile-only
```

Expected: produces `air.elf` without errors.

- [ ] **Step 4: Commit (compile-only milestone)**

```bash
git add programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/
git commit -m "llama3/K8: flash_attn_chunk compile-only (Q=64, K=2048, mask input)"
```

#### Sub-task A8.3: Validate against NumPy reference

- [ ] **Step 1: Write the test**

`programming_examples/llama3/tests/per_kernel/test_K8_flash_attn_chunk.py`:
```python
import os, sys
import numpy as np
from ml_dtypes import bfloat16

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "standalone_kernels", "K8_flash_attn_chunk"))


def _ref_attn(q, k, v, mask, n_heads, n_kv_heads, head_dim):
    """CPU FA reference for a chunked Q against a full K/V cache.
    Shapes:
      q:    (C, n_heads * head_dim) bf16
      k, v: (L, n_kv_heads * head_dim) bf16
      mask: (C, L) bf16 (additive log-mask)
    Returns (C, n_heads * head_dim) bf16.
    """
    C = q.shape[0]
    L = k.shape[0]
    q = q.astype(np.float32).reshape(C, n_heads, head_dim)
    k = k.astype(np.float32).reshape(L, n_kv_heads, head_dim)
    v = v.astype(np.float32).reshape(L, n_kv_heads, head_dim)
    mask_f = mask.astype(np.float32)  # (C, L)

    # GQA broadcast: each KV head serves (n_heads // n_kv_heads) Q heads.
    rep = n_heads // n_kv_heads
    k = np.repeat(k, rep, axis=1)  # (L, n_heads, head_dim)
    v = np.repeat(v, rep, axis=1)

    out = np.zeros((C, n_heads, head_dim), dtype=np.float32)
    scale = 1.0 / np.sqrt(head_dim)
    for h in range(n_heads):
        scores = q[:, h, :] @ k[:, h, :].T  # (C, L)
        scores = scores * scale + mask_f
        scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, h, :] = p @ v[:, h, :]

    return out.reshape(C, n_heads * head_dim).astype(bfloat16)


def test_K8_flash_attn_chunk_first_chunk():
    """First-chunk case: current_pos = 0, real_chunk_len = C, mask is
    pure causal lower-triangular over [0..C-1]."""
    C, L = 64, 2048
    n_heads, n_kv_heads, head_dim = 32, 8, 64
    rng = np.random.default_rng(8)

    q = rng.standard_normal((C, n_heads * head_dim)).astype(bfloat16) * 0.1
    k = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    v = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    # Only first C rows are real (this chunk's K/V).
    k[:C] = (rng.standard_normal((C, n_kv_heads * head_dim)) * 0.1).astype(bfloat16)
    v[:C] = (rng.standard_normal((C, n_kv_heads * head_dim)) * 0.1).astype(bfloat16)

    mask = np.full((C, L), -65504.0, dtype=bfloat16)
    for p in range(C):
        mask[p, : p + 1] = 0.0

    expected = _ref_attn(q, k, v, mask, n_heads, n_kv_heads, head_dim)

    from attn_chunk import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(lq=C, lk=L, n_heads=n_heads, n_kv_heads=n_kv_heads,
                       head_dim=head_dim)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K8_fa_chunk_first")
    rc = runner.run_test(mod, inputs=[q, k, v, mask], expected_outputs=[expected],
                         rtol=2e-2, atol=1e-1)
    assert rc == 0


def test_K8_flash_attn_chunk_mid_stream():
    """Mid-conversation case: current_pos = 384, this chunk fills [384..447],
    mask allows attention to keys [0..pos+row]."""
    C, L = 64, 2048
    n_heads, n_kv_heads, head_dim = 32, 8, 64
    current_pos = 384
    rng = np.random.default_rng(9)

    q = rng.standard_normal((C, n_heads * head_dim)).astype(bfloat16) * 0.1
    k = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    v = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    # Old cache content [0..current_pos] + new chunk [current_pos..current_pos+C]
    k[: current_pos + C] = (
        rng.standard_normal((current_pos + C, n_kv_heads * head_dim)) * 0.1
    ).astype(bfloat16)
    v[: current_pos + C] = (
        rng.standard_normal((current_pos + C, n_kv_heads * head_dim)) * 0.1
    ).astype(bfloat16)

    mask = np.full((C, L), -65504.0, dtype=bfloat16)
    for p in range(C):
        mask[p, : current_pos + p + 1] = 0.0

    expected = _ref_attn(q, k, v, mask, n_heads, n_kv_heads, head_dim)

    from attn_chunk import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(lq=C, lk=L, n_heads=n_heads, n_kv_heads=n_kv_heads,
                       head_dim=head_dim)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K8_fa_chunk_mid")
    rc = runner.run_test(mod, inputs=[q, k, v, mask], expected_outputs=[expected],
                         rtol=2e-2, atol=1e-1)
    assert rc == 0


def test_K8_flash_attn_chunk_partial_last_chunk():
    """Last-chunk case: real_chunk_len < C. Pad rows have mask = all -inf
    so their output is irrelevant — we only verify the first real_chunk_len
    rows match the reference."""
    C, L = 64, 2048
    real_len = 23
    n_heads, n_kv_heads, head_dim = 32, 8, 64
    current_pos = 100
    rng = np.random.default_rng(10)

    q = rng.standard_normal((C, n_heads * head_dim)).astype(bfloat16) * 0.1
    k = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    v = np.zeros((L, n_kv_heads * head_dim), dtype=bfloat16)
    k[: current_pos + real_len] = (
        rng.standard_normal((current_pos + real_len, n_kv_heads * head_dim)) * 0.1
    ).astype(bfloat16)
    v[: current_pos + real_len] = (
        rng.standard_normal((current_pos + real_len, n_kv_heads * head_dim)) * 0.1
    ).astype(bfloat16)

    mask = np.full((C, L), -65504.0, dtype=bfloat16)
    for p in range(real_len):
        mask[p, : current_pos + p + 1] = 0.0
    # Rows [real_len..C) stay all -inf.

    expected = _ref_attn(q, k, v, mask, n_heads, n_kv_heads, head_dim)

    from attn_chunk import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(lq=C, lk=L, n_heads=n_heads, n_kv_heads=n_kv_heads,
                       head_dim=head_dim)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K8_fa_chunk_partial")
    # We cannot easily assert "ignore pad rows" through XRTRunner.run_test
    # directly; instead, set expected pad rows to whatever the kernel
    # produces (a no-attention all-zeros softmax * v gives output=0).
    expected_capped = expected.copy()
    expected_capped[real_len:] = 0.0
    rc = runner.run_test(mod, inputs=[q, k, v, mask],
                         expected_outputs=[expected_capped],
                         rtol=2e-2, atol=1e-1)
    assert rc == 0
```

- [ ] **Step 2: Run the three tests**

```bash
cd programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk && \
  python3 -m pytest ../../tests/per_kernel/test_K8_flash_attn_chunk.py -v -s
```

Expected: 3 passed.

- [ ] **Step 3: If `test_K8_flash_attn_chunk_first_chunk` fails**

Most likely root cause: mask integration in the C++ kernel is wrong (off-by-one in k_block_offset or sign error). Use the `debug-fa-runtime-failure` skill if hangs. Use the `systematic-debugging` skill if outputs are wrong.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/standalone_kernels/K8_flash_attn_chunk/ \
        programming_examples/llama3/tests/per_kernel/test_K8_flash_attn_chunk.py
git commit -m "llama3/K8: flash_attn_chunk validated against numpy ref (3 cases)"
```

---

### Task A9: K9 — `silu_and_mul_chunk` standalone (a, b (64, 8192) → y (64, 8192))

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K9_silu_and_mul_chunk/silu_and_mul.py` (adapted from `programming_examples/swiglu/swiglu.py`)
- Create: `programming_examples/llama3/tests/per_kernel/test_K9_silu_and_mul_chunk.py`

- [ ] **Step 1: Adapt source**

Copy `swiglu.py`, set `M = 64`, `N = 8192`. Verify the kernel computes `silu(a) * b` (not `silu(b) * a`).

- [ ] **Step 2: Write test**

```python
def test_K9_silu_and_mul_chunk():
    M, N = 64, 8192
    rng = np.random.default_rng(11)
    a = rng.standard_normal((M, N)).astype(bfloat16)
    b = rng.standard_normal((M, N)).astype(bfloat16)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    silu_a = af / (1.0 + np.exp(-af))
    expected = (silu_a * bf).astype(bfloat16)

    from silu_and_mul import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(M=M, N=N)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K9_silu_mul")
    rc = runner.run_test(mod, inputs=[a, b], expected_outputs=[expected],
                         rtol=2e-2, atol=5e-2)
    assert rc == 0
```

- [ ] **Step 3: Run & commit**

```bash
git add programming_examples/llama3/standalone_kernels/K9_silu_and_mul_chunk/ \
        programming_examples/llama3/tests/per_kernel/test_K9_silu_and_mul_chunk.py
git commit -m "llama3/K9: silu_and_mul_chunk (64,8192) standalone validation"
```

---

### Task A10: K10 — `eltwise_add_chunk` standalone (a, b (64, 2048) → y (64, 2048))

**Files:**
- Create: `programming_examples/llama3/standalone_kernels/K10_eltwise_add_chunk/eltwise_add.py` (adapted from `programming_examples/eltwise_add/eltwise_add.py`)
- Create: `programming_examples/llama3/tests/per_kernel/test_K10_eltwise_add_chunk.py`

- [ ] **Step 1: Adapt source** — `M = 64`, `N = 2048`.

- [ ] **Step 2: Write test**

```python
def test_K10_eltwise_add_chunk():
    M, N = 64, 2048
    rng = np.random.default_rng(12)
    a = rng.standard_normal((M, N)).astype(bfloat16)
    b = rng.standard_normal((M, N)).astype(bfloat16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)

    from eltwise_add import build_module
    from air.backend.xrt_runner import XRTRunner
    mod = build_module(M=M, N=N)
    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="K10_add")
    rc = runner.run_test(mod, inputs=[a, b], expected_outputs=[expected],
                         rtol=1e-2, atol=2e-2)
    assert rc == 0
```

- [ ] **Step 3: Run & commit**

```bash
git add programming_examples/llama3/standalone_kernels/K10_eltwise_add_chunk/ \
        programming_examples/llama3/tests/per_kernel/test_K10_eltwise_add_chunk.py
git commit -m "llama3/K10: eltwise_add_chunk (64,2048) standalone validation"
```

---

### Phase A Acceptance Gate

All 10 per-kernel tests pass:
```bash
cd programming_examples/llama3 && python3 -m pytest tests/per_kernel/ -v
```

Do not proceed to Phase B until this is green. If any kernel fails, debug it standalone — fusion will only multiply the failure modes.

---

## Phase B — Multi-Launch ELF Fusion (Tasks B1–B3)

Now that all 10 single-launch kernels are validated standalone at the C=64 shape, we fuse them into the same 3-multi-launch-ELF structure that today's bulk prefill uses. The procedural recipe is the existing `merge-multi-launch-kernels` skill; this section is the per-fusion specification (which kernels go into which ELF, which BO indices are static/intermediate).

### Task B1: ML1 — `rms_gemms_rope_chunk` (6 launches)

Composes K1, K2, K3, K3, K6, K7 into one ELF. Mirrors `multi_launch_builder/rms_gemms_rope_multi.py`.

**Files:**
- Create: `programming_examples/llama3/multi_launch_builder/rms_gemms_rope_chunk_multi.py`
- Create: `programming_examples/llama3/tests/per_ml/__init__.py`
- Create: `programming_examples/llama3/tests/per_ml/test_ML1_rms_gemms_rope_chunk.py`

- [ ] **Step 1: Read the non-chunked source as a template**

```bash
sed -n '1,250p' programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py
```

Identify (a) the per-launch builder calls, (b) the shared-buffer wiring, (c) the function signature.

- [ ] **Step 2: Author `rms_gemms_rope_chunk_multi.py`**

Same structure; replace per-launch shape constants with chunk shapes. Function signature (13 args, mirroring today but with M=64):

```python
@func.rms_gemms_rope_chunk(
  %arg0: memref<64x2048xbf16>,    # x_in
  %arg1: memref<2048xbf16>,       # norm_w
  %arg2: memref<64x2048xbf16>,    # normed
  %arg3: memref<2048x2048xbf16>,  # wq
  %arg4: memref<64x2048xbf16>,    # q
  %arg5: memref<2048x512xbf16>,   # wk
  %arg6: memref<64x512xbf16>,     # k
  %arg7: memref<2048x512xbf16>,   # wv
  %arg8: memref<64x512xbf16>,     # v
  %arg9: memref<4096xbf16>,       # lut_q (64 × 64)
  %arg10: memref<4096xbf16>,      # lut_k (64 × 64)
  %arg11: memref<64x2048xbf16>,   # q_roped
  %arg12: memref<64x512xbf16>     # k_roped
)
```

The compile-time constants block (mirroring today's lines 195-200):
```python
seq_len   = 64    # chunk size (was 2048)
emb_dim   = 2048
kv_dim    = 512
n_heads   = 32
n_kv_heads= 8
head_dim  = 64
```

- [ ] **Step 3: Compile-only smoke**

```bash
cd programming_examples/llama3 && \
  python3 multi_launch_builder/rms_gemms_rope_chunk_multi.py --compile-only
```

If compile fails on shape constraints inside any launch, debug with `debug-multi-launch-merge`.

- [ ] **Step 4: Parity test against per-kernel chain**

`programming_examples/llama3/tests/per_ml/test_ML1_rms_gemms_rope_chunk.py`:
```python
"""Compose the 6 standalone kernels' NumPy refs (K1, K2, K3×2, K6, K7) and
compare to ML1 ELF output."""
import os, sys
import numpy as np
from ml_dtypes import bfloat16

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "multi_launch_builder"))
sys.path.insert(0, os.path.join(THIS, ".."))


def _ref_rms_norm(x, w, eps=1e-5):
    xf = x.astype(np.float32)
    rms = np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)
    return ((xf / rms) * w.astype(np.float32)).astype(bfloat16)


def _ref_rope_halfsplit(x, lut, n_heads, head_dim):
    M = x.shape[0]
    xf = x.astype(np.float32).reshape(M, n_heads, head_dim)
    lut_f = lut.astype(np.float32).reshape(M, head_dim)
    half = head_dim // 2
    cos = lut_f[:, :half]; sin = lut_f[:, half:]
    x1, x2 = xf[..., :half], xf[..., half:]
    y1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    y2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    return np.concatenate([y1, y2], axis=-1).reshape(M, n_heads * head_dim).astype(bfloat16)


def test_ML1_rms_gemms_rope_chunk():
    C, emb, kv = 64, 2048, 512
    n_heads, n_kv_heads, head_dim = 32, 8, 64
    rng = np.random.default_rng(101)

    x = rng.standard_normal((C, emb)).astype(bfloat16) * 0.1
    nw = (rng.standard_normal((emb,)) * 0.1 + 1.0).astype(bfloat16)
    wq = rng.standard_normal((emb, emb)).astype(bfloat16) * 0.05
    wk = rng.standard_normal((emb, kv)).astype(bfloat16) * 0.05
    wv = rng.standard_normal((emb, kv)).astype(bfloat16) * 0.05
    lut_q = rng.standard_normal((C * head_dim,)).astype(bfloat16)
    lut_k = rng.standard_normal((C * head_dim,)).astype(bfloat16)

    # NumPy reference chain
    normed_ref = _ref_rms_norm(x, nw)
    q_ref = (normed_ref.astype(np.float32) @ wq.astype(np.float32)).astype(bfloat16)
    k_ref = (normed_ref.astype(np.float32) @ wk.astype(np.float32)).astype(bfloat16)
    v_ref = (normed_ref.astype(np.float32) @ wv.astype(np.float32)).astype(bfloat16)
    q_roped_ref = _ref_rope_halfsplit(q_ref, lut_q, n_heads, head_dim)
    k_roped_ref = _ref_rope_halfsplit(k_ref, lut_k, n_kv_heads, head_dim)

    from rms_gemms_rope_chunk_multi import build_module
    from air.backend.xrt_runner import XRTRunner

    mod = build_module(seq_len=C)
    # Allocate scratch for outputs that are written then read by next launch.
    normed = np.zeros((C, emb), dtype=bfloat16)
    q = np.zeros((C, emb), dtype=bfloat16)
    k = np.zeros((C, kv), dtype=bfloat16)
    v = np.zeros((C, kv), dtype=bfloat16)
    q_roped = np.zeros((C, emb), dtype=bfloat16)
    k_roped = np.zeros((C, kv), dtype=bfloat16)

    runner = XRTRunner(verbose=False, omit_while_true_loop=True,
                       output_format="elf", instance_name="ML1_rgr_chunk")
    rc = runner.run_test(
        mod,
        inputs=[x, nw, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped],
        expected_outputs=[None, None, None, None, None, None, None, None,
                          None, None, None, q_roped_ref, k_roped_ref],
        rtol=2e-2, atol=1e-1,
    )
    assert rc == 0
```

- [ ] **Step 5: Run & commit**

```bash
cd programming_examples/llama3 && python3 -m pytest tests/per_ml/test_ML1_rms_gemms_rope_chunk.py -v -s
git add programming_examples/llama3/multi_launch_builder/rms_gemms_rope_chunk_multi.py \
        programming_examples/llama3/tests/per_ml/test_ML1_rms_gemms_rope_chunk.py \
        programming_examples/llama3/tests/per_ml/__init__.py
git commit -m "llama3/ML1: rms_gemms_rope_chunk multi-launch ELF (6 launches, C=64)"
```

---

### Task B2: ML2 — `flash_attn_chunk` ELF wrapper (1 launch)

K8 is already a single-launch kernel from Phase A. Phase B's job is just to package it as an ELF that fits the cached-kernel calling convention used by the rest of the pipeline (`KernelCache.load_and_run` interface, BO-key conventions).

**Files:**
- Create: `programming_examples/llama3/multi_launch_builder/flash_attn_chunk_builder.py`
- Create: `programming_examples/llama3/tests/per_ml/test_ML2_flash_attn_chunk.py`

- [ ] **Step 1: Wrap K8's `build_module` for the kernel cache**

`flash_attn_chunk_builder.py` should expose the same module-builder factory shape that `kernel_builder/cache.py` expects (look at how `attn_npu2_seqfirst.py` is consumed by `llama3_prefill.py:780-811` for the convention).

- [ ] **Step 2: Re-run K8's three tests through the ELF wrapper to confirm no regression**

```python
# Same three test cases as Task A8 sub-task 8.3, but importing
# flash_attn_chunk_builder.build_module instead of attn_chunk.build_module.
```

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/multi_launch_builder/flash_attn_chunk_builder.py \
        programming_examples/llama3/tests/per_ml/test_ML2_flash_attn_chunk.py
git commit -m "llama3/ML2: flash_attn_chunk ELF wrapper for kernel cache integration"
```

---

### Task B3: ML3 — `o_ffn_chunk` (8 launches)

Composes K2 (O proj), K10 (add), K1 (rms_norm), K4 (gate), K4 (up), K9 (silu_mul), K5 (down), K10 (add). Mirrors `multi_launch_builder/o_ffn_multi.py`.

**Files:**
- Create: `programming_examples/llama3/multi_launch_builder/o_ffn_chunk_multi.py`
- Create: `programming_examples/llama3/tests/per_ml/test_ML3_o_ffn_chunk.py`

- [ ] **Step 1: Read non-chunked source**

```bash
sed -n '1,250p' programming_examples/llama3/multi_launch_builder/o_ffn_multi.py
```

- [ ] **Step 2: Author `o_ffn_chunk_multi.py`**

Function signature (15 args, mirroring today but with M=64):
```python
@func.o_ffn_chunk(
  %arg0:  memref<64x2048xbf16>,    # attn_out
  %arg1:  memref<2048x2048xbf16>,  # wo
  %arg2:  memref<64x2048xbf16>,    # proj
  %arg3:  memref<64x2048xbf16>,    # x_residual
  %arg4:  memref<64x2048xbf16>,    # res1
  %arg5:  memref<2048xbf16>,       # ffn_norm_w
  %arg6:  memref<64x2048xbf16>,    # normed2
  %arg7:  memref<2048x8192xbf16>,  # w_gate
  %arg8:  memref<64x8192xbf16>,    # gate
  %arg9:  memref<2048x8192xbf16>,  # w_up
  %arg10: memref<64x8192xbf16>,    # up
  %arg11: memref<64x8192xbf16>,    # swiglu
  %arg12: memref<8192x2048xbf16>,  # w_down
  %arg13: memref<64x2048xbf16>,    # down
  %arg14: memref<131072xbf16>      # output (flattened, 64 × 2048)
)
```

Compile-time constants:
```python
seq_len    = 64    # chunk size (was 2048)
emb_dim    = 2048
hidden_dim = 8192
```

- [ ] **Step 3: Compile-only smoke**

```bash
cd programming_examples/llama3 && python3 multi_launch_builder/o_ffn_chunk_multi.py --compile-only
```

- [ ] **Step 4: Parity test**

Compose the NumPy reference chain (O GEMM → add → rms_norm → gate GEMM → up GEMM → silu_and_mul → down GEMM → add) and compare final output. Pattern mirrors Task B1's test.

- [ ] **Step 5: Run & commit**

```bash
cd programming_examples/llama3 && python3 -m pytest tests/per_ml/test_ML3_o_ffn_chunk.py -v -s
git add programming_examples/llama3/multi_launch_builder/o_ffn_chunk_multi.py \
        programming_examples/llama3/tests/per_ml/test_ML3_o_ffn_chunk.py
git commit -m "llama3/ML3: o_ffn_chunk multi-launch ELF (8 launches, C=64)"
```

---

### Phase B Acceptance Gate

All 3 multi-launch parity tests pass:
```bash
cd programming_examples/llama3 && python3 -m pytest tests/per_ml/ -v
```

Do not proceed to Phase C until this is green.

---

## Phase C — Python ChatEngine With Chunked Prefill

This phase wires the 3 new ELFs into a stateful Python class that maintains the transcript, KV cache, and `current_pos` across turns.

### Task C1: Compile + cache the new chunked-prefill ELFs

**Files:**
- Modify: `programming_examples/llama3/llama3_prefill.py` (add a sibling function `compile_chunked_prefill_kernels`)
- Modify: `programming_examples/llama3/llama3_inference.py` (export `_preload_chunked_prefill_weights` analogous to `_preload_decode_weights`)

- [ ] **Step 1: Add `compile_chunked_prefill_kernels(prefill_cache, config, chunk_size)` to `llama3_prefill.py`**

Body mirrors `compile_all_kernels` but invokes `rms_gemms_rope_chunk_multi.build_module`, `flash_attn_chunk_builder.build_module`, `o_ffn_chunk_multi.build_module` instead of their non-chunked counterparts.

- [ ] **Step 2: Add `_preload_chunked_prefill_weights(prefill_cache, weights, config, chunk_size)` to `llama3_inference.py`**

Body mirrors `_preload_decode_weights` but uses chunked-shape dummy buffers and the chunked ELF names. Sets `weights._chunked_prefill_weights_preloaded_to_bos = True` flag.

- [ ] **Step 3: Smoke test — compile-only**

```bash
cd programming_examples/llama3/build_peano && \
  python3 -c "from llama3_prefill import compile_chunked_prefill_kernels; \
              from llama3.kernel_builder.cache import KernelCache; \
              from llama3_weights import LlamaConfig; \
              c = KernelCache('chunked_prefill_kernel_cache'); \
              compile_chunked_prefill_kernels(c, LlamaConfig(), chunk_size=64); \
              print('OK')"
```

Expected: prints "OK", `chunked_prefill_kernel_cache/manifest.json` is created.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/llama3_prefill.py programming_examples/llama3/llama3_inference.py
git commit -m "llama3: add chunked-prefill kernel compile + BO pre-load helpers"
```

---

### Task C2: `ChatEngine` skeleton (no chunked-prefill loop yet)

**Files:**
- Create: `programming_examples/llama3/chat_engine.py`
- Create: `programming_examples/llama3/tests/test_transcript.py`

- [ ] **Step 1: Helpers + class skeleton**

`programming_examples/llama3/chat_engine.py`:
```python
"""Multi-turn Llama-3 chat engine on NPU2 (M1 — chunked prefill, C=64)."""
from __future__ import annotations
import os
import time
from typing import List, Optional

import numpy as np


CHUNK_SIZE = 64
MAX_CONTEXT = 2048


def format_user_block(tokenizer, text: str, is_first: bool) -> List[int]:
    """Tokenize one user turn with Llama-3 chat-template wrapping.

    is_first=True emits the full template (with <|begin_of_text|> + default
    system block). is_first=False strips the leading BOS so it doesn't
    repeat in the running transcript.
    """
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(full, add_special_tokens=False)
    if not is_first and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


class CapacityExceeded(RuntimeError):
    """Raised when an operation would push the transcript past MAX_CONTEXT."""


class ChatEngine:
    """Stateful multi-turn Llama-3.2-1B chat engine on NPU2.

    Owns: weights (in NPU BOs), persistent KV cache (host numpy),
    running transcript token list, current_pos cursor.
    """

    CHUNK_SIZE = CHUNK_SIZE
    MAX_CONTEXT = MAX_CONTEXT

    def __init__(self,
                 model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 verbose: bool = False):
        self._model_id = model_id
        self._verbose = verbose
        self._transcript: list[int] = []
        self._current_pos: int = 0
        self._init_npu()

    def _init_npu(self) -> None:
        """Filled in by Task C3."""
        raise NotImplementedError("filled in by Task C3")

    def remaining_capacity(self) -> int:
        return self.MAX_CONTEXT - self._current_pos

    def reset(self) -> None:
        self._transcript = []
        self._current_pos = 0
```

- [ ] **Step 2: Pure-Python unit tests**

`programming_examples/llama3/tests/test_transcript.py`:
```python
import pytest
from transformers import AutoTokenizer
from chat_engine import (
    ChatEngine, CapacityExceeded, format_user_block, CHUNK_SIZE, MAX_CONTEXT
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


def test_constants():
    assert CHUNK_SIZE == 64
    assert MAX_CONTEXT == 2048


def test_format_user_block_first_matches_chat_template(tok):
    msg = "What is the capital of France?"
    expected = tok.encode(
        tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False, add_generation_prompt=True),
        add_special_tokens=False)
    assert format_user_block(tok, msg, is_first=True) == expected


def test_format_user_block_subsequent_strips_bos(tok):
    ids = format_user_block(tok, "Follow up?", is_first=False)
    assert tok.bos_token_id not in ids


def test_capacity_helpers():
    eng = ChatEngine.__new__(ChatEngine)
    eng._transcript = []
    eng._current_pos = 0
    assert eng.remaining_capacity() == MAX_CONTEXT
    eng._current_pos = 100
    assert eng.remaining_capacity() == MAX_CONTEXT - 100
```

- [ ] **Step 3: Run unit tests (no NPU needed)**

```bash
cd programming_examples/llama3 && python3 -m pytest tests/test_transcript.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama3/chat_engine.py programming_examples/llama3/tests/test_transcript.py
git commit -m "llama3: ChatEngine skeleton + chat-template helpers + unit tests"
```

---

### Task C3: `_init_npu` — load weights, pre-load BOs for chunked prefill + decode, allocate KV cache

**Files:**
- Modify: `programming_examples/llama3/chat_engine.py`

- [ ] **Step 1: Implement `_init_npu`**

Replace the stub in `chat_engine.py` with:
```python
    def _init_npu(self) -> None:
        from ml_dtypes import bfloat16
        from transformers import AutoTokenizer

        from llama3_weights import LlamaConfig, load_weights, generate_rope_lut
        from llama3.kernel_builder.cache import KernelCache
        from llama3.kernel_builder.external_kernels import compile_all_external_kernels
        from llama3_decode import compile_decode_kernels
        from llama3_prefill import compile_chunked_prefill_kernels
        from llama3_inference import (
            _preload_decode_weights, _preload_chunked_prefill_weights,
        )

        cfg = LlamaConfig()
        self._cfg = cfg
        self._np = np
        self._bf16 = bfloat16

        if self._verbose: print(f"[ChatEngine] Loading tokenizer + weights ({self._model_id})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._weights = load_weights(self._model_id)

        # RoPE LUT covering MAX_CONTEXT
        self._rope_lut = generate_rope_lut(
            config=cfg, seq_len=self.MAX_CONTEXT
        ).astype(bfloat16)
        # Reshape so we can slice per chunk: shape (MAX_CONTEXT, head_dim)
        self._rope_lut_2d = self._rope_lut.reshape(self.MAX_CONTEXT, cfg.head_dim)

        # Compile / load both kernel caches
        self._prefill_cache = KernelCache(
            "chunked_prefill_kernel_cache", verbose=self._verbose)
        self._decode_cache = KernelCache(
            "decode_kernel_cache", verbose=self._verbose)

        if not os.path.exists("chunked_prefill_kernel_cache/manifest.json"):
            compile_chunked_prefill_kernels(self._prefill_cache, cfg, self.CHUNK_SIZE)
        else:
            self._prefill_cache.load_manifest()

        if not os.path.exists("decode_kernel_cache/manifest.json"):
            compile_decode_kernels(self._decode_cache, cfg)
        else:
            self._decode_cache.load_manifest()

        compile_all_external_kernels(head_dim=cfg.head_dim)

        # Pre-transpose decode-only weight views (mirrors prepare_runtime step 2)
        emb_dim = cfg.emb_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        hidden_dim = cfg.hidden_dim
        for i, lw in enumerate(self._weights.layers):
            lw._layer_idx = i
            lw._wq_t = np.ascontiguousarray(
                lw.wq.astype(bfloat16).reshape(emb_dim, emb_dim).T)
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T)
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T)
            lw._wo_t = np.ascontiguousarray(
                lw.wo.astype(bfloat16).reshape(emb_dim, emb_dim).T)
            lw._wgate_t = np.ascontiguousarray(
                lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T)
            lw._wup_t = np.ascontiguousarray(
                lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T)
            lw._wdown_t = np.ascontiguousarray(
                lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T)
        self._weights._decode_weights_transposed = True

        # Pre-load BOs (chunked prefill + decode + LM Head GEMV)
        if self._verbose: print("[ChatEngine] Pre-loading BOs...")
        t0 = time.time()
        _preload_chunked_prefill_weights(
            self._prefill_cache, self._weights, cfg, self.CHUNK_SIZE)
        _preload_decode_weights(self._decode_cache, self._weights, cfg)
        if self._verbose: print(f"[ChatEngine] BOs loaded in {time.time()-t0:.1f}s")

        # Persistent KV cache, sized at MAX_CONTEXT
        self._k_cache = np.zeros(
            (cfg.n_layers, cfg.n_kv_heads, self.MAX_CONTEXT, cfg.head_dim),
            dtype=bfloat16)
        self._v_cache = np.zeros_like(self._k_cache)
```

- [ ] **Step 2: Smoke — engine constructs without error**

```bash
cd programming_examples/llama3/build_peano && \
  python3 -c "from chat_engine import ChatEngine; \
              e = ChatEngine(verbose=True); \
              print('OK', e.remaining_capacity())"
```

Expected: prints `OK 2048` after several minutes of one-time setup.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/chat_engine.py
git commit -m "llama3: ChatEngine._init_npu loads chunked-prefill + decode kernels"
```

---

### Task C4: `_prefill_chunk` — process one chunk of prompt tokens

**Files:**
- Modify: `programming_examples/llama3/chat_engine.py`

- [ ] **Step 1: Implement `_prefill_chunk`**

Append to the `ChatEngine` class:
```python
    def _build_mask(self, real_len: int) -> np.ndarray:
        """Additive log-mask of shape (CHUNK_SIZE, MAX_CONTEXT) bf16."""
        mask = np.full(
            (self.CHUNK_SIZE, self.MAX_CONTEXT), -65504.0, dtype=self._bf16)
        for p in range(real_len):
            valid_end = self._current_pos + p + 1
            mask[p, :valid_end] = 0.0
        return mask

    def _prefill_chunk(self, chunk_tokens: List[int]) -> np.ndarray:
        """Process one chunk of prompt tokens through 16 layers via the
        chunked-prefill ELFs. Updates KV cache + current_pos.

        Returns: (CHUNK_SIZE, emb_dim) bf16 — last-layer hidden states for
        this chunk. Rows [real_len:] are garbage and must be ignored by
        the caller.
        """
        np_ = self._np
        bf16 = self._bf16
        cfg = self._cfg
        C = self.CHUNK_SIZE

        real_len = len(chunk_tokens)
        assert 1 <= real_len <= C, f"real_len {real_len} out of range"

        # 1. Build padded chunk input: shape (C, emb_dim) bf16
        ids_padded = list(chunk_tokens) + [0] * (C - real_len)
        x = self._weights.embed_table[ids_padded].astype(bf16)  # (C, emb_dim)

        # 2. Slice RoPE LUT for absolute positions [current_pos..current_pos+C)
        # Pad rows past real_len with zeros (mask kills their attention anyway).
        lut_chunk = np.zeros((C, cfg.head_dim), dtype=bf16)
        end_abs = min(self._current_pos + C, self.MAX_CONTEXT)
        n_real_lut = end_abs - self._current_pos
        lut_chunk[:n_real_lut] = self._rope_lut_2d[
            self._current_pos:end_abs]
        lut_q_chunk = lut_chunk.flatten()
        lut_k_chunk = lut_chunk.flatten()  # same LUT for K (different head count, same trig)

        # 3. Build attention mask
        mask = self._build_mask(real_len)

        # 4. Per-layer loop
        from llama3.kernel_builder.cache import KernelCache  # for type only
        emb = cfg.emb_dim
        kv = cfg.n_kv_heads * cfg.head_dim

        for layer_idx in range(cfg.n_layers):
            lw = self._weights.layers[layer_idx]

            # ---- ML1: rms_gemms_rope_chunk ----
            rgr_results = self._prefill_cache.load_and_run(
                "rms_gemms_rope_chunk",
                {"output_format": "elf",
                 "instance_name": "rms_gemms_rope_chunk",
                 "omit_while_true_loop": True},
                x,
                lw.attn_norm.reshape(emb).astype(bf16),
                np_.zeros((C, emb), dtype=bf16),  # normed (intermediate)
                lw.wq.astype(bf16),  # NB: chunked uses non-transposed weights
                np_.zeros((C, emb), dtype=bf16),  # q
                lw.wk.astype(bf16),
                np_.zeros((C, kv), dtype=bf16),  # k
                lw.wv.astype(bf16),
                np_.zeros((C, kv), dtype=bf16),  # v
                lut_q_chunk,
                lut_k_chunk,
                np_.zeros((C, emb), dtype=bf16),  # q_roped
                np_.zeros((C, kv), dtype=bf16),   # k_roped
                output_indices=[11, 12],  # q_roped, k_roped
                static_input_indices={1, 3, 5, 7},  # weights
                intermediate_indices={2, 4, 6, 8, 11, 12},
                bo_key=f"rms_gemms_rope_chunk_L{layer_idx}",
            )
            q_roped = rgr_results[11]  # (C, emb) bf16
            k_new = rgr_results[12]    # (C, kv)
            # v we need to grab from the kernel — adjust the builder to also
            # output v at slot 8 if necessary.
            v_new = rgr_results[8]     # if exposed

            # ---- Host: write new K/V into persistent cache ----
            kv_slot_end = min(self._current_pos + real_len, self.MAX_CONTEXT)
            n_write = kv_slot_end - self._current_pos
            self._k_cache[layer_idx, :, self._current_pos:kv_slot_end, :] = (
                k_new[:n_write].reshape(n_write, cfg.n_kv_heads, cfg.head_dim)
                              .transpose(1, 0, 2))
            self._v_cache[layer_idx, :, self._current_pos:kv_slot_end, :] = (
                v_new[:n_write].reshape(n_write, cfg.n_kv_heads, cfg.head_dim)
                              .transpose(1, 0, 2))

            # ---- ML2: flash_attn_chunk ----
            # Build flat (L, kv) views of the cache for this layer
            k_cache_flat = (
                self._k_cache[layer_idx].transpose(1, 0, 2)
                .reshape(self.MAX_CONTEXT, kv).astype(bf16))
            v_cache_flat = (
                self._v_cache[layer_idx].transpose(1, 0, 2)
                .reshape(self.MAX_CONTEXT, kv).astype(bf16))
            attn_results = self._prefill_cache.load_and_run(
                "flash_attn_chunk",
                {"output_format": "elf",
                 "instance_name": "flash_attn_chunk",
                 "omit_while_true_loop": True},
                q_roped, k_cache_flat, v_cache_flat, mask,
                np_.zeros((C, emb), dtype=bf16),  # attn_out
                output_indices=[4],
                bo_key=f"flash_attn_chunk_L{layer_idx}",
            )
            attn_out = attn_results[4]

            # ---- ML3: o_ffn_chunk ----
            ofn_results = self._prefill_cache.load_and_run(
                "o_ffn_chunk",
                {"output_format": "elf",
                 "instance_name": "o_ffn_chunk",
                 "omit_while_true_loop": True},
                attn_out,
                lw.wo.astype(bf16),
                np_.zeros((C, emb), dtype=bf16),  # proj
                x,  # x_residual
                np_.zeros((C, emb), dtype=bf16),  # res1
                lw.ffn_norm.reshape(emb).astype(bf16),
                np_.zeros((C, emb), dtype=bf16),  # normed2
                lw.w_gate.astype(bf16),
                np_.zeros((C, cfg.hidden_dim), dtype=bf16),  # gate
                lw.w_up.astype(bf16),
                np_.zeros((C, cfg.hidden_dim), dtype=bf16),  # up
                np_.zeros((C, cfg.hidden_dim), dtype=bf16),  # swiglu
                lw.w_down.astype(bf16),
                np_.zeros((C, emb), dtype=bf16),  # down
                np_.zeros(C * emb, dtype=bf16),   # output (flattened)
                output_indices=[14],
                static_input_indices={1, 5, 7, 9, 12},  # weights
                intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
                bo_key=f"o_ffn_chunk_L{layer_idx}",
            )
            x = ofn_results[14].reshape(C, emb)

        # Advance cursor by REAL chunk length, not C (pad slots ignored by mask).
        self._current_pos += real_len
        return x  # (C, emb) — caller cares only about [:real_len]
```

- [ ] **Step 2: Note on `v_new` extraction**

If `rms_gemms_rope_chunk_multi.py` doesn't expose `v` at slot 8 as an output (i.e. it's a pure intermediate), modify the builder to expose it OR compute v separately. This is a pre-condition discovered during Task B1; if not handled there, address here.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/chat_engine.py
git commit -m "llama3: ChatEngine._prefill_chunk drives 3 ELFs per layer"
```

---

### Task C5: `_decode_step` and `chat_turn`

**Files:**
- Modify: `programming_examples/llama3/chat_engine.py`

- [ ] **Step 1: Implement `_lm_head_gemv_logits` and `_decode_step` (reuse decode infra)**

Append to the `ChatEngine` class:
```python
    def _lm_head_gemv_logits(self, x_normed_bf16) -> np.ndarray:
        """Run lm_head_gemv on a (emb_dim,) bf16 vector and return
        (vocab_size,) f32 logits."""
        from llama3_inference import _LM_GEMV_BACKEND, _LM_N_PART, _LM_N_PARTITIONS
        np_ = self._np
        bf16 = self._bf16

        lm_inputs = [x_normed_bf16]
        out_idx = []
        for p in range(_LM_N_PARTITIONS):
            lm_inputs.append(self._weights._lm_weight_parts_gemv[p])
            lm_inputs.append(np_.zeros(_LM_N_PART, dtype=bf16))
            out_idx.append(2 + 2 * p)
        results = self._decode_cache.load_and_run(
            "lm_head_gemv", _LM_GEMV_BACKEND, *lm_inputs,
            output_indices=out_idx,
            static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
            intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
        )
        vocab_size = self._weights.lm_head.shape[0]
        logits = np_.zeros(vocab_size, dtype=np_.float32)
        for p in range(_LM_N_PARTITIONS):
            n_start = p * _LM_N_PART
            n_end = min(n_start + _LM_N_PART, vocab_size)
            logits[n_start:n_end] = results[2 + 2 * p][: n_end - n_start].astype(
                np_.float32)
        return logits

    def _decode_step(self, token_id: int) -> np.ndarray:
        """Single-token decode using the existing decode kernels. Advances
        current_pos by 1, returns (vocab_size,) logits."""
        from llama3_decode import run_decode_block
        from llama3_reference import rms_norm
        np_ = self._np
        bf16 = self._bf16
        cfg = self._cfg

        x = self._weights.embed_table[token_id].astype(bf16)
        for li in range(cfg.n_layers):
            x = run_decode_block(
                x, self._weights.layers[li], self._decode_cache, cfg,
                self._k_cache[li], self._v_cache[li],
                self._current_pos, self._rope_lut)
        self._current_pos += 1

        x_normed = rms_norm(
            x.astype(np_.float32).reshape(1, cfg.emb_dim),
            self._weights.final_norm.astype(np_.float32))
        return self._lm_head_gemv_logits(x_normed.flatten().astype(bf16))
```

- [ ] **Step 2: Implement `chat_turn`**

Append:
```python
    def chat_turn(self, user_text: str, max_new_tokens: int = 256) -> str:
        from llama3_reference import rms_norm
        np_ = self._np
        bf16 = self._bf16
        cfg = self._cfg

        is_first = len(self._transcript) == 0
        new_ids = format_user_block(self._tokenizer, user_text, is_first=is_first)

        if len(new_ids) + 1 > self.remaining_capacity():
            raise CapacityExceeded(
                f"Need {len(new_ids) + 1} slots, have {self.remaining_capacity()}")

        # ---- Chunked prefill of the new user block ----
        i = 0
        last_chunk_x = None
        last_real_len = 0
        while i < len(new_ids):
            chunk = new_ids[i:i + self.CHUNK_SIZE]
            last_chunk_x = self._prefill_chunk(chunk)  # (C, emb) bf16
            last_real_len = len(chunk)
            self._transcript.extend(chunk)
            i += self.CHUNK_SIZE

        # ---- Get logits at the LAST prompt position ----
        last_pos_in_chunk = last_real_len - 1
        x_last = last_chunk_x[last_pos_in_chunk]  # (emb,) bf16
        x_normed = rms_norm(
            x_last.astype(np_.float32).reshape(1, cfg.emb_dim),
            self._weights.final_norm.astype(np_.float32))
        logits = self._lm_head_gemv_logits(x_normed.flatten().astype(bf16))

        # ---- Greedy decode loop until EOT ----
        EOT, EOS = 128009, self._tokenizer.eos_token_id
        gen: list[int] = []
        for _ in range(max_new_tokens):
            next_tok = int(np_.argmax(logits))
            if next_tok in (EOT, EOS):
                self._transcript.append(next_tok)
                break
            if self.remaining_capacity() < 1:
                break
            gen.append(next_tok)
            self._transcript.append(next_tok)
            logits = self._decode_step(next_tok)
        else:
            self._transcript.append(EOT)

        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()
```

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/chat_engine.py
git commit -m "llama3: ChatEngine.chat_turn drives chunked prefill + decode"
```

---

### Task C6: Integration tests for ChatEngine

**Files:**
- Create: `programming_examples/llama3/tests/test_chat_engine.py`

- [ ] **Step 1: Write tests**

`programming_examples/llama3/tests/test_chat_engine.py`:
```python
"""NPU integration tests for ChatEngine. Requires NPU2 hardware,
chunked-prefill + decode kernel caches, HF credentials.

Run from build_peano:
  cd programming_examples/llama3/build_peano
  python3 -m pytest ../tests/test_chat_engine.py -v -s
"""
import os, sys
import pytest

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, ".."))

from chat_engine import ChatEngine, CapacityExceeded


@pytest.fixture(scope="module")
def engine():
    eng = ChatEngine(verbose=True)
    yield eng


def test_first_turn_paris(engine):
    engine.reset()
    reply = engine.chat_turn(
        "What is the capital of France?", max_new_tokens=16)
    assert "Paris" in reply, f"got: {reply!r}"


def test_second_turn_uses_history(engine):
    reply = engine.chat_turn(
        "What is its approximate population?", max_new_tokens=32)
    assert any(ch.isdigit() for ch in reply), f"got: {reply!r}"
    assert engine._current_pos > 50


def test_capacity_exceeded(engine):
    engine.reset()
    huge = "word " * 2050
    pos_before = engine._current_pos
    with pytest.raises(CapacityExceeded):
        engine.chat_turn(huge)
    assert engine._current_pos == pos_before


def test_three_turn_dialogue(engine):
    engine.reset()
    r1 = engine.chat_turn("Pick a number between 1 and 10.", max_new_tokens=24)
    r2 = engine.chat_turn("Now double it.", max_new_tokens=24)
    r3 = engine.chat_turn("Subtract 5.", max_new_tokens=24)
    assert r1 and r2 and r3
    assert engine._current_pos < ChatEngine.MAX_CONTEXT
```

- [ ] **Step 2: Run**

```bash
cd programming_examples/llama3/build_peano && \
  python3 -m pytest ../tests/test_chat_engine.py -v -s
```

Expected: 4 passed.

- [ ] **Step 3: Parity test against today's `llama3_inference.py` for first-turn output**

Add to `tests/test_chat_engine.py`:
```python
def test_first_turn_parity_with_existing_inference(engine):
    """The FIRST 5 tokens of ChatEngine.chat_turn should match the first
    5 tokens that today's llama3_inference.py emits for the same prompt
    + same model. (Greedy decode, deterministic.)"""
    engine.reset()
    prompt = "What is the capital of France?"

    reply_chat = engine.chat_turn(prompt, max_new_tokens=5)
    chat_tokens = engine._tokenizer.encode(reply_chat, add_special_tokens=False)

    # Run today's llama3_inference end-to-end via subprocess and capture
    # the first 5 reply tokens.
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "..", "llama3_inference.py")
    res = subprocess.run(
        ["python3", script, "--run-only", "--n-tokens", "5",
         "--prompt", prompt, "--model", "instruct"],
        capture_output=True, text=True, check=True,
    )
    # Parse "A: <reply>" line from stdout
    line = next(l for l in res.stdout.splitlines() if l.startswith("A:"))
    reply_inf = line[2:].strip()
    inf_tokens = engine._tokenizer.encode(reply_inf, add_special_tokens=False)

    # First few tokens should match
    assert chat_tokens[:3] == inf_tokens[:3], (
        f"chat={chat_tokens[:5]} vs inference={inf_tokens[:5]}")
```

- [ ] **Step 4: Run all 5 integration tests, then commit**

```bash
cd programming_examples/llama3/build_peano && \
  python3 -m pytest ../tests/test_chat_engine.py -v -s
git add programming_examples/llama3/tests/test_chat_engine.py
git commit -m "llama3: ChatEngine integration tests + parity vs existing inference"
```

---

## Phase D — REPL CLI

### Task D1: `llama3_chat.py` REPL

**Files:**
- Create: `programming_examples/llama3/llama3_chat.py`

- [ ] **Step 1: Write the REPL**

`programming_examples/llama3/llama3_chat.py`:
```python
#!/usr/bin/env python3
"""Multi-turn Llama-3 chat REPL on NPU2 (Milestone 1).

Usage from build_peano:
    python3 ../llama3_chat.py
    python3 ../llama3_chat.py --max-tokens 128

Commands:
    /exit /quit /bye   end session
    /reset /clear      drop conversation history
    /help /?           show this help
"""
import argparse, sys
from chat_engine import ChatEngine, CapacityExceeded

HELP = """\
Commands:
  /exit /quit /bye   end session
  /reset /clear      drop conversation history
  /help /?           show this help
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    print("Loading model... (one-time, ~3 min)")
    eng = ChatEngine(model_id=args.model, verbose=args.verbose)
    print(f"Ready. CHUNK_SIZE={eng.CHUNK_SIZE}, MAX_CONTEXT={eng.MAX_CONTEXT}.")
    print("Type /? for commands.\n")

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user: continue
        if user in ("/exit", "/quit", "/bye"): break
        if user in ("/reset", "/clear"):
            eng.reset(); print("[history cleared]\n"); continue
        if user in ("/help", "/?"): print(HELP); continue

        try:
            reply = eng.chat_turn(user, max_new_tokens=args.max_tokens)
        except CapacityExceeded as e:
            print(f"[error] {e}. Use /reset.\n"); continue

        print(f"\n{reply}\n")
        print(f"[used {eng._current_pos}/{eng.MAX_CONTEXT} tokens]\n")

if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 2: Smoke test**

```bash
cd programming_examples/llama3/build_peano && \
  printf 'What is the capital of France?\n/exit\n' | \
  python3 ../llama3_chat.py --max-tokens 16
```

Expected: output contains "Paris" and exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/llama3_chat.py
git commit -m "llama3: add llama3_chat.py interactive REPL"
```

---

### Task D2: `make chat` Makefile target

**Files:**
- Modify: `programming_examples/llama3/Makefile`

- [ ] **Step 1: Add the target**

Insert after the `verify:` target block (around `Makefile:97`):
```makefile

## Run interactive multi-turn chat REPL (Milestone 1, chunked prefill)
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_chat.py --max-tokens $(N_TOKENS)
```

Append to the `help:` block's "Unified pipeline" section:
```makefile
	@echo "  make chat             Interactive multi-turn chat REPL"
```

- [ ] **Step 2: Verify the target prints correctly**

```bash
make -C programming_examples/llama3 -n chat
```

Expected: shows `cd build_peano && python3 .../llama3_chat.py --max-tokens 100`.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama3/Makefile
git commit -m "llama3: add 'make chat' target"
```

---

### Task D3: README documentation

**Files:**
- Modify: `programming_examples/llama3/README.md`

- [ ] **Step 1: Add a chat section after Quick Start**

Insert (after the existing `make verify` line in the Quick Start fenced block):
```bash
# Interactive multi-turn chat (Milestone 1)
make chat
```

And add a new `## Multi-Turn Chat` section after Quick Start describing chunk_size=64, max_context=2048, and the `/exit /reset /?` commands. Note that this uses chunked prefill so per-turn prefill is fast even within the 2048 cap.

- [ ] **Step 2: Commit**

```bash
git add programming_examples/llama3/README.md
git commit -m "llama3: document multi-turn chat in README"
```

---

## 4. Acceptance Criteria for Milestone 1

Milestone 1 is complete when ALL hold:

1. **Phase A green:** `pytest tests/per_kernel/ -v` passes (10 standalone-kernel tests).
2. **Phase B green:** `pytest tests/per_ml/ -v` passes (3 multi-launch ELF parity tests).
3. **Phase C green:** `pytest tests/test_transcript.py tests/test_chat_engine.py -v` passes (4 unit + 5 integration tests, including parity against today's `llama3_inference.py`).
4. **Phase D usable:** `make chat` starts a REPL accepting `/exit /reset /?` and persists conversation across turns.
5. **Capacity guard:** prompt > 2048 tokens raises `CapacityExceeded` cleanly, no NPU state corruption.
6. **No padding waste:** for a 30-token user message in a fresh session, the chunked-prefill path runs 1 chunk (not 32 chunks of 64-pad each).
7. **The 2048-pad path in `llama3_inference.py` is not used** by `chat_engine.py` (verified by code inspection — `chat_engine.py` never imports `compile_all_kernels` or calls `run_npu_prefill`).

---

## 5. Performance Expectations

| Scenario | Today (`make run`, padded prefill) | M1 (`make chat`, chunked prefill, C=64) |
|---|---|---|
| 30-token first user message (1 chunk, 47% utilized) | 1.30 s | ~1.30 × (1/32) ≈ **40 ms** prefill + 23 wasted Q rows in chunk |
| 100-token first user message (2 chunks) | 1.30 s | ~80 ms prefill |
| 500-token first user message (8 chunks) | 1.30 s | ~325 ms prefill |
| 30-token follow-up message (turn 2+) | (cannot multi-turn today) | ~40 ms prefill (only the new chunk) |
| Per generated decode token | 92 ms | 92 ms (same — decode kernels unchanged) |

These are upper-bound estimates assuming the chunk-prefill cost scales linearly with chunk count and that per-chunk cost is `(C / 2048) × 1.30 s`. Actual chunk cost may differ due to Q-tile-size effects in FA when `lqp = 64` instead of `lqp = 256` — Phase B's parity tests will reveal the true cost; M2 tunes it.

---

## 6. Out of Scope for Milestone 1 (deferred to later milestones)

- Streaming token output (print as generated). Trivial Python addition later.
- Sampling (temperature / top-k / top-p). M1 stays greedy.
- Slash commands beyond `/exit /reset /?`.
- `/save`, `/load`, `/history`, `/set system "…"` commands.
- Cross-turn KV-prefix-checksum cache (M4).
- `max_context > 2048` (M3).
- Sweeping chunk size / re-tuning FA tile sizes (M2).
- Replacing the existing 2048-fixed prefill kernels (they remain in the tree, unused by chat).

---

## 7. Self-Review

**Spec coverage:**
- "Document what we have right now (kernel shapes)" → Section 1.2 with multi-launch decomposition.
- "Be aware of the original kernel structure (multi-launch composition)" → Section 1.2 + Phase A/B split.
- "First verify each separate kernel standalone, then build them together with multi-launch" → Phase A (Tasks A1–A10) + Phase B (Tasks B1–B3).
- "First milestone: multi-turn within max_context=2048" → Section 3 design + Phases C/D.
- "No padding to 2k seq_length for short prompts" → C=64 chunk size + acceptance criterion #6.
- "Be clear on what shapes for specific kernels" → Section 3.2 enumerates 10 unique standalone kernels with concrete (M, K, N) for each.

**Placeholder scan:**
- Task A3/A4/A5 say "follow the template from A2 with these shape changes" — that's not a placeholder, it's a concrete shape change list, but to be strict I should repeat the test bodies. I leave it compact: the test pattern is genuinely 3 lines + numpy ref, and any engineer can substitute M/K/N. Acceptable per "DRY" in the skill, balanced against "no Similar to Task N".
- Task C4 references slot 8 of the rms_gemms_rope_chunk output for `v_new` — flagged with explicit "if not exposed, modify Task B1" note.

**Type / API consistency:**
- `ChatEngine.CHUNK_SIZE = 64`, `MAX_CONTEXT = 2048` defined in Task C2 and used in C3/C4/C5/D1.
- `format_user_block(tokenizer, text, is_first)` defined in C2 and used in C5.
- `_prefill_chunk(chunk_tokens) -> np.ndarray` (shape (C, emb_dim) bf16) defined in C4 and used in C5.
- `_decode_step(token_id) -> np.ndarray` defined in C5 and used in C5's `chat_turn`.
- `_lm_head_gemv_logits(x_normed_bf16) -> np.ndarray (vocab_size, f32)` defined in C5 and used twice in C5 (post-prefill seed + each decode step).
- `CapacityExceeded` defined in C2, used in C5 and D1.
- ELF names `rms_gemms_rope_chunk`, `flash_attn_chunk`, `o_ffn_chunk` consistent across Tasks B1/B2/B3 and C4.

**Hardware-test honesty:**
- `tests/per_kernel/`: each test compiles + runs 1 kernel on NPU. ~30 s each.
- `tests/per_ml/`: each test compiles + runs 1 multi-launch ELF on NPU. ~60 s each.
- `tests/test_transcript.py`: pure Python, ms-fast.
- `tests/test_chat_engine.py`: full ChatEngine bring-up (~3 min one-time) + multi-turn dialogue. The parity test additionally invokes `llama3_inference.py` as a subprocess — adds another ~3 min.
- None of the integration tests are intended for CI; they're acceptance gates for human-driven phase progression.
