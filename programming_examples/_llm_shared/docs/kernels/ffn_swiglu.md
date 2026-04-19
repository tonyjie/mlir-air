# FFN SwiGLU Block — Multi-Launch Optimization

> **Development history document.** This records kernel optimization analysis
> done during development. For the current pipeline, see `docs/profile.md`
> and `docs/explain.md`. Kernel names and invocation counts may have changed
> due to multi-launch merging.


## Role in LLAMA Pipeline

Steps 11-14 of each transformer block: the full feed-forward network.

```
output = SwiGLU(input @ W_gate, input @ W_up) @ W_down
       = (SiLU(input @ W_gate) × (input @ W_up)) @ W_down
```

| Step | Operation | Shape | Kernel |
|------|-----------|-------|--------|
| 11 | Gate GEMM | (2048, 2048) × (2048, 8192) | Vectorized GEMM, 8×4 herd |
| 12 | Up GEMM | (2048, 2048) × (2048, 8192) | Vectorized GEMM, 8×4 herd |
| 13 | SiLU × mul | 16,777,216 elements | External C++ kernel, 8×1 herd |
| 14 | Down GEMM | (2048, 8192) × (8192, 2048) | Vectorized GEMM, 8×4 herd |

---

## Multi-Launch Approach

Instead of 4 separate XRT invocations (one per kernel), all 4 operations are placed as sequential `air.launch` ops inside a single func → compiled to one ELF → executed in one `xrt.run()` call.

### Architecture

```
func @ffn_block(%input, %w_gate, %gate_buf, %w_up, %up_buf, %swiglu_buf, %w_down, %output):
    air.launch 1: Gate GEMM     input × w_gate → gate_buf     (vectorized, 8×4 herd, L2 tiling)
    air.launch 2: Up GEMM       input × w_up → up_buf         (same as Gate)
    air.launch 3: SwiGLU        SiLU(gate_buf) × up_buf → swiglu_buf  (8×1 herd, external C++)
    air.launch 4: Down GEMM     swiglu_buf × w_down → output  (vectorized, 8×4 herd, L2 tiling)
    return
```

8 function arguments (all 2D BF16 memrefs):

| Arg | Name | Shape | Role |
|-----|------|-------|------|
| %arg0 | input | 2048×2048 | FFN input (from RMSNorm) |
| %arg1 | w_gate | 2048×8192 | Gate weight matrix |
| %arg2 | gate_buf | 2048×8192 | Intermediate: Gate GEMM output → SwiGLU input |
| %arg3 | w_up | 2048×8192 | Up weight matrix |
| %arg4 | up_buf | 2048×8192 | Intermediate: Up GEMM output → SwiGLU input |
| %arg5 | swiglu_buf | 2048×8192 | Intermediate: SwiGLU output → Down GEMM input |
| %arg6 | w_down | 8192×2048 | Down weight matrix |
| %arg7 | output | 2048×2048 | FFN output |

Intermediates (%arg2, %arg4, %arg5) are DDR buffers shared between launches. The NPU writes to them in one launch and reads from them in the next — no host memcpy.

---

## Performance

| Metric | 4 Separate Kernels | Multi-Launch | Multi-Launch + ReadOpt | IRON Fused |
|--------|-------------------|-------------|----------------------|------------|
| Kernel time | 109ms | 35.7ms | **35.7ms** | ~48ms |
| Total (with host) | ~149ms | 83ms | **52ms** | 57.4ms |
| Speedup vs separate | — | 1.8× | **2.9×** | — |
| vs IRON | 0.39× | 0.69× | **1.10×** | 1.0× |
| Correlation vs CPU F32 | 0.999+ | 0.9996 | 0.9996 | 0.999+ |

The multi-launch FFN + read-only-output eliminates:
- 3 inter-kernel host round-trips (~25ms saved: no DDR→DDR memcpy for intermediates)
- 3 kernel dispatch overheads (~5ms saved)
- Unnecessary BO read-back of 210MB (~30ms saved: only read 8MB output)

See `../host_optimization.md` for detailed BO write/read analysis.

---

## How It's Built (Text-Based MLIR Stitching)

`ffn_swiglu/run.py` builds the multi-launch module in 4 steps:

### Step 1: Build each kernel independently

Each kernel is built using existing functions that already produce optimized, transformed MLIR:
- Gate/Up GEMM: `_build_gemm_module()` from `llama3_prefill.py` (linalg → transform → vectorized)
- SwiGLU: `build_module_2d()` from `silu_and_mul.py` (external C++ kernel, 2D memrefs)
- Down GEMM: `_build_gemm_module()` with different tile config

### Step 2: Rename all identifiers

Each module uses the same SSA names (`%arg0`, `%c4`, `#map`, etc.). The `_rename_all()` function adds a unique prefix to every identifier:

| Module | Prefix | Example: `%arg3` → | Example: `#map1` → | Example: `@herd_0` → |
|--------|--------|--------------------|--------------------|---------------------|
| Gate GEMM | `g` | `%g_arg3` | `#g_map1` | `@g_herd_0` |
| Up GEMM | `u` | `%u_arg3` | `#u_map1` | `@u_herd_0` |
| SwiGLU | `s` | `%s_arg3` | `#s_map` | `@s_herd_0` |
| Down GEMM | `d` | `%d_arg3` | `#d_map1` | `@d_herd_0` |

External C++ function names (`@silu_and_mul_bf16`) are NOT renamed.

### Step 3: Remap func-arg references

After renaming, the launch's `args(...)` clause has renamed func-arg references (e.g., `%g_arg0`). These are remapped to the combined func's `%arg0`–`%arg7`:

```
Gate:   args(...=%arg0, ...=%arg1, ...=%arg2)   ← input, w_gate, gate_buf
Up:     args(...=%arg0, ...=%arg3, ...=%arg4)   ← input, w_up, up_buf
SwiGLU: args(...=%arg2, ...=%arg4, ...=%arg5)   ← gate_buf, up_buf, swiglu_buf
Down:   args(...=%arg5, ...=%arg6, ...=%arg7)   ← swiglu_buf, w_down, output
```

### Step 4: Assemble and parse

All affine maps, private func declarations, and the 4 launch bodies are concatenated into one `func @ffn_block` and parsed by `Module.parse()`.

---

## SwiGLU 2D Memref Handling

The GEMM outputs are 2D (`memref<2048x8192xbf16>`) but the SwiGLU C++ kernel operates on flat 1D data. The `build_module_2d()` function in `silu_and_mul.py` handles this:

```mlir
air.launch () in () args(%arg8=%arg2, %arg9=%arg4, %arg10=%arg5) : memref<2048x8192xbf16>, ... {
    %collapse_shape = memref.collapse_shape %arg8 [[0, 1]] : memref<2048x8192xbf16> into memref<16777216xbf16>
    ...
    air.segment @s_swiglu_seg args(%arg11=%collapse_shape, ...) : memref<16777216xbf16>, ... {
        air.herd @s_herd_0 ... args(...) : memref<16777216xbf16>, ... {
            // DMA + func.call @silu_and_mul_bf16 on 1D data
        }
    }
}
```

The `memref.collapse_shape` converts 2D→1D inside the launch, before passing to the segment/herd.

---

## Files

| File | Purpose |
|------|---------|
| `llama3/ffn_swiglu/run.py` | Self-contained multi-launch FFN builder + test |
| `llama3/ffn_swiglu/Makefile` | Build targets: `run`, `profile`, `print`, `print-kernels` |
| `llama3/ffn_swiglu/silu_and_mul.py` | SiLU×mul kernel with `build_module_2d()` (2D memref variant) |
| `llama3/ffn_swiglu/silu_and_mul.cc` | C++ SiLU×mul kernel (BF16, VecLen=16) |
| `llama3/ffn_swiglu/*.mlir` | Saved MLIR: `gate_gemm.mlir`, `up_gemm.mlir`, `swiglu.mlir`, `down_gemm.mlir`, `ffn_multi_launch.mlir` |

## Commands

```bash
cd programming_examples/llama3/ffn_swiglu

make run              # Compile + run + validate on NPU (PASS/FAIL)
make profile          # Compile + run + profile with timing comparison
make print            # Print combined multi-launch MLIR
make print-kernels    # Print each sub-kernel's MLIR separately + combined
make clean            # Clean build artifacts
```

---

## Related Documents

- `silu_and_mul.md` — SiLU + elementwise multiply standalone kernel optimization
- `gemm.md` — GEMM kernel optimization and tile configs
- `../perf_opt_prefill.md` — Overall LLAMA optimization roadmap
