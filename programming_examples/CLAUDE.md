# programming_examples/ — End-to-End Operator Examples

Self-contained examples demonstrating AI/ML operators compiled and executed on AMD NPU hardware. Each example generates AIR MLIR from Python, compiles to xclbin, and validates on hardware.

## Directory Structure

37+ example directories organized by operator type. Each contains:

| File | Purpose |
|------|---------|
| `<name>.py` | Python IR generator using `@module_builder` pattern |
| `Makefile` | Build targets: `run`, `print`, `clean` (some: `profile`, `compile-kernel`) |
| `*.lit` | LIT test specs for CI (peano/chess, npu1/npu2 variants) |
| `*.cc` | Optional C++ kernels for external functions |
| `test.cpp` | Optional C++ XRT profiling harness |

### Example categories
- **Linear algebra**: `matrix_multiplication/`, `vector_matrix_multiplication/`
- **Activations**: `relu/`, `silu/`, `gelu/`, `leaky_relu/`
- **Normalization**: `softmax/`, `rms_norm/`, `layer_norm/`, `weighted_rms_norm/`
- **Transformer blocks**: `flash_attention/`, `llama2_mha/`, `llama2_rope/`, `ffn_swiglu/`, `swiglu/`
- **Element-wise**: `eltwise_add/`, `eltwise_add_with_l2/`
- **Data transfer**: `passthrough/`, `data_transfer_transpose/`, `shim_dma_2d/`
- **Primitives**: `primitives/scalar_examples/`, `primitives/vector_examples/`

## Python IR Generation Pattern

### Standard structure
```python
@module_builder
def build_module(size, tile_size, dtype):
    xrt_dtype = type_mapper(dtype)
    memrefTy = MemRefType.get([size], xrt_dtype)

    @FuncOp.from_py_func(memrefTy, memrefTy, memrefTy)
    def func(arg0, arg1, arg2):
        @launch(operands=[arg0, arg1, arg2])
        def launch_body(a, b, c):
            @segment(name="seg0")
            def segment_body():
                @herd(name="herd0", sizes=[1, 1])
                def herd_body(tx, ty):
                    # L1 alloc, DMA in, compute, DMA out
                    yield_([])
```

### Key imports
```python
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
```

### Memory hierarchy in Python
```python
# L2 (MemTile, shared)
l2_mem = IntegerAttr.get(T.i32(), MemorySpace.L2)  # value 1
# L1 (Local, per-core)
l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)  # value 2
# DDR (host) — no memory_space attr
```

### DMA transfers
```python
dma_memcpy_nd(dst, src,
    src_offsets=[...], src_sizes=[...], src_strides=[1],
    dst_offsets=[...], dst_sizes=[...], dst_strides=[1])
```

### Vectorized computation
```python
vecTy = VectorType.get([vector_size], xrt_dtype)
for j in range_(0, size, vector_size):
    sub = subview(buffer, [j], [vector_size], [1])
    v = transfer_read(vecTy, sub, [c0], identity_map, cst0, [True])
    result = arith.mulf(v_a, v_b)
    transfer_write(None, result, sub, [c0], identity_map, [True])
    yield_([])
```

## Execution Pattern

### Command-line arguments (all examples)
```python
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-p", "--print-module-only", action="store_true")
parser.add_argument("--compile-mode", choices=["compile-only", "compile-and-run"], default="compile-and-run")
parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
```

### Validation via XRTRunner
```python
runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=True,
                   output_format=args.output_format, instance_name="example_name")
exit(runner.run_test(mlir_module, inputs=[A, B], expected_outputs=[C],
                     rtol=1e-3, atol=1e-5))
```

### Compile-only mode
```python
backend = XRTBackend(verbose=args.verbose, omit_while_true_loop=True,
                     output_format=args.output_format)
module_function = backend.compile(mlir_module)
backend.unload()
```

## Build & Run

```bash
cd programming_examples/<example_name>
make run                    # compile and validate on NPU
make print                  # print generated MLIR only (no compilation)
make clean                  # remove build artifacts
```

### Complex examples (matrix_multiplication, softmax)
```bash
make profile                # profile with C++ test executable
make run4x4                 # specific herd configuration
make sweep4x4               # benchmarking sweep
make compile-kernel         # compile external C++ kernel only
make run PEANO_INSTALL_DIR=/path/to/peano  # explicit compiler path
```

### Key Makefile variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `PEANO_INSTALL_DIR` | auto | Peano compiler (determines `build_peano/` vs `build_chess/`) |
| `OUTPUT_FORMAT` | `xclbin` | Output: `xclbin` (hardware) or `elf` (direct codegen) |
| `AIE_TARGET` | example-specific | Architecture: `aie2` (NPU1) or `aie2p` (NPU2) |

### Build directory convention
```
build_peano/    # when PEANO_INSTALL_DIR is set
build_chess/    # when using Chess compiler
```

## LIT Test Integration

Tests use LLVM Integrated Tester. Files named `run_[variant]_[compiler].lit`.

```
// REQUIRES: ryzen_ai, peano
// RUN: mkdir -p test_peano
// RUN: cd test_peano && make -f %S/Makefile clean && make -f %S/Makefile run PEANO_INSTALL_DIR=%PEANO_INSTALL_DIR
// CHECK: PASS!
```

### REQUIRES tags
- `ryzen_ai` — any NPU hardware
- `ryzen_ai_npu1` — NPU1 (Phoenix) only
- `ryzen_ai_npu2` — NPU2 (Strix) only
- `peano` / `chess` — compiler backend

### Running LIT tests
```bash
# From build directory (after cmake)
cmake --build . --target check-programming-examples-peano
cmake --build . --target check-programming-examples        # all backends
```

### Operator Dashboard
Auto-generated from LIT files: `python3 programming_examples/generate_readme.py`

## Shared Infrastructure

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | LIT test suite discovery, defines `check-programming-examples-*` targets |
| `lit.cfg.py` | LIT config: detects hardware, sets PYTHONPATH, tool substitutions |
| `makefile-common` | Shared Vitis/AIE build variables and compiler flags |
| `generate_readme.py` | Auto-generates README.md operator dashboard from LIT files |

## Data Types

| NumPy | MLIR | Vector width (AIE2P) | Import |
|-------|------|---------------------|--------|
| `bfloat16` | `BF16Type.get()` | 16 lanes | `from ml_dtypes import bfloat16` |
| `np.float32` | `F32Type.get()` | 8 lanes | stdlib |
| `np.int32` | `T.i32()` | 8 lanes | stdlib |
| `np.int8` | `T.i8()` | 32 lanes | stdlib |

Type mapping: `xrt_dtype = type_mapper(np_dtype)` (from `air.backend.xrt_runner`)

## Complexity Tiers

**Simple** (single file, 1x1 herd): `passthrough_dma`, `eltwise_add`, `relu`
- One `.py` file generates full IR, no external kernels

**Medium** (external kernels or transform scripts): `softmax`, `silu`, `sine_cosine`
- Python IR + `.cc` kernel compiled with Peano/Chess
- Optional transform script for tiling

**Complex** (multi-variant, parameterized): `matrix_multiplication`, `flash_attention`, `ffn_swiglu`
- Multiple Python scripts per architecture variant
- Configurable tile sizes, herd shapes, sweep infrastructure
- C++ profiling harness

## LLAMA-3.2-1B Prefill on NPU2 (`llama3/`)

End-to-end LLAMA-3.2-1B BF16 prefill inference (seq_len=2048, 16 layers) on NPU2.

**Status**: Full NPU pipeline (FlashAttention + NPU LM Head + 8-tile RMSNorm). Top-1 = " Paris". **30% faster than IRON** (1.92s vs 2.744s total prefill). 5 XRT invocations/layer + 1 for LM Head. Decode: 351ms/token (5% faster than IRON's 370ms).

**Key files**:
- `llama3/llama3_prefill.py` — Main orchestrator: KernelCache + transformer block pipeline
- `llama3/llama3_weights.py` — Weight loading from HuggingFace safetensors + RoPE LUT
- `llama3/llama3_reference.py` — CPU F32 reference (per-step + full model)
- `llama3/multi_launch_builder/` — Multi-launch ELF builders:
  - `rms_attn_gemms_multi.py` — RMSNorm + QKV GEMMs (4 launches)
  - `rope_qk_multi.py` — RoPE Q+K (2 herds)
  - `o_proj_add_multi.py` — O GEMM + Add (2 launches)
  - `ffn_full_multi.py` — RMSNorm + FFN + Add (6 launches)
  - `lm_head_multi.py` — LM Head (8 launches)
- `llama3/ffn_swiglu/` — SiLU×mul kernel + FFN sub-module builder
- `llama3/docs/` — Documentation (plan, progress, performance, issues)

**Quick start** (requires cached kernels from prior `--compile-only` run):
```bash
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --profile
```

**Architecture**: 7 unique kernel configs compiled once via `KernelCache`: rms_attn_gemms (4-launch ELF), rope_qk (2-herd ELF), flash_attn (ELF), o_proj_add (2-launch ELF), ffn_full (6-launch ELF), lm_head (8-launch ELF), rmsnorm (xclbin). Uses `bo.map()` zero-copy and static weight BO pre-loading.

See `llama3/docs/LLAMA_PLAN.md` for full plan and `llama3/docs/LLAMA_progress.md` for session log.

## LLAMA-3.2-1B GEMM — Validated Parameters (NPU2, BF16, direct-codegen)

All LLAMA-3.2-1B GEMM shapes validated on NPU2 hardware with `--direct-codegen` mode.

### Tile configuration (conservative, fits L1=64KB, L2=256KB)

| Projection | M | K | N | tile_m | tile_k_l2 | tile_k_l1 | tile_n | herd_m | herd_n | Status |
|------------|---|---|---|--------|-----------|-----------|--------|--------|--------|--------|
| Q/O-proj | 128 | 2048 | 2048 | 32 | 64 | 32 | 32 | 4 | 4 | PASS |
| K/V-proj | 128 | 2048 | 512 | 32 | 64 | 32 | 32 | 4 | 4 | PASS |
| Gate/Up FFN | 128 | 2048 | 8192 | 32 | 64 | 32 | 32 | 4 | 4 | PASS |
| Down FFN | 128 | 8192 | 2048 | 32 | 64 | 32 | 32 | 4 | 4 | PASS |
| Q/O-proj | 64 | 2048 | 2048 | 32 | 64 | 32 | 32 | 2 | 4 | PASS |
| Q/O-proj | 256 | 2048 | 2048 | 32 | 64 | 32 | 32 | 4 | 4 | PASS |

### Memory budget per config above

- **L1 per tile**: A=32x32x2=2KB + B=32x32x2=2KB = 4KB (of 64KB)
- **L2 per segment**: A=4x32x64x2=16KB + B=4x64x32x2=16KB + C=4x4x32x32x2=32KB = 64KB (of 256KB)

### Key constraint: M=64 requires herd_m=2

For M=64 with tile_m=32: `launch_size[0] = 64 // (32 * herd_m)` must be >= 1, so `herd_m <= 2`.

### Run command template

```bash
python3 programming_examples/matrix_multiplication/bf16/run.py \
  --m M --k K --n N \
  --tile-m 32 --tile-k-l2 64 --tile-k-l1 32 --tile-n 32 \
  --herd-m 4 --herd-n 4 \
  --arch aie2p --compile-mode compile-and-run --direct-codegen
```
