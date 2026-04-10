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

## LLAMA-3.2-1B on NPU2 (`llama3/`)

End-to-end LLAMA-3.2-1B BF16 inference (prefill + decode) on NPU2. Prefill: 1.30s kernel / 1.54s wall (2.1x faster than IRON). Decode: 92ms/token (4.0x faster than IRON).

```bash
cd programming_examples/llama3
make compile && make run
```

See `llama3/CLAUDE.md` for full architecture, file map, design patterns, and GEMM tile configs.
