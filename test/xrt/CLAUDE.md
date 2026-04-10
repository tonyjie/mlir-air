# test/xrt/ — Compiler Pass End-to-End Tests

Tests that validate the full compilation pipeline from Linalg MLIR through transform scripts to NPU execution via XRT.

## Directory Structure

52+ numbered test directories (`01_air_to_npu` through `51_scf_if_channel_herd`). Each contains:

| File | Purpose |
|------|---------|
| `run.py` | Python driver: loads IR, applies transforms, runs passes, executes on NPU |
| `transform_aie2p.mlir` | Transform script for NPU2 (Strix, AIE2P) |
| `transform_aie2.mlir` | Transform script for NPU1 (Phoenix, AIE2) — not all tests |
| `Makefile` | Build targets: `run`, `compile-xclbin`, `profile`, `debug-aircc` |
| `input_ir/` | Input MLIR files (some tests embed IR in `run.py` instead) |
| `extern_func.cc` | External C++ kernel (NPU1 tests needing math intrinsics) |
| `test.cpp` | C++ XRT profiling harness |
| `run_npu2_peano.lit` | LIT test spec: `REQUIRES`, `RUN`, `CHECK` directives |

## Compilation Flow (per test)

```
input MLIR (file or embedded)
    |  [1] transform script → air-opt --air-transform
    v
Tiled + vectorized AIR MLIR
    |  [2] run.py passes → memory space override, parallel wrapping, copy-to-dma
    v
    |  [3] XRTRunner/XRTBackend → aircc.py → aiecc.py → xclbin
    v
Validation (numpy comparison with rtol/atol)
```

## run.py Patterns

### Imports
```python
import argparse
import numpy as np
from ml_dtypes import bfloat16
from air.ir import *
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.compiler.util import run_transform
import air.passmanager
```

### Two IR source patterns
1. **External file**: `Module.parse(open(args.input_mlir).read())`
2. **Embedded string**: MLIR IR defined as Python string literal in `run.py`

### Standard pass pipeline (after transform script)
```python
pipeline = (
    "builtin.module("
    + ",".join([
        "air-override-memref-memory-space{scope=func memory-space=1}",
        "func.func(air-wrap-func-with-parallel{loop-bounds=M,N,K})",
        "air-par-to-launch{depth=-1 has-air-segment=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline)
pm.run(air_module.operation)
```

### Execution via XRTRunner
```python
runner = XRTRunner(verbose=args.verbose, debug_ir=args.debug_aircc,
                   output_format=args.output_format, instance_name="test_name")
exit(runner.run_test(air_module, inputs=[A, B], expected_outputs=[C],
                     rtol=0.04, atol=0.001))
```

## Transform Script Development

Transform scripts define how high-level operations map to AIE hardware. Located at `transform_aie2p.mlir` (NPU2) or `transform_aie2.mlir` (NPU1).

### Key operations
- Tile for AIE sizes, allocate buffers to L1/L2
- Vectorize for AIE vector units (16-lane bf16, 32-lane int8)
- Create herds and DMAs

### Typical phases
1. Canonicalization + fold unit-extent dims
2. Operation fusion (`fuse_elementwise_linalg`, `fuse_multi_op_linalg`)
3. Tiling with `tile_using_forall` (parallel) or `tile_using_for` (sequential)
4. Buffer allocation: `bufferize_to_allocation {memory_space = 1}` (L2) or `{memory_space = 2}` (L1)
5. Herd conversion: `forall_to_parallel` → `par_to_herd`
6. Copy-to-DMA: `air.copy_to_dma`
7. Vectorization: `herd_vectorize` + `vector_type_cast`

### Patterns
- Tile outermost dimensions first for data locality
- L2 (`memory_space=1`) for intermediate buffers, L1 (`memory_space=2`) for innermost computation
- AIE2P: 16 lanes for bf16, 32 lanes for int8/int4

### Pitfalls
- Ping-pong buffers must have matching tile sizes between producer and consumer
- Too aggressive channel fusion can cause routing congestion
- ObjectFifo sizing must account for double-buffering
- AIE only supports 1D vectors; transforms producing 2D vectors will fail at backend

### Advanced techniques (matmul-style)
- `transform.structured.pack` + `pack_transpose` for data layout
- `loop.fuse_sibling` for L2 pingpong buffering
- Unrolling with `factor=2` for register blocking
- Hoisting: `hoist_loop_invariant_transfers`, `hoist_cast_pair`, `hoist_vector_transfer_pointers`

## Build & Run

```bash
cd test/xrt/<test_name>
make run                          # build and run on NPU (default: npu2)
make run TARGET=npu1              # run on NPU1
make run DEBUG_AIRCC=1            # per-pass IR output → build_peano/air_project/debug_ir/
make compile-xclbin               # compile only, no validation
make profile                      # profile with C++ test executable
make run M=512 N=1024             # custom dimensions
make run-pretransform-baseline    # skip transform, use pre-transformed IR
make clean                        # remove build artifacts
```

### Key Makefile variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `TARGET` | `npu2` | Hardware target: `npu1` or `npu2` |
| `M`, `N`, `K` | test-specific | Problem dimensions |
| `VERBOSE` | `0` | Verbose output |
| `DEBUG_AIRCC` | `0` | Emit per-pass IR to `debug_ir/` |
| `PEANO_INSTALL_DIR` | auto | Peano compiler path (determines `build_peano/` vs `build_chess/`) |

## LIT Test Files

Naming: `run_[target]_[compiler].lit` (e.g., `run_npu2_peano.lit`)

```
// REQUIRES: ryzen_ai_npu2, peano
// RUN: mkdir -p test_npu2_peano
// RUN: cd test_npu2_peano && %python %S/run.py --transform-script %S/transform_aie2p.mlir ...
// CHECK: PASS!
```

## Key Intermediate Files

After compilation, look in `build_peano/air_project/`:
- `placed.air.mlir` — after placement pass
- `aie.air.mlir` — after AIR-to-AIE lowering
- `npu.air.mlir` — after NPU instruction generation
- `input_physical.mlir` — physical AIE IR
- `air.xclbin` + `air.insts.bin` — final binaries
- `debug_ir/pass_XXX_after_<passname>.mlir` — per-pass IR (with `DEBUG_AIRCC=1`)
