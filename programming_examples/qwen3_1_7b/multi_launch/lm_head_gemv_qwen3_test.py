# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4 — LM head GEMV ELF at Qwen3-1.7B vocab=151936, emb_dim=1024.

Reuses the existing `llama3.multi_launch_builder.lm_head_gemv_multi`
unchanged. Qwen3-1.7B and qwen25_1_5b share vocab=151936 (10 partitions ×
16384 padded), so the only change vs qwen25 is emb_dim 1536 → 1024.

Tile config from Step 1 finding: M=16384 needs tile_m=16, m_input=16, herd_m=8
(default 8/4/8 → 256 outer iters → AIE2P shim repeat>255 hard fail).

Verifies one partition's NPU output matches `weight_partition @ x` within
BF16 cosine > 0.99.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_DEPLOY_DIR = _THIS_DIR.parent
_EXAMPLES = _DEPLOY_DIR.parent
for _p in (
    str(_EXAMPLES),
    str(_EXAMPLES / "llama3"),
    str(_EXAMPLES / "matrix_vector_multiplication" / "bf16"),
    str(_EXAMPLES / "weighted_rms_norm"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama3.multi_launch_builder.lm_head_gemv_multi import build_lm_head_gemv_module
from air.backend.xrt import XRTBackend
import filelock
import pyxrt as xrt


def main():
    EMB_DIM = 1024
    N_PART = 16384
    N_PARTITIONS = 10
    VOCAB = 151936  # padded total = N_PART * N_PARTITIONS = 163840

    print(
        f"Qwen3 LM head GEMV ELF: vocab={VOCAB} (padded to {N_PART*N_PARTITIONS}), "
        f"emb_dim={EMB_DIM}, {N_PARTITIONS} × {N_PART}"
    )

    print("\nBuilding module...")
    module = build_lm_head_gemv_module(
        emb_dim=EMB_DIM,
        n_partitions=N_PARTITIONS,
        n_part=N_PART,
        tile_m=16,
        m_input=16,
        herd_m=8,  # Step 1 finding: required for M=16384
    )

    print("\nCompiling...")
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
        runtime_loop_tiling_sizes=[4, 4],
        use_lock_race_condition_fix=True,
        output_format="elf",
        instance_name="lm_head_gemv",
    )
    t0 = time.time()
    artifact = backend.compile(module)
    print(f"  Compile: {time.time()-t0:.1f}s")

    # --- Random inputs ---
    np.random.seed(7)
    x = (np.random.randn(EMB_DIM) * 1.0).astype(bfloat16)
    weights = [
        (np.random.randn(N_PART, EMB_DIM) * 0.02).astype(bfloat16)
        for _ in range(N_PARTITIONS)
    ]
    outputs = [np.zeros(N_PART, dtype=bfloat16) for _ in range(N_PARTITIONS)]

    # CPU reference for each partition
    expected = [
        (W.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16) for W in weights
    ]

    # --- Run on NPU ---
    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)

    inputs_all = [x]
    for W, O in zip(weights, outputs):
        inputs_all.append(W)
        inputs_all.append(O)
    sizes = [a.size * a.itemsize for a in inputs_all]
    bos = [xrt.ext.bo(backend.device, s) for s in sizes]
    for i, a in enumerate(inputs_all):
        bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
        bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = xrt.run(backend.kernel)
    for i, bo in enumerate(bos):
        run.set_arg(i, bo)
    t1 = time.time()
    run.start()
    run.wait2()
    print(f"\n  NPU run: {(time.time()-t1)*1000:.2f} ms ({N_PARTITIONS} partitions)")

    # --- Read outputs back ---
    for p in range(N_PARTITIONS):
        idx = 2 + 2 * p
        bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _cos(a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    print("\nPer-partition correctness:")
    all_pass = True
    for p in range(N_PARTITIONS):
        idx = 2 + 2 * p
        npu_out = (
            bos[idx].read(sizes[idx], 0).view(np.int16).view(bfloat16).reshape(N_PART)
        )
        c = _cos(npu_out, expected[p])
        marker = "PASS" if c > 0.99 else "FAIL"
        print(f"  [{marker}] partition {p}  cosine={c:.6f}")
        if c <= 0.99:
            all_pass = False

    backend.unload()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
