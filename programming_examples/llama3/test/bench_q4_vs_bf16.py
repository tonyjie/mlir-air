#!/usr/bin/env python3
"""Benchmark Q4 vs BF16 standalone GEMV on NPU.

Run from build_peano/:
  python3 ../test/bench_q4_vs_bf16.py
"""

import sys
import os
import time
import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import filelock
from air.backend.xrt import XRTBackend

from matrix_vector_multiplication.bf16.matvec import build_module as build_bf16
from matrix_vector_multiplication.q4.matvec_q4 import build_module as build_q4
from llama3.kernel_builder.quantize import pack_q4_interleaved

configs = [
    ("Q/O GEMV", 2048, 2048, 8, 4, 8),
    ("Gate/Up", 8192, 2048, 8, 4, 8),
    ("Down (K=8K)", 2048, 8192, 2, 1, 8),
]

N_WARMUP = 5
N_RUNS = 20


def bench_kernel(build_fn, M, K, tile_m, m_input, herd_m, inputs, instance):
    module = build_fn(M, K, tile_m, m_input, herd_m)
    backend = XRTBackend(
        omit_while_true_loop=False, output_format="elf", instance_name=instance
    )
    compiled = backend.compile(module)
    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(compiled)

    for _ in range(N_WARMUP):
        invoker(*inputs)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        invoker(*inputs)
        times.append((time.perf_counter() - t0) * 1000)

    backend.unload()
    return np.median(times), np.min(times), np.max(times)


print(
    f"{'Kernel':<20s} {'BF16 (ms)':>10s} {'Q4 (ms)':>10s} {'Speedup':>8s} {'BW Reduction':>14s}"
)
print("-" * 70)

for name, M, K, tile_m, m_input, herd_m in configs:
    np.random.seed(42)
    w_bf16 = np.random.uniform(-1, 1, (M, K)).astype(bfloat16)
    x_bf16 = np.random.uniform(-1, 1, K).astype(bfloat16)

    # BF16
    bf16_inputs = [w_bf16, x_bf16, np.zeros(M, dtype=bfloat16)]
    bf16_med, bf16_min, bf16_max = bench_kernel(
        lambda m, k, tm, mi, hm: build_bf16(m, k, tm, mi, hm, bfloat16, bfloat16),
        M,
        K,
        tile_m,
        m_input,
        herd_m,
        bf16_inputs,
        "matvec_bf16",
    )

    # Q4
    w_packed = pack_q4_interleaved(w_bf16)
    q4_inputs = [w_packed.flatten(), x_bf16, np.zeros(M, dtype=bfloat16)]
    q4_med, q4_min, q4_max = bench_kernel(
        build_q4,
        M,
        K,
        tile_m,
        m_input,
        herd_m,
        q4_inputs,
        "q4_matvec",
    )

    speedup = bf16_med / q4_med
    bw_ratio = (M * K * 2) / w_packed.nbytes
    print(
        f"{name:<20s} {bf16_med:>8.2f}ms {q4_med:>8.2f}ms {speedup:>7.2f}x {bw_ratio:>12.1f}x"
    )

print()
