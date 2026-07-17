"""Benchmark: NPU backbone inference latency vs CPU backbone (same 16-layer math).

Apples-to-apples INFERENCE latency for the SmolLM2-360M backbone at seq=256:
- compiles NPU kernels ONCE (excluded from timing)
- times N NPU run_backbone_prefill calls (cpu_attn=False, full NPU attention)
- times N cpu_backbone_forward calls (pure-numpy fp32, the verified reference)

After the Phase-4 GQA-group batching optimization (commit f404998e), NPU
attention issues ~11 XRT dispatches/layer (5 QKt + 1 softmax + 5 PV; was
31/layer before batching) x 16 layers. Reports the split so remaining
fusion headroom (Route 3: fold attention into the per-layer fused ELF) is
explicit. Does NOT include one-time kernel compile (~min) or the
two-process npz bridge — those are deployment-harness costs, not inference cost.

Run under the NPU lock, worktree python:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python bench_backbone.py --iters 5
"""

import argparse
import time

import numpy as np
from ml_dtypes import bfloat16

from smolvla_backbone_weights import (
    load_backbone_weights,
    SmolVLABackboneConfig,
    generate_rope_lut,
)
from smolvla_cpu_helpers import build_prefix_mask, cpu_backbone_forward
from smolvla_backbone_prefill import compile_all_kernels, run_backbone_prefill
from shared.infra.cache import KernelCache, Profiler

NPU_SEQ = 256
ORACLE_LEN = 241


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    cfg = SmolVLABackboneConfig()
    w = load_backbone_weights("lerobot/smolvla_base", config=cfg)

    # prefix embed: use the oracle fixture if present, else random (timing is
    # data-independent for a fixed shape).
    try:
        o = np.load("smolvla_oracle.npz")
        prefix = o["prefix_embed"].astype(np.float32)  # (241,960)
    except FileNotFoundError:
        prefix = (
            np.random.default_rng(0)
            .standard_normal((ORACLE_LEN, cfg.emb_dim))
            .astype(np.float32)
        )

    # pad to 256
    x_f32 = np.zeros((NPU_SEQ, cfg.emb_dim), np.float32)
    x_f32[:ORACLE_LEN] = prefix
    x_bf16 = x_f32.astype(bfloat16)

    # masks/positions at 256
    mask = np.full((NPU_SEQ, NPU_SEQ), -np.inf, np.float32)
    mask[:ORACLE_LEN, :ORACLE_LEN] = (
        0.0  # (bench: treat all 241 as bidirectional prefix)
    )
    positions = np.arange(NPU_SEQ)
    full_lut = generate_rope_lut(cfg, seq_len=NPU_SEQ, dtype=bfloat16)
    rope_lut = np.asarray(full_lut)[positions]

    # CPU-side mask/positions (241)
    cpu_mask = build_prefix_mask(n_prefix=ORACLE_LEN - 1, n_state=1)

    # ---- NPU: compile ONCE (excluded from timing) ----
    print("[bench] compiling NPU kernels (one-time, excluded from timing)...")
    t_compile0 = time.perf_counter()
    cache = KernelCache("bench_kernel_cache", verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, NPU_SEQ, cpu_attn=False)
    t_compile = time.perf_counter() - t_compile0
    print(f"[bench] compile took {t_compile:.1f}s (one-time)")

    # warmup (first run pays BO-alloc etc.)
    run_backbone_prefill(
        x_bf16,
        w,
        cfg,
        cache,
        mask,
        positions,
        rope_lut,
        cpu_attn=False,
        verbose=False,
        return_kv=False,
    )

    npu_times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        run_backbone_prefill(
            x_bf16,
            w,
            cfg,
            cache,
            mask,
            positions,
            rope_lut,
            cpu_attn=False,
            verbose=False,
            return_kv=False,
        )
        npu_times.append(time.perf_counter() - t0)
    npu = np.array(npu_times)

    # ---- CPU: same 16-layer backbone (pure numpy fp32) ----
    # warmup
    cpu_backbone_forward(
        prefix,
        w,
        cfg,
        cpu_mask,
        rope_base=cfg.rope_base,
        positions=np.arange(ORACLE_LEN),
    )
    cpu_times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        cpu_backbone_forward(
            prefix,
            w,
            cfg,
            cpu_mask,
            rope_base=cfg.rope_base,
            positions=np.arange(ORACLE_LEN),
        )
        cpu_times.append(time.perf_counter() - t0)
    cpu = np.array(cpu_times)

    print("\n=== SmolVLA backbone inference latency (16 layers, seq=256/241) ===")
    print(f"  iters = {args.iters}")
    print(
        f"  NPU backbone (cpu_attn=False):  median {np.median(npu)*1e3:8.1f} ms  "
        f"[min {npu.min()*1e3:.1f}, max {npu.max()*1e3:.1f}]"
    )
    print(
        f"  CPU backbone (numpy fp32):      median {np.median(cpu)*1e3:8.1f} ms  "
        f"[min {cpu.min()*1e3:.1f}, max {cpu.max()*1e3:.1f}]"
    )
    ratio = np.median(npu) / np.median(cpu)
    print(
        f"  ratio NPU/CPU = {ratio:.2f}x  ({'NPU slower' if ratio>1 else 'NPU faster'})"
    )
    print(
        f"\n  one-time NPU compile = {t_compile:.1f}s (NOT in the per-inference numbers)"
    )
    print(
        "  NOTE: ~11 dispatches/layer x 16 = ~176 (after GQA-group batching; was 31/layer)."
    )
    print(
        "  Route 3 (fuse attention into the per-layer fused ELF) is the remaining path to cut dispatch overhead further."
    )


if __name__ == "__main__":
    main()
