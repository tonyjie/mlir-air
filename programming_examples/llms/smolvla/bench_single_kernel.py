"""EXPERIMENT: single-kernel NPU vs CPU latency/GFLOPS at SmolVLA's small shapes.

Question: at SmolVLA's small backbone shapes (seq=256, emb=960 -- much
smaller than llama's seq=2048, emb=2048), is a single isolated NPU kernel
actually faster than the same op on CPU (PyTorch)? This measurement explains
why the whole NPU backbone (~217ms approach-B / ~105ms FlashAttention path)
ends up slower than the pure-PyTorch CPU backbone (56.8ms) -- see
bench_backbone.py for the full-backbone comparison; this script isolates one
GEMM and one FlashAttention dispatch to see whether the NPU is even winning
at the kernel level.

Two experiments, both using the exact SmolVLA-validated NPU configs from the
kernel_registry (kernel_registry/details/GEMM_bf16_in_bf16_out.md's "256x960x960
SmolVLA Q/O proj" row and FlashAttention_bf16.md's SmolVLA non-causal
15Q/5KV row, reused verbatim from smolvla_backbone_prefill.py's
_compile_flash_attention_kernel):

  1. GEMM  256x960x960 (SmolVLA Q/O proj shape), bf16-in/bf16-out, drain
     method, tile_m=32/tile_k_l2=320/tile_k_l1=32/tile_n=80/herd_m=8/herd_n=4.
  2. FlashAttention lq=lk=256, 15Q/5KV heads, head_dim=64, non-causal,
     num_heads_per_unroll=1 (15 has no divisor <=2 other than 1).

NPU numbers are collected by shelling out to the already-existing, already-
validated standalone harnesses (matrix_multiplication/bf16_in_bf16_out/run.py
and flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py) with their
built-in `--perf-iters N` XRTRunner timing path: N=20 kernel dispatches timed
after 10 warmup iterations, kernel-launch-to-wait only (host<->device BO sync
excluded, matching how every kernel_registry GFLOPS number was collected).

CPU numbers use PyTorch (torch.matmul / scaled_dot_product_attention) on CPU
in bf16 and fp32, median of 20 dispatches after 5 warmup -- run in the
lerobot venv (~/Projects/smolvla_playground/.venv/bin/python) since this
worktree's default python has air/pyxrt but not torch.

KNOWN CAVEAT (documented, not fabricated away): the FlashAttention harness's
`--perf-iters` timing loop reissues the *same* kernel dispatch back-to-back
without resetting the cascade-stage accumulator state between launches. This
produces a WRONG final numerical result (confirmed: fails the correctness
check both at SmolVLA's hpu=1 config and at a control MHA/GQA hpu=2 config),
but does not change the per-dispatch instruction stream, so the *latency*
number is still valid for the isolated-single-dispatch performance question
this script asks -- it is not valid as a "run FA in a tight loop in
production" recipe. A single (perf-iters=0) FA dispatch at this shape passes
correctness (mean_rel_L1 ~4.3e-2, consistent with the registry's non-causal
FA rows). See run_experiment_fa() and the printed report for detail.

Usage (worktree default python has air/pyxrt/numpy; NPU steps need the lock):
  flock -x -w 1800 /tmp/mlir-air-npu.lock python bench_single_kernel.py
  python bench_single_kernel.py --skip-npu     # CPU-only (no NPU/lock needed)
  python bench_single_kernel.py --skip-cpu     # NPU-only
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROG_EXAMPLES = _HERE.parent.parent  # programming_examples/
_GEMM_DIR = _PROG_EXAMPLES / "matrix_multiplication" / "bf16_in_bf16_out"
_FA_DIR = _PROG_EXAMPLES / "flash_attention" / "kernel_fusion_based"
_LEROBOT_VENV_PY = (
    Path.home() / "Projects" / "smolvla_playground" / ".venv" / "bin" / "python"
)

NPU_LOCK = ["flock", "-x", "-w", "1800", "/tmp/mlir-air-npu.lock"]

# ---------------------------------------------------------------------------
# Experiment 1: GEMM 256x960x960 (SmolVLA Q/O proj)
# ---------------------------------------------------------------------------
GEMM_M, GEMM_K, GEMM_N = 256, 960, 960
# Registry-validated tile config for this shape (GEMM_bf16_in_bf16_out.md,
# "high-precision, --method drain" table, "SmolVLA Q/O proj" row).
GEMM_TILE = dict(tile_m=32, tile_k_l2=320, tile_k_l1=32, tile_n=80, herd_m=8, herd_n=4)

# ---------------------------------------------------------------------------
# Experiment 2: FlashAttention lq=lk=256, 15Q/5KV, head_dim=64, non-causal
# ---------------------------------------------------------------------------
FA_SEQ, FA_HEAD_DIM, FA_N_HEADS, FA_N_KV_HEADS = 256, 64, 15, 5
FA_NUM_Q_TILES, FA_NUM_CASCADE_STAGES, FA_HPU = 4, 4, 1

_LATENCY_RE = re.compile(
    r"Latency \(us\):\s*([\d.]+)\s*\|\s*Throughput:\s*([\d.eE+-]+)\s*GFLOP/s"
)


def _run_npu(cmd, cwd, label):
    print(f"\n--- NPU: {label} ---")
    print("  $", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=900
    )
    out = proc.stdout + "\n" + proc.stderr
    m = _LATENCY_RE.search(out)
    if not m:
        print(out[-4000:])
        raise RuntimeError(
            f"Could not find 'Latency (us): ... GFLOP/s' in {label} output"
        )
    latency_us, gflops = float(m.group(1)), float(m.group(2))
    # Surface the correctness line too (informational -- see module docstring
    # caveat for why FA's may legitimately FAIL under perf-iters repetition).
    for line in out.splitlines():
        if "[precision]" in line or line.strip() in ("PASS!", "failed."):
            print(" ", line.strip())
    print(f"  Latency: {latency_us:.1f} us | Throughput: {gflops:.1f} GFLOP/s")
    return latency_us, gflops


def run_experiment_gemm_npu():
    cmd = NPU_LOCK + [
        "make",
        "run",
        f"TILE_M={GEMM_TILE['tile_m']}",
        f"TILE_K_L2={GEMM_TILE['tile_k_l2']}",
        f"TILE_K_L1={GEMM_TILE['tile_k_l1']}",
        f"TILE_N={GEMM_TILE['tile_n']}",
        f"HERD_M={GEMM_TILE['herd_m']}",
        f"HERD_N={GEMM_TILE['herd_n']}",
        f"M={GEMM_M}",
        f"K={GEMM_K}",
        f"N={GEMM_N}",
        "METHOD=drain",
        "PERF_ITERS=20",
    ]
    latency_us, gflops = _run_npu(
        cmd, _GEMM_DIR, f"GEMM {GEMM_M}x{GEMM_K}x{GEMM_N} (drain)"
    )
    return {"latency_us": latency_us, "gflops": gflops}


def run_experiment_fa_npu():
    extra = (
        f"--num-heads-per-unroll {FA_HPU} --num-cascade-stages {FA_NUM_CASCADE_STAGES} "
        f"--num-q-tiles {FA_NUM_Q_TILES} --perf-iters 20"
    )
    cmd = NPU_LOCK + [
        "make",
        "run",
        "SCRIPT=attn_npu2_seqfirst.py",
        f"LK={FA_SEQ}",
        f"LKP={FA_HEAD_DIM}",
        f"LQ={FA_SEQ}",
        f"LQP={FA_SEQ}",
        f"DK={FA_HEAD_DIM}",
        f"DV={FA_HEAD_DIM}",
        f"NUM_HEADS={FA_N_HEADS}",
        f"NUM_KV_HEADS={FA_N_KV_HEADS}",
        f"EXTRA_PY_FLAGS={extra}",
    ]
    latency_us, gflops = _run_npu(
        cmd,
        _FA_DIR,
        f"FlashAttention {FA_SEQ}x{FA_SEQ} ({FA_N_HEADS}q/{FA_N_KV_HEADS}kv, non-causal)",
    )
    return {"latency_us": latency_us, "gflops": gflops}


# ---------------------------------------------------------------------------
# CPU (PyTorch) side -- run in the lerobot venv, which has torch but not air.
# ---------------------------------------------------------------------------
_CPU_GEMM_SCRIPT = f"""
import json, time
import torch
torch.manual_seed(0)

def bench(dtype, iters=20, warmup=5):
    a = torch.randn({GEMM_M}, {GEMM_K}, dtype=torch.float32)
    b = torch.randn({GEMM_K}, {GEMM_N}, dtype=torch.float32)
    if dtype == torch.bfloat16:
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    for _ in range(warmup):
        torch.matmul(a, b)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    flops = 2.0 * {GEMM_M} * {GEMM_K} * {GEMM_N}
    return med * 1e6, flops / med / 1e9  # us, GFLOPS

out = {{}}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    us, gflops = bench(dt)
    out[name] = {{"latency_us": us, "gflops": gflops}}
out["torch_threads"] = torch.get_num_threads()
print(json.dumps(out))
"""

_CPU_FA_SCRIPT = f"""
import json, time
import torch
import torch.nn.functional as F
torch.manual_seed(0)

num_heads, num_kv, seq, hd = {FA_N_HEADS}, {FA_N_KV_HEADS}, {FA_SEQ}, {FA_HEAD_DIM}
group = num_heads // num_kv

def bench(dtype, iters=20, warmup=5):
    q = torch.randn(1, num_heads, seq, hd, dtype=dtype)
    k = torch.randn(1, num_kv, seq, hd, dtype=dtype)
    v = torch.randn(1, num_kv, seq, hd, dtype=dtype)
    k_exp = k.repeat_interleave(group, dim=1)
    v_exp = v.repeat_interleave(group, dim=1)
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    flops = 2.0 * num_heads * seq * seq * (hd + hd)  # non-causal, full (not halved)
    return med * 1e6, flops / med / 1e9  # us, GFLOPS

out = {{}}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    us, gflops = bench(dt)
    out[name] = {{"latency_us": us, "gflops": gflops}}
out["torch_threads"] = torch.get_num_threads()
print(json.dumps(out))
"""


def _run_cpu(script, label):
    print(f"\n--- CPU (PyTorch, {_LEROBOT_VENV_PY}): {label} ---")
    if not _LEROBOT_VENV_PY.exists():
        raise RuntimeError(f"lerobot venv python not found at {_LEROBOT_VENV_PY}")
    proc = subprocess.run(
        [str(_LEROBOT_VENV_PY), "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"CPU bench failed for {label}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    for prec in ("bf16", "fp32"):
        r = result[prec]
        print(f"  {prec}: {r['latency_us']:.1f} us | {r['gflops']:.1f} GFLOPS")
    return result


def run_experiment_gemm_cpu():
    return _run_cpu(_CPU_GEMM_SCRIPT, f"GEMM {GEMM_M}x{GEMM_K}x{GEMM_N}")


def run_experiment_fa_cpu():
    return _run_cpu(
        _CPU_FA_SCRIPT,
        f"SDPA {FA_SEQ}x{FA_SEQ} ({FA_N_HEADS}q/{FA_N_KV_HEADS}kv, non-causal)",
    )


def print_table(gemm_npu, gemm_cpu, fa_npu, fa_cpu):
    print(f"\n{'='*100}")
    print("SUMMARY: single-kernel NPU vs CPU latency/GFLOPS at SmolVLA's small shapes")
    print(f"{'='*100}\n")

    def row(op, shape, npu, cpu_bf16, cpu_fp32):
        npu_s = (
            f"{npu['latency_us']:.1f}us / {npu['gflops']:.1f} GFLOPS" if npu else "n/a"
        )
        bf16_s = (
            f"{cpu_bf16['latency_us']:.1f}us / {cpu_bf16['gflops']:.1f} GFLOPS"
            if cpu_bf16
            else "n/a"
        )
        fp32_s = (
            f"{cpu_fp32['latency_us']:.1f}us / {cpu_fp32['gflops']:.1f} GFLOPS"
            if cpu_fp32
            else "n/a"
        )
        faster = "n/a"
        if npu and cpu_bf16:
            faster = "NPU" if npu["gflops"] > cpu_bf16["gflops"] else "CPU(bf16)"
        print(
            f"| {op:22s} | {shape:24s} | {npu_s:26s} | {bf16_s:26s} | {fp32_s:26s} | {faster:10s} |"
        )

    print(
        f"| {'op':22s} | {'shape':24s} | {'NPU':26s} | {'CPU bf16':26s} | {'CPU fp32':26s} | {'NPU faster?':10s} |"
    )
    print(
        "|"
        + "-" * 24
        + "|"
        + "-" * 26
        + "|"
        + "-" * 28
        + "|"
        + "-" * 28
        + "|"
        + "-" * 12
        + "|"
    )
    row(
        "GEMM",
        f"{GEMM_M}x{GEMM_K}x{GEMM_N}",
        gemm_npu,
        gemm_cpu["bf16"] if gemm_cpu else None,
        gemm_cpu["fp32"] if gemm_cpu else None,
    )
    row(
        "FlashAttention",
        f"{FA_SEQ}x{FA_SEQ} ({FA_N_HEADS}/{FA_N_KV_HEADS}/{FA_HEAD_DIM})",
        fa_npu,
        fa_cpu["bf16"] if fa_cpu else None,
        fa_cpu["fp32"] if fa_cpu else None,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-npu", action="store_true", help="skip NPU runs (CPU-only)")
    ap.add_argument(
        "--skip-cpu", action="store_true", help="skip CPU (torch) runs (NPU-only)"
    )
    ap.add_argument(
        "--registry-fallback",
        action="store_true",
        help="if the NPU GEMM run fails/can't isolate, cite the registry's 1896 GFLOPS number "
        "instead of running it live",
    )
    args = ap.parse_args()

    gemm_npu = fa_npu = gemm_cpu = fa_cpu = None

    if not args.skip_npu:
        if args.registry_fallback:
            latency_us = 2.0 * GEMM_M * GEMM_K * GEMM_N / 1896e9 * 1e6
            gemm_npu = {"latency_us": latency_us, "gflops": 1896.0}
            print(f"\n--- NPU: GEMM (registry citation, not measured live) ---")
            print(f"  Latency: {latency_us:.1f} us | Throughput: 1896.0 GFLOP/s")
        else:
            gemm_npu = run_experiment_gemm_npu()
        fa_npu = run_experiment_fa_npu()

    if not args.skip_cpu:
        gemm_cpu = run_experiment_gemm_cpu()
        fa_cpu = run_experiment_fa_cpu()

    if gemm_npu or fa_npu or gemm_cpu or fa_cpu:
        print_table(gemm_npu, gemm_cpu, fa_npu, fa_cpu)


if __name__ == "__main__":
    sys.exit(main())
