# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A3-5 Step 5: TRUE end-to-end SmolVLA latency, three configs, same input+noise.

Measures the wall time of one full `predict_action_chunk` (vision -> connector ->
prefix -> backbone prefill -> 10-step action-expert denoise -> (1,50,6) chunk):

  1. cpu       : pure lerobot on CPU (the real baseline, nothing hooked).
  2. backbone  : hybrid, NPU 16-layer backbone prefill, CPU vision.
  3. vision+bb : hybrid, NPU SigLIP ViT + connector AND NPU backbone.

Configs 2 and 3 pay a SUBPROCESS BRIDGE per NPU stage: the lerobot venv (torch,
no air) and the worktree python (air, no torch) are disjoint, so every NPU stage
costs a process spawn + interpreter/import startup + a safetensors weight load +
XRT context/ELF load + npz round-trip, on EVERY inference. None of that is
inherent to running on the NPU — a single-process deployment pays it once at
startup. So each config is reported twice:

  as measured   : the honest wall clock of predict_action_chunk today.
  compute-only  : with each bridge's non-compute overhead subtracted, i.e. what
                  the same NPU work would cost inside one process. The bridges
                  self-report their phases (bridge_common.Timings) so this is
                  measured, not modelled.

`make bench-e2e` runs this under the NPU lock. Note the Makefile self-locks —
do NOT wrap it in another flock.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

from smolvla_inference import (  # noqa: E402
    run_hybrid_forward,
    build_oracle_batch,
    normalized_mse,
    _fixed_noise,
    DEFAULT_MODEL,
)

# The bridge phase that is REAL NPU work, per stage. Everything else the
# subprocess reports (npz_read, weight_load, kernels/ELF load) plus the spawn
# gap is bridge overhead.
_COMPUTE_PHASE = {"vision": "t_encode_ms", "backbone": "t_prefill_ms"}


def median_ms(xs):
    return statistics.median(xs) * 1e3


def _bridge_split(timings):
    """-> (compute_ms, overhead_ms) summed over all bridges of one inference."""
    compute = 0.0
    overhead = 0.0
    for stage, rec in timings.items():
        c = float(rec.get(_COMPUTE_PHASE[stage], 0.0))
        compute += c
        overhead += float(rec["wall_ms"]) - c
    return compute, overhead


def run_config(name, fn, iters, warmup=1):
    """Time `fn()` (returns (chunk, timings_dict)) `iters` times after `warmup`."""
    print(f"\n[bench] --- config '{name}' (warmup {warmup}, iters {iters}) ---")
    chunk = None
    for _ in range(warmup):
        chunk, _ = fn()
    walls, computes, overheads, per_stage = [], [], [], []
    for i in range(iters):
        t0 = time.perf_counter()
        chunk, timings = fn()
        walls.append(time.perf_counter() - t0)
        c, o = _bridge_split(timings)
        computes.append(c)
        overheads.append(o)
        per_stage.append(timings)
        print(
            f"[bench]   iter {i}: wall {walls[-1]*1e3:8.1f} ms  "
            f"(bridge compute {c:7.1f} ms, bridge overhead {o:7.1f} ms)"
        )
    return {
        "name": name,
        "wall_ms": median_ms(walls),
        "bridge_compute_ms": statistics.median(computes) if computes else 0.0,
        "bridge_overhead_ms": statistics.median(overheads) if overheads else 0.0,
        "chunk": chunk,
        "stages": per_stage[-1] if per_stage else {},
    }


def cpu_stage_profile(policy, batch, noise):
    """One instrumented pure-CPU run: per-stage wall time via module hooks.

    Informational context for the table (how the ~1s CPU baseline splits), not
    part of any median. Vision is summed over all 3 camera calls."""
    vwe = policy.model.vlm_with_expert
    vm = vwe.get_vlm_model().vision_model
    conn = vwe.get_vlm_model().connector
    acc = {"vision": 0.0, "connector": 0.0, "backbone_prefill": 0.0, "calls": 0}
    marks = {}
    hooks = [
        vm.register_forward_pre_hook(
            lambda m, a: marks.__setitem__("vm", time.perf_counter())
        ),
        vm.register_forward_hook(
            lambda m, i, o: acc.__setitem__(
                "vision", acc["vision"] + (time.perf_counter() - marks["vm"])
            )
        ),
        conn.register_forward_pre_hook(
            lambda m, a: marks.__setitem__("cn", time.perf_counter())
        ),
        conn.register_forward_hook(
            lambda m, i, o: (
                acc.__setitem__(
                    "connector", acc["connector"] + (time.perf_counter() - marks["cn"])
                ),
                acc.__setitem__("calls", acc["calls"] + 1),
            )
            and None
        ),
    ]
    orig_fwd = vwe.forward

    def _timed_fwd(*a, **kw):
        t0 = time.perf_counter()
        out = orig_fwd(*a, **kw)
        if kw.get("fill_kv_cache", False):
            acc["backbone_prefill"] += time.perf_counter() - t0
        return out

    vwe.forward = _timed_fwd
    try:
        policy.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            policy.predict_action_chunk(batch, noise=noise)
        total = time.perf_counter() - t0
    finally:
        for h in hooks:
            h.remove()
        vwe.forward = orig_fwd
    acc["total"] = total
    acc["expert_and_rest"] = (
        total - acc["vision"] - acc["connector"] - acc["backbone_prefill"]
    )
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--configs",
        default="cpu,backbone,vision+backbone",
        help="comma-separated subset of cpu,backbone,vision+backbone",
    )
    args = ap.parse_args()
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    noise = _fixed_noise(policy)

    def cpu_fn():
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
        return chunk.detach().float().numpy(), {}

    def hybrid_fn(npu_vision):
        def _f():
            t = {}
            chunk = run_hybrid_forward(
                batch, policy=policy, noise=noise, npu_vision=npu_vision, timings=t
            )
            return chunk, t

        return _f

    factories = {
        "cpu": cpu_fn,
        "backbone": hybrid_fn(False),
        "vision+backbone": hybrid_fn(True),
    }

    results = []
    for name in wanted:
        results.append(
            run_config(name, factories[name], args.iters, warmup=args.warmup)
        )

    # -- CPU stage profile (informational) --
    prof = cpu_stage_profile(policy, batch, noise)

    ref = None
    for r in results:
        if r["name"] == "cpu":
            ref = r["chunk"]
    if ref is None:
        o = np.load(_HERE / "smolvla_oracle.npz")
        ref = o["action_chunk"]

    print("\n" + "=" * 92)
    print(
        "SmolVLA end-to-end predict_action_chunk latency  "
        f"(median of {args.iters}, same batch + fixed zero noise)"
    )
    print("=" * 92)
    print(
        f"{'config':<20}{'as measured':>14}{'bridge ovh':>13}"
        f"{'compute-only':>14}{'vs CPU (meas)':>15}{'vs CPU (comp)':>15}"
    )
    print("-" * 92)
    cpu_wall = next((r["wall_ms"] for r in results if r["name"] == "cpu"), None)
    for r in results:
        comp = r["wall_ms"] - r["bridge_overhead_ms"]
        s1 = f"{cpu_wall / r['wall_ms']:.2f}x" if cpu_wall else "-"
        s2 = f"{cpu_wall / comp:.2f}x" if cpu_wall else "-"
        print(
            f"{r['name']:<20}{r['wall_ms']:>11.1f} ms"
            f"{r['bridge_overhead_ms']:>10.1f} ms"
            f"{comp:>11.1f} ms{s1:>15}{s2:>15}"
        )
    print("-" * 92)
    print(
        "  'bridge ovh' = process spawn + interpreter/import + safetensors weight\n"
        "  load + XRT/ELF load + npz round-trip, measured by the bridges themselves.\n"
        "  'compute-only' = as measured minus that, i.e. a single-process deployment."
    )

    print("\n--- pure-CPU lerobot stage profile (one instrumented run) ---")
    print(f"  vision  (3 cameras) : {prof['vision']*1e3:8.1f} ms")
    print(f"  connector (3 calls) : {prof['connector']*1e3:8.1f} ms")
    print(f"  backbone prefill    : {prof['backbone_prefill']*1e3:8.1f} ms")
    print(f"  expert + rest       : {prof['expert_and_rest']*1e3:8.1f} ms")
    print(f"  total               : {prof['total']*1e3:8.1f} ms")

    for r in results:
        if not r["stages"]:
            continue
        print(f"\n--- bridge phase detail, config '{r['name']}' (last iter) ---")
        for stage, rec in r["stages"].items():
            comp_key = _COMPUTE_PHASE[stage]
            print(f"  [{stage}] driver wall      : {rec['wall_ms']:8.1f} ms")
            for k in sorted(rec):
                if k.startswith("t_") and k != "t_bridge_total_ms":
                    tag = " <- NPU compute" if k == comp_key else ""
                    v = rec[k]
                    v = f"{v:8.1f}" if isinstance(v, float) else str(v)
                    print(f"      {k:<20}: {v} ms{tag}")
            print(f"      spawn+imports       : {rec['spawn_ms']:8.1f} ms")

    print("\n--- action-chunk agreement vs the pure-CPU chunk ---")
    for r in results:
        c = np.asarray(r["chunk"], np.float32)
        num = (c.reshape(-1) * ref.reshape(-1)).sum()
        den = np.linalg.norm(c) * np.linalg.norm(ref)
        print(
            f"  {r['name']:<20} cosine {float(num/den):.6f}   "
            f"nmse {normalized_mse(c, ref):.6f}"
        )


if __name__ == "__main__":
    main()
