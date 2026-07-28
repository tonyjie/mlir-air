# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tiered correctness gate for the action expert on NPU2 (phase E6).

Three levels, weakest assumption first, so a failure localises itself:

  L1  teacher-forced per layer   feed the oracle's own input to layer i and
                                 compare that layer's output. Isolates each
                                 layer from every other layer's drift; this is
                                 the PRIMARY gate.
  L2  teacher-forced per step    feed the oracle's suffix_emb for step s and
                                 run all 16 layers. Shows within-step
                                 accumulation over depth.
  L3  free-running 10 steps      only the first step's input comes from the
                                 oracle. Shows accumulation over the sequential
                                 denoise loop, which is the thing BFP16 drift
                                 compounds through. Reported, not blocking.

The gate thresholds follow the project's bf16 tier (per-layer >= 0.98,
end-of-stack >= 0.99), the same ones the backbone and vision ports used.

Run under the NPU lock, worktree python:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_expert_npu.py
  ... --layers 0,1  --steps 0        to narrow it down while debugging
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from expert_denoise import ExpertRunner
from expert_weights import SmolVLAExpertConfig, load_expert_weights

ORACLE = _HERE / "expert_oracle.npz"
LAYER_GATE = 0.98
STACK_GATE = 0.99


def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _setup(d, runner):
    runner.set_prefix(
        prefix_k=d["prefix_k"],
        prefix_v=d["prefix_v"],
        mask_self=d["mask_self"],
        mask_cross=d["mask_cross"],
        pos_self=d["pos_self"],
        pos_cross=d["pos_cross"],
    )


def level1(d, r, steps, layers):
    """Teacher-forced per layer -- the primary gate."""
    print("\nL1  teacher-forced per layer (oracle input -> one layer)")
    print(f"    {'step':>4} {'layer':>5} {'kind':6} {'cosine':>10}")
    worst, rows = 1.0, []
    for s in steps:
        for li in layers:
            x_in = d["hidden_in"][s, li]  # (50, 720) fp32
            ref = d["hidden_out"][s, li]
            xp = np.zeros((r.seq_pad, r.cfg.emb_dim), dtype=x_in.dtype)
            xp[: r.seq_real] = x_in
            from ml_dtypes import bfloat16

            out = r.run_layer(np.asarray(xp, bfloat16), li)
            c = cos(out[: r.seq_real], ref)
            kind = "self" if r.cfg.is_self_attn(li) else "cross"
            rows.append((s, li, kind, c))
            worst = min(worst, c)
            print(f"    {s:4d} {li:5d} {kind:6} {c:10.6f}")
    print(
        f"    worst {worst:.6f}  gate {LAYER_GATE}  "
        f"{'PASS' if worst >= LAYER_GATE else 'FAIL'}"
    )
    return worst, rows


def level2(d, r, steps):
    """Teacher-forced per step -- all 16 layers from the oracle's suffix_emb."""
    print("\nL2  teacher-forced per step (oracle suffix_emb -> 16 layers)")
    worst = 1.0
    for s in steps:
        out = r.run_step(d["suffix_embs"][s])
        c = cos(out, d["final_norm"][s])
        worst = min(worst, c)
        print(f"    step {s:2d}  final_norm cosine {c:.6f}")
    print(
        f"    worst {worst:.6f}  gate {STACK_GATE}  "
        f"{'PASS' if worst >= STACK_GATE else 'FAIL'}"
    )
    return worst


def level3(d, r):
    """Genuinely free-running: nothing after step 0's x_t comes from the oracle.

    Each step's input is the previous step's output, closed through the CPU
    head/tail (embed_suffix -> NPU 16 layers -> action_out_proj -> x_t update).
    This is the only level where BFP16 error actually compounds across the
    denoise loop. Reported, not blocking.
    """
    print("\nL3  free-running 10 steps (reported, not blocking)")
    x_final, per = r.run_denoise_loop(timesteps=list(d["timesteps"]))
    print(f"    {'step':>4} {'v_t':>10} {'x_t':>10}")
    worst_v = 1.0
    for s, rec in enumerate(per):
        cv = cos(rec["v_t"], d["v_t"][s])
        cx = cos(rec["x_t"], d["x_t"][s + 1]) if s + 1 < len(d["x_t"]) else float("nan")
        worst_v = min(worst_v, cv)
        print(f"    {s:4d} {cv:10.6f} {cx:10.6f}")
    cf = cos(x_final, d["x_t_final"])
    print(f"    worst v_t {worst_v:.6f}")
    print(f"    FINAL x_t cosine vs oracle: {cf:.6f}")
    return cf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--steps", default="0")
    ap.add_argument("--level", default="1", help="1, 12, or 123")
    ap.add_argument(
        "--no-hoist",
        action="store_true",
        help="recompute cross K/V every step, as lerobot does",
    )
    ap.add_argument("--compile", action="store_true", help="(re)compile kernels")
    args = ap.parse_args()

    if not ORACLE.exists():
        sys.exit(f"missing {ORACLE} -- run expert_prefix.py first")
    d = np.load(ORACLE)

    cfg = SmolVLAExpertConfig()
    print("Loading expert weights...")
    w = load_expert_weights(config=cfg)
    r = ExpertRunner(weights=w, config=cfg, hoist_cross_kv=not args.no_hoist)
    if args.compile:
        r.compile_all()
    else:
        r.ensure_kernels()

    layers = (
        list(range(cfg.n_layers))
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )
    steps = [int(x) for x in args.steps.split(",")]

    _setup(d, r)
    print(
        f"cross K/V: {'hoisted out of the denoise loop (16 GEMMs)' if r.hoist_cross_kv else 'recomputed per step (160 GEMMs, lerobot-faithful)'}"
    )

    ok = True
    if "1" in args.level:
        worst, _ = level1(d, r, steps, layers)
        ok &= worst >= LAYER_GATE
    if "2" in args.level:
        ok &= level2(d, r, steps) >= STACK_GATE
    if "3" in args.level:
        level3(d, r)

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
