# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A3-7: TRUE end-to-end SmolVLA latency, all configs, same input + fixed noise.

Measures the wall time of one full `predict_action_chunk` (vision -> connector ->
prefix -> backbone prefill -> 10-step action-expert denoise -> (1,50,6) chunk).

Configs (`--configs`, comma separated):
  cpu              pure lerobot on CPU — THE baseline, nothing hooked.
  vision           NPU SigLIP ViT + connector, CPU backbone  <-- the target
  vision+backbone  NPU vision AND NPU 16-layer backbone prefill
  backbone         NPU backbone only, CPU vision
  bridge:*         the same hybrids through the legacy two-process subprocess
                   bridge (bridge:backbone, bridge:vision+backbone), for the
                   record: they pay process spawn + safetensors reload + XRT/ELF
                   load on EVERY inference.

Everything except `bridge:*` runs SINGLE PROCESS: the lerobot venv imports
air/pyxrt, so the NPU runtimes (weights, ELFs, XRT context, device BOs) are
built once by `warmup_npu()` before the timed loop and reused, exactly as a
deployment would. Consequently `as measured` IS the deployable number for those
rows; `stage ovh` is only non-zero for the bridged rows.

BLAS-thread A/B (`--blas-ab`): the NPU dispatch loop is host-bound and is slowed
by BLAS workers busy-spinning, but the CPU action expert needs those threads.
`smolvla_npu_runtime.npu_thread_limits` clamps them only around the NPU call;
this flag re-runs the target config with the clamp disabled so the trade-off is
measured, not asserted.

`make bench-e2e` runs this under the NPU lock. The Makefile self-locks — do NOT
wrap it in another flock.
"""

from __future__ import annotations

import argparse
import os
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
    warmup_npu,
    build_oracle_batch,
    normalized_mse,
    _fixed_noise,
    DEFAULT_MODEL,
)

# The phase of each NPU stage that is REAL device work. Everything else the
# stage reports (npz I/O, weight (re)load, ELF load) plus the spawn gap is
# execution-model overhead — zero by construction in the single-process path.
_COMPUTE_PHASE = {"vision": "t_encode_ms", "backbone": "t_prefill_ms"}

# name -> (npu_vision, npu_backbone, npu_expert, bridge)
CONFIGS = {
    "cpu": (False, False, False, False),
    "vision": (True, False, False, False),
    "vision+backbone": (True, True, False, False),
    "backbone": (False, True, False, False),
    "expert": (False, False, True, False),
    "vision+expert": (True, False, True, False),
    "all": (True, True, True, False),
    "bridge:backbone": (False, True, False, True),
    "bridge:vision+backbone": (True, True, False, True),
}


def median_ms(xs):
    return statistics.median(xs) * 1e3


def _stage_split(timings):
    """-> (npu_compute_ms, overhead_ms) summed over the NPU stages of one run."""
    compute = 0.0
    overhead = 0.0
    for stage, rec in timings.items():
        c = float(rec.get(_COMPUTE_PHASE[stage], 0.0))
        compute += c
        overhead += float(rec["wall_ms"]) - c
    return compute, overhead


STAGE_KEYS = (
    "cpu_vision",
    "cpu_connector",
    "cpu_backbone",
    "npu_vision",
    "npu_backbone",
)


class StageHooks:
    """Splits every timed iteration into vision / backbone-prefill /
    everything-else, whichever side each stage runs on.

    The hooks sit on lerobot's OWN modules and are installed BEFORE
    run_hybrid_forward wraps anything, so they always measure the CPU work; a
    stage moved to NPU simply never calls them and is read from `timings`
    instead. Overhead is a few perf_counter() calls per inference, so this can
    stay on during the timed loop — which matters: a separate instrumented run
    lands in a different thread-pool/cache state and does NOT reproduce the
    median (measured 1146 ms vs a 960 ms median for the same CPU config)."""

    def __init__(self, policy):
        self.policy = policy
        self.acc = dict.fromkeys(STAGE_KEYS, 0.0)
        self._marks = {}
        self._hooks = []
        self._orig_fwd = None

    def __enter__(self):
        vwe = self.policy.model.vlm_with_expert
        vm = vwe.get_vlm_model().vision_model
        conn = vwe.get_vlm_model().connector
        acc, marks = self.acc, self._marks
        self._hooks = [
            vm.register_forward_pre_hook(
                lambda m, a: marks.__setitem__("vm", time.perf_counter())
            ),
            vm.register_forward_hook(
                lambda m, i, o: acc.__setitem__(
                    "cpu_vision",
                    acc["cpu_vision"] + (time.perf_counter() - marks["vm"]),
                )
            ),
            conn.register_forward_pre_hook(
                lambda m, a: marks.__setitem__("cn", time.perf_counter())
            ),
            conn.register_forward_hook(
                lambda m, i, o: acc.__setitem__(
                    "cpu_connector",
                    acc["cpu_connector"] + (time.perf_counter() - marks["cn"]),
                )
            ),
        ]
        self._orig_fwd = vwe.forward
        orig = self._orig_fwd

        def _timed_fwd(*a, **kw):
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            if kw.get("fill_kv_cache", False):
                acc["cpu_backbone"] += time.perf_counter() - t0
            return out

        vwe.forward = _timed_fwd
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self.policy.model.vlm_with_expert.forward = self._orig_fwd
        return False

    def reset(self):
        for k in self.acc:
            self.acc[k] = 0.0

    def read(self, timings, total_s):
        prof = {
            k: self.acc[k] * 1e3
            for k in ("cpu_vision", "cpu_connector", "cpu_backbone")
        }
        prof["npu_vision"] = float(timings.get("vision", {}).get("wall_ms", 0.0))
        prof["npu_backbone"] = float(timings.get("backbone", {}).get("wall_ms", 0.0))
        prof["total"] = total_s * 1e3
        prof["expert_and_rest"] = prof["total"] - sum(prof[k] for k in STAGE_KEYS)
        return prof


def _summarize(name, acc):
    walls, computes, overheads, per_stage, profs, chunk = acc
    med_prof = {
        k: statistics.median([p[k] for p in profs])
        for k in list(STAGE_KEYS) + ["expert_and_rest", "total"]
    }
    return {
        "name": name,
        "wall_ms": median_ms(walls),
        "wall_min_ms": min(walls) * 1e3,
        "npu_compute_ms": statistics.median(computes) if computes else 0.0,
        "overhead_ms": statistics.median(overheads) if overheads else 0.0,
        "chunk": chunk,
        "stages": per_stage[-1] if per_stage else {},
        "profile": med_prof,
    }


def run_all(named_fns, policy, iters, warmup=1, interleave=True):
    """Time every config `iters` times with the per-stage hooks live.

    interleave=True (default) walks the configs ROUND-ROBIN, one iteration
    each, instead of finishing one config before starting the next. This
    machine is shared (other agents compile/run concurrently) and drifts on a
    minutes timescale; measuring config A entirely before config B lets that
    drift masquerade as a difference between them. Interleaving makes every
    config see the same drift, so the median ratio is meaningful.
    `--no-interleave` restores the blocked order."""
    acc = {n: ([], [], [], [], [], None) for n, _ in named_fns}

    def one(name, fn, hooks, record):
        hooks.reset()
        t0 = time.perf_counter()
        chunk, timings = fn()
        wall = time.perf_counter() - t0
        if not record:
            return
        walls, computes, overheads, per_stage, profs, _ = acc[name]
        walls.append(wall)
        c, o = _stage_split(timings)
        computes.append(c)
        overheads.append(o)
        per_stage.append(timings)
        profs.append(hooks.read(timings, wall))
        acc[name] = (walls, computes, overheads, per_stage, profs, chunk)
        print(
            f"[bench]   {name:<24} wall {wall*1e3:8.1f} ms  "
            f"(NPU {c:7.1f} ms, ovh {o:5.1f} ms, "
            f"expert+rest {profs[-1]['expert_and_rest']:6.1f} ms)"
        )

    with StageHooks(policy) as hooks:
        if interleave:
            for w in range(warmup):
                print(f"\n[bench] --- warmup round {w} ---")
                for name, fn in named_fns:
                    one(name, fn, hooks, record=False)
            for i in range(iters):
                print(f"\n[bench] --- round {i} (interleaved) ---")
                for name, fn in named_fns:
                    one(name, fn, hooks, record=True)
        else:
            for name, fn in named_fns:
                print(f"\n[bench] --- config '{name}' ---")
                for _ in range(warmup):
                    one(name, fn, hooks, record=False)
                for _ in range(iters):
                    one(name, fn, hooks, record=True)

    return [_summarize(n, acc[n]) for n, _ in named_fns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--configs",
        default="cpu,vision,vision+backbone",
        help="comma-separated subset of " + ",".join(CONFIGS),
    )
    ap.add_argument(
        "--blas-ab",
        action="store_true",
        help="also run the NPU-vision config with the scoped BLAS-thread clamp "
        "DISABLED (SMOLVLA_NPU_BLAS_LIMIT=0), to measure the trade-off",
    )
    ap.add_argument(
        "--no-interleave",
        action="store_true",
        help="run each config's iterations back-to-back instead of round-robin "
        "(round-robin is the default: it cancels the drift of this shared machine)",
    )
    args = ap.parse_args()
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    for w in wanted:
        assert w in CONFIGS, f"unknown config {w!r}; known: {list(CONFIGS)}"

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    noise = _fixed_noise(policy)

    # Build the NPU runtimes ONCE, before any timing (a deployment's startup).
    need_v = any(CONFIGS[w][0] and not CONFIGS[w][3] for w in wanted) or args.blas_ab
    need_b = any(CONFIGS[w][1] and not CONFIGS[w][3] for w in wanted)
    need_e = any(CONFIGS[w][2] for w in wanted)
    if need_v or need_b or need_e:
        print("[bench] building single-process NPU runtimes (one-time startup)...")
        t0 = time.perf_counter()
        warmup_npu(npu_vision=need_v, npu_backbone=need_b)
        if need_e:
            # Expert runtime is built here too, so its weight load + ELF load
            # lands in startup rather than in the first timed iteration.
            from smolvla_npu_runtime import get_expert_runtime

            get_expert_runtime()
        print(f"[bench] NPU startup {(time.perf_counter()-t0)*1e3:.0f} ms (once)")

    def make_fn(name):
        npu_v, npu_b, npu_e, bridge = CONFIGS[name]
        if not npu_v and not npu_b and not npu_e:

            def _cpu():
                policy.reset()
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(batch, noise=noise)
                return chunk.detach().float().numpy(), {}

            return _cpu

        def _hybrid(timings=None):
            t = {} if timings is None else timings
            chunk = run_hybrid_forward(
                batch,
                policy=policy,
                noise=noise,
                npu_vision=npu_v,
                npu_backbone=npu_b,
                npu_expert=npu_e,
                bridge=bridge,
                timings=t,
            )
            return chunk, t

        return _hybrid

    named_fns = [(n, make_fn(n)) for n in wanted]
    if args.blas_ab and "vision" in wanted:
        # Same config, clamp disabled per call — measured, not asserted.
        base = make_fn("vision")

        def _no_clamp(_b=base):
            os.environ["SMOLVLA_NPU_BLAS_LIMIT"] = "0"
            try:
                return _b()
            finally:
                os.environ["SMOLVLA_NPU_BLAS_LIMIT"] = "1"

        named_fns.append(("vision (blas clamp OFF)", _no_clamp))

    results = run_all(
        named_fns,
        policy,
        args.iters,
        warmup=args.warmup,
        interleave=not args.no_interleave,
    )
    profiles = {r["name"]: r["profile"] for r in results}

    ref = next((r["chunk"] for r in results if r["name"] == "cpu"), None)
    if ref is None:
        ref = np.load(_HERE / "smolvla_oracle.npz")["action_chunk"]

    print("\n" + "=" * 96)
    print(
        "SmolVLA end-to-end predict_action_chunk latency  "
        f"(median of {args.iters}, same batch + fixed zero noise)"
    )
    print("=" * 96)
    print(
        f"{'config':<26}{'as measured':>14}{'best':>10}{'stage ovh':>12}"
        f"{'NPU compute':>13}{'vs CPU':>10}"
    )
    print("-" * 96)
    cpu_wall = next((r["wall_ms"] for r in results if r["name"] == "cpu"), None)
    for r in results:
        s = f"{cpu_wall / r['wall_ms']:.2f}x" if cpu_wall else "-"
        print(
            f"{r['name']:<26}{r['wall_ms']:>11.1f} ms{r['wall_min_ms']:>7.1f} ms"
            f"{r['overhead_ms']:>9.1f} ms{r['npu_compute_ms']:>10.1f} ms{s:>10}"
        )
    print("-" * 96)
    print(
        "  'as measured' = honest wall clock of predict_action_chunk. For the\n"
        "  single-process rows that IS the deployable number (weights/ELFs/XRT\n"
        "  loaded once at startup). 'stage ovh' = per-inference non-device cost:\n"
        "  for bridge:* rows the spawn + safetensors reload + npz round-trip;\n"
        "  for single-process rows only the host im2col patch-embed (~14 ms)."
    )

    print(
        f"\n--- per-stage breakdown (median of the same {args.iters} timed "
        "iterations) ---"
    )
    keys = [
        ("cpu_vision", "vision (CPU)"),
        ("cpu_connector", "connector (CPU)"),
        ("npu_vision", "vision+conn (NPU)"),
        ("cpu_backbone", "backbone prefill (CPU)"),
        ("npu_backbone", "backbone prefill (NPU)"),
        ("expert_and_rest", "action expert + rest"),
        ("total", "TOTAL"),
    ]
    names = [r["name"] for r in results]
    hdr = "".join(f"{n:>26}" for n in names)
    print(f"{'stage':<24}{hdr}")
    for k, label in keys:
        row = "".join(f"{profiles[n][k]:>23.1f} ms" for n in names)
        print(f"{label:<24}{row}")

    for r in results:
        if not r["stages"]:
            continue
        print(f"\n--- NPU stage detail, config '{r['name']}' (last iter) ---")
        for stage, rec in r["stages"].items():
            print(f"  [{stage}] driver wall      : {rec['wall_ms']:8.1f} ms")
            for k in sorted(rec):
                if k.startswith("t_") and k != "t_bridge_total_ms":
                    tag = " <- NPU compute" if k == _COMPUTE_PHASE[stage] else ""
                    v = rec[k]
                    v = f"{v:8.1f}" if isinstance(v, float) else str(v)
                    print(f"      {k:<20}: {v} ms{tag}")
            if "spawn_ms" in rec:
                print(f"      spawn+imports       : {rec['spawn_ms']:8.1f} ms")

    print("\n--- action-chunk agreement vs the pure-CPU chunk ---")
    for r in results:
        c = np.asarray(r["chunk"], np.float32)
        num = (c.reshape(-1) * ref.reshape(-1)).sum()
        den = np.linalg.norm(c) * np.linalg.norm(ref)
        print(
            f"  {r['name']:<26} cosine {float(num/den):.6f}   "
            f"nmse {normalized_mse(c, ref):.6f}"
        )


if __name__ == "__main__":
    main()
