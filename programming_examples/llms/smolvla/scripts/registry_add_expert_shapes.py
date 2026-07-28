"""Backfill the SmolVLA action-expert GEMM shapes into the kernel registry.

Every shape the expert executes was missing from
`kernel_registry/details/GEMM_bf16_in_bf16_out.json`, and `gemm_config()` has no
fallback -- it raises KeyError (registry_lookup.py:88). All the rows below were
already measured on real NPU2 hardware by the feasibility study (2026-07-27);
the raw sweep is `results/expert_gemm.csv` (50 configs) and the medianed table
is `docs/expert_npu_feasibility.md` section 2. This script only transcribes
them -- it runs no kernels and needs no NPU.

**Do not gate on the CSV's `status` column.** It reports NUMFAIL for rows whose
`mean_rel_L1` is squarely in the bf16-out drain tier (gate_up 9.426e-3,
kv_cross 9.399e-3, act_in 9.263e-3): a near-zero-`atol` element-wise artifact,
the same one every Qwen Gate/Up row carries, not a datapath error. The real
discriminator is `mean_rel_L1` -- the genuinely corrupt K-tilings in the sweep
report 0.78-1.02, two orders of magnitude away. See the feasibility study's
appendix ("silent-corruption traps").

Run:  python3 scripts/registry_add_expert_shapes.py [--dry-run]
"""

import argparse
import json
from pathlib import Path

REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "kernel_registry"
    / "details"
    / "GEMM_bf16_in_bf16_out.json"
)

# The tolerance-artifact note, attached to the rows whose sweep status was
# NUMFAIL purely because of it.
_ATOL_NOTE = (
    "Sweep status was NUMFAIL on a handful of near-zero-reference elements "
    "(atol artifact, same class as the Qwen Gate/Up rows); mean_rel_L1 is "
    "in-tier and the datapath is correct."
)

# (M, K, N, tile_m, tile_k_l2, tile_k_l1, tile_n, herd, gflops, mean_rel_L1,
#  used_by, note)
ROWS = [
    # ---- the 16-layer hot loop -------------------------------------------
    (
        64,
        720,
        960,
        16,
        144,
        48,
        80,
        [4, 4],
        559,
        0.00939,
        "SmolVLA action expert q_proj (hidden 720 -> q_dim 960, seq 50 padded to 64)",
        "M=64 forces tile_m=16 + herd_m=4 (tile_m*herd_m == M). K=720 = 2^4*45 "
        "admits NO tile_k_l1=32; use 144/48 or 240/80. 240/48, 360/40, 720/48 and "
        "720/80 all compile, run, and return garbage silently (mean_rel_L1 0.78-1.02).",
    ),
    (
        64,
        720,
        320,
        16,
        144,
        48,
        80,
        [4, 4],
        310,
        0.00943,
        "SmolVLA action expert k_proj/v_proj on EVEN (self-attn) layers",
        None,
    ),
    (
        64,
        960,
        768,
        16,
        320,
        32,
        96,
        [4, 4],
        685,
        0.00946,
        "SmolVLA action expert o_proj with N padded 720 -> 768",
        "Padding N 720->768 is FASTER than the native N=720 row despite +6.7% "
        "FLOPs (135.6 us vs 165.9 us), because 768 admits herd_n=4 and 720 does not.",
    ),
    (
        64,
        960,
        720,
        16,
        320,
        32,
        80,
        [4, 3],
        533,
        0.00947,
        "SmolVLA action expert o_proj, native N=720",
        "N=720 cannot use herd_n=4: no tile_n with tile_n%16==0 satisfies "
        "720%(4*tile_n)==0. herd_n=3 leaves a quarter of the array idle.",
    ),
    (
        64,
        720,
        2048,
        16,
        144,
        48,
        128,
        [4, 4],
        1020,
        0.00943,
        "SmolVLA action expert gate_proj / up_proj",
        _ATOL_NOTE,
    ),
    (
        64,
        2048,
        768,
        16,
        256,
        32,
        96,
        [4, 4],
        1122,
        0.00928,
        "SmolVLA action expert down_proj with N padded 720 -> 768",
        "Same padding win as o_proj: 179.5 us padded vs 212.3 us native.",
    ),
    (
        64,
        2048,
        720,
        16,
        256,
        32,
        80,
        [4, 3],
        889,
        0.00936,
        "SmolVLA action expert down_proj, native N=720",
        None,
    ),
    (
        256,
        320,
        320,
        32,
        320,
        32,
        80,
        [8, 4],
        466,
        0.00940,
        "SmolVLA action expert k_proj/v_proj on ODD (cross-attn) layers -- these "
        "run over the 241-token backbone prefix (padded to 256), NOT the 50 action "
        "tokens",
        "tile_k_l2=160 is a silent-corruption trap here (mean_rel_L1 0.793); "
        "320 is correct. " + _ATOL_NOTE,
    ),
    # ---- head/tail: outside the current port scope, but measured ---------
    (
        64,
        1440,
        768,
        16,
        144,
        48,
        96,
        [4, 4],
        931,
        0.00934,
        "SmolVLA action_time_mlp_in with N padded 720 -> 768",
        None,
    ),
    (
        64,
        1440,
        720,
        16,
        144,
        48,
        80,
        [4, 3],
        749,
        0.00929,
        "SmolVLA action_time_mlp_in, native N=720",
        "K=1440 = 2^5*45 admits no tile_k_l1=32. 288/32 and 480/32 are silent-"
        "corruption traps (0.80 / 0.90); 144/48 and 240/80 are correct.",
    ),
    (
        64,
        720,
        720,
        16,
        144,
        48,
        80,
        [4, 3],
        447,
        0.00945,
        "SmolVLA action_time_mlp_out",
        None,
    ),
    (
        64,
        32,
        720,
        16,
        32,
        32,
        80,
        [4, 3],
        25,
        0.00926,
        "SmolVLA action_in_proj (K=32, almost pure launch floor)",
        "25 GFLOP/s: 115.9 us for 3.3 MFLOP is ~100% per-launch floor. " + _ATOL_NOTE,
    ),
    (
        64,
        720,
        32,
        16,
        144,
        48,
        32,
        [4, 1],
        37,
        0.00901,
        "SmolVLA action_out_proj (N=32 -> tile_n=32, herd_n=1)",
        None,
    ),
    # ---- weight-concat variants, for the fused-weight optimisation -------
    (
        64,
        720,
        1600,
        16,
        144,
        48,
        80,
        [4, 4],
        713,
        0.00944,
        "SmolVLA action expert q||k||v with concatenated weights (960+320+320)",
        "One GEMM instead of three: 206.8 us vs 158.2+95.2+95.2 = 348.6 us.",
    ),
    (
        64,
        720,
        4096,
        16,
        144,
        48,
        128,
        [4, 4],
        1163,
        0.00944,
        "SmolVLA action expert gate||up with concatenated weights (2048+2048)",
        "One GEMM instead of two: 324.6 us vs 2x185.0 = 370.0 us.",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(REGISTRY.read_text())
    existing = {(s["M"], s["K"], s["N"]): s for s in data["shapes"]}

    added, skipped = [], []
    for m, k, n, tm, tk2, tk1, tn, herd, gf, mrl, used_by, note in ROWS:
        if (m, k, n) in existing:
            skipped.append((m, k, n))
            continue
        method = {
            "tile": {
                "tile_m": tm,
                "tile_k_l2": tk2,
                "tile_k_l1": tk1,
                "tile_n": tn,
            },
            "gflops": gf,
            "mean_rel_L1": mrl,
            "tier": "high",
        }
        # Only carry a herd override when it differs from the file-level 8x4.
        if herd != data.get("herd", [8, 4]):
            method["herd"] = list(herd)
        entry = {
            "M": m,
            "K": k,
            "N": n,
            "used_by": used_by,
            "methods": {"drain": method},
            "best": {"high": "drain"},
        }
        if note:
            entry["_note"] = note
        data["shapes"].append(entry)
        added.append((m, k, n))

    data["shapes"].sort(key=lambda s: (s["M"], s["K"], s["N"]))

    print(f"registry: {REGISTRY}")
    print(f"  added   : {len(added)}  {added}")
    print(f"  skipped : {len(skipped)} (already present) {skipped}")
    if args.dry_run:
        print("  (dry run -- not written)")
        return
    REGISTRY.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {len(data['shapes'])} shapes")


if __name__ == "__main__":
    main()
