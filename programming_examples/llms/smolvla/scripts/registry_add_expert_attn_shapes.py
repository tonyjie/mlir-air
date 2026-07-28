"""Add the SmolVLA action-expert *attention* GEMM shapes to the kernel registry.

Sibling of `registry_add_expert_shapes.py` (which covered the expert's
projection GEMMs). These four rows are the decomposed-attention path -- QK^T and
P@V -- for the action expert, batched across the GQA group exactly as
`smolvla_backbone_prefill.py::_compile_npu_attention_kernels` does it: 15 q-heads
/ 5 kv-heads -> group 3, 50 action tokens padded to 64, so M = 3*64 = 192.

    qkt_even : 192 x  64 x 320   (K/V len 241 prefix + 50 action = 291 -> pad 320)
    pv_even  : 192 x 320 x  64
    qkt_odd  : 192 x  64 x 256   (cross-attn, K/V len 241 -> pad 256)
    pv_odd   : 192 x 256 x  64

All four were swept on real NPU2 (Strix, aie2p) at CPU governor `performance` +
NPU `pmode=Turbo` on 2026-07-28; raw data in `results/expert_attn_gemm.csv`
(53 runs, 2-4 repeats on the leaders). This script only transcribes the medians.

**Do not gate on the sweep's `status` column.** Every one of the 53 runs reports
NUMFAIL: at these tiny shapes the reference has many near-zero elements and the
harness's high-precision `atol=1.5e-3` trips on ~4% of them, the same artifact
the projection rows carry. The gate is `mean_rel_L1`: the correct configs sit at
9.27e-3 to 9.45e-3, and the genuinely corrupt ones report 0.71-0.91.

Run:  python3 scripts/registry_add_expert_attn_shapes.py [--dry-run]
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

_ATOL_NOTE = (
    "Sweep status was NUMFAIL on the near-zero-reference elements (atol "
    "artifact, same class as the projection rows); mean_rel_L1 is in-tier and "
    "the datapath is correct."
)

_M192_NOTE = (
    "M=192 (GQA group 3 x 64 padded action tokens) admits only three "
    "(tile_m, herd_m) pairs under mm.o's tile_m%16==0, run.py's "
    "M%(tile_m*herd_m)==0, and the single-M-launch-iteration rule "
    "M//tile_m//herd_m==1: 32x6, 48x4, 64x3. Measured: 32x6 and 48x4 are within "
    "noise of each other, 64x3 is consistently slowest (only 12 of 32 tiles)."
)

# (M, K, N, tile_m, tile_k_l2, tile_k_l1, tile_n, herd, gflops, mean_rel_L1,
#  used_by, note)
ROWS = [
    (
        192,
        64,
        320,
        48,
        64,
        32,
        80,
        [4, 4],
        95,
        0.00942,
        "SmolVLA action expert QK^T on EVEN (self-attn) layers: group-batched "
        "S = Q_group @ K_h^T, K/V len 291 (241 prefix + 50 action) padded to 320",
        _M192_NOTE
        + " tile_n=80 (=320/4) is required for herd_n=4 and is worth 1.6x over "
        "tile_n=16 (82.9 us vs 129.2 us). Padding N to 384 instead of 320 is "
        "slower (86.7 us) -- the extra FLOPs are not amortized at this size. "
        + _ATOL_NOTE,
    ),
    (
        192,
        320,
        64,
        32,
        320,
        32,
        16,
        [6, 4],
        109,
        0.00927,
        "SmolVLA action expert P@V on EVEN (self-attn) layers: group-batched "
        "O = P_group @ V_h, K = 291 padded to 320",
        _M192_NOTE + " K=320 IS A SILENT-CORRUPTION TRAP: tile_k_l2=160 (0.91), "
        "tile_k_l2=64/tile_k_l1=64 (0.71) and tile_k_l2=320/tile_k_l1=64 (0.80) "
        "all compile, run, report plausible latency and return garbage. Only "
        "tile_k_l1 in {16, 32} with tile_k_l2=320 (single reduction tile) was "
        "measured correct. " + _ATOL_NOTE,
    ),
    (
        192,
        64,
        256,
        32,
        64,
        32,
        64,
        [6, 4],
        82,
        0.00939,
        "SmolVLA action expert QK^T on ODD (cross-attn) layers: group-batched "
        "S = Q_group @ K_h^T against the frozen 241-token prefix, padded to 256",
        _M192_NOTE + " " + _ATOL_NOTE,
    ),
    (
        192,
        256,
        64,
        32,
        64,
        64,
        16,
        [6, 4],
        89,
        0.00945,
        "SmolVLA action expert P@V on ODD (cross-attn) layers: group-batched "
        "O = P_group @ V_h over the 241-token prefix, K padded to 256",
        _M192_NOTE + " Unlike the K=320 sibling, every K-tiling tried at K=256 was "
        "numerically correct (9.452e-3 across tile_k_l2 in {64,128,256} and "
        "tile_k_l1 in {32,64}) -- the corruption tracks non-power-of-two K. "
        + _ATOL_NOTE,
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(REGISTRY.read_text())
    existing = {(s["M"], s["K"], s["N"]) for s in data["shapes"]}

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
