#!/usr/bin/env python3
"""Project SmolVLA action-expert NPU2 latency from THIS SESSION's measurements.

Every latency below is a device `xrt.run.start()+wait2()` time measured on real
NPU2 in this session (see results/expert_gemm.csv and results/logs/). Nothing is
carried over from another kernel, another run, or the kernel_registry.

The expert's real per-denoise-step op graph (read off lerobot
`smolvlm_with_expert.py` at the smolvla_base config: 16 layers,
self_attn_every_n_layers=2, chunk_size=50, num_steps=10):

  EVEN layer_idx (8 layers) -> forward_attn_layer, "self" attention, but the
    cached 241-token prefix K/V is CONCATENATED, so K/V length is 241+50=291.
    q/k/v are projected from the 50 action tokens (M=50 -> pad 64).
  ODD layer_idx (8 layers) -> forward_cross_attn_layer. Q comes from the 50
    action tokens; K/V are produced by running the expert's k_proj/v_proj over
    the *backbone's cached 241x320 prefix* -> those two GEMMs are M=241 (pad
    256), K=320, N=320, and lerobot recomputes them on EVERY denoise step even
    though they are constant (hoistable; quantified below).

Run:  python3 scripts/expert_projection.py
"""

N_LAYERS = 16
N_STEPS = 10
N_EVEN = N_LAYERS // 2  # self-attn layers
N_ODD = N_LAYERS // 2  # cross-attn layers

# ---------------------------------------------------------------------------
# MEASURED device latencies (us), NPU2, this session.
# GEMM: bf16-in/bf16-out, HIGH_PRECISION=true, --method drain, PERF_ITERS=20.
#       median over repeat runs where repeats exist (results/expert_gemm.csv).
# ---------------------------------------------------------------------------
US = {
    # --- GEMM, M=50 padded to 64 --------------------------------------------
    "q_proj": 158.2,  # 64x720x960   tm16/tk2 144/tk1 48/tn80  herd 4x4 (median of 4)
    "kv_even": 95.2,  # 64x720x320   same tiling
    "o_proj": 135.6,  # 64x960x768   (N 720->768) tn96 herd 4x4 (median of 3)
    "gate_or_up": 185.0,  # 64x720x2048  tn128 herd 4x4
    "down": 179.5,  # 64x2048x768  (N 720->768) tn96 herd 4x4
    "kv_cross": 112.6,  # 256x320x320  tm32/tk2 320/tn80 herd 8x4 (M=241->256)
    # fused-weight variants (one GEMM instead of 3 / 2)
    "qkv_concat": 206.8,  # 64x720x1600
    "gateup_concat": 324.6,  # 64x720x4096
    # --- FlashAttention, 15q/5kv, head_dim 64, non-causal --------------------
    # ONLY lq=256 is numerically correct today (see report S4): lq=64 and
    # lq=128 compile and run but return inf/NaN. Q=50 must be padded to 256.
    "fa_even_ok": 891.6,  # lq=256 lk=384 (K 291->384) nqt=4 ncs=3  PASS 4.2e-2
    "fa_odd_ok": 919.7,  # lq=256 lk=256 (K 241->256) nqt=4 ncs=4  PASS 4.3e-2
    # what a *fixed* small-Q FA would cost (timing valid, numerics broken)
    "fa_even_q64": 350.0,  # lq=64 lk=384 nqt=1 ncs=3
    "fa_odd_q64": 368.7,  # lq=64 lk=256 nqt=1 ncs=4
    # --- non-GEMM ------------------------------------------------------------
    "rmsnorm": 151.3,  # 64x720
    "eltwise_add": 86.6,  # 64x720  (n=46080, tile_n=1920, herd_x=8)
    "silu_mul": 121.6,  # 64x2048 (n=131072, tile_n=4096, herd_x=8)
    "rope_q": 106.3,  # 960x64  (64 rows x 15 q-heads)
    "rope_k": 83.0,  # 320x64  (64 rows x 5 kv-heads)
    # --- per-denoise-step head/tail (outside the 16 layers) -------------------
    "action_in": 115.9,  # 64x32x720
    "atmlp_in": 152.0,  # 64x1440x768
    "atmlp_out": 148.3,  # 64x720x720
    "action_out": 79.2,  # 64x720x32
}

# Per-dispatch host cost on the DEPLOYED driver path (llms/shared/infra/cache.py
# load_and_run), measured by bench_expert_dispatch.py: 24 us fixed + BO write.
# The expert's activations are 92-256 KB; the real backbone measured BO write at
# ~0.07 ms/MB -> ~6-18 us here. Use 36 us total per dispatch.
OVERHEAD_US = 36.0

# Measured penalty of folding launches into ONE multi-launch ELF instead of
# dispatching each op as its own optimally-tiled kernel. From the two real
# SmolVLA backbone fused ELFs profiled this session at seq=256:
#   rms_gemms_rope  2030 us fused vs 1134 us as standalone parts -> 1.79x
#   o_ffn           3370 us fused vs 2418 us as standalone parts -> 1.39x
FUSION_PENALTY = (1.39, 1.79)


def even_layer(fa, qkv_concat=False, gateup_concat=False):
    ops = [("rmsnorm", US["rmsnorm"])]
    if qkv_concat:
        ops.append(("qkv_concat", US["qkv_concat"]))
    else:
        ops += [
            ("q_proj", US["q_proj"]),
            ("k_proj", US["kv_even"]),
            ("v_proj", US["kv_even"]),
        ]
    ops += [("rope_q", US["rope_q"]), ("rope_k", US["rope_k"]), ("attn", fa)]
    ops += [("o_proj", US["o_proj"]), ("add", US["eltwise_add"])]
    ops.append(("rmsnorm2", US["rmsnorm"]))
    if gateup_concat:
        ops.append(("gateup_concat", US["gateup_concat"]))
    else:
        ops += [("gate", US["gate_or_up"]), ("up", US["gate_or_up"])]
    ops += [
        ("silu_mul", US["silu_mul"]),
        ("down", US["down"]),
        ("add2", US["eltwise_add"]),
    ]
    return ops


def odd_layer(fa, hoist_kv=False, gateup_concat=False):
    # Cross-attention: q from the 50 action tokens; k/v from the 241 prefix.
    # q_proj and k/v_proj have DIFFERENT M (64 vs 256) -> cannot be concatenated.
    ops = [("rmsnorm", US["rmsnorm"]), ("q_proj", US["q_proj"])]
    if not hoist_kv:
        ops += [("k_proj_x", US["kv_cross"]), ("v_proj_x", US["kv_cross"])]
    ops += [("rope_q", US["rope_q"]), ("attn", fa)]
    ops += [("o_proj", US["o_proj"]), ("add", US["eltwise_add"])]
    ops.append(("rmsnorm2", US["rmsnorm"]))
    if gateup_concat:
        ops.append(("gateup_concat", US["gateup_concat"]))
    else:
        ops += [("gate", US["gate_or_up"]), ("up", US["gate_or_up"])]
    ops += [
        ("silu_mul", US["silu_mul"]),
        ("down", US["down"]),
        ("add2", US["eltwise_add"]),
    ]
    return ops


def head_tail():
    return [
        ("action_in", US["action_in"]),
        ("atmlp_in", US["atmlp_in"]),
        ("swish", US["silu_mul"]),
        ("atmlp_out", US["atmlp_out"]),
        ("action_out", US["action_out"]),
        ("add", US["eltwise_add"]),
        ("add2", US["eltwise_add"]),
    ]


def scenario(
    name, fa_even, fa_odd, qkv_concat=False, gateup_concat=False, hoist_kv=False
):
    ev = even_layer(fa_even, qkv_concat, gateup_concat)
    od = odd_layer(fa_odd, hoist_kv, gateup_concat)
    ht = head_tail()
    dev_step = N_EVEN * sum(t for _, t in ev) + N_ODD * sum(t for _, t in od)
    dev_step += sum(t for _, t in ht)
    disp_step = N_EVEN * len(ev) + N_ODD * len(od) + len(ht)

    dev_ms = dev_step * N_STEPS / 1000.0
    oh_ms = disp_step * N_STEPS * OVERHEAD_US / 1000.0
    # hoisted cross-attn k/v run once per inference, not once per step
    once_ms = (2 * US["kv_cross"] * N_ODD / 1000.0) if hoist_kv else 0.0
    return dict(
        name=name,
        dispatches=disp_step * N_STEPS + (2 * N_ODD if hoist_kv else 0),
        device_ms=dev_ms + once_ms,
        overhead_ms=oh_ms,
        total_ms=dev_ms + once_ms + oh_ms,
        per_step_ms=dev_step / 1000.0,
    )


def main():
    rows = [
        scenario("A. unfused, today's kernels", US["fa_even_ok"], US["fa_odd_ok"]),
        scenario(
            "B. + weight-concat q/k/v & gate/up, hoisted cross K/V",
            US["fa_even_ok"],
            US["fa_odd_ok"],
            qkv_concat=True,
            gateup_concat=True,
            hoist_kv=True,
        ),
        scenario(
            "C. B + a working small-Q (50->64) FlashAttention",
            US["fa_even_q64"],
            US["fa_odd_q64"],
            qkv_concat=True,
            gateup_concat=True,
            hoist_kv=True,
        ),
    ]
    print(f"{'scenario':<52} {'disp':>6} {'device':>9} {'ovhd':>7} {'TOTAL':>9}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['name']:<52} {r['dispatches']:>6} {r['device_ms']:>8.1f}ms "
            f"{r['overhead_ms']:>6.1f}ms {r['total_ms']:>8.1f}ms"
        )

    print(
        "\nSame op set folded into 3 multi-launch ELFs/layer (the 'fuse like vision'"
        "\nscenario), applying the fusion penalty measured on the two real backbone"
        f"\nfused ELFs this session ({FUSION_PENALTY[0]}x - {FUSION_PENALTY[1]}x):"
    )
    b = rows[1]
    lo = b["device_ms"] * FUSION_PENALTY[0]
    hi = b["device_ms"] * FUSION_PENALTY[1]
    fused_disp = 3 * N_LAYERS * N_STEPS + N_STEPS
    fused_oh = fused_disp * OVERHEAD_US / 1000.0
    print(
        f"  dispatches {fused_disp}, device {lo:.0f}-{hi:.0f}ms, "
        f"overhead {fused_oh:.1f}ms -> TOTAL {lo + fused_oh:.0f}-{hi + fused_oh:.0f}ms"
    )
    print(
        "  i.e. fusion REMOVES ~"
        f"{(b['dispatches'] - fused_disp) * OVERHEAD_US / 1000.0:.0f}ms of dispatch cost "
        f"and ADDS {lo - b['device_ms']:.0f}-{hi - b['device_ms']:.0f}ms of device cost."
    )

    # ---- where the time goes (scenario A) ----
    print("\nScenario A device-time budget per inference (ms):")
    ev = even_layer(US["fa_even_ok"])
    od = odd_layer(US["fa_odd_ok"])
    buckets = {}
    for tag, ops, n in (("even", ev, N_EVEN), ("odd", od, N_ODD)):
        for name, t in ops:
            key = (
                "attention"
                if name == "attn"
                else (
                    "RMSNorm"
                    if name.startswith("rmsnorm")
                    else (
                        "RoPE"
                        if name.startswith("rope")
                        else (
                            "eltwise add"
                            if name.startswith("add")
                            else "SiLU" if name == "silu_mul" else "GEMM (proj/FFN)"
                        )
                    )
                )
            )
            buckets[key] = buckets.get(key, 0.0) + t * n * N_STEPS / 1000.0
    buckets["head/tail"] = sum(t for _, t in head_tail()) * N_STEPS / 1000.0
    tot = sum(buckets.values())
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v:8.1f} ms  ({100*v/tot:4.1f}%)")
    print(f"  {'TOTAL':<20} {tot:8.1f} ms")

    # ---- how much is pure per-launch floor ----
    FLOOR = 85.0  # measured: 64x720x32 GEMM 79.2us, 64x32x32 GEMM 106us,
    #              64x720 eltwise add 86.6us -> ~85us is what ANY launch costs
    a = rows[0]
    floor_ms = a["dispatches"] * FLOOR / 1000.0
    print(
        f"\nPer-launch device floor: {a['dispatches']} launches x ~{FLOOR:.0f}us = "
        f"{floor_ms:.0f} ms = {100*floor_ms/a['device_ms']:.0f}% of scenario A's device time."
    )

    # ---- useful FLOPs ----
    ev_f = (
        2 * 50 * 720 * 960
        + 2 * (2 * 50 * 720 * 320)
        + 2 * 50 * 960 * 720
        + 2 * (2 * 50 * 720 * 2048)
        + 2 * 50 * 2048 * 720
        + 2 * 2 * 15 * 50 * 291 * 64
    )
    od_f = (
        2 * 50 * 720 * 960
        + 2 * (2 * 241 * 320 * 320)
        + 2 * 50 * 960 * 720
        + 2 * (2 * 50 * 720 * 2048)
        + 2 * 50 * 2048 * 720
        + 2 * 2 * 15 * 50 * 241 * 64
    )
    gflop = (N_EVEN * ev_f + N_ODD * od_f) * N_STEPS / 1e9
    print(f"\nUseful FLOPs per inference (no padding): {gflop:.1f} GFLOP")
    for r in rows:
        print(
            f"  {r['name']:<52} -> {gflop / (r['total_ms'] / 1000.0):7.0f} GFLOP/s effective"
        )


if __name__ == "__main__":
    main()
