"""Per-operation CPU wall-clock breakdown of SmolVLA's action expert.

Mirrors `bench_cpu_siglip_ops.py` (the vision-side breakdown) in rigour:
every op is timed individually at its REAL shape, and every level closes
against an independently-measured wall clock.

Three levels, each with its own closure check:

  L0  chunk        -- predict_action_chunk wall, and the expert's share of it
  L1  denoise step -- embed_suffix / 16 layers / final norm / action_out_proj
  L2  per-op       -- replay of one EVEN (self-attn) and one ODD (cross-attn)
                      expert layer, op by op, checked against the real layer

Run with the lerobot venv (this worktree's python has air/pyxrt but no torch):

    /home/jiajli/Projects/smolvla_playground/.venv/bin/python \
        scripts/bench_cpu_expert_ops.py

Shapes are read off the live checkpoint, not hardcoded, and printed with
every row so the NPU side can be lined up against them.
"""

import argparse
import statistics
import subprocess
import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL = "lerobot/smolvla_base"


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
def med(xs):
    return statistics.median(xs)


class Acc:
    """Ordered accumulator of per-op sample lists (microseconds)."""

    def __init__(self):
        self.d = OrderedDict()
        self.shape = {}

    def timed(self, name, fn, shape=""):
        t0 = time.perf_counter()
        r = fn()
        dt = (time.perf_counter() - t0) * 1e6
        self.d.setdefault(name, []).append(dt)
        if shape:
            self.shape[name] = shape
        return r

    def report(self, title, whole_us, whole_label="whole (independent)"):
        print(f"\n--- {title} ---")
        print(f"{'op':34s} {'shape':26s} {'us (median)':>12s} {'share':>8s}")
        tot = sum(med(v) for v in self.d.values())
        for k, v in self.d.items():
            m = med(v)
            print(
                f"{k:34s} {self.shape.get(k, ''):26s} {m:12.1f} {100 * m / tot:7.1f}%"
            )
        print("-" * 84)
        print(f"{'SUM of parts':34s} {'':26s} {tot:12.1f}")
        print(f"{whole_label:34s} {'':26s} {whole_us:12.1f}")
        gap = whole_us - tot
        print(f"{'closure gap':34s} {'':26s} {gap:12.1f} {100 * gap / whole_us:7.2f}%")
        return tot, whole_us


def build_batch(p):
    cfg = p.config
    vwe = p.model.vlm_with_expert
    b = {}
    for k, f in cfg.input_features.items():
        b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
    tok = vwe.processor.tokenizer(
        ["pick up the cube"],
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b


# ---------------------------------------------------------------------------
# L0 + L1: stage wall clock, measured in situ on the real model
# ---------------------------------------------------------------------------
def level01(p, batch, n_infer, warmup):
    cfg = p.config
    m = p.model
    vwe = m.vlm_with_expert

    noise = torch.zeros((1, cfg.chunk_size, cfg.max_action_dim))

    stage = defaultdict(list)  # name -> per-denoise-step us
    step_wall = []
    chunk_wall = []
    live = [False]

    # -- wrap the pieces of denoise_step --
    o_embed_suffix = m.embed_suffix
    o_vwe_forward = vwe.forward
    o_action_out = m.action_out_proj
    o_self_layer = vwe.forward_attn_layer
    o_cross_layer = vwe.forward_cross_attn_layer
    o_denoise = m.denoise_step

    cur = defaultdict(float)  # accumulates within one denoise step

    def wrap(name, fn):
        def inner(*a, **kw):
            if not live[0]:
                return fn(*a, **kw)
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            cur[name] += (time.perf_counter() - t0) * 1e6
            return r

        return inner

    m.embed_suffix = wrap("embed_suffix", o_embed_suffix)
    vwe.forward = wrap("vwe.forward (16 layers)", o_vwe_forward)
    vwe.forward_attn_layer = wrap("  even: attn block", o_self_layer)
    vwe.forward_cross_attn_layer = wrap("  odd: attn block", o_cross_layer)

    # o_proj / post_attention_layernorm / mlp of the expert, split by parity
    handles = []
    tstack = {}

    def mk(cat):
        def pre(mod, i):
            tstack[id(mod)] = time.perf_counter()

        def post(mod, i, o):
            if live[0]:
                cur[cat] += (
                    time.perf_counter() - tstack.pop(id(mod), time.perf_counter())
                ) * 1e6
            else:
                tstack.pop(id(mod), None)

        return pre, post

    for idx, layer in enumerate(vwe.lm_expert.layers):
        par = "even" if idx % 2 == 0 else "odd"
        for mod, tag in (
            (layer.self_attn.o_proj, f"  {par}: o_proj"),
            (layer.post_attention_layernorm, f"  {par}: post_attn_norm"),
            (layer.mlp, f"  {par}: mlp"),
        ):
            pre, post = mk(tag)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))
    pre, post = mk("  final norm")
    handles.append(vwe.lm_expert.norm.register_forward_pre_hook(pre))
    handles.append(vwe.lm_expert.norm.register_forward_hook(post))
    pre, post = mk("action_out_proj")
    handles.append(o_action_out.register_forward_pre_hook(pre))
    handles.append(o_action_out.register_forward_hook(post))

    def wrapped_denoise(*a, **kw):
        live[0] = True
        cur.clear()
        t0 = time.perf_counter()
        try:
            r = o_denoise(*a, **kw)
        finally:
            step_wall.append((time.perf_counter() - t0) * 1e6)
            live[0] = False
            for k, v in cur.items():
                stage[k].append(v)
        return r

    m.denoise_step = wrapped_denoise

    def one():
        p.reset()
        with torch.no_grad():
            p.predict_action_chunk(batch, noise=noise)

    for _ in range(warmup):
        one()
    stage.clear()
    step_wall.clear()

    for _ in range(n_infer):
        t0 = time.perf_counter()
        one()
        chunk_wall.append((time.perf_counter() - t0) * 1e3)

    # restore
    for h in handles:
        h.remove()
    m.embed_suffix = o_embed_suffix
    vwe.forward = o_vwe_forward
    vwe.forward_attn_layer = o_self_layer
    vwe.forward_cross_attn_layer = o_cross_layer
    m.denoise_step = o_denoise

    n_steps = cfg.num_steps
    print("\n" + "=" * 84)
    print("L0/L1: expert stage wall clock (real model, in situ)")
    print("=" * 84)
    print(f"  chunks measured      : {n_infer} (warmup {warmup})")
    print(
        f"  denoise steps/chunk  : {n_steps}   (denoise_step calls seen: {len(step_wall) // n_infer}/chunk)"
    )
    print(
        f"  predict_action_chunk : {med(chunk_wall):8.1f} ms   (median of {n_infer}, "
        f"min {min(chunk_wall):.1f} max {max(chunk_wall):.1f})"
    )
    step_ms = med(step_wall) / 1e3
    print(
        f"  denoise step         : {step_ms:8.2f} ms   (median of {len(step_wall)}, "
        f"min {min(step_wall) / 1e3:.2f} max {max(step_wall) / 1e3:.2f})"
    )
    print(
        f"  expert (10 steps)    : {step_ms * n_steps:8.1f} ms   = {100 * step_ms * n_steps / med(chunk_wall):.1f}% of the chunk"
    )

    print(
        f"\n{'stage (per denoise step)':34s} {'us (median)':>12s} {'ms/inference':>14s} {'share of step':>14s}"
    )
    order = [
        "embed_suffix",
        "vwe.forward (16 layers)",
        "  even: attn block",
        "  even: o_proj",
        "  even: post_attn_norm",
        "  even: mlp",
        "  odd: attn block",
        "  odd: o_proj",
        "  odd: post_attn_norm",
        "  odd: mlp",
        "  final norm",
        "action_out_proj",
    ]
    stepus = med(step_wall)
    for k in order:
        if k not in stage:
            continue
        v = med(stage[k])
        print(f"{k:34s} {v:12.1f} {v * n_steps / 1e3:14.2f} {100 * v / stepus:13.1f}%")

    # closure: denoise step = embed_suffix + vwe.forward + action_out_proj + glue
    top = sum(
        med(stage[k])
        for k in ("embed_suffix", "vwe.forward (16 layers)", "action_out_proj")
        if k in stage
    )
    print("-" * 78)
    print(f"{'SUM (embed+forward+out_proj)':34s} {top:12.1f}")
    print(f"{'denoise_step wall':34s} {stepus:12.1f}")
    print(
        f"{'glue (masks, slicing, python)':34s} {stepus - top:12.1f} {'':14s} {100 * (stepus - top) / stepus:13.2f}%"
    )

    inner = sum(med(stage[k]) for k in order if k.startswith("  ") and k in stage)
    fwd = med(stage["vwe.forward (16 layers)"])
    print(f"\n{'SUM of 16-layer internals':34s} {inner:12.1f}")
    print(f"{'vwe.forward wall':34s} {fwd:12.1f}")
    print(
        f"{'layer-loop glue (residual/clone)':34s} {fwd - inner:12.1f} {'':14s} {100 * (fwd - inner) / fwd:13.2f}%"
    )

    return {
        "chunk_ms": med(chunk_wall),
        "step_ms": step_ms,
        "expert_ms": step_ms * n_steps,
        "stage_us": {k: med(v) for k, v in stage.items()},
    }


# ---------------------------------------------------------------------------
# L2: per-op replay of one even and one odd expert layer
# ---------------------------------------------------------------------------
def capture_layer_inputs(p, batch):
    """Run one real inference, stash the arguments the expert layers see."""
    cfg = p.config
    m = p.model
    vwe = m.vlm_with_expert
    grab = {}

    o_self = vwe.forward_attn_layer
    o_cross = vwe.forward_cross_attn_layer

    def snap(
        kind, inputs_embeds, layer_idx, position_ids, attention_mask, past_key_values
    ):
        if kind in grab or inputs_embeds[1] is None or past_key_values is None:
            return
        grab[kind] = {
            "x": inputs_embeds[1].detach().clone(),
            "layer_idx": layer_idx,
            "position_ids": position_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "k_cache": past_key_values[layer_idx]["key_states"].detach().clone(),
            "v_cache": past_key_values[layer_idx]["value_states"].detach().clone(),
        }

    def w_self(
        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, *a, **kw
    ):
        snap(
            "even",
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            kw.get("past_key_values"),
        )
        return o_self(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            *a,
            **kw,
        )

    def w_cross(
        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, *a, **kw
    ):
        snap(
            "odd",
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            kw.get("past_key_values"),
        )
        return o_cross(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            *a,
            **kw,
        )

    vwe.forward_attn_layer = w_self
    vwe.forward_cross_attn_layer = w_cross
    p.reset()
    with torch.no_grad():
        p.predict_action_chunk(
            batch, noise=torch.zeros((1, cfg.chunk_size, cfg.max_action_dim))
        )
    vwe.forward_attn_layer = o_self
    vwe.forward_cross_attn_layer = o_cross
    return grab


def eager_parts(acc, vwe, mask, q, k, v, head_dim, tag):
    """Replay eager_attention_forward op by op (source: smolvlm_with_expert.py:504)."""
    B = q.shape[0]
    nh, nkv = vwe.num_attention_heads, vwe.num_key_value_heads
    grp = nh // nkv
    S = k.shape[1]
    sq = q.shape[1]

    def expand_kv():
        kk = (
            k[:, :, :, None, :]
            .expand(B, S, nkv, grp, head_dim)
            .reshape(B, S, nkv * grp, head_dim)
        )
        vv = (
            v[:, :, :, None, :]
            .expand(B, S, nkv, grp, head_dim)
            .reshape(B, S, nkv * grp, head_dim)
        )
        return kk, vv

    kk, vv = acc.timed(f"{tag}GQA expand k,v", expand_kv, f"5->15 heads, {S}x64")
    qf = acc.timed(
        f"{tag}cast q,k -> fp32",
        lambda: (q.to(torch.float32), kk.to(torch.float32)),
        f"{sq}x960 / {S}x960",
    )
    qf, kf = qf
    tr = acc.timed(
        f"{tag}transpose q,k",
        lambda: (qf.transpose(1, 2), kf.transpose(1, 2)),
        "B,H,S,D",
    )
    qt, kt = tr
    aw = acc.timed(
        f"{tag}QK^T (fp32)",
        lambda: torch.matmul(qt, kt.transpose(2, 3)),
        f"15x{sq}x64x{S}",
    )
    aw = acc.timed(f"{tag}scale", lambda: aw * (head_dim**-0.5), f"15x{sq}x{S}")
    big_neg = torch.finfo(torch.float32).min
    aw = acc.timed(
        f"{tag}mask where",
        lambda: torch.where(mask[:, None, :, :], aw, big_neg),
        f"15x{sq}x{S}",
    )
    pr = acc.timed(
        f"{tag}softmax (fp32)",
        lambda: nn.functional.softmax(aw, dim=-1),
        f"15x{sq}x{S}",
    )
    pr = acc.timed(
        f"{tag}cast probs -> bf16", lambda: pr.to(dtype=vv.dtype), f"15x{sq}x{S}"
    )
    out = acc.timed(
        f"{tag}PV", lambda: torch.matmul(pr, vv.permute(0, 2, 1, 3)), f"15x{sq}x{S}x64"
    )
    out = acc.timed(
        f"{tag}permute+reshape out",
        lambda: out.permute(0, 2, 1, 3).reshape(B, -1, nkv * grp * head_dim),
        f"{sq}x960",
    )
    return out


def replay(acc, vwe, layer, g, is_self, head_dim, reps):
    """One full expert layer, op by op, faithful to smolvlm_with_expert.py."""
    x = g["x"]
    pos = g["position_ids"]
    mask_full = g["attention_mask"]
    kc, vc = g["k_cache"], g["v_cache"]
    S = x.shape[1]

    for _ in range(reps):
        h = acc.timed(
            "input_layernorm (RMSNorm)", lambda: layer.input_layernorm(x), f"{S}x720"
        )
        hs = (*h.shape[:-1], -1, layer.self_attn.head_dim)
        wdt = layer.self_attn.q_proj.weight.dtype
        h = acc.timed("cast hidden -> bf16", lambda: h.to(dtype=wdt), f"{S}x720")
        q = acc.timed(
            "q_proj", lambda: layer.self_attn.q_proj(h).view(hs), f"{S}x720x960"
        )

        if is_self:
            k = acc.timed(
                "k_proj", lambda: layer.self_attn.k_proj(h).view(hs), f"{S}x720x320"
            )
            v = acc.timed(
                "v_proj", lambda: layer.self_attn.v_proj(h).view(hs), f"{S}x720x320"
            )
            q = acc.timed("cat q (1 elem)", lambda: torch.cat([q], dim=1), "no-op copy")
            k = acc.timed("cat k (1 elem)", lambda: torch.cat([k], dim=1), "no-op copy")
            v = acc.timed("cat v (1 elem)", lambda: torch.cat([v], dim=1), "no-op copy")
            q = acc.timed("RoPE q", lambda: apply_rope(q, pos), f"{S}x15x64")
            k = acc.timed("RoPE k", lambda: apply_rope(k, pos), f"{S}x5x64")
            k = acc.timed(
                "cat k with cache", lambda: torch.cat([kc, k], dim=1), f"241+{S}=291"
            )
            v = acc.timed(
                "cat v with cache", lambda: torch.cat([vc, v], dim=1), f"241+{S}=291"
            )
            mask = mask_full
        else:
            P = kc.shape[1]
            kdt = layer.self_attn.k_proj.weight.dtype
            vdt = layer.self_attn.v_proj.weight.dtype
            _k = acc.timed(
                f"cast+view k cache -> {kdt}".replace("torch.", ""),
                lambda: kc.to(dtype=kdt).view(*kc.shape[:2], -1),
                f"{P}x320",
            )
            k = acc.timed(
                "k_proj (over prefix!)",
                lambda: layer.self_attn.k_proj(_k).view(
                    *_k.shape[:-1], -1, layer.self_attn.head_dim
                ),
                f"{P}x320x320",
            )
            _v = acc.timed(
                f"cast+view v cache -> {vdt}".replace("torch.", ""),
                lambda: vc.to(dtype=vdt).view(*vc.shape[:2], -1),
                f"{P}x320",
            )
            v = acc.timed(
                "v_proj (over prefix!)",
                lambda: layer.self_attn.v_proj(_v).view(
                    *_v.shape[:-1], -1, layer.self_attn.head_dim
                ),
                f"{P}x320x320",
            )
            pos0 = acc.timed(
                "pos - min(pos)",
                lambda: pos - torch.min(pos, dim=1, keepdim=True).values,
                f"1x{S}",
            )
            q = acc.timed("RoPE q", lambda: apply_rope(q, pos0), f"{S}x15x64")
            mask = acc.timed(
                "mask slice",
                lambda: mask_full[:, -S:, : k.shape[1]],
                f"{S}x{k.shape[1]}",
            )

        att = eager_parts(acc, vwe, mask, q, k, v, head_dim, "attn: ")

        odt = layer.self_attn.o_proj.weight.dtype
        att = acc.timed("cast att -> bf16", lambda: att.to(odt), f"{S}x960")
        att = acc.timed("attn slice [0:50]", lambda: att[:, 0:S], f"{S}x960")
        o = acc.timed("o_proj", lambda: layer.self_attn.o_proj(att), f"{S}x960x720")
        acc.timed("residual add 1 (in-place)", lambda: o.add_(x), f"{S}x720")
        r2 = acc.timed("clone", lambda: o.clone(), f"{S}x720")
        n2 = acc.timed(
            "post_attention_layernorm",
            lambda: layer.post_attention_layernorm(o),
            f"{S}x720",
        )
        gt = acc.timed("gate_proj", lambda: layer.mlp.gate_proj(n2), f"{S}x720x2048")
        up = acc.timed("up_proj", lambda: layer.mlp.up_proj(n2), f"{S}x720x2048")
        sm = acc.timed("SiLU * up", lambda: layer.mlp.act_fn(gt) * up, f"{S}x2048")
        dn = acc.timed("down_proj", lambda: layer.mlp.down_proj(sm), f"{S}x2048x720")
        acc.timed("residual add 2 (in-place)", lambda: dn.add_(r2), f"{S}x720")


def level2(p, grab, reps):
    vwe = p.model.vlm_with_expert
    head_dim = vwe.vlm.config.text_config.head_dim
    out = {}
    for kind, is_self in (("even", True), ("odd", False)):
        g = grab[kind]
        layer = vwe.lm_expert.layers[g["layer_idx"]]
        acc = Acc()
        with torch.no_grad():
            replay(acc, vwe, layer, g, is_self, head_dim, reps)

            # independent whole-block check: the real forward_*_attn_layer +
            # the outer-loop tail, timed as one unit
            fn = vwe.forward_attn_layer if is_self else vwe.forward_cross_attn_layer
            pkv = {
                g["layer_idx"]: {
                    "key_states": g["k_cache"],
                    "value_states": g["v_cache"],
                }
            }
            whole = []
            for _ in range(max(10, reps // 2)):
                t0 = time.perf_counter()
                atts, _ = fn(
                    [[None] * 32, vwe.lm_expert.layers],
                    [None, g["x"]],
                    g["layer_idx"],
                    g["position_ids"],
                    g["attention_mask"],
                    1,
                    head_dim,
                    use_cache=True,
                    fill_kv_cache=False,
                    past_key_values={k: dict(v) for k, v in pkv.items()},
                )
                a = atts[0].to(layer.self_attn.o_proj.weight.dtype)[
                    :, 0 : g["x"].shape[1]
                ]
                e = layer.self_attn.o_proj(a)
                e += g["x"]
                r2 = e.clone()
                e = layer.post_attention_layernorm(e)
                e = layer.mlp(e)
                e += r2
                whole.append((time.perf_counter() - t0) * 1e6)

        label = (
            "EVEN layer (self-attn, KV=291)"
            if is_self
            else "ODD layer (cross-attn, KV=241)"
        )
        tot, w = acc.report(f"L2: {label}", med(whole), "real layer+tail (independent)")
        out[kind] = {
            "parts_us": tot,
            "whole_us": w,
            "ops": {k: med(v) for k, v in acc.d.items()},
            "shape": dict(acc.shape),
        }
    return out


# op -> (kernel family, NPU-side counterpart). Families match the mlir-air
# kernel registry so the NPU port can be lined up row for row.
FAMILY = [
    ("q_proj", "GEMM"),
    ("k_proj", "GEMM"),
    ("v_proj", "GEMM"),
    ("o_proj", "GEMM"),
    ("gate_proj", "GEMM"),
    ("up_proj", "GEMM"),
    ("down_proj", "GEMM"),
    ("QK^T", "attention"),
    ("PV", "attention"),
    ("softmax", "attention"),
    ("mask where", "attention"),
    ("scale", "attention"),
    ("GQA expand", "attention"),
    ("layernorm", "RMSNorm"),
    ("RoPE", "RoPE"),
    ("SiLU", "SiLU-and-Mul"),
    ("residual add", "EltwiseAdd"),
]


def classify(op):
    low = op.lower()
    for key, fam in FAMILY:
        if key.lower() in low:
            return fam
    return "layout/cast glue"


def level3_rollup(p, r2, out_json):
    """Roll the per-op medians up to ms per inference, grouped by kernel family."""
    n = p.config.num_steps
    rows = {}
    for kind in ("even", "odd"):
        for op, us in r2[kind]["ops"].items():
            fam = classify(op)
            ms = us * 8 * n / 1e3  # 8 layers of this parity x 10 denoise steps
            e = rows.setdefault(
                (fam, op.replace("attn: ", "")),
                {"even": 0.0, "odd": 0.0, "shape": {}},
            )
            e[kind] += ms
            e["shape"][kind] = r2[kind]["shape"].get(op, "")

    print("\n" + "=" * 84)
    print("L3: per-inference roll-up by kernel family (16 layers x 10 denoise steps)")
    print("=" * 84)
    fam_tot = defaultdict(float)
    for (fam, _), e in rows.items():
        fam_tot[fam] += e["even"] + e["odd"]
    grand = sum(fam_tot.values())

    print(f"{'family':18s} {'ms/inference':>13s} {'share':>8s}")
    for fam, ms in sorted(fam_tot.items(), key=lambda x: -x[1]):
        print(f"{fam:18s} {ms:13.1f} {100 * ms / grand:7.1f}%")
    print("-" * 42)
    print(f"{'TOTAL (L2 basis)':18s} {grand:13.1f}")

    print(
        f"\n{'family':16s} {'op':26s} {'even ms':>9s} {'odd ms':>9s} {'total':>8s} {'shape (even / odd)'}"
    )
    for (fam, op), e in sorted(
        rows.items(), key=lambda x: -(x[1]["even"] + x[1]["odd"])
    ):
        se, so = e["shape"].get("even", ""), e["shape"].get("odd", "")
        sh = se if se == so else f"{se or '-'} / {so or '-'}"
        print(
            f"{fam:16s} {op:26s} {e['even']:9.1f} {e['odd']:9.1f} "
            f"{e['even'] + e['odd']:8.1f} {sh}"
        )

    if out_json:
        import json

        with open(out_json, "w") as f:
            json.dump(
                {
                    "families": dict(fam_tot),
                    "ops": [
                        {
                            "family": fam,
                            "op": op,
                            "even_ms": e["even"],
                            "odd_ms": e["odd"],
                            "total_ms": e["even"] + e["odd"],
                            "shape": e["shape"],
                        }
                        for (fam, op), e in rows.items()
                    ],
                },
                f,
                indent=2,
            )
        print(f"\nwrote {out_json}")


def level2b(p, grab, reps):
    """Why L2 (hot single layer) undercounts vs L1 (16 distinct layers).

    Hypothesis: the real loop streams a different layer's weights every block,
    so it misses L3; the single-layer replay re-reads the same ~12 MB and
    partially hits. Test: run the identical whole-block over ONE layer hot vs
    round-robin over all 16, same op count.
    """
    vwe = p.model.vlm_with_expert
    head_dim = vwe.vlm.config.text_config.head_dim
    layers = vwe.lm_expert.layers

    wbytes = sum(t.numel() * t.element_size() for t in layers[0].parameters())
    print("\n" + "=" * 84)
    print("L2b: weight working set -- why the hot replay undercounts")
    print("=" * 84)
    print(f"  weights per expert layer : {wbytes / 1e6:7.2f} MB")
    print(f"  all 16 layers            : {wbytes * 16 / 1e6:7.1f} MB  (vs 24 MB L3)")
    print(
        f"  streamed per inference   : {wbytes * 16 * p.config.num_steps / 1e9:7.2f} GB  (16 layers x 10 steps)"
    )

    def block(layer, g, is_self):
        fn = vwe.forward_attn_layer if is_self else vwe.forward_cross_attn_layer
        atts, _ = fn(
            [[None] * 32, layers],
            [None, g["x"]],
            g["layer_idx"],
            g["position_ids"],
            g["attention_mask"],
            1,
            head_dim,
            use_cache=True,
            fill_kv_cache=False,
            past_key_values={
                g["layer_idx"]: {
                    "key_states": g["k_cache"],
                    "value_states": g["v_cache"],
                }
            },
        )
        a = atts[0].to(layer.self_attn.o_proj.weight.dtype)[:, 0 : g["x"].shape[1]]
        e = layer.self_attn.o_proj(a)
        e += g["x"]
        r2 = e.clone()
        e = layer.post_attention_layernorm(e)
        e = layer.mlp(e)
        e += r2

    with torch.no_grad():
        for kind, is_self in (("even", True), ("odd", False)):
            g0 = grab[kind]
            idxs = [i for i in range(16) if (i % 2 == 0) == is_self]

            hot = []
            for _ in range(reps):
                t0 = time.perf_counter()
                block(layers[g0["layer_idx"]], g0, is_self)
                hot.append((time.perf_counter() - t0) * 1e6)

            rr = []
            for r in range(reps):
                li = idxs[r % len(idxs)]
                g = dict(g0)
                g["layer_idx"] = li
                t0 = time.perf_counter()
                block(layers[li], g, is_self)
                rr.append((time.perf_counter() - t0) * 1e6)

            print(
                f"  {kind:4s}: hot single layer {med(hot):7.1f} us | "
                f"round-robin over {len(idxs)} layers {med(rr):7.1f} us | "
                f"ratio {med(rr) / med(hot):.2f}x"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-infer", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--skip-l2", action="store_true")
    ap.add_argument("--json", default="results/cpu_expert_ops.json")
    args = ap.parse_args()

    try:
        for line in subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        ).stdout.splitlines():
            if "Model name" in line:
                print(line.strip())
    except Exception:
        pass
    gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    epp = (
        open("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
        .read()
        .strip()
    )
    print(
        f"governor={gov}  EPP={epp}  torch={torch.__version__}  threads={torch.get_num_threads()}"
    )

    torch.set_grad_enabled(False)
    p = SmolVLAPolicy.from_pretrained(MODEL).eval()
    vwe = p.model.vlm_with_expert
    print(
        f"expert: {len(vwe.lm_expert.layers)} layers, hidden={vwe.lm_expert.layers[0].self_attn.q_proj.in_features}, "
        f"q={vwe.num_attention_heads}h kv={vwe.num_key_value_heads}h head_dim={vwe.vlm.config.text_config.head_dim}, "
        f"chunk={p.config.chunk_size}, steps={p.config.num_steps}, dtype={next(vwe.lm_expert.parameters()).dtype}"
    )

    # dtype audit -- cross-attn layers turn out NOT to be uniformly bf16
    print("  weight dtypes:")
    for idx in (0, 1):
        sa = vwe.lm_expert.layers[idx].self_attn
        print(
            f"    layer {idx} ({'even/self' if idx % 2 == 0 else 'odd/cross'}): "
            + " ".join(
                f"{n}={str(getattr(sa, n).weight.dtype).replace('torch.', '')}"
                for n in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        )

    batch = build_batch(p)
    r01 = level01(p, batch, args.n_infer, args.warmup)

    if not args.skip_l2:
        grab = capture_layer_inputs(p, batch)
        print(
            f"\ncaptured: even layer_idx={grab['even']['layer_idx']}, odd layer_idx={grab['odd']['layer_idx']}, "
            f"x={tuple(grab['even']['x'].shape)}, k_cache={tuple(grab['even']['k_cache'].shape)}, "
            f"mask={tuple(grab['even']['attention_mask'].shape)}"
        )
        r2 = level2(p, grab, args.reps)
        level2b(p, grab, args.reps)
        level3_rollup(p, r2, args.json)

        # roll L2 up to a per-inference number and check against L1
        n = p.config.num_steps
        even_ms = r2["even"]["whole_us"] * 8 * n / 1e3
        odd_ms = r2["odd"]["whole_us"] * 8 * n / 1e3
        print("\n" + "=" * 84)
        print("L2 -> L1 roll-up check")
        print("=" * 84)
        print(f"  8 even layers x 10 steps : {even_ms:7.1f} ms")
        print(f"  8 odd  layers x 10 steps : {odd_ms:7.1f} ms")
        print(f"  sum (16 layers)          : {even_ms + odd_ms:7.1f} ms")
        fwd_ms = r01["stage_us"].get("vwe.forward (16 layers)", 0) * n / 1e3
        print(
            f"  L1 vwe.forward           : {fwd_ms:7.1f} ms   (delta {even_ms + odd_ms - fwd_ms:+.1f} ms)"
        )
        print(f"  L1 expert total          : {r01['expert_ms']:7.1f} ms")


if __name__ == "__main__":
    main()
