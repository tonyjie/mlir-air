"""CPU-only timing of SmolVLA's "action expert" for a feasibility study.

Pure PyTorch/CPU work -- no NPU, no flock, no compilation. Run with the
lerobot venv (has torch + lerobot; this worktree's default python has
air/pyxrt but not torch):

    /home/jiajli/Projects/smolvla_playground/.venv/bin/python bench_expert_cpu.py

Flags to skip individual tasks (each is independent):
    --skip-real         skip Task 1 (loads the real lerobot/smolvla_base checkpoint)
    --skip-crosscheck   skip Task 2 (standalone equivalent-shape torch stack)
    --skip-gemm         skip Task 3 (per-GEMM CPU timing)

Expert shape (confirmed by reading modeling_smolvla.py / smolvlm_with_expert.py
against the real "lerobot/smolvla_base" checkpoint):
  hidden=720, intermediate=2048, 16 layers, GQA 15 q-heads/5 kv-heads, head_dim=64.
  chunk_size (action tokens) = 50, num_steps (denoise loop) = 10.
  self_attn_every_n_layers=2 -> layer_idx % 2 == 0 is self-attn (even), else
  cross-attn (odd), confirmed empirically against the loaded checkpoint.

IMPORTANT correction vs. the naive "uniform per-layer shape" assumption:
  - SELF layers (even): q 50x720x960, k/v 50x720x320, o 50x960x720 -- all over
    the 50 action tokens, as expected.
  - CROSS layers (odd): q is still 50x720x960 (over the 50 action tokens), but
    k_proj/v_proj are REPLACED at construction time with fresh nn.Linear(320,320)
    that re-project the *backbone's* cached raw K/V (241 tokens x 320) into the
    expert's own K/V space -- i.e. cross-layer k/v proj is actually
    241x320x320, run over 241 tokens, NOT the 50 action tokens, and it is
    RECOMPUTED every denoise step (only the backbone's raw 241-token K/V is
    cached across steps; the expert's re-projection of it is not). See
    forward_cross_attn_layer() in smolvlm_with_expert.py. This is measured
    both in Task 1 (real model) and included as a bonus row in Task 3.
  - state_proj (1x32x960) runs ONCE during prefix embedding (backbone side),
    NOT inside the 10-step denoise loop -- it is not part of the expert's
    per-step cost despite being one of the "small" projections.
"""

import argparse
import collections
import copy
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared expert-shape constants (see module docstring for provenance)
# ---------------------------------------------------------------------------
HIDDEN = 720
INTERMEDIATE = 2048
NUM_LAYERS = 16
NUM_Q_HEADS = 15
NUM_KV_HEADS = 5
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 960
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 320
GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 3
SEQ = 50  # action tokens (chunk_size)
PREFIX_LEN = 241  # backbone prefix cache length (vision+lang+state)
NUM_STEPS = 10  # denoise loop iterations per inference
ACTION_DIM = 32


def _median_us(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6  # us


def _gflops(flops, us):
    return flops / (us * 1e-6) / 1e9


# =============================================================================
# Task 1: instrument the REAL lerobot/smolvla_base checkpoint
# =============================================================================
def task1_real_expert(n_infer=6, warmup=3):
    print("\n" + "=" * 78)
    print("TASK 1: real lerobot/smolvla_base action expert (CPU)")
    print("=" * 78)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    MODEL = "lerobot/smolvla_base"
    torch.manual_seed(0)
    p = SmolVLAPolicy.from_pretrained(MODEL).eval()
    cfg = p.config
    vwe = p.model.vlm_with_expert

    def build_batch():
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

    batch = build_batch()

    # -- instrumentation state --
    in_denoise = [False]
    denoise_step_times = []  # flat pool, one entry per denoise_step() call
    self_layer_time = collections.defaultdict(float)
    cross_layer_time = collections.defaultdict(float)
    attn_math_time = [0.0]
    attn_math_calls = [0]
    op_time = collections.defaultdict(float)
    op_calls = collections.defaultdict(int)

    def classify_linear(name):
        if "action_in_proj" in name:
            return "io_action_in (50x32x720)"
        if "action_time_mlp_in" in name:
            return "io_time_mlp_in (50x1440x720)"
        if "action_time_mlp_out" in name:
            return "io_time_mlp_out (50x720x720)"
        if "action_out_proj" in name:
            return "io_action_out (50x720x32)"
        if "state_proj" in name:
            return "io_state_proj (SHOULD NOT FIRE under denoise flag)"
        if "lm_expert.layers." in name:
            idx = int(name.split("lm_expert.layers.")[1].split(".")[0])
            kind = "self" if idx % 2 == 0 else "cross"
            if "self_attn" in name:
                return f"{kind}_attn_proj"
            if "mlp" in name:
                return f"{kind}_mlp_proj"
        return None

    handles = []
    t_stack = {}
    for name, mod in p.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        cat = classify_linear(name)
        if cat is None:
            continue

        def pre(m, i):
            t_stack[id(m)] = time.perf_counter()

        def post(m, i, o, cat=cat):
            if not in_denoise[0]:
                t_stack.pop(id(m), None)
                return
            dt = time.perf_counter() - t_stack.pop(id(m), time.perf_counter())
            op_time[cat] += dt
            op_calls[cat] += 1

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    # -- wrap forward_attn_layer / forward_cross_attn_layer / eager_attention_forward --
    orig_self_layer = vwe.forward_attn_layer
    orig_cross_layer = vwe.forward_cross_attn_layer
    orig_attn_math = vwe.eager_attention_forward

    def wrapped_self_layer(model_layers, inputs_embeds, layer_idx, *a, **kw):
        t0 = time.perf_counter()
        out = orig_self_layer(model_layers, inputs_embeds, layer_idx, *a, **kw)
        if in_denoise[0]:
            self_layer_time[layer_idx] += time.perf_counter() - t0
        return out

    def wrapped_cross_layer(model_layers, inputs_embeds, layer_idx, *a, **kw):
        t0 = time.perf_counter()
        out = orig_cross_layer(model_layers, inputs_embeds, layer_idx, *a, **kw)
        if in_denoise[0]:
            cross_layer_time[layer_idx] += time.perf_counter() - t0
        return out

    def wrapped_attn_math(*a, **kw):
        t0 = time.perf_counter()
        out = orig_attn_math(*a, **kw)
        if in_denoise[0]:
            attn_math_time[0] += time.perf_counter() - t0
            attn_math_calls[0] += 1
        return out

    vwe.forward_attn_layer = wrapped_self_layer
    vwe.forward_cross_attn_layer = wrapped_cross_layer
    vwe.eager_attention_forward = wrapped_attn_math

    orig_denoise_step = p.model.denoise_step

    def wrapped_denoise_step(*a, **kw):
        in_denoise[0] = True
        t0 = time.perf_counter()
        try:
            out = orig_denoise_step(*a, **kw)
        finally:
            denoise_step_times.append(time.perf_counter() - t0)
            in_denoise[0] = False
        return out

    p.model.denoise_step = wrapped_denoise_step

    def clear_accumulators():
        denoise_step_times.clear()
        self_layer_time.clear()
        cross_layer_time.clear()
        attn_math_time[0] = 0.0
        attn_math_calls[0] = 0
        op_time.clear()
        op_calls.clear()

    def run_one_inference():
        p.reset()
        with torch.no_grad():
            p.predict_action_chunk(
                batch, noise=torch.zeros((1, cfg.chunk_size, cfg.max_action_dim))
            )

    # warmup
    for _ in range(warmup):
        run_one_inference()
    clear_accumulators()

    # measured
    whole_times = []
    per_inference_expert_ms = []
    for _ in range(n_infer):
        start_idx = len(denoise_step_times)
        t0 = time.perf_counter()
        run_one_inference()
        whole_times.append(time.perf_counter() - t0)
        this_run_steps = denoise_step_times[start_idx:]
        assert (
            len(this_run_steps) == NUM_STEPS
        ), f"expected {NUM_STEPS} denoise_step calls, got {len(this_run_steps)}"
        per_inference_expert_ms.append(sum(this_run_steps) * 1e3)

    for h in handles:
        h.remove()
    vwe.forward_attn_layer = orig_self_layer
    vwe.forward_cross_attn_layer = orig_cross_layer
    vwe.eager_attention_forward = orig_attn_math
    p.model.denoise_step = orig_denoise_step

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    ms_per_step = med(denoise_step_times) * 1e3
    ms_per_inference_expert = med(per_inference_expert_ms)
    ms_per_inference_whole = med(whole_times) * 1e3

    print(f"\nn_infer={n_infer} (warmup={warmup}), {NUM_STEPS} denoise steps/inference")
    print(
        f"  ms/denoise-step (median of {len(denoise_step_times)} calls): {ms_per_step:.2f} ms"
    )
    print(
        f"  ms/inference, EXPERT ONLY (sum of 10 steps, median of {n_infer}): {ms_per_inference_expert:.1f} ms"
    )
    print(
        f"  ms/inference, FULL predict_action_chunk (vision+backbone+expert, median): {ms_per_inference_whole:.1f} ms"
    )

    print(
        "\n--- self-attn vs cross-attn ATTENTION-BLOCK wall time ---\n"
        "    (forward_attn_layer/forward_cross_attn_layer only: q/k/v proj + RoPE +\n"
        "     attention math. Does NOT include o_proj or mlp -- those run in the\n"
        "     outer loop and are folded into *_attn_proj/*_mlp_proj in the GEMM\n"
        "     breakdown below, which is the complete per-category accounting.)"
    )
    self_total = sum(self_layer_time.values()) / n_infer * 1e3
    cross_total = sum(cross_layer_time.values()) / n_infer * 1e3
    print(
        f"  self-attn layers  (8 of 16, even idx): {self_total:8.2f} ms/inference total  ({self_total / len(self_layer_time):.2f} ms/layer avg)"
    )
    print(
        f"  cross-attn layers (8 of 16, odd idx):  {cross_total:8.2f} ms/inference total  ({cross_total / len(cross_layer_time):.2f} ms/layer avg)"
    )

    print("\n--- GEMM (nn.Linear) vs attention-math time, per inference ---")
    total_linear = sum(op_time.values()) / n_infer * 1e3
    attn_math_ms = attn_math_time[0] / n_infer * 1e3
    print(f"  all nn.Linear (GEMM) in expert+io-mlps: {total_linear:8.2f} ms/inference")
    print(
        f"  eager_attention_forward (attn math):    {attn_math_ms:8.2f} ms/inference  ({attn_math_calls[0] / n_infer:.0f} calls/inference)"
    )

    print("\n--- per-category GEMM breakdown (ms/inference, avg per-call us) ---")
    for cat, t in sorted(op_time.items(), key=lambda x: -x[1]):
        ms = t / n_infer * 1e3
        calls = op_calls[cat] / n_infer
        per_call_us = (t / op_calls[cat]) * 1e6 if op_calls[cat] else 0.0
        print(
            f"  {cat:38s}: {ms:7.2f} ms/inference  {calls:5.1f} calls/inference  {per_call_us:7.1f} us/call"
        )

    print(f"\n  torch.get_num_threads() = {torch.get_num_threads()}")

    return {
        "ms_per_step": ms_per_step,
        "ms_per_inference_expert": ms_per_inference_expert,
        "ms_per_inference_whole": ms_per_inference_whole,
        "self_total_ms": self_total,
        "cross_total_ms": cross_total,
        "linear_total_ms": total_linear,
        "attn_math_ms": attn_math_ms,
    }


# =============================================================================
# Task 2: standalone equivalent-shape torch stack (cross-check on Task 1)
# =============================================================================
class _ExpertLayer(nn.Module):
    def __init__(self, is_self: bool, dtype):
        super().__init__()
        self.is_self = is_self
        self.q_proj = nn.Linear(HIDDEN, Q_DIM, bias=False, dtype=dtype)
        if is_self:
            self.k_proj = nn.Linear(HIDDEN, KV_DIM, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(HIDDEN, KV_DIM, bias=False, dtype=dtype)
        else:
            # cross layers: k/v proj re-project the backbone's cached raw K/V
            # (241 x 320), NOT the 50 action tokens -- see module docstring.
            self.k_proj = nn.Linear(KV_DIM, KV_DIM, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(KV_DIM, KV_DIM, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(Q_DIM, HIDDEN, bias=False, dtype=dtype)
        self.ln1 = nn.RMSNorm(HIDDEN, dtype=dtype)
        self.ln2 = nn.RMSNorm(HIDDEN, dtype=dtype)
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False, dtype=dtype)

    def forward(self, x, backbone_k_raw, backbone_v_raw):
        bsz = x.shape[0]
        residual = x
        h = self.ln1(x)
        q = self.q_proj(h).view(bsz, SEQ, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        if self.is_self:
            k = self.k_proj(h).view(bsz, SEQ, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = self.v_proj(h).view(bsz, SEQ, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        else:
            k = (
                self.k_proj(backbone_k_raw)
                .view(bsz, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(backbone_v_raw)
                .view(bsz, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
        k = k.repeat_interleave(GROUP, dim=1)
        v = v.repeat_interleave(GROUP, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, SEQ, Q_DIM)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        residual2 = x
        h2 = self.ln2(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        x = residual2 + mlp_out
        return x


class _ExpertStack(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ExpertLayer(is_self=(i % 2 == 0), dtype=dtype) for i in range(NUM_LAYERS)]
        )

    def forward(self, x, backbone_k_raw, backbone_v_raw):
        for layer in self.layers:
            x = layer(x, backbone_k_raw, backbone_v_raw)
        return x


def task2_standalone_crosscheck(n_runs=5, warmup=2):
    print("\n" + "=" * 78)
    print("TASK 2: standalone equivalent-shape torch stack (cross-check)")
    print("=" * 78)

    torch.manual_seed(0)
    results = {}
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        stack = _ExpertStack(dtype).eval()
        x0 = torch.randn(1, SEQ, HIDDEN, dtype=dtype)
        backbone_k_raw = torch.randn(1, PREFIX_LEN, KV_DIM, dtype=dtype)
        backbone_v_raw = torch.randn(1, PREFIX_LEN, KV_DIM, dtype=dtype)

        def run_10_steps():
            x = x0
            with torch.no_grad():
                for _ in range(NUM_STEPS):
                    x = stack(x, backbone_k_raw, backbone_v_raw)
            return x

        for _ in range(warmup):
            run_10_steps()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            run_10_steps()
            times.append(time.perf_counter() - t0)
        times.sort()
        med_total_ms = times[len(times) // 2] * 1e3
        print(
            f"  {dtype_name}: {med_total_ms:7.2f} ms / 10 steps  ({med_total_ms / NUM_STEPS:6.2f} ms/step)  (median of {n_runs} runs, {warmup} warmup)"
        )
        results[dtype_name] = {
            "ms_per_10_steps": med_total_ms,
            "ms_per_step": med_total_ms / NUM_STEPS,
        }
    return results


# =============================================================================
# Task 3: per-GEMM CPU timing at the expert's shapes
# =============================================================================
_GEMM_SHAPES = [
    ("q_proj (self+cross)", HIDDEN, Q_DIM),
    ("k/v_proj (self layer)", HIDDEN, KV_DIM),
    ("o_proj", Q_DIM, HIDDEN),
    ("gate/up_proj", HIDDEN, INTERMEDIATE),
    ("down_proj", INTERMEDIATE, HIDDEN),
]
# Bonus row discovered while reading the real code: cross-layer k/v proj is
# NOT run over the 50 action tokens -- it re-projects the backbone's 241-token
# raw K/V cache every step. Included for completeness, not part of the
# original ask.
_BONUS_SHAPE = (
    "k/v_proj (CROSS layer, over 241 backbone tokens)",
    KV_DIM,
    KV_DIM,
    PREFIX_LEN,
)


def task3_per_gemm(iters=20, warmup=5):
    print("\n" + "=" * 78)
    print("TASK 3: per-GEMM CPU timing at the expert's shapes")
    print("=" * 78)

    torch.manual_seed(0)
    rows = []

    def bench(m, k, n, dtype):
        a = torch.randn(m, k, dtype=torch.float32)
        b = torch.randn(k, n, dtype=torch.float32)
        if dtype == torch.bfloat16:
            a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
        us = _median_us(lambda: torch.matmul(a, b), iters, warmup)
        flops = 2.0 * m * k * n
        return us, _gflops(flops, us)

    print(
        f"\n{'shape':40s} | {'M=50 bf16':>16s} | {'M=50 fp32':>16s} | {'M=64 bf16':>16s} | {'M=64 fp32':>16s}"
    )
    print("-" * 130)
    for name, k, n in _GEMM_SHAPES:
        row = {"name": name, "k": k, "n": n}
        for m in (SEQ, 64):
            for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                us, gflops = bench(m, k, n, dt)
                row[f"M{m}_{dt_name}_us"] = us
                row[f"M{m}_{dt_name}_gflops"] = gflops
        rows.append(row)
        print(
            f"{name:40s} | {row['M50_bf16_us']:7.1f}us {row['M50_bf16_gflops']:6.1f}G | "
            f"{row['M50_fp32_us']:7.1f}us {row['M50_fp32_gflops']:6.1f}G | "
            f"{row['M64_bf16_us']:7.1f}us {row['M64_bf16_gflops']:6.1f}G | "
            f"{row['M64_fp32_us']:7.1f}us {row['M64_fp32_gflops']:6.1f}G"
        )

    # bonus: cross-layer k/v proj, real M=241 (not part of original ask, but a
    # real cost -- see module docstring)
    name, k, n, m = _BONUS_SHAPE
    row = {"name": name, "k": k, "n": n}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        us, gflops = bench(m, k, n, dt)
        row[f"M{m}_{dt_name}_us"] = us
        row[f"M{m}_{dt_name}_gflops"] = gflops
    rows.append(row)
    print("-" * 130)
    print(
        f"{name:40s} | {row[f'M{m}_bf16_us']:7.1f}us {row[f'M{m}_bf16_gflops']:6.1f}G (M={m}, bonus row, not M=50/64)"
        f" | fp32 {row[f'M{m}_fp32_us']:7.1f}us {row[f'M{m}_fp32_gflops']:6.1f}G"
    )

    print(f"\n  torch.get_num_threads() = {torch.get_num_threads()}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--skip-real", action="store_true", help="skip Task 1 (real lerobot checkpoint)"
    )
    ap.add_argument(
        "--skip-crosscheck", action="store_true", help="skip Task 2 (standalone stack)"
    )
    ap.add_argument(
        "--skip-gemm", action="store_true", help="skip Task 3 (per-GEMM timing)"
    )
    args = ap.parse_args()

    import subprocess

    try:
        cpu_model = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in cpu_model.splitlines():
            if "Model name" in line:
                print(line.strip())
    except Exception:
        pass
    print(
        f"torch {torch.__version__}, torch.get_num_threads()={torch.get_num_threads()}"
    )

    if not args.skip_real:
        task1_real_expert()
    if not args.skip_crosscheck:
        task2_standalone_crosscheck()
    if not args.skip_gemm:
        task3_per_gemm()


if __name__ == "__main__":
    main()
