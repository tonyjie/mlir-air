# SmolVLA action expert — CPU wall-clock breakdown

Companion to `vision_perf_breakdown.md`, at the same rigour: every op timed at
its **real** shape, every level closed against an independently-measured wall
clock. This is the CPU baseline the NPU port will be compared against.

**Measurement conditions.** AMD Ryzen AI 9 HX 370, CPU governor `performance`
+ EPP `performance`, NPU `pmode=Turbo` (idle, not used here), machine idle.
torch 2.10.0, 12 threads, real `lerobot/smolvla_base` checkpoint via
`policy.predict_action_chunk`. 9 chunks after 3 warmups, medians throughout.

**Reproduce:**
```bash
cd programming_examples/llms/smolvla
/home/jiajli/Projects/smolvla_playground/.venv/bin/python scripts/bench_cpu_expert_ops.py
```
Raw output: `results/cpu_expert_ops.txt`, machine-readable roll-up:
`results/cpu_expert_ops.json`. Every number below is from **one** run of that
script, so the levels are mutually consistent.

---

## 1. Headline

| | ms | share of chunk |
|---|---|---|
| `predict_action_chunk` (whole inference) | **907.7** | 100% |
| action expert (10 denoise steps) | **278.4** | **30.7%** |
| one denoise step | 27.84 | 3.1% |

Spread is tight: chunk min 900.9 / max 912.6, step min 27.08 / max 31.48.
This matches the canonical baseline already in the artifact (913 ms chunk,
285 ms expert) to within 0.6% / 2.3%.

---

## 2. L1 — stage breakdown of one denoise step

Measured in situ on the real model (light wrappers + module hooks).

| stage | µs/step | ms/inference | share of step |
|---|---|---|---|
| `embed_suffix` (action_in, time MLP) | 642.3 | 6.42 | 2.3% |
| **`vwe.forward` (16 layers)** | **27063.6** | **270.64** | **97.2%** |
| ├ even: attn block | 6977.2 | 69.77 | 25.1% |
| ├ even: o_proj | 733.3 | 7.33 | 2.6% |
| ├ even: post_attn_norm | 481.1 | 4.81 | 1.7% |
| ├ even: mlp | 4767.2 | 47.67 | 17.1% |
| ├ odd: attn block | 6946.3 | 69.46 | 24.9% |
| ├ odd: o_proj | 784.5 | 7.84 | 2.8% |
| ├ odd: post_attn_norm | 472.7 | 4.73 | 1.7% |
| ├ odd: mlp | 4811.2 | 48.11 | 17.3% |
| └ final norm | 57.7 | 0.58 | 0.2% |
| `action_out_proj` | 31.8 | 0.32 | 0.1% |

**Closure:**

| check | µs | gap |
|---|---|---|
| Σ (embed_suffix + vwe.forward + action_out_proj) | 27737.7 | |
| `denoise_step` wall | 27842.5 | **+104.7 (0.38%)** — mask build, slicing, Python |
| Σ 16-layer internals | 26031.2 | |
| `vwe.forward` wall | 27063.6 | **+1032.4 (3.81%)** — residual adds, `.clone()`, loop |

**Even and odd layers cost the same** (69.77 vs 69.46 ms attention block;
47.67 vs 48.11 ms MLP). The cross-attn layers' 241-token K/V re-projection is
almost exactly as expensive as the self-attn layers' RoPE + 291-token
attention it replaces.

---

## 3. L2 — per-op replay, one layer of each kind

Replay is faithful to `smolvlm_with_expert.py` line for line, driven with
tensors captured from a real inference (`x=(1,50,720)`, `k_cache=(1,241,5,64)`,
`mask=(1,50,291)`).

### EVEN layer — self-attn, KV = 241 cached + 50 new = 291

| op | shape | µs | share |
|---|---|---|---|
| input_layernorm (RMSNorm) | 50×720 | 39.5 | 2.7% |
| cast hidden → bf16 | 50×720 | 6.1 | 0.4% |
| **q_proj** | 50×720×960 | **98.7** | 6.8% |
| **k_proj** | 50×720×320 | **50.5** | 3.5% |
| **v_proj** | 50×720×320 | **47.6** | 3.3% |
| cat q / k / v (1 elem) | no-op copy | 9.4 / 2.7 / 2.6 | 1.0% |
| **RoPE q** | 50×15×64 | **96.7** | 6.7% |
| **RoPE k** | 50×5×64 | **51.6** | 3.6% |
| cat k / v with cache | 241+50 = 291 | 14.3 / 12.7 | 1.9% |
| attn: GQA expand k,v | 5→15 heads, 291×64 | 35.2 | 2.4% |
| attn: cast q,k → fp32 | 50×960 / 291×960 | 24.5 | 1.7% |
| attn: transpose q,k | B,H,S,D | 2.3 | 0.2% |
| **attn: QK^T (fp32)** | 15×50×64×291 | **89.0** | 6.2% |
| attn: scale | 15×50×291 | 20.6 | 1.4% |
| attn: mask where | 15×50×291 | 31.2 | 2.2% |
| attn: softmax (fp32) | 15×50×291 | 22.7 | 1.6% |
| attn: cast probs → bf16 | 15×50×291 | 14.2 | 1.0% |
| **attn: PV** | 15×50×291×64 | **65.0** | 4.5% |
| attn: permute+reshape | 50×960 | 8.8 | 0.6% |
| cast att → bf16 / slice | 50×960 | 1.2 / 1.9 | 0.2% |
| **o_proj** | 50×960×720 | **84.8** | 5.9% |
| residual add 1 (in-place) | 50×720 | 20.3 | 1.4% |
| clone | 50×720 | 6.0 | 0.4% |
| post_attention_layernorm | 50×720 | 51.9 | 3.6% |
| **gate_proj** | 50×720×2048 | **170.6** | 11.8% |
| **up_proj** | 50×720×2048 | **165.1** | 11.4% |
| SiLU × up | 50×2048 | 32.5 | 2.3% |
| **down_proj** | 50×2048×720 | **155.6** | 10.8% |
| residual add 2 (in-place) | 50×720 | 7.9 | 0.5% |
| **Σ parts** | | **1443.9** | |
| real layer+tail (independent) | | **1427.3** | closure **−1.16%** |

### ODD layer — cross-attn, KV = 241 (backbone prefix)

| op | shape | µs | share |
|---|---|---|---|
| input_layernorm (RMSNorm) | 50×720 | 52.0 | 3.5% |
| cast hidden → bf16 | 50×720 | 0.5 | 0.0% |
| **q_proj** | 50×720×960 | **100.4** | 6.7% |
| cast+view k cache → **fp32** | 241×320 | 13.3 | 0.9% |
| **k_proj (over prefix!)** | 241×320×320 | **114.9** | 7.7% |
| cast+view v cache → **fp32** | 241×320 | 11.9 | 0.8% |
| **v_proj (over prefix!)** | 241×320×320 | **102.3** | 6.8% |
| pos − min(pos) | 1×50 | 6.4 | 0.4% |
| **RoPE q** | 50×15×64 | **100.9** | 6.8% |
| mask slice | 50×241 | 2.5 | 0.2% |
| attn: GQA expand k,v | 5→15 heads, 241×64 | 42.3 | 2.8% |
| attn: cast q,k → fp32 | 50×960 / 241×960 | 12.6 | 0.8% |
| attn: transpose q,k | B,H,S,D | 2.2 | 0.1% |
| **attn: QK^T (fp32)** | 15×50×64×241 | **74.6** | 5.0% |
| attn: scale | 15×50×241 | 20.5 | 1.4% |
| attn: mask where | 15×50×241 | 31.7 | 2.1% |
| attn: softmax (fp32) | 15×50×241 | 19.5 | 1.3% |
| attn: cast probs → bf16 | 15×50×241 | 0.8 | 0.1% |
| **attn: PV** | 15×50×241×64 | **66.5** | 4.5% |
| attn: permute+reshape | 50×960 | 11.0 | 0.7% |
| cast att → bf16 / slice | 50×960 | 7.7 / 2.0 | 0.6% |
| **o_proj** | 50×960×720 | **87.2** | 5.8% |
| residual add 1 (in-place) | 50×720 | 8.3 | 0.6% |
| clone | 50×720 | 6.2 | 0.4% |
| post_attention_layernorm | 50×720 | 56.0 | 3.8% |
| **gate_proj** | 50×720×2048 | **173.2** | 11.6% |
| **up_proj** | 50×720×2048 | **166.3** | 11.1% |
| SiLU × up | 50×2048 | 33.0 | 2.2% |
| **down_proj** | 50×2048×720 | **159.0** | 10.6% |
| residual add 2 (in-place) | 50×720 | 7.7 | 0.5% |
| **Σ parts** | | **1493.5** | |
| real layer+tail (independent) | | **1495.2** | closure **+0.11%** |

---

## 4. L3 — per-inference roll-up by kernel family

Each op × 8 layers of its parity × 10 denoise steps.

| family | ms/inference | share |
|---|---|---|
| **GEMM** | **134.1** | **57.1%** |
| attention (QK^T, PV, softmax, mask, scale, GQA expand) | 41.5 | 17.7% |
| **RoPE** | **19.9** | **8.5%** |
| RMSNorm | 16.0 | 6.8% |
| layout / cast glue | 14.7 | 6.3% |
| SiLU-and-Mul | 5.2 | 2.2% |
| EltwiseAdd (residuals) | 3.5 | 1.5% |
| **TOTAL (L2 basis)** | **235.0** | |

Largest individual line items, ms/inference:

| op | even | odd | total | shape |
|---|---|---|---|---|
| gate_proj | 13.6 | 13.9 | **27.5** | 50×720×2048 |
| up_proj | 13.2 | 13.3 | **26.5** | 50×720×2048 |
| down_proj | 12.4 | 12.7 | **25.2** | 50×2048×720 |
| q_proj | 7.9 | 8.0 | **15.9** | 50×720×960 |
| RoPE q | 7.7 | 8.1 | **15.8** | 50×15×64 |
| o_proj | 6.8 | 7.0 | **13.8** | 50×960×720 |
| QK^T (fp32) | 7.1 | 6.0 | **13.1** | 15×50×64×291 / ×241 |
| PV | 5.2 | 5.3 | **10.5** | 15×50×291×64 / ×241×64 |
| k_proj (over prefix) | — | 9.2 | **9.2** | 241×320×320, **fp32** |
| post_attention_layernorm | 4.2 | 4.5 | **8.6** | 50×720 |
| v_proj (over prefix) | — | 8.2 | **8.2** | 241×320×320, **fp32** |
| input_layernorm | 3.2 | 4.2 | **7.3** | 50×720 |

---

## 5. Reconciling L2 → L1: the hot-replay gap is the weight working set

L2 rolls up to 233.8 ms (whole-block basis) but L1 measures 270.6 ms for the
same 16 layers — a −13.6% undercount. That is **not** measurement error; it is
cache behaviour, and it was verified rather than assumed.

| | |
|---|---|
| weights per expert layer | **12.54 MB** |
| all 16 layers | **200.6 MB** (vs 24 MB L3 total, 16 MB per CCX) |
| streamed per inference | **2.01 GB** (16 layers × 10 steps) |

Running the *identical* whole-block hot on one layer vs round-robin across all
8 layers of that parity:

| | hot single layer | round-robin over 8 | ratio |
|---|---|---|---|
| even | 1453.5 µs | 1579.5 µs | **1.09×** |
| odd | 1489.7 µs | 1597.6 µs | **1.07×** |

Chain: 233.8 ms × ~1.08 ≈ 253 ms, + hook overhead and Python in the L1 path
→ 270.6 ms. **The CPU expert is partly weight-streaming bound, not purely
compute bound** — it re-reads 200 MB of weights ten times per inference to do
112.7 GFLOP of useful work (FLOP count from `expert_npu_feasibility.md` §1).

---

## 6. Findings not in the feasibility study

The feasibility study (`expert_npu_feasibility.md`) measured *kernels in
isolation*. Replaying the real code path surfaced four things it could not see.

**1. Cross-attn `k_proj` / `v_proj` weights are fp32, not bf16.** The rest of
the expert is bf16, but the cross layers' K/V re-projections are freshly
constructed `nn.Linear(320,320)` that never got cast:

```
layer 0 (even/self): q_proj=bfloat16 k_proj=bfloat16 v_proj=bfloat16 o_proj=bfloat16
layer 1 (odd/cross): q_proj=bfloat16 k_proj=float32  v_proj=float32  o_proj=bfloat16
```

This is inherent to lerobot's model construction, not to how we load it:
`smolvlm_with_expert.py:113,118` replaces the cross layers' `k_proj`/`v_proj`
with bare `nn.Linear(...)` at torch's default dtype, after the rest of the
expert was built bf16 from config. The checkpoint values are then upcast into
them.

Those two GEMMs cost **17.4 ms/inference** and force a bf16→fp32 cast of the
K/V cache on every one of the 10 steps (2.0 ms more). For the NPU port this
matters twice: it is the only fp32 GEMM in the expert, and the CPU baseline it
must beat is an *fp32* baseline, not a bf16 one.

**2. RoPE costs as much as the GEMM that feeds it.** RoPE q on 50×15×64 is
96.7–100.9 µs against q_proj's 98.7–100.4 µs at 50×720×960 — a ~0.3 MFLOP
rotation matching a 69 MFLOP matmul, 200× the arithmetic for the same time.
`apply_rope` recomputes `freq_exponents`,
`timescale`, `radians`, `sin` and `cos` from scratch in fp32 on **every call**
and never caches them. At 8.5% (19.9 ms) of the expert this is the single
largest piece of avoidable CPU work, and it is avoidable on the NPU side too
(the registry RoPE kernel takes precomputed tables).

**3. Attention is fp32 end to end.** `eager_attention_forward` upcasts q and k
to fp32, does an explicit `torch.where` mask over a 15×50×291 fp32 tensor, and
softmaxes in fp32 — 41.5 ms, 17.7%. Only 23.6 ms of that is the two matmuls;
**17.9 ms is mask + scale + softmax + GQA expand**, i.e. the parts a fused
FlashAttention kernel would absorb for free.

**4. GEMM is only 57% of the expert.** Even a perfect matmul accelerator that
made all seven projection GEMMs free would leave 101 ms of the 235 ms
untouched. This bounds what any port can achieve and says the port must move
RoPE, norms, softmax and the residuals on-device too — the same lesson the
vision tower taught (fusing bias/residual on-device was the real win there).

---

## 7. What this means for the NPU port

Restating the target honestly: **278.4 ms is the number to reproduce, not
necessarily to beat.** The goal is a complete, correct end-to-end mapping;
slower than CPU is acceptable.

Per-family, the CPU time the NPU has to replace:

| family | CPU ms | NPU status (from `expert_npu_feasibility.md`, measured) |
|---|---|---|
| GEMM | 134.1 | all shapes measured; 1.1–2.3× slower at M=50→64 |
| attention | 41.5 | **blocker** — FA numerically broken below lq=256; must pad 50→256 |
| RoPE | 19.9 | registry kernel measured (106.3 / 83.0 µs) |
| RMSNorm | 16.0 | measured 151.3 µs at 64×720 — 100% launch floor |
| layout/cast glue | 14.7 | mostly disappears if the port keeps data on-device |
| SiLU-and-Mul | 5.2 | measured 121.6 µs |
| EltwiseAdd | 3.5 | measured 86.6 µs |

Three structural facts the port has to design around, all confirmed here:

- **M=50 pads to 64**, and 64 admits only `tile_m=16, herd_m=4` — half the herd
  idle before the first instruction.
- **Cross layers project the prefix, not the actions** (241×320×320), and
  lerobot recomputes them on all 10 steps even though they are constant across
  the denoise loop. Hoisting them out of the loop is a free 9/10 reduction of
  17.4 ms of work — and it is a *correctness-preserving* change, so the port
  should do it and report both variants.
- **The layer-loop glue (residual + `.clone()`) is 3.81%** on CPU but would be
  ~85 µs per launch on the NPU if left as separate dispatches. It has to be
  fused into the surrounding kernels.

Next step: end-to-end NPU mapping of the expert, reusing the already-measured
per-kernel data above rather than re-measuring blind.
