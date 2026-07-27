# SmolVLA action expert on NPU2 — feasibility study

**Question.** Would SmolVLA's action expert run faster or slower on NPU2 (Strix,
AIE2P) than on the CPU it runs on today, and why?

**Verdict: substantially SLOWER — do not port it.**
Measured CPU **324 ms/inference**; projected NPU **386–526 ms** depending on how
much engineering is spent, i.e. **1.2–1.6× slower**. The verdict does not depend
on the projection model: **the NPU loses head-to-head on every single kernel the
expert executes**, by 1.1× (gate/up) to 6.6× (attention), measured at the
expert's own shapes.

The reason is **not** dispatch overhead. The premise going into this study was
that ~480 fused dispatches × ~1.75 ms would bury the expert in ~840 ms of
overhead. That premise is **wrong by ~50×**: the measured per-dispatch cost of
the deployed driver path is **24–36 µs**, not 1.75 ms (§5). The real reason is
that **seq=50 is far too small to feed the array** — every kernel runs at
200–1100 GFLOP/s against the 2046–3366 measured this session for the same
kernels at the *backbone's* seq=256 (§2) — and that **every NPU launch
costs ~85 µs no matter how little work it does**, which for a 239-launch denoise
step is 20 ms/step of pure floor.

Everything below was measured on real NPU2 hardware in one session
(2026-07-27). No number is carried over from a prior run or from the kernel
registry.

---

## 1. What the expert actually is

Read off the real checkpoint (`lerobot/smolvla_base`) and
`lerobot/policies/smolvla/smolvlm_with_expert.py`:

- hidden 720, intermediate 2048, **16 layers**, GQA **15 q / 5 kv**, head_dim 64
- **50 action tokens**, `num_steps = 10` → the whole stack runs **10× per
  inference**, strictly sequentially (`sample_actions`, `modeling_smolvla.py:835`)
- `self_attn_every_n_layers = 2` → **EVEN layers self-attend, ODD layers
  cross-attend**

Two structural facts matter and are **not** in the original task description:

1. **EVEN ("self-attn") layers are not self-attention over 50 tokens.**
   `forward_attn_layer` concatenates the cached 241-token prefix K/V
   (`smolvlm_with_expert.py:264`), so K/V length is **241+50 = 291**, not 50.
2. **ODD (cross-attn) layers project the *prefix*, not the action tokens.**
   `expert_layer.self_attn.k_proj(_key_states)` (`smolvlm_with_expert.py:354`)
   runs over the backbone's cached `(241, 320)` K/V, so those two GEMMs are
   **241×320×320**, not 50×720×320 — and lerobot recomputes them on **every one
   of the 10 denoise steps** even though they are constant. That is 144 redundant
   GEMMs per inference (quantified in §6, scenario B).

Per denoise step the expert issues **239 ops** (120 across 8 even layers, 112
across 8 odd layers, 7 head/tail), = **2390 ops per inference**.
Useful FLOPs (no padding): **112.7 GFLOP per inference**.

**M=50 is not tileable.** `mm_aie2p.cc:136` requires `tile_m % 16 == 0` and
`run.py:62` requires `M % (tile_m·herd_m) == 0`, so M=50 → **pad to 64**, and
64 admits only `tile_m=16, herd_m=4` — **half the 8×4 herd is idle before a
single instruction runs.**

---

## 2. Measured GEMM throughput at the expert's shapes (NPU2)

bf16-in / bf16-out, `HIGH_PRECISION=true`, `--method drain`, `PERF_ITERS=20`
(device `run.start()+wait2()`, buffer sync excluded — the same timing basis as
every kernel-registry GFLOPS number). Median over 3 repeats where repeated.
Full sweep in `results/expert_gemm.csv`, logs in `results/logs/`.

| expert op | shape (M=50→64) | tiling (tm/tk2/tk1/tn, herd) | latency | **GFLOP/s** | mean_rel_L1 |
|---|---|---|---|---|---|
| q_proj | 64×720×960 | 16/144/48/80, 4×4 | 158.2 µs | **559** | 9.39e-3 |
| k/v_proj (even layers) | 64×720×320 | 16/144/48/80, 4×4 | 95.2 µs | **310** | 9.43e-3 |
| o_proj (N 720→768) | 64×960×768 | 16/320/32/96, 4×4 | 135.6 µs | **685** | 9.46e-3 |
| o_proj (native N=720) | 64×960×720 | 16/320/32/80, 4×**3** | 165.9 µs | 533 | 9.47e-3 |
| gate/up | 64×720×2048 | 16/144/48/128, 4×4 | 185.0 µs | **1020** | 9.43e-3 |
| down (N 720→768) | 64×2048×768 | 16/256/32/96, 4×4 | 179.5 µs | **1122** | 9.28e-3 |
| down (native N=720) | 64×2048×720 | 16/256/32/80, 4×**3** | 212.3 µs | 889 | 9.36e-3 |
| k/v_proj (cross layers) | 256×320×320 | 32/320/32/80, 8×4 | 112.6 µs | **466** | 9.40e-3 |
| action_time_mlp in | 64×1440×768 | 16/144/48/96, 4×4 | 152.0 µs | 931 | 9.34e-3 |
| action_time_mlp out | 64×720×720 | 16/144/48/80, 4×3 | 148.3 µs | 447 | 9.45e-3 |
| action_in | 64×32×720 | 16/32/32/80, 4×3 | 115.9 µs | 25 | 9.26e-3 |
| action_out | 64×720×32 | 16/144/48/32, 4×1 | 79.2 µs | 37 | 9.01e-3 |
| *fused-weight* q‖k‖v | 64×720×1600 | 16/144/48/80, 4×4 | 206.8 µs | 713 | 9.44e-3 |
| *fused-weight* gate‖up | 64×720×4096 | 16/144/48/128, 4×4 | 324.6 µs | 1163 | 9.44e-3 |

**Reference points, same session, same harness:**

| reference | shape | GFLOP/s |
|---|---|---|
| SmolVLA backbone q/o proj (seq 256) | 256×960×960 | **2046** |
| backbone gate/up (seq 256) | 256×960×2560 | **2762** |
| backbone down (seq 256) | 256×2560×960 | **3366** |
| SmolVLA vision (registry, seq 1024) | 1024×3072×768 | 5790 |

The expert's best GEMM (1122) is **3× below the backbone's down-proj** (measured
this session) and **5× below the vision tower's** (registry figure, not re-measured here). The expert's *typical* GEMM (559) is
**3.7× below the backbone's equivalent projection**.

### Why: M is the whole story

Same K/N, only M changes — this is the cleanest single measurement in the study:

| M | shape | latency | GFLOP/s |
|---|---|---|---|
| **64** (the expert) | 64×720×960 | 158.2 µs | **559** |
| 128 | 128×720×960 | 173.8 µs | **1018** |
| 256 (the backbone) | 256×720×960 | 198.9 µs | **1780** |

**4× the work costs 1.25× the time.** The M=64 kernel is almost entirely
latency, not compute: `tile_m=16 · herd_m=4` occupies 4 of 8 herd rows, and the
K-loop is too short to amortize launch. There is **3.2× of throughput sitting
idle that seq=50 structurally cannot reach.**

Low-precision (direct-codegen) GEMM was checked as a possible lever and is not
one: 139.8 µs (q_proj), 169.1 (gate/up), 191.6 (down) — ~10% better on two
shapes, worse on the third, at 1.03e-2 vs 9.4e-3.

---

## 3. Measured non-GEMM ops (NPU2)

| op | expert shape | latency | note |
|---|---|---|---|
| RMSNorm | 64×720 | **151.3 µs** | 92 KB of data. 2× per layer. |
| EltwiseAdd (residual) | 64×720 (n=46080) | **86.6 µs** | 2× per layer |
| SiLU-and-Mul | 64×2048 (n=131072) | **121.6 µs** | 1× per layer |
| RoPE (q, 15 heads) | 960×64 | **106.3 µs** | |
| RoPE (k, 5 heads) | 320×64 | **83.0 µs** | even layers only |

A 92 KB RMSNorm taking 151 µs is 1.2 GB/s — against the ~57 GB/s this fabric
reaches on large EltwiseAdd. **These are 100% launch floor, 0% work.** They are
30% of the expert's projected device time (§6).

---

## 4. FlashAttention — the worst offender, and a blocker

Attention is 15 q-heads / 5 kv-heads / head_dim 64, non-causal, Q=50,
K/V=291 (even) or 241 (odd).

**Finding: the FlashAttention kernel is numerically broken below lq=256.**
Both variants (`attn_npu2_seqfirst.py` and `attn_npu2.py`) compile and run at
lq=64 and lq=128 but return `inf`/`NaN` (lq=64) or garbage (lq=128,
mean_rel_L1 = 1.197). Only lq=256 is correct.

| config | lq | lk | nqt/hpu/ncs | latency | correctness |
|---|---|---|---|---|---|
| **odd layers, deployable** | 256 | 256 | 4/1/4 | **919.7 µs** | **PASS**, mean_rel_L1 4.264e-2 |
| **even layers, deployable** | 256 | 384 | 4/1/3 | **891.6 µs** | **PASS**, mean_rel_L1 4.203e-2 |
| odd, small-Q (broken) | 64 | 256 | 1/1/4 | 368.7 µs | **inf/NaN** |
| even, small-Q (broken) | 64 | 384 | 1/1/3 | 350.0 µs | **inf/NaN** |
| small-Q, heads-first (broken) | 64 | 256 | 1/1/4 | 360.2 µs | **NaN** |
| lq=128 (broken) | 128 | 256 | 2/1/4 | 538.0 µs | mean_rel_L1 1.192 |

So today the expert's 50 query tokens must be **padded 50 → 256 (5.1× waste)**,
costing ~900 µs per attention. Worse, `num_heads_per_unroll` must be 1 (15 has
no divisor ≤ 2 other than 1) and `num_q_tiles` is pinned by the cascade width,
so **attention occupies 1 of 8 columns — 4 of 32 tiles, 12.5% of the array.**

Both broken-config latencies are still reported above because the instruction
stream is representative; they are used in §6 scenario C as "what a *fixed*
small-Q FA would buy". They are **not** deployable numbers.

---

## 5. Dispatch overhead — the premise was wrong by ~50×

Two independent measurements, both this session.

**(a) Direct microbenchmark** (`bench_expert_dispatch.py`) — the expert's own
q_proj ELF driven 200× through the *deployed* driver path
(`llms/shared/infra/cache.py:427` `load_and_run`: filelock, BO write + sync,
fresh `xrt.run`, `set_arg` per buffer, start+wait, sync back, zero-copy map),
weights `static_input_indices`, output `intermediate_indices`:

| kernel | driver wall | start+wait | BO write | BO read | **overhead** |
|---|---|---|---|---|---|
| q_proj 64×720×960 | 0.156 ms | 0.132 ms | 0.007 ms | 0.002 ms | **0.024 ms** |
| trivial 64×32×32 | 0.144 ms | 0.106 ms | 0.004 ms | 0.003 ms | **0.038 ms** |

**(b) The real 16-layer SmolVLA backbone**, profiler enabled, 832 real
dispatches over 4 inferences:

```
--- Fine-Grained NPU Breakdown (avg per invocation) ---
  Kernel                 BO Write    NPU Run    BO Read      Total
  masked_softmax           0.23ms     1.22ms     0.02ms     1.47ms  (x64)
  o_ffn                    0.46ms     3.37ms     0.01ms     3.85ms  (x64)
  pv                       0.03ms     0.38ms     0.00ms     0.42ms  (x320)
  qkt                      0.03ms     0.43ms     0.01ms     0.46ms  (x320)
  rms_gemms_rope           0.14ms     2.03ms     0.02ms     2.19ms  (x64)
  TOTAL                    73.2ms    682.7ms      6.6ms    762.5ms
  %                         10%        90%         1%
```

**90% of a "1.75 ms dispatch" is the NPU executing.** The host part is ~10%,
and it scales with payload (~0.07 ms/MB of BO write) — the expert's activations
are 92–256 KB, so ~6–18 µs. **Per-dispatch cost for the expert: ~36 µs.**

**Consequence: fusion cannot save this workload.** At 2390 dispatches, *all*
dispatch overhead is 86 ms of a 526 ms projection. Eliminating it entirely still
leaves 440 ms > CPU's 324 ms.

### And fusion actively costs device time

The two real fused multi-launch ELFs in the shipped backbone were compared
against the sum of their parts run as individually-optimally-tiled standalone
kernels (all parts measured this session at seq=256):

| fused ELF | fused device time | Σ standalone parts | **penalty** |
|---|---|---|---|
| `rms_gemms_rope` (RMSNorm + q/k/v + 2×RoPE) | 2030 µs | 428.8+230.6+130.8+130.8+119.9+93.1 = **1134 µs** | **1.79×** |
| `o_ffn` (o + add + RMSNorm + gate + up + SiLU + down + add) | 3370 µs | 230.6+113.7+428.8+455.6+455.6+246.6+373.9+113.7 = **2418 µs** | **1.39×** |

Folding launches into one ELF forces them to share a device configuration and
serializes them, costing **1.4–1.8×** in device time to save **24 µs** per
removed dispatch. For the expert that trade is decisively negative (§6).

*(This also re-explains the vision win: fusing vision took it 368 → 141 ms not
by removing XRT overhead but by moving the per-Linear bias-adds and residuals
off the host — "host-gap 212 ms → 4 ms". The expert's glue is already on-device
in any port we would write, so that lever is already spent.)*

---

## 6. Projection

`scripts/expert_projection.py` rebuilds this from the measured numbers above.

| scenario | dispatches | device | overhead | **TOTAL** | effective |
|---|---|---|---|---|---|
| **A** — one dispatch per op, today's kernels | 2390 | 440.2 ms | 86.0 ms | **526.3 ms** | 214 GFLOP/s |
| **B** — A + concat q‖k‖v and gate‖up weights, hoist the cross-attn K/V out of the denoise loop | 1926 | 405.4 ms | 68.8 ms | **474.2 ms** | 238 GFLOP/s |
| **C** — B + a *hypothetical fixed* small-Q FlashAttention (§4) | 1926 | 318.0 ms | 68.8 ms | **386.7 ms** | 291 GFLOP/s |
| **D** — "fuse like vision", 3 ELFs/layer × 16 × 10 | **490** | 563–726 ms | 17.6 ms | **581–743 ms** | 152–194 GFLOP/s |

Scenario D is the one the original premise expected to win. It removes 52 ms of
dispatch cost and adds 158–320 ms of device cost (§5). **It is the worst option.**

Where scenario A's 440.2 ms of device time goes:

| bucket | ms | share |
|---|---|---|
| GEMM (projections + FFN) | 168.2 | 38.2% |
| attention | 144.9 | 32.9% |
| RMSNorm | 48.4 | 11.0% |
| eltwise add (residuals) | 27.7 | 6.3% |
| RoPE | 23.6 | 5.4% |
| SiLU-and-Mul | 19.5 | 4.4% |
| head/tail (action_in/time-MLP/action_out) | 7.9 | 1.8% |

**2390 launches × ~85 µs measured floor = 203 ms = 46% of scenario A's device
time is the fixed cost of starting a launch, independent of the work in it.**
(The 85 µs floor is measured, not assumed: 64×720×32 GEMM = 79.2 µs,
64×32×32 GEMM = 106 µs, 64×720 eltwise add = 86.6 µs.)

---

## 7. The CPU side, measured this session

`/home/jiajli/Projects/smolvla_playground/.venv/bin/python`, real
`policy.predict_action_chunk`, `model.denoise_step` wrapped with a wall-clock
timer, 5 chunks after 2 warmups, machine idle.
CPU: **AMD Ryzen AI 9 HX 370** (12 cores / 24 threads), `torch.get_num_threads() = 12`,
weights bf16 (474 of 500 tensors).

| metric | value |
|---|---|
| **per denoise step** | **32.54 ms** (min 30.12, max 36.51) |
| **action expert per inference (10 steps)** | **323.94 ms** (min 317.9, max 343.5) |
| full `predict_action_chunk` | 1027.79 ms |
| expert share of the chunk | 31.5% |

Independent cross-check with `profile_cpu_baseline.py` (leaf-op attribution):
expert stage 217.1 ms of leaf-op time over 1651 leaf calls/chunk — the ~107 ms
difference from the 324 ms wall is Python/dispatch glue between the leaf ops,
which is exactly what one expects at 165 torch ops per denoise step.

### Head-to-head, per kernel — the NPU loses all of them

CPU: torch bf16, median of 50 after 10 warmups, at the **real** M=50/241.
NPU: from §2/§3/§4, at the **padded** M=64/256 it is forced to use.

| op | shape | CPU bf16 | CPU GFLOP/s | NPU | NPU GFLOP/s | **CPU faster by** |
|---|---|---|---|---|---|---|
| q_proj | 50×720×960 | **89.9 µs** | 769 | 158.2 µs | 559 | **1.8×** |
| k/v_proj (even) | 50×720×320 | **41.6 µs** | 554 | 95.2 µs | 310 | **2.3×** |
| o_proj | 50×960×720 | **62.3 µs** | 1109 | 135.6 µs | 685 | **2.2×** |
| gate/up | 50×720×2048 | **169.2 µs** | 871 | 185.0 µs | 1020 | **1.1×** |
| down | 50×2048×720 | **112.3 µs** | 1313 | 179.5 µs | 1122 | **1.6×** |
| k/v_proj (cross) | 241×320×320 | **55.5 µs** | 890 | 112.6 µs | 466 | **2.0×** |
| attention (self) | Q50, KV291 | **164.0 µs** | 341 | 891.6 µs | — | **5.4×** |
| attention (cross) | Q50, KV241 | **140.0 µs** | 331 | 919.7 µs | — | **6.6×** |

This table is the study's core result, and it needs no projection model: at
seq=50 a 12-thread Zen 5 with AVX-512+bf16 beats the NPU array on **every**
kernel the expert runs. The closest call is gate/up (the widest N), which is
exactly the shape that best fills the herd.

---

## 8. Verdict

| | latency per inference | vs CPU |
|---|---|---|
| **CPU (measured)** | **323.9 ms** | 1.00× |
| NPU scenario A (today's kernels, unfused) | 526.3 ms | **1.62× slower** |
| NPU scenario B (cheap structural wins) | 474.2 ms | **1.46× slower** |
| NPU scenario C (+ fixed small-Q FlashAttention) | 386.7 ms | **1.19× slower** |
| NPU scenario D ("fuse like vision") | 581–743 ms | **1.79–2.29× slower** |

**The expert should stay on CPU.** It is the mirror image of the vision tower:
vision wins 1.5× because seq=1024 and 768/3072-wide matmuls saturate the array;
the expert loses because seq=50 leaves half the herd unused, attention on 1
column of 8, and 46% of the device time in per-launch floor.

Note also the *lower bound*: scenario C's **device time alone is 318.0 ms**,
already a dead heat with the CPU's 323.9 ms. **No amount of dispatch-overhead
reduction, driver work, or fusion can produce a win** — the compute itself is
not fast enough at these shapes.

---

## 9. What would have to change to flip it

Target: ≤ 324 ms. Best modelled today is 386.7 ms (scenario C). The gap is 1.19×,
and it is **not** in the host.

| lever | effect | feasible? |
|---|---|---|
| **1. Cut the per-launch device floor from ~85 µs to ~30 µs.** 46% of device time is launch floor (herd start/stop, DMA BD programming, lock setup). At 30 µs, scenario C → **~193 ms device / ~262 ms total = a 1.24× WIN.** | **−105 ms** | Compiler/runtime work, not kernel tuning. Unquantified difficulty, but it is the only lever big enough on its own. |
| **2. Fix FlashAttention below lq=256** (§4). Removes a 5.1× Q-padding waste and 550 µs × 160 attentions. | **−87 ms** | Real bug (inf/NaN at lq≤128 in *both* variants). Bounded, well-defined kernel work. Necessary but not sufficient. |
| **3. Concat q‖k‖v and gate‖up weights; hoist the cross-attn K/V GEMMs out of the 10-step loop.** Pure host-side restructuring; the fused-weight GEMMs are already measured (§2). | **−52 ms** | Easy and safe. Nowhere near enough alone. |
| **4. Batch > 1** (multiple environments / action chunks per call). M goes 64 → 128 → 256, and measured GFLOP/s goes 559 → 1018 → **1780** (3.2×). | converts the workload from latency-bound to compute-bound | The single biggest structural lever — but it buys **throughput, not single-chunk latency**. Only useful if the deployment can batch. |
| ~~5. Fuse into 3 ELFs/layer like vision~~ | **+158 to +320 ms** | **Counterproductive** — measured 1.39–1.79× device penalty for 24 µs/dispatch of savings (§5). |
| ~~6. Reduce per-dispatch host overhead~~ | ≤ **−86 ms**, and it is already only 16% | Even at zero, 440 ms > 324 ms. Cannot flip it. |

**Minimum credible path to a win: levers 1 + 2 + 3 together** (~−244 ms →
~280 ms, a ~1.16× win). Lever 1 is the load-bearing one and is the least
scoped. Lever 4 flips it comfortably but changes the deployment model.

Given that the vision tower already delivers a measured 1.5× win and the
backbone a measured 2.5× loss, engineering effort is better spent on lever 1
(which would also lift the backbone, where 208 dispatches × 85 µs = 18 ms of
floor sits) than on porting the expert.

---

## 10. Reproduce

```bash
# GEMM sweep at the expert's shapes (all configs, incremental CSV, resume-safe)
cd programming_examples/llms/smolvla
bash scripts/expert_gemm_sweep.sh results/cfg_expert.csv results/expert_gemm.csv

# single GEMM, e.g. q_proj (the harness's `make profile` hardcodes 1024^3 — don't use it)
cd programming_examples/matrix_multiplication/bf16_in_bf16_out
flock -x -w 1800 /tmp/mlir-air-npu.lock make run \
  M=64 K=720 N=960 TILE_M=16 TILE_K_L2=144 TILE_K_L1=48 TILE_N=80 \
  HERD_M=4 HERD_N=4 METHOD=drain AIE_TARGET=aie2p PERF_ITERS=20

# FlashAttention (deployable config: Q padded 50 -> 256)
cd programming_examples/flash_attention/kernel_fusion_based
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SCRIPT=attn_npu2_seqfirst.py \
  LK=256 LKP=64 LQ=256 LQP=256 DK=64 DV=64 NUM_HEADS=15 NUM_KV_HEADS=5 \
  AIE_TARGET=aie2p \
  EXTRA_PY_FLAGS="--num-heads-per-unroll 1 --num-cascade-stages 4 --num-q-tiles 4 --perf-iters 20"

# per-dispatch overhead on the deployed driver path
cd programming_examples/llms/smolvla
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 bench_expert_dispatch.py

# projection from the measured numbers
python3 scripts/expert_projection.py

# CPU reference (real lerobot model)
cd ~/Projects/smolvla_playground && ./.venv/bin/python profile_cpu_baseline.py
```

Raw data: `results/expert_gemm.csv` (35 GEMM configs), `results/logs/` (per-run
stdout for every GEMM, FlashAttention and elementwise measurement).

---

## Appendix — silent-corruption traps found while sweeping

`K=720` and `K=1440` are `2^4·45` / `2^5·45`; they admit **no** `tile_k_l1=32`
(the registry default) because no multiple of 32 divides them. Several legal-looking
K-tilings **pass every assert, compile, run, and return garbage with no error**:

| shape | tile_k_l2 / tile_k_l1 | mean_rel_L1 | |
|---|---|---|---|
| 64×720×960 | 144 / 48 | **9.39e-3** | ✅ use this |
| 64×720×960 | 240 / 80 | **9.39e-3** | ✅ |
| 64×720×960 | 240 / 48 | 1.016e+00 | ❌ silent |
| 64×720×960 | 360 / 40 | 7.80e-1 | ❌ silent |
| 64×720×960 | 720 / 48 | 8.96e-1 | ❌ silent |
| 64×720×960 | 720 / 80 | 8.73e-1 | ❌ silent |
| 64×1440×720 | 144 / 48 | **9.29e-3** | ✅ use this |
| 64×1440×720 | 288 / 32 | 8.02e-1 | ❌ silent |
| 64×1440×720 | 480 / 32 | 9.00e-1 | ❌ silent |
| 256×320×320 | 160 / 32 | 7.93e-1 | ❌ silent |
| 256×320×320 | 320 / 32 | **9.40e-3** | ✅ |

Always read `mean_rel_L1` — the divisibility asserts passing does **not** mean
the datapath is correct. (Distinguish this from the benign near-zero-`atol`
artifact: `gate_up` and `kv_cross` report FAIL on 1–52 of 10⁵ elements with
`mean_rel_L1 = 9.4e-3`; that is the documented tolerance artifact, not corruption.)

Also: **N=720 cannot use `herd_n=4`** (`720 % (4·tile_n) == 0` has no solution
with `tile_n % 16 == 0`). Either drop to `herd_n=3` or pad N 720→768 — padding
is *faster* despite +6.7% FLOPs (o_proj 135.6 µs padded vs 165.9 µs native;
down 179.5 vs 212.3).
