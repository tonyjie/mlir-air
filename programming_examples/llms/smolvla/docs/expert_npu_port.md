# SmolVLA action expert on NPU2 — the port, and what it cost

**The expert now runs end to end on NPU2 and is correct.** It is also about
**4× slower than the CPU**, which was the expected outcome and was accepted up
front: the goal was a complete, correct mapping plus an honest comparison, not
a speedup.

All numbers real-machine measured on AMD Ryzen AI 9 HX 370 / NPU2 (Strix,
AIE2P), CPU governor `performance` + EPP `performance`, NPU `pmode=Turbo`,
machine idle (load 1.47), configs interleaved.

---

## 1. Correctness

Tiered, weakest assumption first, so a failure localises itself.

| level | what it isolates | result | gate |
|---|---|---|---|
| L1 teacher-forced per layer | one layer, oracle input (3 steps × 16 layers) | **worst 0.999919** | 0.98 PASS |
| L2 teacher-forced per step | 16 layers stacked | **worst 0.998724** | 0.99 PASS |
| L3 free-running 10 steps | the real denoise recurrence | **final x_t 0.998343** | reported |
| e2e, NPU expert + CPU vision | action chunk vs pure-CPU baseline | **cosine 0.998737**, nmse 0.002823 | PASS |
| e2e, NPU vision + NPU expert | both stages on NPU | **cosine 0.996516**, nmse 0.008252 | PASS |

Two things worth stating plainly:

**The error does not compound over the 10 sequential denoise steps — it damps.**
Free-running `x_t` cosine rises monotonically 0.980 → 0.997 across the loop.
Flow matching is self-correcting: each step's `v_t` is a fresh correction toward
the same target, so a BFP16 perturbation gets absorbed rather than carried. This
was the main correctness risk going in, and it did not materialise.

**Depth costs ~0.0012.** 0.99992 for an isolated layer → 0.9987 through all 16.
That is ordinary BFP16 accumulation with real margin against the gate.

---

## 2. Performance

| config | total | expert stage | vs CPU expert |
|---|---|---|---|
| pure CPU | 988.4 ms | 334.5 ms | 1.00× |
| **NPU expert**, CPU vision | 1978.5 ms | **1329.8 ms** | **3.98× slower** |
| NPU vision, CPU expert *(shipping)* | **840.7 ms** | 284.1 ms | — |
| NPU vision + NPU expert | 1852.7 ms | 1300.6 ms | 4.58× slower |

### Where the time actually goes

Profiled rather than assumed, per denoise step:

| | ms | share |
|---|---|---|
| wall | 112.7 | 100% |
| XRT calls | 102.5 | 91% |
| ├ NPU Run (device) | 98.4 | **87%** |
| ├ BO write | 4.1 | 4% |
| └ BO read | 1.0 | 1% |
| host glue outside XRT | 10.2 | 9% |

**~91% of the cost is inside XRT and 96% of that is the device actually
executing.** This is not dispatch overhead and not host glue. No amount of
driver or runtime work moves it.

Per-dispatch device time, and the count per denoise step:

| ELF | µs/dispatch | ×/step | ms/step |
|---|---|---|---|
| `expert_o_ffn` | 1670 | 16 | 26.7 |
| `qkt_self` / `qkt_cross` | 230 / 300 | 40 / 40 | 21.2 |
| `pv_self` / `pv_cross` | 300 / 290 | 40 / 40 | 23.6 |
| `expert_rms_qkv_rope` | 1180 | 8 | 9.4 |
| `masked_softmax` self/cross | 490 / 450 | 8 / 8 | 7.5 |
| `expert_rms_q_rope` | 630 | 8 | 5.0 |
| | | **208** | **93.4** |

---

## 3. Why 1127 ms and not the 526 ms the feasibility study projected

The gap is 601 ms and it decomposes into exactly two causes, both measured.

**(a) FlashAttention turned out to be unusable — +378 ms.** The projection
assumed one FA dispatch per layer at ~900 µs. FA cannot express this model's
mask: dumped from the real checkpoint, only **197 of the 241 prefix tokens are
real** (the language block is padded to `tokenizer_max_length`), so *both* layer
kinds carry a genuine mask, and the registry FA ELF applies none. The
decomposed path (5 qkt + 1 masked_softmax + 5 pv, GQA-group batched) costs
52.3 ms/step = 523 ms/inference against the projected 145 ms.

This was not knowable at feasibility time — that study derived the mask from a
synthetic all-ones `prefix_pad`, which is also the mistake this port initially
repeated before dumping the real thing.

**(b) The fusion penalty — +143 ms.** The projection's scenario A assumed one
dispatch per op, i.e. unfused. This port fuses, mirroring the backbone and
vision ports. Measured against the same ops as standalone kernels
(`expert_npu_feasibility.md` §2/§3):

| fused ELF | fused | Σ standalone | penalty | cost/inference |
|---|---|---|---|---|
| `expert_o_ffn` | 1670 µs | 1131 µs | **1.48×** | 86.2 ms |
| `expert_rms_qkv_rope` | 1180 µs | 689 µs | **1.71×** | 39.3 ms |
| `expert_rms_q_rope` | 630 µs | 416 µs | **1.52×** | 17.1 ms |
| | | | | **143 ms** |

That reproduces the 1.39–1.79× the feasibility study measured for the
*backbone's* fused ELFs, and it confirms that study's warning that "fusion
actively costs device time… for the expert that trade is decisively negative."
Fusing was still the right call for a correctness-first port — it is the pattern
both prior stages use, and it kept the per-layer glue on-device — but it is not
free and the cost is now quantified rather than assumed.

`440 ms projected device + 378 + 143 ≈ 961 ms`, against 934 ms measured. The
projection's model was sound; two of its inputs were wrong.

### What de-fusing would buy

~143 ms, i.e. 1127 → ~984 ms, **still ~3.4× slower than CPU**. Worth doing if
this path is ever deployed, not worth doing to change the verdict.

---

## 4. Why the NPU loses here

Unchanged from the feasibility study, and now confirmed end to end:

- **seq=50 pads to 64**, and 64 admits only `tile_m=16, herd_m=4` — half the
  8×4 herd is idle before the first instruction. Measured GFLOP/s at the
  expert's shapes are 25–1163 against 2046–3366 for the *same kernels* at the
  backbone's seq=256.
- **Every launch costs ~85 µs regardless of work**, and a denoise step issues
  208 of them.
- **GEMM is only 57% of the CPU expert's time** (`expert_cpu_breakdown.md` §4),
  so even a perfect matmul accelerator caps the achievable win.

The expert is the mirror image of the vision tower: vision wins 1.19×/image
because seq=1024 and 768/3072-wide matmuls saturate the array; the expert loses
because seq=50 cannot.

---

## 5. What the port is

**Scope:** the 16 transformer layers × 10 denoise steps (97.2% of a step).
`embed_suffix` and `action_out_proj` stay on lerobot's CPU path — 2.4%, and both
are pure launch-floor shapes on NPU. The final `lm_expert.norm` also stays on
CPU (0.58 ms, 0.2%), the same choice the backbone port and every llama/qwen
sibling makes.

**Per layer, 13 dispatches:**

```
EVEN (self-attn)                     ODD (cross-attn)
  expert_rms_qkv_rope  (6 launches)    expert_rms_q_rope  (3 launches)
    RMSNorm + Q/K/V + RoPE Q,K           RMSNorm + Q + RoPE Q
  attention: 5 qkt + 1 masked_softmax + 5 pv   (GQA-group batched)
    K/V = 241 prefix + 50 new = 291→320   K/V = 241 prefix →256
  expert_o_ffn (8 launches)
    O + residual + RMSNorm + gate/up + SiLU-mul + down + residual
```

**One structural optimisation, exact:** the cross-attention K/V re-projection is
hoisted out of the denoise loop. It reads only the backbone's constant 241-token
cache — lerobot recomputes it on all 10 steps although it cannot change, and the
oracle dump asserts step-invariance. 16 GEMMs instead of 160.
`hoist_cross_kv=False` restores the faithful behaviour for A/B.

**One deliberate numeric deviation:** the odd layers' `k_proj`/`v_proj` weights
are stored **fp32** in the released checkpoint while everything else is bf16
(`smolvlm_with_expert.py:113,118` replaces them with bare `nn.Linear` after the
expert was built bf16). The port casts them to bf16. Measured cost: **none** —
mirroring the downcast in numpy gives 0.999997 either way. It is also not purely
a concession: the same GEMM is 59.2 µs bf16 vs 92.3 µs fp32 on this CPU, so the
fp32 storage costs the *baseline* ~1.56× on those two projections.

---

## 6. Reproduce

```bash
cd programming_examples/llms/smolvla
make expert-gate            # tiered teacher-forced gate (L1/L2/L3)
make verify-expert-only     # e2e action chunk, NPU expert + CPU vision
make verify-npu-expert      # e2e, NPU vision + NPU expert
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  $LEROBOT_PYTHON bench_e2e.py --iters 7 --configs cpu,expert,vision,vision+expert
```
(The Makefile targets already wrap themselves in the NPU flock — do not add an
outer one, it self-deadlocks.)

Raw data: `results/expert_npu_gate.txt`, `results/expert_npu_freerun.txt`,
`results/expert_bench_e2e.txt`, `results/expert_profile.txt`.

---

## 7. Two bugs worth remembering

**`mm.o` bakes DIM_K, but the shared helpers name objects from `tile_n` alone.**
`compile_gemm_mm` bakes `DIM_M=tile_m`, `DIM_N=tile_n` *and*
`DIM_K=tile_k_l1`, while `disambiguate_by_tile_n` /
`_force_tile_n_suffix` key the filename on `tile_n` with an `m32`/`m64` tag
taken from the *method*. The expert is the first model to trip this: q/k/v
resolve to `(16, k1=48, n=80)` and o/down to `(16, k1=32, n=80)`, so both would
claim `mm_m16_n80.o` and whichever compiled last would silently hand the other a
wrong-`DIM_K` microkernel. Fixed here by keying on all three dims; **the shared
helpers still carry the latent hazard.**

**Sharing one `bo_key` between two dispatches trips two BO hazards at once.**
The cross K/V projection ran K and V under one key, which meant (a)
`static_input_indices` uploaded the weight only on the first call, so V silently
reused K's weight, and (b) `load_and_run` returns **zero-copy views**, so K's
returned array aliased the buffer V then overwrote. Both outputs ended up as
V's result — cos −0.02. The tiered gate localised it in one run (self 0.99997,
cross 0.96665), and mirroring the bf16 downcast in numpy ruled out precision
before any hardware debugging started.
