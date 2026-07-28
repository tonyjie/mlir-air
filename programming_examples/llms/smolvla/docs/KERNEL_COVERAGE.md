# SmolVLA → NPU2 Kernel Coverage Matrix

The施工清单 for mapping the *whole* SmolVLA model onto NPU2, not just the A1
backbone. It answers one question per row: **do we have the kernel, and has
this exact shape been measured on real NPU2 hardware?**

All shapes below are **measured** from `~/Projects/smolvla_playground/dump_full_pipeline.py`
(real `lerobot/smolvla_base` checkpoint, batch=1), NOT derived from config.
Re-dump to refresh.

## How to read this

- **Coverage tier** (does the kernel exist?):
  - ✓ **registry** — a verified `kernel_registry` leaf kernel exists.
  - ◐ **example** — a `programming_examples/` kernel exists but is not yet a
    registry entry (needs adaptation + registry-standard validation).
  - ✗ **new** — no kernel; genuinely new work.
- **Shape status** (has *this* shape been run?):
  - ✅ **measured** — run on real NPU2, recorded in `supported_kernels.md`.
  - ❌ **untested** — kernel exists, this specific shape never run. **This is
    the work.**
  - N/A — host-side (pure reshape), no NPU kernel.

The gap for A2/A3 is almost never "no kernel" — it is **"kernel exists, this
shape + tile config never measured."** That is what the next round of real-NPU
testing closes.

---

## Config (measured)

| Stage | dims |
|---|---|
| VISION (SigLIP) | hidden=768, inter=3072, heads=12 (MHA, no GQA), head_dim=64, layers=12, patch=16, img=512, seq=**1024** |
| BACKBONE (SmolLM2) | hidden=960, inter=2560, heads=15/kv=5 (GQA g=3), head_dim=64, layers=16, seq=241→**256** pad |
| EXPERT | hidden=720, inter=2048, heads=15/kv=5 (GQA g=3), head_dim=64, layers=16, self_attn_every=2, seq=**50** action tokens |
| flow-matching | chunk_size=50, num_steps=10 (expert runs 10×), action_dim=32 |

Runtime: image input **bf16**; backbone/expert projections **bf16**; norms +
state_proj + action projections **fp32** in/out (the CPU baseline is bf16-compute
for all matmuls).

---

## STAGE 2 — BACKBONE (A1, DONE)  ✅ all measured

Runs on NPU today, `make verify` PASS. Included as the coverage baseline — this
is what "measured" looks like. Full tile configs in `docs/PROGRESS.md` +
`kernel_registry/supported_kernels.md`.

| Op | Kernel | Shape | Tier | Shape status | Registry row |
|---|---|---|---|---|---|
| input/post RMSNorm | RMSNorm | 256×960 | ✓ registry | ✅ measured (4.2e-3) | L188 |
| q_proj | GEMM bf16-out | 256×960×960 | ✓ | ✅ (1896 GF) | L92 |
| k/v_proj | GEMM bf16-out | 256×960×320 | ✓ | ✅ (1228) | L93 |
| RoPE Q/K | RoPE half-split | 256×64 | ✓ | ✅ (2.8e-3) | L293 |
| **QKᵀ** | GEMM (per GQA group) | 5×(256×64×256) | ✓ (generic) | ⚠️ **runs, not a registry row** | — |
| **softmax** | masked_softmax | 15×256×256 | ◐ example (new in A1) | ⚠️ **runs, not a registry row** | — |
| **PV** | GEMM (per GQA group) | 5×(256×256×64) | ✓ (generic) | ⚠️ **runs, not a registry row** | — |
| o_proj | GEMM bf16-out | 256×960×960 | ✓ | ✅ | L92 |
| residual add | EltwiseAdd | 256×960 (N=245760) | ✓ | ✅ (tile_n=1920!) | L241 |
| gate/up | GEMM bf16-out | 256×960×2560 | ✓ | ✅ (2838) | L94 |
| SiLU-and-Mul | SiLU-Mul | 256×2560 (N=655360) | ✓ | ✅ (1.0e-2) | L263 |
| down | GEMM bf16-out | 256×2560×960 | ✓ | ✅ (3425) | L95 |

**Backbone gap:** the 3 attention sub-ops (QKᵀ / masked_softmax / PV) *run
correctly on NPU* but were never promoted to registry rows. If A2 reuses them
(it does — same GQA/prefix-mask semantics at seq=50), they should be measured +
recorded as first-class rows.

---

## STAGE 3 — EXPERT (A2 target)  ❌ every shape untested

hidden=720, seq=50, GQA 15/5. Reuses **all** backbone kernels — but every shape
is new, and **M=50 is small + non-256-aligned**, so tile configs are unknown
(the backbone's tile_m=32 assumes M=256=32·8 fills one herd row; M=50 does not).
Expert runs **10×** per action chunk (20% of end-to-end runtime).

### Projections (GEMM bf16-out) — kernel ✓, shapes ❌

| Op | Shape (M×K×N) | Tier | Status | Nearest registry row | Tile risk |
|---|---|---|---|---|---|
| q_proj | 50×720×960 | ✓ | ❌ | none at M=50 | K=720 tile_k_l2? N=960→TILE_N=80 |
| k/v_proj | 50×720×320 | ✓ | ❌ | none | N=320→TILE_N=80 |
| o_proj | 50×960×720 | ✓ | ❌ | none | N=720 not 512-aligned |
| gate/up | 50×720×2048 | ✓ | ❌ | none | N=2048→TILE_N=128 |
| down | 50×2048×720 | ✓ | ❌ | none | K=2048 tile_k_l2=256? |
| action_in_proj | 50×32×720 | ✓ | ❌ | none | K=32 tiny |
| action_time_mlp_in | 50×1440×720 | ✓ | ❌ | none | K=1440 (720·2 concat) |
| action_time_mlp_out | 50×720×720 | ✓ | ❌ | none | square-ish |
| action_out_proj | 50×720×32 | ✓ | ❌ | none | N=32 tiny |
| state_proj | 1×32×960 | ✓ | ❌ | none | M=1 → GEMV territory |

### Norm / activation / elementwise — kernel ✓, shapes ❌

| Op | Kernel | Shape | Tier | Status |
|---|---|---|---|---|
| input/post RMSNorm | RMSNorm | 50×720 | ✓ | ❌ |
| RoPE (self-attn Q/K) | RoPE half-split | 50×64 | ✓ | ❌ |
| SiLU-and-Mul | SiLU-Mul | 50×2048 (N=102400) | ✓ | ❌ |
| residual add | EltwiseAdd | 50×720 (N=36000) | ✓ | ❌ (watch tile_n) |

### Attention — reuses A1's approach-B, new shapes ❌

| Layer type | Op | Shape | Notes |
|---|---|---|---|
| EVEN (self-attn) | QKᵀ/softmax/PV | q=50, kv=50, GQA 15/5 | 50×50 score matrix, tiny |
| ODD (cross-attn) | QKᵀ/softmax/PV | q=50, **kv=241 (backbone cache)** | K/V from prefix — **K/V-injection mechanism already built in A1** |

**A2 verdict:** zero new kernels. 14 GEMM shapes + 4 norm/act/elt shapes + 2
attention configs, all untested at these small M. The K/V-injection seam (A1's
core deliverable) already handles the cross-attn wiring. Main risk = small-M
tile configs + NPU under-utilization at seq=50 (may not beat CPU on perf, but
correctness-first).

---

## STAGE 1 — VISION (A3 target)  ❌ untested + 2 non-registry kernels + 1 new

SigLIP ViT, hidden=768, seq=1024, 12-head MHA (**no GQA**), **bidirectional
(non-causal)** attention, **LayerNorm (not RMSNorm)**, **GELU-tanh (not SiLU)**.
73.6% of end-to-end runtime — the biggest perf prize.

### GEMM projections — kernel ✓, shapes ❌

| Op | Shape (M×K×N) | Tier | Status | Note |
|---|---|---|---|---|
| q/k/v/out_proj | 1024×768×768 | ✓ | ❌ | 768 not 512-aligned → TILE_N shrink |
| mlp fc1 | 1024×768×3072 | ✓ | ❌ | |
| mlp fc2 | 1024×3072×768 | ✓ | ❌ | |
| connector proj | 64×12288×960 | ✓ | ❌ | **K=12288 huge**, M=64 tiny; L2-bound |

### LayerNorm — ◐ example, not registry, shapes ❌

| Op | Shape | Tier | Status |
|---|---|---|---|
| layer_norm1/2 | 1024×768 | ◐ `programming_examples/layer_norm/` (bf16) | ❌ + **not a registry kernel** |
| post_layernorm | 1024×768 | ◐ same | ❌ |

LayerNorm ≠ RMSNorm (subtracts mean + has bias/beta). Must promote the example
to a registry-standard kernel first, then measure at 1024×768.

### GELU-tanh — ◐ example, not registry, shapes ❌

| Op | Shape | Tier | Status |
|---|---|---|---|
| mlp activation_fn | 1024×3072 (N=3.1M) | ◐ `programming_examples/gelu/` (NPU2 `__builtin_aie2p_tanh`) | ❌ + **not a registry kernel** |

### Conv2d patch embedding — ◐ example, shapes ❌

| Op | Shape | Tier | Status |
|---|---|---|---|
| patch_embedding | (1,3,512,512)→(1,768,32,32), patch16 | ◐ `conv2d/` (int32) + `conv2d_14x14/` (bf16 NPU2) | ❌ + adapt to 512/patch16 |

### Attention — 1024² bidirectional, ❌

| Op | Shape | Tier | Status |
|---|---|---|---|
| self_attn | 1024×1024, 12/12 MHA, **non-causal** | ✓ FA (has 512×512 12/6 non-causal) OR ◐ approach-B masked | ❌ no 1024² MHA row |

FA supports non-causal + this could reuse approach-B (no mask needed — vision is
fully bidirectional, so even a maskless full-softmax works). 1024² score matrix
= 2MB fp32, larger than backbone's 256² — may or may not fit L2; **this is the
one shape where flash's online-softmax tiling might actually earn its keep.**

### pixel-shuffle — ✗ new (but host-side)

| Op | Shape | Tier | Status |
|---|---|---|---|
| connector pixel-shuffle | (1,1024,768)→(1,64,12288) | ✗ new | N/A — pure reshape (space-to-depth factor 4), host-side, no kernel needed |

**A3 verdict:** 4 GEMM shapes (kernel ✓) + LayerNorm/GELU-tanh (◐ promote 2
examples to registry) + Conv2d (◐ adapt) + 1024² bidirectional attention + host
pixel-shuffle. Bigger integration lift than A2, but the biggest perf payoff
(74%).

---

## Summary — what the next NPU-testing round must produce

| | new kernels to write | examples to promote | GEMM/norm/act shapes to measure |
|---|---|---|---|
| **A2 expert** | 0 | 0 (reuse A1) | ~18 shapes at small M (50, 1) |
| **A3 vision** | 0 (pixel-shuffle is host) | 2 (LayerNorm, GELU-tanh) + Conv2d adapt | ~7 shapes at seq=1024 + 1024² attn |
| **A1 backbone** | — | — | promote 3 attention sub-ops (QKᵀ/softmax/PV) to registry rows |

**The through-line:** almost everything is "kernel exists, shape untested." The
施工 is a shape-sweep + tile-tune + precision-gate pass per row, exactly like
Phase-1 kernel validation was for A1 — plus promoting LayerNorm and GELU-tanh
from `programming_examples/` to registry kernels for A3.

Ordering options (perf-first vs risk-first):
- **Risk-first (A2 expert):** 0 new kernels, all shape-sweeps, validates the
  small-M tile story + reuses the K/V-injection seam. 20% perf.
- **Perf-first (A3 vision):** 74% perf, but needs LayerNorm/GELU-tanh promotion
  + Conv2d + 1024² attn integration.
