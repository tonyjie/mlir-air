# SmolVLA Backbone → NPU2 Port — Design Spec

**Date:** 2026-07-13
**Branch:** `smolvla` (worktree)
**Target:** `programming_examples/llms/smolvla/` in mlir-air
**Status:** design approved in brainstorming; pending user review before writing-plans

---

## 1. Goal & Scope

Port as much of **SmolVLA** (`lerobot/smolvla_base`, 450M) onto **AMD NPU2 (Strix,
AIE2P)** via **mlir-air** as is practical, while **guaranteeing numerical
correctness** against the existing CPU baseline. Not all of the model must run on
NPU — a **hybrid CPU/NPU pipeline** is expected and acceptable.

- **This is a new example** under `programming_examples/llms/smolvla/`, using the
  mlir-air deployment flow (kernel_registry + `programming_examples/llms/verify/`).
- **The CPU baseline is the oracle** — `~/Projects/smolvla_playground/` (LeRobot
  v0.5.0). The NPU port must match it within bf16 tolerance.
- **The Allo/NPU1 repo (`vla-to-npu`) is reference only** — algorithm ideas, not
  trusted code. It is a scaled-down toy (ViT 1 layer, LLaMA 2 layers, prefix=128,
  NPU1 tile constants) and was never verified by the user.

### Execution path (agreed)

- **A1 (this spec's milestone):** port the **SmolLM2-360M language backbone**
  (16-layer prefill) to NPU2. Vision + connector + action-expert + denoise loop
  stay on CPU.
- **A2 (follow-on):** extend to the **action expert** (cross-attn + causal
  self-attn, 10-step flow-matching), reusing the new masked-attn kernel.
- **A3 (stretch, likely never):** vision encoder (SigLIP). It is
  `freeze_vision_encoder=true`, hardest (1024×1024 bidirectional attention + 3 new
  kernels: GELU, LayerNorm, pixel-shuffle), and lowest ROI — **intentionally left
  on CPU**.

---

## 2. Ground-Truth Facts (measured from the real checkpoint)

Measured by hooking `lerobot/smolvla_base` on CPU (`dump_backbone_shapes.py`) and
reading the loaded model config — NOT from paper prose.

### Backbone (SmolLM2, the ported stage)

| Param | Value |
|---|---|
| hidden_size | **960** |
| intermediate_size | **2560** |
| num_attention_heads | **15** |
| num_key_value_heads | **5** (GQA group = 3) |
| head_dim | **64** (< 128 → no FA-hang risk) |
| layers used | **16** (first half of 32; `text_model.layers[:16]`) |
| rope_theta | **100000** (`rope_type=default`, `rope_interleaved=False` → half-split) |
| rms_norm_eps | 1e-5 |
| input RMSNorm | done in **fp32** (reduction), matching registry RMSNorm-fp32 |

### Prefix sequence length = **241** (decomposition verified)

```
3 cameras × 64 visual tokens (post pixel-shuffle) = 192
+ 48 language tokens (tokenizer_max_length)        = 240
+ 1 state token                                    = 241  ✓ matches text_model seq
```

### Per-layer op shapes (seq=241)

| Op | in → out |
|---|---|
| input RMSNorm | (241,960) fp32 |
| q_proj (GEMM) | (241,960) → (241,960) |
| k/v_proj (GEMM) | (241,960) → (241,320) |
| RoPE | head_dim=64, θ=100000, half-split |
| **attention QKᵀ** | (15, 241, 241) — **non-causal (prefix-bidirectional mask)** |
| o_proj (GEMM) | (241,960) → (241,960) |
| residual add | (241,960) |
| post RMSNorm | (241,960) |
| SwiGLU: gate/up (GEMM) | (241,960) → (241,2560) |
| SiLU-and-Mul | (241,2560) |
| down (GEMM) | (241,2560) → (241,960) |
| residual add | (241,960) |
| final norm (after 16 layers) | (241,960) |

### Attention mask (the one hard new thing)

`make_att_2d_masks` (`modeling_smolvla.py:101-131`): image+language tokens attend
**bidirectionally** to each other; the state token starts a new causal block.
This is **prefix-LM masking, NOT causal** — the registry FlashAttention is
causal-pinned and does not cover it.

### How the backbone runs (critical structural facts)

- **Prefill-only, one shot.** No autoregressive decode, no token sampling, no
  lm_head in the action path (`modeling_smolvla.py:823-830`).
- **Output = per-layer KV cache** (not a single hidden vector); the 10 denoise
  steps reuse this cache and **never recompute the prefix**
  (`modeling_smolvla.py:834-868`).
- **Backbone + action-expert are a fused per-layer loop**
  (`smolvlm_with_expert.py:403-498`): cross-attn layers use the same-layer VLM
  cached K/V; self-attn layers (every 2nd) concat VLM + expert K/V into one
  attention op. → For **A1 this does not matter** (expert is off-NPU); for **A2**
  the KV cache must be a shared, addressable buffer.

---

## 3. Kernel Coverage (backbone stage)

Only **one genuinely new kernel**. The other 7 op-classes are existing registry
kernels at new shapes (Phase-1 re-validation work).

| Backbone op | Registry kernel | SmolVLA shape | Status |
|---|---|---|---|
| input/post RMSNorm | RMSNorm BF16 (fp32 reduce) | 241×960 | ⚠️ new shape |
| q_proj | GEMM bf16 | 241×960×960 | ⚠️ new shape |
| k/v_proj | GEMM bf16 | 241×960×320 | ⚠️ new shape (GQA narrow) |
| RoPE | RoPE BF16 half-split | head_dim=64, θ=100000 | ⚠️ new shape |
| **non-causal attention** | — (FA is causal-pinned) | 241×241 bidirectional | ❌ **NEW KERNEL** |
| o_proj | GEMM bf16 | 241×960×960 | ⚠️ new shape |
| residual add | Element-wise Add BF16 | 241×960 | ⚠️ new shape |
| SwiGLU | SiLU-and-Mul + GEMM | 960→2560→960 | ⚠️ new shape |

### New kernel: small-matrix masked attention

seq=241 is small — single-head QKᵀ = 241×241×fp32 ≈ 232 KB (fits L2). So do NOT
modify the flaky causal-pinned FlashAttention. Instead build a dedicated
**non-flash masked attention**:

```
per head (15 q-heads, GQA reuse of 5 kv-heads):
  S = Q @ Kᵀ / √64          GEMM (241,64)×(64,241) → (241,241)
  S += prefix_mask           host-precomputed additive mask (bidirectional=0, no causal triangle)
  P = softmax(S)             device softmax (registry masked-softmax algorithm as reference)
  O = P @ V                  GEMM (241,241)×(241,64) → (241,64)
```

- Mask precomputed **host-side** as an additive (241×241) tensor and injected —
  the mature "host mask + device softmax" pattern from the Allo reference, with
  `triu` replaced by the prefix mask.
- Added to `kernel_registry` via the **add-kernel** skill/agent-team, so it is a
  first-class, reusable entry (**A2's action-expert attention reuses it**).

---

## 4. Hybrid Pipeline (A1)

```
[CPU] 3 cameras → SigLIP → connector → 192 visual tokens ┐
[CPU] language tokenize → 48 tokens                       ├→ assemble prefix (241,960)
[CPU] state → 1 token                                     ┘         │
                                                                    ▼
[NPU2] ★ SmolLM2 backbone, 16-layer prefill ★  → per-layer KV cache / hidden
                                                                    │
[CPU] action expert + 10-step flow-matching denoise  ←──────────────┘
                                                                    ▼
                                                            (50,6) action chunk
```

The CPU→NPU cut point (assembled prefix) is **injectable and comparable** — we can
hook it out of the CPU baseline, so both correctness tracks are feasible.

---

## 5. Correctness (dual-track, both required)

- **Diagnosis lens (per-layer cosine):** each ported module's output vs the CPU
  baseline's forward-hook tensor at that layer. Used to localize regressions.
  Mirrors the `make diagnosis` per-layer-cosine lens.
- **PASS/FAIL gate (end-to-end):**
  1. NPU backbone 16-layer output (hidden / KV cache) vs CPU baseline, within bf16
     tolerance.
  2. Full hybrid pipeline's (50,6) action chunk vs the pure-CPU baseline chunk,
     within bf16 tolerance (MSE / cosine).

  Mirrors the `make verify` gate. Both must pass to call a phase done.

Tolerances follow the registry precedent: GEMM/RMSNorm ~4e-3–1e-2 mean_rel_L1,
attention ~4e-2 (two BFP16 MMAs + bf16 softmax).

---

## 6. Phased Execution (deploy-* workflow)

Uses the `deploy-*` agent chain (researcher → planner → runner → verifier) and the
`phase-*` skills. The verifier independently re-runs and audits WHERE each kernel
executes (NPU vs CPU).

- **Phase 0 — CPU reference.** Largely exists in `~/Projects/smolvla_playground/`.
  Formalize a `smolvla_weights.py` (HF loader for the 16-layer backbone) +
  `smolvla_cpu_helpers.py`, and a prefix-injection hook. Confirm the HF bf16
  baseline loads/runs via the shared `verify/` subsystem.
- **Phase 1 — kernel validation.** Re-validate the 7 existing registry kernels at
  SmolVLA shapes (241×960 etc.); **build + validate the new masked-attn kernel**
  via add-kernel. Record each (kernel, shape) in the registry tested-shapes tables.
- **Phase 2 — single-block validation.** Wire the verified kernels into ONE
  backbone layer on NPU; per-layer cosine vs CPU baseline at layer 0.
- **Phase 3 — full-backbone validation.** All 16 layers; end-to-end KV/hidden gate
  + full hybrid-pipeline action-chunk gate vs CPU baseline.

(Phases 4–5 optimization and A2/action-expert are out of scope for this spec; they
follow once A1 Phase 3 passes.)

---

## 7. Open Items / Risks

1. **Non-causal masked-attn kernel is net-new** — the one true unknown. Mitigated
   by the small 241² matrix (fits L2, no flash needed) and the Allo host-mask
   reference pattern.
2. **All backbone shapes are new** (seq=241, hidden=960, GQA 15/5). Each registry
   kernel needs re-validation; some may need tile retuning (241 is not a nice
   power-of-two; may need padding to 256).
3. **Prefix assembly fidelity** — the CPU-side vision/connector/tokenize path must
   produce byte-comparable prefix input to what the CPU baseline feeds its
   backbone, or the correctness comparison is invalid. Verified injectable via
   forward hooks.
4. **`num_expert_layers=0` semantics** — for A1 irrelevant (expert off-NPU);
   confirm before A2 (it means expert layers = num_vlm_layers = 16, one per VLM
   layer, per `smolvlm_with_expert.py:98-104`).
5. **CLAUDE.md rename** — per project convention, any CLAUDE.md must ship as
   ARCHITECTURE.md (top-level .gitignore excludes CLAUDE.md).

## 8. Non-Goals

- Vision encoder on NPU (A3) — stays CPU.
- Closed-loop LIBERO simulation / task success rate — out of scope (GPU/MuJoCo).
- Autoregressive decode / token-set gate — SmolVLA has none.
- Copying Allo/`vla-to-npu` code — reference only.
