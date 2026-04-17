# Phase 3 — Full-model correctness

**Date**: 2026-04-17
**Layers**: 24 of 24
**Attention path**: NPU FlashAttention (MHA, n_kv_heads=32)
**LM Head + final RMSNorm**: CPU (NPU LM Head deferred to Phase 5)

## Setup

- `smollm2_phase3_test.py` (in this dir) wires 24 layers via the existing
  `run_transformer_block` from `llama3_prefill.py` — config-driven, no code
  change needed. Final RMSNorm and LM Head GEMM run on CPU (F32).
- Kernel cache reused from Phase 2 (3 cached kernels: `rms_gemms_rope`,
  `o_ffn`, `flash_attn` — all built for SmolLM2's MHA shapes).

## Top-1 prediction results (canonical prompts)

| Prompt | NPU top-1 | CPU top-1 | Match | Logits cosine |
|--|--|--|--|--|
| `"The capital of France is"` | `' Paris'` (id=7042) | `' Paris'` | ✅ | 0.995684 |
| `"1 + 1 ="` | `' '` (id=216) | `' '` | ✅ | 0.993608 |
| `"The sky is"` | `' blue'` (id=4461) | `' blue'` | ✅ | 0.998675 |

**3/3 top-1 match** with the CPU reference. Note: SmolLM2 tokenizes "1 + 1 = 2"
with a space token after `=`, so its top-1 next-token after `1 + 1 =` is
literally `' '` (CPU and NPU agree, this is the model's actual prediction).

## Per-layer drift (whole-tensor cosine sim vs CPU F32)

Diagnostic run on prompt `"The capital of France is"`:

| Layer | whole cos | per-pos min | | Layer | whole cos | per-pos min |
|--|--|--|--|--|--|--|
| 0 | 0.998583 | 0.997820 | | 12 | 0.981620 | 0.992071 |
| 1 | 0.992783 | 0.988116 | | 13 | 0.980716 | 0.993528 |
| 2 | 0.992759 | 0.989725 | | 14 | 0.979550 | 0.994238 |
| 3 | 0.997198 | 0.993146 | | 15 | 0.978737 | 0.995031 |
| 4 | 0.997239 | 0.992841 | | 16 | 0.977931 | 0.995321 |
| 5 | 0.999005 | 0.992488 | | 17 | 0.976981 | 0.994936 |
| 6 | 0.998805 | 0.992842 | | 18 | 0.975857 | 0.994416 |
| 7 | 0.985855 | 0.992893 | | 19 | 0.975008 | 0.993756 |
| 8 | 0.984925 | 0.993353 | | 20 | 0.974487 | 0.994028 |
| 9 | 0.983988 | 0.992658 | | 21 | 0.974250 | 0.991696 |
| 10 | 0.983363 | 0.992024 | | 22 | 0.988647 | 0.986714 |
| 11 | 0.982551 | 0.992615 | | 23 | 0.998324 | **0.518730** |

**Min whole-tensor cosine = 0.974 (layer 21)** — well above the 0.95 gate.
**Max per-pos cosine min** is 0.518 at layer 23 — informational only:

> The per-position cosine drop at layer 23 occurs at a single padding-token
> position whose hidden state has been driven near zero by the residual
> updates. Cosine sim of near-zero vectors is numerically unstable. The
> whole-tensor cosine at layer 23 is 0.998 and the next-token logits at the
> real prediction position have cosine 0.996, so this artifact has no
> downstream impact.

## Performance (kernel-only, 24 layers, NPU FlashAttention)

| Run | NPU prefill (s) | per layer (ms) |
|--|--|--|
| First prompt (BO allocation cost on per-layer first call) | 4.16 | 173 |
| Second prompt (BOs cached) | 1.98 | 83 |
| Third prompt (BOs cached) | 1.99 | 83 |

Steady-state ~83 ms/layer. By comparison, llama3 (16 layers, GQA) runs
~81 ms/layer, so SmolLM2 hits the same per-layer rate despite 4× larger
KV-projection compute (MHA). The deeper stack (24 vs 16 layers) gives total
prefill **2.0s** vs llama3's **1.3s** — proportional to depth.

CPU reference prefill: ~22-29 s per prompt.

## Phase 3 gate verdict

- ✅ Top-1 match: 3/3 prompts
- ✅ Per-layer whole-tensor cosine_sim > 0.95: min = 0.974 at layer 21
- ✅ No NaN
- ✅ Logits correlation > 0.99 on all 3 prompts

**PASS — end-to-end SmolLM2-1.7B prefill correctness milestone reached.**

## Items surfaced

- 🔸 **NPU LM Head still deferred** — Phase 5 must implement it (vocab=49152
  doesn't fit llama3's hardcoded 8×16384 partitioning; recommended option is
  3×16384 = 49152 exact)
- 🔸 **Per-layer drift is monotonic in mid-stack** (layers 7-21: 0.986→0.974)
  — indicates accumulated BF16 truncation. Not a regression vs llama3
  (whose Down GEMM fix from BF16 to F32 output was reverted for performance).
  If higher accuracy is ever needed, F32-output GEMMs are the lever.
- 🔸 **Per-layer first-call BO allocation** adds ~3s on first prompt — the
  NPU prefill fast path is ~2s for subsequent calls. Phase 4 (prefill perf)
  will introduce explicit weight pre-loading to amortize this.
