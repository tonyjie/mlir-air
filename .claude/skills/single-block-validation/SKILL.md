---
name: single-block-validation
description: Phase 2 of LLM deployment — wire the verified Phase 1 kernels into one transformer block on NPU and verify cosine vs CPU reference. Catches integration bugs (layout mismatches, missing transposes, type drops between kernel boundaries) before scaling to N layers.
---

## Purpose

Phase 1 verified each kernel × shape standalone. Phase 2 wires those
kernels into a single transformer block on NPU and compares the
block-level output to the CPU reference at the same input. This
catches integration bugs (wrong tensor layouts at kernel boundaries,
dropped biases, missing padding) without the cost of running N layers.

## Phase 2 PASS criteria (HARD GATES)

Run a single block at layer 0 with a canonical input (e.g., embeddings
of `"The capital of France is"`). All four must hold:

1. **Whole-tensor cosine ≥ 0.99** between NPU output and
   `<model>_reference.py:transformer_block(...)` output. Catches:
   coarse integration breaks (wrong layout, missing op).

2. **Per-position cosine ≥ THRESHOLD(head_dim)** at each real-token
   position (`[:real_len]`, NOT padded positions — those are
   out-of-distribution and amplify BF16 noise unhelpfully). Catches:
   per-row dropouts the whole-tensor cosine averages over.

   | head_dim | per-position min |
   |---|---|
   | ≤ 64  | 0.99 |
   | 128   | 0.98 |
   | ≥ 256 | 0.97 |

   Threshold scales with `head_dim` because BF16 accumulation noise
   grows as `√(head_dim · K)` (LESSON 1 from llama32_3b: smollm2 hd=64
   hits per-pos min ≈ 0.998; llama32_3b hd=128 hits ≈ 0.980 with
   5× LOWER MAE — the larger cosine drop is geometric, not a bug).

3. **No NaN** anywhere in NPU output.

4. **Result documented** in
   `<model>/docs/development_progress/phase2_block.md` (cosine numbers
   + tile/integration choice + any bisect findings if Step 4 fired).

**Also record `max_abs` and `max_rel` error** alongside the cosine
numbers (informational, not gated — absolute thresholds depend on
input distribution). The current BF16-output GEMM production path is
in 0.005–0.025 MAE territory across 7 GEMMs + softmax + RoPE + RMSNorm.
Recording these gives future deployments a baseline for regression
checks (NPU max_abs ≤ 1.5× reference deployment's measured value at
same shape signals no regression).

## Knowledge base references

PRIMARY:

- `programming_examples/llama3/llama3_prefill.py:run_transformer_block`
  — reference single-block pipeline. Inheritance-path deployments call
  this directly with new shape parameters.
- `programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
  — Phase 1 input. Lists every (kernel, shape) the model needs; Phase 2
  must wire ALL of them.
- `programming_examples/_llm_shared/docs/kernel_registry/supported_kernels.md`
  — kernel-by-kernel reference (builders, tile rules, layouts).

WORKAROUNDS (cite when model config triggers them):

- `programming_examples/qwen25_1_5b/qwen25_pad.py` — GQA-aware
  reindexed padding for non-1024-aligned dims (LESSON 4)
- `programming_examples/qwen25_1_5b/qwen25_bias.py` — host-side
  post-RoPE bias add for QKV bias models (LESSON 1)

KERNEL-FIRST PROTOTYPE:

- `programming_examples/qwen3_0_6b/multi_launch/` — model-specific
  fused ELFs when the per-layer kernel sequence diverges from llama
- `programming_examples/qwen3_0_6b/docs/development_progress/phase_b_fusion.md`
  — kernel-first integration walkthrough (Q/K Norm via heads-as-M trick)

## Workflow

### Step 1: Choose integration path

Compare the model's per-layer kernel sequence against llama's:
RMSNorm → Q/K/V GEMM → RoPE → FA → O → add → RMSNorm → Gate/Up → SwiGLU → Down → add.

**Inheritance** (default): if the sequence matches bit-for-bit, call
`llama3_prefill.run_transformer_block` with the new shape parameters.
This is what smollm2, llama32_3b, and qwen25 do.

**Kernel-first**: if any of these is true, build new model-specific
multi-launch ELFs in `<model>/multi_launch/` instead:

  (a) NEW op type (e.g., Qwen3's Q/K Norm — per-head RMSNorm with
      `(head_dim,)` weight)
  (b) NEW op needs to land BETWEEN currently-fused launches (e.g.,
      Q/K Norm sits between Q/K projection and RoPE, but
      `rms_gemv_rope` fuses both)
  (c) Op REORDER (post-norm vs pre-norm)

Don't write new C kernels speculatively — almost always the leaf kernel
exists in the registry; the trick is the right way to STITCH. For
Q/K Norm specifically, `weighted_rms_norm` with the heads-as-M trick
(M=n_heads, N=head_dim, sharing the (head_dim,) weight across rows) IS
the op — see qwen3-0.6B prototype.

### Step 2: Apply config-specific prereqs (only if model needs them)

Two known triggers from model config (NOT from upstream phases):

**Non-1024-aligned `emb_dim` or `hidden_dim`** → BD pool exhaustion
risk at long seq. Use GQA-aware reindexed padding (`qwen25_pad.py`):
pad up to a 1024-aligned multiple by inserting phantom Q heads INSIDE
each KV group (not at the end — naive padding breaks GQA semantics by
changing `n_heads / n_kv_heads = group_size`). CPU-only sanity test
the padded vs orig forward FIRST (cosine should be 0.999998+) before
touching NPU.

**`qkv_bias=True`** (Qwen2 / Qwen3 family) → host-side post-RoPE
bias add (`qwen25_bias.py`), exploiting RoPE's linearity:
`RoPE(q + bq) = RoPE(q) + RoPE(bq)`. The `rms_gemms_rope` ELF stays
bias-free; bias is added on host after the ELF returns.

If both: padding determines the n_heads count the bias precompute
uses — see `qwen25_bias.py` for the combined path.

### Step 3: Wire one block + numerical check

In `<model>/<model>_prefill.py`, implement
`run_single_block(layer_idx=0, hidden, weights, ...)`:

- **Inheritance path**: call `llama3_prefill.run_transformer_block(...)`
  with this model's shape parameters
- **Kernel-first path**: call your new
  `run_transformer_block_<model>(...)` that invokes the per-model
  multi-launch ELFs in order via `_run_cached`

Pick a canonical input (embeddings of `"The capital of France is"` is
the standard). Run both:

```python
npu_out = run_single_block(layer_idx=0, hidden=x, weights=weights, ...)
ref_out = <model>_reference.transformer_block(layer_idx=0, hidden=x, weights=weights, ...)
```

Compute whole-tensor cosine and per-position cosines (real-token
positions only). Check against the PASS criteria above.

### Step 4: Bisect on FAIL

If cosine fails, the integration is broken at one specific kernel
boundary. Bisect by swapping NPU kernels back to CPU one at a time:
walk forward through the block, replacing `npu_<kernel>(...)` with
`reference.<kernel>(...)`, recompute cosine. The first replacement
that pushes cosine above threshold identifies the offender — that's
where the layout / type / argument mismatch lives. Invoke
`superpowers:systematic-debugging` on it.

Record the bisect table (per-step cosine) in `phase2_block.md` so
future deployments learn from this specific failure.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Cosine drops at Q/K/V GEMM | weight loading / tensor layout (seq-first vs heads-first) | Compare NPU output shape to reference's; check `np.ascontiguousarray()` after weight load |
| Cosine drops at FlashAttention | causal masking missing / wrong dk_chunks compile flag | See `kernels/flash_attention.md` + `debug-fa-runtime-failure` |
| Cosine drops at Down GEMM | BF16 truncation; missing F32 accumulator | Llama Phase 3 fix — check production GEMM path uses bf16-out with F32 internal accumulate |
| NaN in output | uninitialized BO / reused stale buffer | Invoke `debug-bo-corruption` |
| Cosine drops at residual add | bias forgotten on padded path / GQA reindex bug | If padding+bias model: re-run CPU sanity test on padded forward (Step 2) |
| Whole-tensor cosine OK but per-position min low | one bad position run; check whether last few positions diverge (causal mask edge case) | Print per-position cosine, look for contiguous bad runs |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 2 PASS:

- Append cosine + per-position min + integration-path choice to
  `<model>/docs/development_progress/phase2_block.md`
- Mark Phase 2 in `<model>/TODO.md`
- If Step 2 padding or bias workarounds were used, surface as a
  Phase 4/5 prerequisite ("perf optimization must preserve the
  padded/bias wrappers")
