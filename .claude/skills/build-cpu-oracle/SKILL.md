---
name: build-cpu-oracle
description: Phase 0 of LLM deployment — build a decomposed CPU FP32 reference (`<model>_reference.py`) aligned 1:1 with NPU kernel layout, then verify it numerically against HuggingFace transformers. This reference becomes the ground-truth oracle for every downstream phase's cosine check.
---

## Purpose

Phase 0 produces the **CPU oracle** that every downstream phase compares
NPU output against. It is two things, both required:

1. **A decomposed CPU FP32 implementation** (`<model>_reference.py` +
   `<model>_weights.py`) where each kernel call (RMSNorm, GEMM, RoPE,
   FlashAttention, SwiGLU, etc.) is an isolated numpy function with
   inputs/outputs **byte-aligned** to what the NPU kernels expect
   (e.g., RoPE half-split LUT layout, K/V cache memory layout). This
   decomposition is what enables Phase 1's per-kernel cosine checks —
   HF transformers is a black box that exposes only end-to-end forward,
   so Phase 1+ cannot use HF directly.

2. **Numerical equivalence to HuggingFace transformers** as Phase 0's
   own gold standard. Without this, a bug in our reference (wrong RoPE
   convention, wrong norm order, mis-placed bias) silently poisons every
   downstream cosine check: NPU agrees with our buggy reference → all
   PASS → real top-1 doesn't match HF → only the Phase 7 evaluator
   discovers it.

### Why hand-written reference, not direct HF oracle

A reasonable alternative is "skip the hand-written reference and let
Phase 1+ compare NPU directly against HF transformers via forward
hooks". We chose **hand-written** for these specific reasons; do not
silently switch to a pure HF oracle without re-evaluating them:

1. **Layout alignment with NPU**: our reference is intentionally
   structured in the NPU's seq-first flat layout (`[seq, n_heads*head_dim]`)
   throughout the attention block. HF's `LlamaAttention` uses head-first
   `[batch, n_heads, seq, head_dim]` for RoPE and FA inputs/outputs.
   Comparing NPU vs HF directly requires explicit transpose adapters at
   4 attention-internal points (RoPE in/out, FA in/out). Each adapter is
   a potential bug source. The hand-written reference makes per-kernel
   cosine comparison a direct `np.dot` with zero adapter code.
2. **Debugging**: with a numpy reference we can `print(intermediate)` at
   any computation step to compare visually with NPU. With HF, capturing
   intermediates inside `LlamaAttention.forward()` requires monkey-patching
   the inline functions (`apply_rotary_pos_emb`, `F.scaled_dot_product_attention`).
3. **HF version decoupling**: HF transformers refactors their attention
   path occasionally (eager → sdpa → flash_attention_2 dispatch added in
   4.36, etc.). A monkey-patch oracle breaks across version bumps; the
   hand-written reference is stable.

The cost we pay: ~300 lines of model-specific reference code per new
deployment, plus the risk of writing a buggy reference. Phase 0's
verify-vs-HF check (criteria 3-5 below) is what makes this safe — HF
is the audit, not the runtime.

## Phase 0 PASS criteria (HARD GATES)

All four are required, because each catches a different bug class:

1. **Loadable** (catches weight-name / shape mismatches):
   `<model>_weights.py` loads every expected tensor; every layer index
   0..n_layers-1 has q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
   down_proj, attn_norm, ffn_norm.
2. **Stable** (catches numerical blow-ups): `<model>_reference.py` runs
   end-to-end on the canonical prompt without NaN, and produces a
   non-degenerate logit distribution.
3. **Per-layer numerical match vs HF**: every transformer layer's output
   `hidden_state` has cosine ≥ 0.99 vs HF's `output_hidden_states[i+1]`
   at the same input. Catches per-block bugs (wrong RoPE, norm, bias).
4. **Final logits + top-1 strict match vs HF**: cosine(our_logits[-1],
   hf.logits[0, -1]) ≥ 0.999 AND argmax matches bit-exactly. Catches
   LM-head bugs (tied vs untied weight, missing bias) that per-layer
   checks pass through.

If any fails, Phase 1 is meaningless (NPU would be verified against a
broken reference). Fix the reference, re-run, then advance.

## Knowledge base references

Read these BEFORE acting:

- `programming_examples/llama3/llama3_weights.py` — reference Config
  dataclass + HF weight loading pattern to copy
- `programming_examples/llama3/llama3_reference.py` — reference
  decomposed CPU FP32 inference to copy. Each function in here matches
  one NPU kernel signature; preserve that structure in your model's
  reference.
- `programming_examples/_llm_shared/docs/explain.md` — RoPE half-split
  convention (the most common source of per-layer cosine drops vs HF)

## Workflow

### Step 1: Read the HF config

Fetch `config.json` for the target model from HuggingFace. Extract:

- `num_hidden_layers` → `n_layers`
- `hidden_size` → `emb_dim`
- `num_attention_heads` → `n_heads`
- `num_key_value_heads` → `n_kv_heads` (default to `n_heads` if absent → MHA)
- `intermediate_size` → `hidden_dim`
- `vocab_size` → `vocab_size`
- `rope_theta` → `rope_base` (default 10000.0 if absent)
- `head_dim` (compute as `emb_dim // n_heads` if absent)
- `tie_word_embeddings` (affects whether to load `lm_head.weight`)

### Step 2: Architecture compatibility check

Confirm the model is in-scope before scaffolding anything downstream:

- Architecture must be in `["LlamaForCausalLM", "MistralForCausalLM"
  (only if no sliding window), "Qwen2ForCausalLM", "Qwen3ForCausalLM"]`
  — i.e., a decoder-only with RMSNorm + SwiGLU + RoPE + GQA/MHA
- Reject if: MoE layers, sliding-window attention, MLA, encoder-decoder
- Reject explicitly with a clear message; don't scaffold a model the
  rest of the pipeline can't handle

### Step 3: Generate `<model>_weights.py`

Copy `programming_examples/llama3/llama3_weights.py` to
`<model>/<model>_weights.py`. Modify:

- `LlamaConfig` dataclass defaults → match Step 1 values
- HF weight name remapping in `load_weights()` — most LLAMA-derived
  models share the same names (`model.layers.<i>.self_attn.q_proj.weight`
  etc.); confirm via inspecting the safetensors index. If different,
  write an explicit mapping.
- `generate_rope_lut()` — verify `rope_base` is parameterized and uses
  the new value
- LM head: load `lm_head.weight` only if `tie_word_embeddings` is False
- **If `qkv_bias=true` (Qwen2 / Qwen3 family)**: add `bq, bk, bv`
  fields to `LayerWeights`, parallel to wq/wk/wv. The bias is
  loaded just like the projection weights but with a different HF key
  (`q_proj.bias` etc.).

### Step 4: Generate `<model>_reference.py`

Copy `programming_examples/llama3/llama3_reference.py` to
`<model>/<model>_reference.py`. The per-kernel functions in this file
must match NPU kernel signatures from
`_llm_shared/docs/kernel_registry/supported_kernels.md` — that signature
match is what makes the reference usable as Phase 1's per-kernel oracle.

Modify:

- Imports: change `from llama3_weights import` to `from <model>_weights import`
- Config references: ensure all hardcoded shapes are replaced by config attributes
- For unfamiliar architectures (different attention masking, post-norm
  vs pre-norm, etc.) — adapt carefully. Cross-check against HF's
  `modeling_<arch>.py` source for the exact computation order.
- **If `qkv_bias=true`**: add the bias term to Q/K/V projections
  **before RoPE** (matches HF reference impl):
  `q = q @ wq + bq; q = rope(q, lut_q)`. Working pattern in
  `programming_examples/qwen25_1_5b/qwen25_bias.py`. Surface in TODO.md
  as a Phase 2 prerequisite — Phase 2 will need the same bias-on-host
  wrapper around the bias-free `rms_gemms_rope` ELF.

### Step 5: Numerical verify vs HuggingFace transformers (HARD GATE)

Add a `--verify-vs-hf` mode to `<model>_reference.py`. This mode:

1. Loads HF model via `transformers.AutoModelForCausalLM.from_pretrained(<hf_id>)`
2. Runs HF forward on the canonical prompt with `output_hidden_states=True`
   to capture per-layer hidden states
3. Runs our reference on the same canonical prompt
4. Compares against the **PASS criteria above** (per-layer cosine,
   final logits cosine, top-1 match)
5. Prints a per-layer + final summary table; exits 0 only if all pass

**Canonical prompt**: use `"The capital of France is"` (matches Phase 3
canonical prompt set; deterministic, top-1 is decisive so close-2
ambiguity won't muddy the gate).

```bash
cd programming_examples/<model>
python3 <model>_reference.py --verify-vs-hf --prompt "The capital of France is"
```

Expected output:

```
Per-layer cosine vs HF:
  Layer 0:  0.99987  ✓
  Layer 1:  0.99975  ✓
  ...
Final logits cosine: 0.99996  ✓ (threshold 0.999)
Top-1 token: " Paris" (id=12366) — match HF ✓
PASS
```

If any check fails, see "Failure modes" below.

(Optional sanity check: extend with `--n-tokens N` to compare multi-token
generation against `model.generate(..., do_sample=False)`. Almost always
passes once single-shot does, so not gating.)

## Failure modes

When `--verify-vs-hf` fails, the symptom narrows down where the bug is:

| Symptom | Likely cause | Where to look |
|---|---|---|
| Layer 0 cosine high (>0.99) but Layer 1 < 0.99 | Per-block computation diverges; usually RoPE convention or per-block weight wiring | Confirm `<model>_reference.py` uses half-split RoPE (matching `rope_halfsplit.cc`); the math must match HF even though the layout differs |
| All layers around the same low cosine (~0.5-0.9) | Norm order wrong (post-norm vs pre-norm) OR weight transpose missing | Cross-check `modeling_<arch>.py` for the exact pre/post-norm flow; verify `np.ascontiguousarray()` after any HF tensor load |
| Cosine drops sharply at one specific layer | KV-cache layout bug at that depth, or attention mask bug | Inspect intermediate shapes; HF uses `[batch, n_heads, seq, head_dim]` whereas our NPU layout is seq-first flat |
| Final logits cosine < 0.999 but layers all OK | LM head weight tied vs untied, or LM head bias missing | Check `tie_word_embeddings` flag (Step 1); load `lm_head.weight` only if untied |
| Top-1 mismatch with all cosines high | Rare in Phase 0 — usually means weight loading is shifted by a layer or has a transpose bug that cosine smooths over | Print weight shapes after load; compare with HF's `state_dict()` |
| All NaN | Weight load shape mismatch OR `head_dim` config wrong | Re-check Step 1 config extraction; print weight shapes after load |
| HF model won't load | Missing `transformers` install, or HF auth needed for gated model | `pip install transformers`; for gated models, `huggingface-cli login` |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 0 PASS, append a brief Phase 0 entry to
`<model>/docs/development_progress/progress.md` recording: HF model id,
resolved config (n_layers / emb_dim / n_heads / n_kv_heads / hidden_dim
/ vocab / rope_base), and verify-vs-HF results (per-layer cosine
min/max, final logits cosine, top-1 token).

Mark Phase 0 in `<model>/TODO.md`.

The verified `<model>_reference.py` is now the immutable ground truth
for Phase 1's kernel-validation, Phase 2's single-block validation, and
Phase 3's full-model validation. Modifying it later silently
invalidates downstream cosine gates — re-run `--verify-vs-hf` after any
edit.
