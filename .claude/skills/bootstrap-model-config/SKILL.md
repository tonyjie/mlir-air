---
name: bootstrap-model-config
description: Phase 0 of LLM deployment — adapt config dataclass and HuggingFace weight loader for the target model. Produce <model>_weights.py and <model>_reference.py. Invoked by deploy-new-llm after scaffolding.
---

## Purpose
Translate a HuggingFace model into the data structures the rest of the pipeline expects: a `Config` dataclass with NPU-relevant fields, a weight loader that produces correctly-shaped numpy arrays, and a CPU F32 reference implementation for downstream correctness gates.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/llama3/llama3_weights.py` — reference Config dataclass + HF weight loading
- `programming_examples/llama3/llama3_reference.py` — reference CPU F32 inference
- `programming_examples/_llm_shared/docs/explain.md` — RoPE half-split convention details

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

### Step 2: Architecture compatibility check
Before writing any file, confirm the model is in-scope (per spec §2):
- Architecture must be in `["LlamaForCausalLM", "MistralForCausalLM"
  (only if no sliding window), "Qwen2ForCausalLM", ...]` — i.e., a
  decoder-only with RMSNorm + SwiGLU + RoPE + GQA/MHA
- Reject if: MoE layers present, sliding-window attention, MLA,
  encoder-decoder structure
- Reject explicitly with a clear message; do NOT scaffold a model the
  rest of the pipeline can't handle

**QKV bias is SUPPORTED** as of qwen25_1_5b deployment (LESSON 1, 2026-04-19):
the bias gets added on the HOST after the bias-free `rms_gemms_rope` /
`rms_gemv_rope` ELFs return, exploiting RoPE's linearity:
`RoPE(q + bq) = RoPE(q) + RoPE(bq)`. Reference implementation:
`programming_examples/qwen25_1_5b/qwen25_bias.py`. When `qkv_bias=true`:
- Load q_proj.bias / k_proj.bias / v_proj.bias in `<model>_weights.py`
  (add `bq, bk, bv` fields to LayerWeights, parallel to wq/wk/wv)
- Add the bias term to Q/K/V projections in `<model>_reference.py`
  (BEFORE RoPE, matches HF reference impl)
- Surface in TODO.md: "Phase 2 prerequisite — wire QKV bias via
  qwen25_bias-style wrapper (~1-2h)"

### Step 3: Generate `<model>_weights.py`
Copy `programming_examples/llama3/llama3_weights.py` to `<model>/<model>_weights.py`. Modify:
- `LlamaConfig` dataclass defaults → match Step 1 values
- HF weight name remapping in `load_weights()` — most LLAMA-derived models share the same names (`model.layers.<i>.self_attn.q_proj.weight` etc.); confirm via inspecting the safetensors index. If different, write an explicit mapping.
- `generate_rope_lut()` — verify `rope_base` is parameterized and uses the new value

### Step 4: Generate `<model>_reference.py`
Copy `programming_examples/llama3/llama3_reference.py` to `<model>/<model>_reference.py`. Modify:
- Imports: change `from llama3_weights import` to `from <model>_weights import`
- Config references: ensure all hardcoded shapes are replaced by config attributes
- For unfamiliar architectures (different attention masking, etc.) — adapt carefully

### Step 5: Smoke-test the reference
Run the CPU reference on a canonical prompt. The reference script (mirrored from `llama3_reference.py`) predicts the next token via single-shot forward pass:

```bash
cd programming_examples/<model>
python3 <model>_reference.py --prompt "The capital of France is"
```

Expected: prints the next token logits and top-k predictions. The top-1 token should be lexically sensible (e.g., " Paris" for the example prompt).

If the new model's reference needs multi-token generation for sanity-checking, extend the script to support `--n-tokens N` (the LLAMA reference is single-shot only).

## Verification (Phase 0 gate)

Phase 0 PASSES when ALL true:
1. `<model>_weights.py` loads all expected tensors with shapes matching the config (programmatic check: every layer index 0..n_layers-1 has q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, attn_norm, ffn_norm)
2. `<model>_reference.py` runs and produces stable output (no NaN, sensible logit distribution: top-5 logits within reasonable range, not all uniform)
3. Reference output for canonical prompt is lexically sensible (human eyeball check OR pre-computed expected from HuggingFace transformers library)

## Failure modes
- Architecture rejected at Step 2 → escalate, deployment cannot proceed
- Weight name mismatch at Step 3 → human resolves (variant naming conventions)
- Reference produces NaN → likely a config error (wrong head_dim, wrong rope_theta); revisit Step 1
- Reference produces gibberish → likely a weight-load shape mismatch; check `np.ascontiguousarray()` and tensor dtypes

## Update protocol

On Phase 0 PASS, append to `<model>/docs/development_progress/progress.md`:
```
## Phase 0: Bootstrap (PASSED YYYY-MM-DD)
- HF model: <id>
- Config: n_layers=N, emb_dim=D, n_heads=H, n_kv_heads=K, hidden_dim=F, vocab=V, rope_base=R
- Reference smoke output: "<first 5 tokens>"
```

Update `<model>/TODO.md`: mark Phase 0 checkbox; populate "Resolved config" section.
