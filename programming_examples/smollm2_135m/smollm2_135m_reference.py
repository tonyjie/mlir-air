# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU reference implementation of SmolLM2-135M forward pass.

Pure NumPy in F32 for numerical verification against NPU results.
All intermediate computations are done in F32 (weights are cast from BF16
at use time) to provide a high-accuracy reference.

SmolLM2-135M config:
  30 layers, emb_dim=576, n_heads=9, head_dim=64,
  n_kv_heads=3 (GQA group_size=3), hidden_dim=1536,
  vocab_size=49152, BF16, rope_base=100000,
  tied embeddings (lm_head shares embed_tokens.weight).

The kernel-side `attention_reference` supports GQA via
group_size = n_heads // n_kv_heads (= 3 for SmolLM2-135M).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from smollm2_135m_weights import (
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    load_weights,
    generate_rope_lut,
)


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def apply_rope(x, lut):
    """Apply Rotary Position Embedding using a precomputed LUT.

    Uses half-split convention (matching HuggingFace Llama):
    pairs (x[i], x[i + dim//2]) with rotation angle theta_i.

    LUT layout: [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
    """
    x = np.asarray(x, dtype=np.float32)
    lut = np.asarray(lut, dtype=np.float32)
    dim = x.shape[-1]
    half = dim // 2

    cos_vals = lut[:, :half]
    sin_vals = lut[:, half:]

    x1 = x[:, :half]
    x2 = x[:, half:]

    out = np.empty_like(x)
    out[:, :half] = x1 * cos_vals - x2 * sin_vals
    out[:, half:] = x1 * sin_vals + x2 * cos_vals
    return out


def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    x = np.asarray(x, dtype=np.float32)
    return x * (1.0 / (1.0 + np.exp(-x)))


def swiglu(gate, up):
    """SwiGLU gating: SiLU(gate) * up."""
    return silu(gate) * np.asarray(up, dtype=np.float32)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Multi-head attention with Grouped Query Attention (GQA).

    For SmolLM2-135M: n_heads=9, n_kv_heads=3, group_size = 3.
    Each KV head is shared by 3 query heads:
      Q heads 0,1,2 -> KV head 0
      Q heads 3,4,5 -> KV head 1
      Q heads 6,7,8 -> KV head 2

    Args:
        q: (seq_len, n_heads * head_dim) -- already projected and RoPE'd.
        k: (seq_len, n_kv_heads * head_dim) -- already projected and RoPE'd.
        v: (seq_len, n_kv_heads * head_dim) -- already projected.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.

    Returns:
        (seq_len, n_heads * head_dim) attention output.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads  # 3 for SmolLM2-135M

    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)

    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale
        scores = scores + causal_mask
        probs = softmax(scores, axis=-1)
        out_heads[h] = probs @ v[kv_idx]

    out = out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
    return out


def transformer_block(x, layer_weights, rope_lut, config):
    """Single transformer block with attention and FFN."""
    x = np.asarray(x, dtype=np.float32)
    intermediates = {}
    seq_len = x.shape[0]
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    # --- Self-attention ---
    normed = rms_norm(x, layer_weights.attn_norm)
    intermediates["attn_norm"] = normed

    wq = np.asarray(layer_weights.wq, dtype=np.float32)
    wk = np.asarray(layer_weights.wk, dtype=np.float32)
    wv = np.asarray(layer_weights.wv, dtype=np.float32)
    q = normed @ wq  # (seq_len, n_heads * head_dim) = (seq_len, 576)
    k = normed @ wk  # (seq_len, n_kv_heads * head_dim) = (seq_len, 192)
    v = normed @ wv  # (seq_len, n_kv_heads * head_dim) = (seq_len, 192)
    intermediates["q"] = q
    intermediates["k"] = k
    intermediates["v"] = v

    q_heads = q.reshape(seq_len, n_heads, head_dim)
    q_roped_heads = np.empty_like(q_heads)
    for h in range(n_heads):
        q_roped_heads[:, h, :] = apply_rope(
            q_heads[:, h, :].reshape(seq_len, head_dim), rope_lut[:seq_len]
        )
    q_roped = q_roped_heads.reshape(seq_len, n_heads * head_dim)
    intermediates["q_roped"] = q_roped

    k_heads = k.reshape(seq_len, n_kv_heads, head_dim)
    k_roped_heads = np.empty_like(k_heads)
    for h in range(n_kv_heads):
        k_roped_heads[:, h, :] = apply_rope(
            k_heads[:, h, :].reshape(seq_len, head_dim), rope_lut[:seq_len]
        )
    k_roped = k_roped_heads.reshape(seq_len, n_kv_heads * head_dim)
    intermediates["k_roped"] = k_roped

    attn_out = attention_reference(q_roped, k_roped, v, n_heads, n_kv_heads)
    intermediates["attn_out"] = attn_out

    wo = np.asarray(layer_weights.wo, dtype=np.float32)
    proj = attn_out @ wo
    intermediates["proj"] = proj

    res1 = x + proj
    intermediates["res1"] = res1

    # --- Feed-forward network ---
    normed2 = rms_norm(res1, layer_weights.ffn_norm)
    intermediates["ffn_norm"] = normed2

    w_gate = np.asarray(layer_weights.w_gate, dtype=np.float32)
    w_up = np.asarray(layer_weights.w_up, dtype=np.float32)
    gate = normed2 @ w_gate
    up = normed2 @ w_up
    intermediates["gate"] = gate
    intermediates["up"] = up

    swiglu_out = swiglu(gate, up)
    intermediates["swiglu"] = swiglu_out

    w_down = np.asarray(layer_weights.w_down, dtype=np.float32)
    down = swiglu_out @ w_down
    intermediates["down"] = down

    output = res1 + down
    intermediates["output"] = output

    return output, intermediates


def forward(token_ids, weights, config, rope_lut=None, return_hidden_states=False):
    """Full SmolLM2-135M forward pass.

    Args:
        token_ids: 1D array of token ids.
        weights, config: see load_weights / LlamaConfig.
        rope_lut: optional precomputed RoPE LUT.
        return_hidden_states: if True, also return list of per-layer outputs
            (the residual stream after each transformer block, in F32),
            matching HF transformers' `output_hidden_states[1:]`.

    Returns:
        logits: (seq_len, vocab_size) F32 array.
        hidden_states: list of (seq_len, emb_dim) per-layer outputs (only when
            return_hidden_states=True).
    """
    seq_len = len(token_ids)

    if rope_lut is None:
        rope_lut = generate_rope_lut(config=config, seq_len=seq_len)
    rope_lut = np.asarray(rope_lut, dtype=np.float32)

    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table[token_ids]

    hidden_states = []
    for i in range(config.n_layers):
        x, _ = transformer_block(x, weights.layers[i], rope_lut, config)
        if return_hidden_states:
            hidden_states.append(x.copy())

    x = rms_norm(x, weights.final_norm)

    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x @ lm_head.T

    if return_hidden_states:
        return logits, hidden_states
    return logits


def _cosine(a, b):
    """Cosine similarity between two flat vectors (F32)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU reference forward pass for SmolLM2-135M"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model name or local path (default: HuggingFaceTB/SmolLM2-135M)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt (default: 'The capital of France is')",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length to pad/truncate to (default: 128)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare output against HuggingFace transformers reference (per-layer + final)",
    )
    args = parser.parse_args()

    config = LlamaConfig()
    print(f"Loading weights from {args.model}...")
    weights = load_weights(args.model, config=config)
    print(f"  Config: {config}")
    print(
        f"  Layers: {config.n_layers}, emb_dim: {config.emb_dim}, "
        f"n_heads: {config.n_heads}, n_kv_heads: {config.n_kv_heads}, "
        f"hidden_dim: {config.hidden_dim}, vocab_size: {config.vocab_size}"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")

    if len(token_ids) > args.seq_len:
        token_ids = token_ids[: args.seq_len]
        print(f"Truncated to {args.seq_len} tokens")
    elif len(token_ids) < args.seq_len:
        pad_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        original_len = len(token_ids)
        token_ids = token_ids + [pad_token] * (args.seq_len - len(token_ids))
        print(
            f"Padded from {original_len} to {args.seq_len} tokens "
            f"(pad_token={pad_token})"
        )

    token_ids = np.array(token_ids, dtype=np.int64)

    print(f"\nRunning forward pass (seq_len={args.seq_len})...")
    logits, our_hidden = forward(token_ids, weights, config, return_hidden_states=True)
    print(f"Output logits shape: {logits.shape}")

    prompt_len = len(tokenizer.encode(args.prompt))
    pred_pos = min(prompt_len - 1, args.seq_len - 1)

    next_token_logits = logits[pred_pos]
    top5_indices = np.argsort(next_token_logits)[-5:][::-1]
    top5_probs = softmax(next_token_logits)

    print(f"\nTop-5 predicted next tokens (position {pred_pos}):")
    for rank, idx in enumerate(top5_indices):
        token_str = tokenizer.decode([idx])
        prob = top5_probs[idx]
        print(
            f"  {rank + 1}. '{token_str}' (id={idx}, logit={next_token_logits[idx]:.4f}, "
            f"prob={prob:.4f})"
        )

    if args.verify:
        print("\n--- Verification against HuggingFace transformers ---")
        try:
            import torch
            from transformers import AutoModelForCausalLM

            print("Loading HuggingFace model (FP32)...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32
            )
            hf_model.eval()

            with torch.no_grad():
                input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                hf_output = hf_model(input_ids, output_hidden_states=True)
                hf_logits = hf_output.logits[0].numpy()
                # HF's output_hidden_states for LlamaModel is (n_layers + 1) tensors:
                #   [0]               = input embeddings (= input to L0)
                #   [i] for i in 1..n = `hidden_states` captured at the START of
                #                       loop iter i, i.e. = output of L(i-1)
                #   [n_layers]        = post-FINAL-NORM (NOT pre-norm output of L(n-1))
                # So the HF tuple is [embed, out_L0, out_L1, ..., out_L(n-2), post_norm].
                # We can compare:
                #   our_hidden[i] (= output L_i, pre-norm)
                #     vs hf_output.hidden_states[i+1]  for i in 0..n_layers-2
                #   our post-final-norm
                #     vs hf_output.hidden_states[n_layers]
                hf_per_layer_pre_norm = [
                    hs[0].numpy() for hs in hf_output.hidden_states[1 : config.n_layers]
                ]  # n_layers-1 entries: outputs of L0..L(n-2)
                hf_post_norm = hf_output.hidden_states[config.n_layers][0].numpy()

            print(f"HF logits shape: {hf_logits.shape}")
            print(f"Our logits shape: {logits.shape}")

            # --- Per-layer hidden state cosine (L0..L(n-2) pre-norm + post-final-norm) ---
            print(
                f"\nPer-layer hidden-state cosine vs HF "
                f"(threshold ≥ 0.99 for HARD GATE):"
            )
            per_layer_min = 1.0
            per_layer_max = 0.0
            per_layer_fail = 0
            for li in range(config.n_layers - 1):
                cos = _cosine(our_hidden[li], hf_per_layer_pre_norm[li])
                marker = "OK" if cos >= 0.99 else "FAIL"
                if cos < 0.99:
                    per_layer_fail += 1
                per_layer_min = min(per_layer_min, cos)
                per_layer_max = max(per_layer_max, cos)
                print(f"  Layer {li:2d}: {cos:.6f}  {marker}")

            # Last comparison: post-final-norm (HF doesn't expose pre-norm output of L(n-1))
            our_post_norm = rms_norm(our_hidden[-1], weights.final_norm)
            cos_post = _cosine(our_post_norm, hf_post_norm)
            marker = "OK" if cos_post >= 0.99 else "FAIL"
            if cos_post < 0.99:
                per_layer_fail += 1
            per_layer_min = min(per_layer_min, cos_post)
            per_layer_max = max(per_layer_max, cos_post)
            print(f"  post-final-norm: {cos_post:.6f}  {marker}")
            print(
                f"  Summary: min={per_layer_min:.6f}, max={per_layer_max:.6f}, "
                f"failed={per_layer_fail}/{config.n_layers}"
            )

            # --- Final logits (at pred_pos) ---
            our_next = logits[pred_pos]
            hf_next = hf_logits[pred_pos]

            abs_diff = np.abs(our_next - hf_next)
            max_abs_err = np.max(abs_diff)
            mean_abs_err = np.mean(abs_diff)

            denom = np.maximum(np.abs(hf_next), 1e-8)
            rel_diff = abs_diff / denom
            max_rel_err = np.max(rel_diff)
            mean_rel_err = np.mean(rel_diff)

            final_cos = _cosine(our_next, hf_next)

            print(f"\nError at position {pred_pos}:")
            print(f"  Max  absolute error: {max_abs_err:.6f}")
            print(f"  Mean absolute error: {mean_abs_err:.6f}")
            print(f"  Max  relative error: {max_rel_err:.6f}")
            print(f"  Mean relative error: {mean_rel_err:.6f}")
            print(f"  Final logits cosine: {final_cos:.8f}  (threshold ≥ 0.999)")

            our_top1 = int(np.argmax(our_next))
            hf_top1 = int(np.argmax(hf_next))
            match = our_top1 == hf_top1
            print(f"\nTop-1 prediction match: {'YES' if match else 'NO'}")
            print(f"  Ours: '{tokenizer.decode([our_top1])}' (id={our_top1})")
            print(f"  HF:   '{tokenizer.decode([hf_top1])}' (id={hf_top1})")

            # --- HARD GATE summary ---
            gate_perlayer = per_layer_fail == 0
            gate_final = final_cos >= 0.999
            gate_top1 = match
            all_pass = gate_perlayer and gate_final and gate_top1
            print(
                f"\nHARD GATE: per-layer cos≥0.99: {gate_perlayer}  "
                f"final cos≥0.999: {gate_final}  top-1 match: {gate_top1}  "
                f"=> {'PASS' if all_pass else 'FAIL'}"
            )

            if not all_pass:
                sys.exit(1)

        except ImportError as e:
            print(f"Cannot verify: {e}")
            print("Install torch and transformers: pip install torch transformers")
            sys.exit(2)
