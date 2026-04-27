# SmolLM2-135M deployment — phase progress log

Each phase appends its summary entry here when its HARD gate passes.

---

## Phase 0 — Build CPU Oracle  (PASS, 2026-04-27)

**HF model**: `HuggingFaceTB/SmolLM2-135M`

**Resolved config**:
n_layers=30, emb_dim=576, n_heads=9 (odd, non-power-of-2),
n_kv_heads=3 (GQA group_size=3), head_dim=64, hidden_dim=1536,
vocab=49152, rope_base=100000, tied embeddings, no QKV bias, no Q/K Norm.

**Files produced**:
- `smollm2_135m_weights.py` — config dataclass + HF safetensors loader + RoPE LUT
- `smollm2_135m_reference.py` — CPU F32 decomposed reference + `--verify` mode

**Sibling templates copied**: `smollm2_1_7b/{smollm2_weights.py,smollm2_reference.py}`
adapted for 135M shape config. `attention_reference` already supports GQA via
`group_size = n_heads // n_kv_heads` — no algorithmic change for g=3.

**HARD gate `--verify` (vs HF transformers FP32, prompt = "The capital of France is", seq_len=128)**:

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| Loadable | every layer 0..29 has q/k/v/o + gate/up/down + norms | ✓ | PASS |
| Stable (no NaN) | finite logits | ✓ | PASS |
| Per-layer hidden cos vs HF | ≥ 0.99 for all 30 (29 pre-norm + 1 post-norm) | min=0.999996, max=1.000000, 0 failed | PASS |
| Final logits cos at pred_pos | ≥ 0.999 | **0.99999976** | PASS |
| Top-1 strict match vs HF | bit-exact argmax | YES (' the' id=260, both ours+HF) | PASS |

**Notable observations**:

1. The per-layer comparison initially showed a spurious "FAIL" at the last
   transformer layer — root cause was an off-by-one alignment in how HF's
   `output.hidden_states` exposes its tuple. HF stores `(n_layers + 1)` tensors:
   `[embed, out_L0, ..., out_L(n-2), POST-FINAL-NORM]` — the last entry is
   post-norm, NOT pre-norm output of the last layer. Fixed by comparing
   our_hidden[0..n-2] vs hf.hidden_states[1..n_layers-1] (pre-norm) AND
   our_post_norm vs hf.hidden_states[n_layers] (post-norm). Worth lifting to
   `_llm_shared/` if other deployments add per-layer verify.
2. SmolLM2-135M is too small to predict ' Paris' for the canonical prompt —
   both our reference and HF top-1 to ' the'. This is model capability, not
   a numerical bug. Phase 3 will need to re-evaluate the "decisive top-1"
   gate against a smaller-model-friendly prompt set.
3. n_h=9 (odd, non-power-of-2) loaded fine in CPU reference (which doesn't
   care about tile alignment); whether this trips any NPU tile config will
   surface in Phase 1.

**Performance** (CPU F32 reference, single-shot, 128 tokens, single core):
~few seconds for forward pass + ~10 s HF model load. Acceptable for an
oracle.

**Next**: Phase 1 — Kernel Validation (`kernel-validation` skill).
