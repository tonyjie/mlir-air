# Qwen2.5-0.5B deployment — lessons learned

## 2026-04-27 — qwen25_pad.py / qwen25_bias.py are fully model-agnostic

**Symptom**: Initially worried the helpers in qwen25_1_5b/ would have
hardcoded 1.5B shape numbers.

**Root cause / observation**: They take Config dataclasses and operate
generically on (orig_config, padded_config). All 1.5B-specific numbers
(emb 2048/9216) come from how the test scripts CALL them. Reused
unchanged for 0.5B with padded_emb_dim=1024, padded_hidden_dim=5120.

**Generalization**: When a 3rd Qwen2 deployment lands (e.g. Qwen2.5-3B,
Qwen2.5-7B), promote `qwen25_pad.py` and `qwen25_bias.py` to
`_llm_shared/phase_helpers/` — currently they live in qwen25_1_5b/ and
are imported via sys.path, which works but obscures the model-agnostic
intent.


## 2026-04-27 — NPU seq-first FA precision profile differs by GQA group_size

**Symptom (Phase 2)**: Single-block cosine drops to 0.978 (whole) /
0.94 (per_pos) when using NPU FA, vs 0.999/0.999 when using CPU FA.
Standalone Phase 1 FA at the same shape gave 0.997 — so it's an
in-block real-distribution effect, not a kernel bug.

**Root cause (likely)**: GQA group=8 + n_kv=2 means only 2 KV heads
serve 16 padded Q heads. The cascade-merge in NPU FA has fewer
independent reductions to spread BF16 noise across, so realistic
attention scores (outlier-heavy after RMSNorm + projection + bias +
RoPE) accumulate more noise than in MHA-shaped models like
smollm2_1_7b (which hits cos 0.998 on the same kernel).

**Impact**: Phase 3 top-1 STILL matches on all 6 canonical prompts.
Phase 6 multi-token greedy match diverges at tok 2 on competitive
continuations only — both NPU and CPU produce VALID English
continuations, just different ones.

**Generalization**: Watch for this pattern at any GQA-imbalanced shape
(group_size ≥ 8 + small n_kv). If a 2nd deployment shows the same
profile, surface as `skill-update.md` item B2 with concrete mitigations
(e.g., FA accumulator F32 promotion, K/V cache F32 storage).

**Workaround for this deployment**: Accept the precision drift as a
documented W1 informational warning. Final top-1 matches; deployment
is functionally correct.


## 2026-04-27 — K=4864 doesn't need down_k_split but DOES need mv_k4864.o

**Symptom**: Down GEMV at M=896, K=4864 with default tile_m=8 hits
`L2 capacity exceeded: 4864×8×8×2 = 622592 > 524288`.

**Root cause**: K=4864 < 8160 → Rule B (auto-split repeat ≤ 255) not
engaged → no `down_k_split` needed. But Rule D (L2 cap) IS engaged at
default tile_m=8 → must drop tile_m to 2.

**Fix**: Compile a renamed `mv_k4864.o` with `-DDIM_M_OUTPUT=2 +
-Dmatvec_*=dg_matvec_*` (mirrors qwen25_1_5b's `mv_k8960.o` pattern but
without the `down_k_split` knob). The shared `o_gemv_ffn_multi` builder
defaults work fine for our K=4864 because Rule C combined-channel reads
(28 + 152) ≈ 180 < 255.

**Generalization**: Rule D triggers at any K > ~32760 / (tile_m ×
herd_m × bytes). For our default 8×8×2: K=4096 is the boundary. So:
- K ≤ 4096: default tile_m=8 fits Rule D, no rename needed
- 4096 < K ≤ 8160: needs `mv_k<N>.o` with DIM_M_OUTPUT=2 (no down_k_split)
- K > 8160: needs both DIM_M_OUTPUT and down_k_split (qwen25_1_5b pattern)


## Pattern table — when to add helpers

| Phase that surfaces it | New file in `<model>/` | Promote to `_llm_shared/` after |
|---|---|---|
| Phase 0 (weights / reference) | `<model>_weights.py`, `<model>_reference.py` | Always model-specific; no promotion |
| Phase 2 (host wrapper) | (e.g., `<model>_bias.py`, `<model>_pad.py`) | 2+ deployments use the same wrapper |
| Phase 5 (decode setup) | `<model>_decode_setup.py` (LM head partition + mv rename) | LM head partition scheme generalizes (10×16384 for vocab=151936) |
| Phase 6 (inference) | `<model>_inference.py` | Always model-specific |
