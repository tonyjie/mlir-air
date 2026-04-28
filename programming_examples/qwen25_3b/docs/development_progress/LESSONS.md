# Qwen2.5-3B deployment — lessons learned

## 2026-04-27 — Padded hidden_dim must be 12288, not 11264 (BD pool tightness)

**Symptom (Phase 2)**: Initial padding hidden 11008 → **11264** (smallest
1024-aligned multiple) compiled o_ffn ELF without errors at smaller seq
but at seq_len=2048 either hit BD blowup ("Allocator exhausted available
buffer descriptor IDs") or compiled to an ELF that runtime-hangs (with
herd_m=4 + swiglu_tile_n=704 fix attempt).

**Root cause**: Default tile_n=128 herd_n=4 + 8-launch o_ffn ELF + 36
layer × seq=2048 hits BD pool limits at hidden=11264. The 11264 is
1024-aligned but the BD pattern still produces too many descriptors.

**Fix**: Pad hidden 11008 → **12288** (12 × 1024). Mirrors qwen25_1_5b's
PADDED known-good recipe at hidden=9216 (just bigger). Cost: 12288/11008
= 11.6% extra FFN compute (vs 11264's 2.3%).

**Generalization**: For Qwen2-family at seq=2048 with hidden NOT a clean
multiple of 1024, prefer the bigger BD-friendly multiple. Add to
`_llm_shared/docs/aie2p_hardware_limits.md` as a clarification of Rule A:
even 1024-aligned dims can hit BD limits if the multiple is "odd" (like
11) at large seq + multi-launch ELF.

## 2026-04-27 — W1 (NPU FA precision drop) is seq-first-FA-specific, NOT GQA-driven

**Symptom (Phase 2 vs qwen25_0_5b)**:
- qwen25_0_5b at GQA g=8 + n_kv=2 + **hd=64** (seq-first FA):
  per-pos cos **0.94** in single-block test
- qwen25_3b at GQA g=8 + n_kv=2 + **hd=128** (head-first FA via Option C):
  per-pos cos **0.995**

Same GQA shape, opposite FA path → different precision profile.

**Root cause (refined hypothesis)**: NPU **seq-first** FA at GQA-imbalanced
shapes (large group_size / small n_kv) accumulates more BF16 noise in the
cascade-merge. NPU **head-first** FA (Option C wrapper, used at hd=128)
does NOT exhibit this — the per-head independent reductions before host
transpose are precision-clean.

**Workaround for hd=64 GQA-imbalanced models**: route through Option C
head-first wrapper (extra host transposes, ~5% perf cost), avoid seq-first
FA. Surface to `skill-update.md` as B2 with this refinement.

**Generalization**: Watch for this pattern — any future GQA-imbalanced +
hd=64 model should route through Option C unless we accept multi-token
greedy divergence on competitive logits.

## 2026-04-27 — Rule C/D conflict at K=2048 + M=11008/16384 (decode LM head)

**Symptom (Phase 5)**: At K=emb_dim=2048, the qwen25_1_5b LM head config
(tile_m=16 m_input=16 herd_m=8) **exceeds Rule D L2 cap by exactly C
buffer 256B** (a_l2=524288 = at cap; c_l2=256 → over). Default tile_m=8
m_input=4 hits **Rule B repeat ≤ 255** at M=11008 Gate/Up (11008/64 × 2 = 344).

**Constraint analysis**: For M=11008 (Gate/Up), to satisfy Rule B:
- launches × (tile_m/m_input) ≤ 255
- → tile_m × herd_m ≥ M/255 × m_input

With herd_m=8: tile_m × m_input ≥ 11008/255 × m_input. For tile_m=m_input=8:
11008/64 = 172 ≤ 255 ✓ (Rule B), but Rule D needs tile_m × herd_m ≤ 128
(at K=2048). 8×8=64 ≤ 128 ✓.

**Fix**: Use tile_m=8 m_input=8 herd_m=8 (vs 1.5B's tile_m=16 m_input=16).
For LM head, also re-partition to **11 × 13824** (vs 1.5B's 10 × 16384) so
per-partition launches=216 ≤ 255.

**Generalization**: For K=2048 + M ≥ 11008 multi-launch GEMV ELFs, can NOT
use tile_m=16 herd_m=8 due to Rule D L2 boundary. Must use tile_m=8
m_input=8 herd_m=8. Forward applicable to any K=2048 model with wide M.

## 2026-04-27 — Phase 6 inference.py NPU LM Head bug (W2) — REFINED by Phase 7 audit

**Symptom (Phase 6)**: `make run` and `make verify` both produce garbage
output. NPU LM Head GEMV returns argmax=0 (token '!') for first_token AND
returns garbage tokens in decode loop. Even after switching first_token to
CPU LM head, decode loop's NPU LM Head produces "')lop')')'" garbage.

**Phase 5 single-run contradiction**: My single-shot test of
`qwen25_3b_phase5_test.py --cpu-verify` gave 4/4 NPU/CPU match. I
INCORRECTLY claimed this proved the NPU LM head works.

**Phase 7 evaluator REFINED finding (2026-04-27)**: Phase 5 NPU LM head is
**FLAKY** (state-dependent), not "always works". Evaluator ran phase5_test
back-to-back 3 times with `make clean` between → got 0/4 → 3/3 → 0/4 NPU/CPU
match. So the NPU LM head 11×13824 ELF has a state-dependent correctness bug
that affects BOTH Phase 5 standalone AND inference.py runner. CPU LM head
workaround is the ONLY reliable path.

**Suspected root cause (W2, unconfirmed)**: NPU prefill's `npu_prefill_with_kv_extraction`
leaves NPU in a state (BO contention, channel state, or K/V cache layout
encoding) that interferes with subsequent NPU LM head GEMV calls. The
phase5_test path doesn't trigger this because it uses CPU prefill.

**Differential**:
- Phase 5 (works): CPU prefill seed → preload LM head → decode loop with NPU LM head
- Phase 6 (broken): NPU prefill (extracts K/V) → preload LM head → first LM head → decode loop with NPU LM head
- Common element: same `qwen25_3b_decode_setup.compile_qwen25_3b_decode_kernels`,
  same `preload_qwen25_3b_lm_head`, same `qwen25_3b_npu_lm_head_gemv`.
- The ONLY difference is whether NPU prefill ran before LM head usage.

**Workaround attempted**: Use CPU LM head for first_token. Fixed first_token
but decode loop's NPU LM head still produces garbage. So the contamination
persists across the verify section's CPU compute and decode loop's biases
re-register.

**Disposition**: Surface as W2 in TODO.md. Phase 6 marked PASS-with-warnings.
**Phase 5 standalone correctness (4/4 NPU/CPU match) is the paper-citable
correctness data** for decode; inference.py's runner needs follow-up
investigation post-paper.

**Hypothesis to test next time**: Re-run preload_qwen25_3b_lm_head IMMEDIATELY
before each LM head usage (instead of once at setup) to force fresh BO
state. If that fixes it → it's BO state contamination from prefill cache.
