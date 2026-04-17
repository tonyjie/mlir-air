---
name: optimize-decode-perf
description: Phase 5 of LLM deployment — apply the 5 known decode optimization patterns from LLAMA Phase 5 (multi-launch merging, static weight BOs, NPU LM Head GEMV, extern kernel rename, CPU→NPU promotion). Invoked after Phase 4 gate.
---

## Purpose
Apply the decode optimization patterns that took LLAMA from ~500ms/token → 92ms/token. Same structure as Phase 4 but tuned for the M=1 (single-token) case.

## Knowledge base references
- `programming_examples/llama3/docs/development_progress/multi-launch/decode_merging.md` — decode-specific merge patterns
- `programming_examples/llama3/docs/development_progress/decode_archive/DECODE_PROGRESS.md` — decode milestones
- `programming_examples/llama3/docs/development_progress/decode_archive/gemv_investigation.md` — AIR vs IRON GEMV perf

## Workflow

### Pattern 1: Multi-launch merging (decode variants)
Invoke `merge-multi-launch-kernels` for decode groups (rms+gemvs+rope, o+ffn). Same procedure as prefill but with GEMV instead of GEMM.

Expected: 10 launches/layer/token → 2–3 launches/layer/token.

### Pattern 2: Static weight BOs
Decode reuses every weight on every token. Convert weight BOs to allocated-once with `bo.map()` zero-copy access.

Expected gain: removes per-token BO write of all weights.

### Pattern 3: NPU LM Head GEMV (vocab projection)
Replace CPU LM head with NPU GEMV partitioned across vocab (LLAMA used 8-partition with `mv_k8192.o` extern rename). Each partition handles `vocab/8` rows × `emb_dim` columns.

Expected gain: LLAMA observed ~250ms → ~14ms.

### Pattern 4: Extern kernel rename (shape-collision avoidance)
If two GEMV shapes coexist in one ELF (e.g., K=2048 and K=8192) and need different kernel implementations, compile with `-D` symbol renames so they can be linked together.

See `_llm_shared/kernel_builder/external_kernels.py` and `programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py` for the existing K=8192 rename pattern.

### Pattern 5: CPU→NPU op promotion
Same as Phase 4 Pattern 5 but for decode-specific ops (typically the small attention step on a single query, if currently on CPU).

### Bookkeeping
Same as Phase 4: record per-pattern latency in `<model>/docs/development_progress/phase5_decode.md`.

## Verification (Phase 5 gate)

Phase 5 PASSES when:
- Decode latency measured (ms/token)
- ≥3 of 5 patterns applied or N/A
- No correctness regression (Phase 3 re-run still PASS)

## Failure modes
Same as Phase 4 plus:
- Extern kernel rename collision → check `-D` symbol mapping uniqueness; invoke `debug-multi-launch-merge`

## Update protocol
Append to `progress.md`. Update `TODO.md` Phase 5.
