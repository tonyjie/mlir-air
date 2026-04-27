---
name: decode-optimization
description: Phase 5 of LLM deployment — apply decode-specific optimization patterns to a Phase-4-correct pipeline (multi-launch merge with N-way extern rename, static weight BOs, NPU LM Head GEMV, CPU→NPU promotion). Each step preserves correctness by re-running the Phase 3 numerical gate. Invoked after Phase 4 PASS.
---

## Purpose

Phase 4 optimized prefill while preserving Phase 3 correctness. Phase 5
does the same for decode — but the dominant patterns differ because
decode runs at M=1 per token, calling all N layers once per generated
token. This amplifies the value of static weight BOs (weights are
loaded once but read on every token) and LM Head GEMV (was a CPU
bottleneck dwarfing all kernel time).

The reference deployment llama3-1B took decode from ~500 ms/token →
92 ms/token (5.4×) by composing the patterns below.

## Phase 5 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied pattern, the Phase 3
   PASS criteria still hold. Re-run Phase 3 gate between patterns. If
   correctness regresses, **revert the pattern** and document why.
2. **Decode time/token strictly < Phase 4 baseline**, measured with
   5-warmup + 20-iter profile at the same canonical prompt.
3. **Per-pattern outcome documented** in
   `<model>/docs/development_progress/phase5_decode.md`: for each of
   the 4 patterns, record `applied / skipped / reverted`, the latency
   delta, and a one-line reason.

The "≥ N patterns applied" check is NOT a gate — some models
legitimately need only A + B (most kernel-first deployments don't need
LM Head promotion if Phase 4 already moved it). The gate is the
outcome (decode time improved + correctness preserved).

## Knowledge base references

PRIMARY:

- `programming_examples/_llm_shared/docs/perf_optimization.md` — full
  llama3-1B optimization journey (500 ms → 92 ms decode); the
  reference for what "good" looks like
- `programming_examples/<model>/docs/development_progress/phase4_prefill.md`
  — Phase 4 baseline (prefill numbers + the integration path used)
- `programming_examples/_llm_shared/docs/multi-launch/decode_merging.md`
  — decode-specific merge patterns + 2-K extern kernel rename
- `programming_examples/_llm_shared/docs/aie2p_hardware_limits.md`
  — Rules B (K_max=8160), C (combined channel reads ≤ 255), D (L2 cap)
  — relevant when GEMV K > 8160 or M is large

INHERITANCE PROTOTYPES (Pattern A reuse path):

- `programming_examples/llama3/multi_launch_builder/rms_gemv_rope_multi.py`
  — fused 6-launch decode ELF for RMSNorm + Q/K/V GEMV + RoPE Q/K
- `programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py`
  — fused 8-launch decode ELF (with 2-K extern rename for K=8192 Down)
- `programming_examples/llama3/multi_launch_builder/lm_head_gemv_multi.py`
  — vocab-partitioned LM Head GEMV (Pattern D)

KERNEL-FIRST PROTOTYPE (Pattern A new-build path):

- `programming_examples/qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py`
  — N-way (3-K) extern kernel rename for `n_heads*head_dim != emb_dim`
  models. Decode here gave **10× wall speedup** on top of fusion
  because per-call XRT overhead dominated.

## Workflow

### Step 1: Measure Phase 4 baseline

Capture the decode time/token before any Phase 5 pattern:

```bash
cd programming_examples/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 <model>_decode.py --profile --warmup 5 --iterations 20 \
                            --n-tokens 50
```

Record: ms/token + per-layer + LM head time breakdown (where the
budget goes today). These numbers gate every pattern below.

### Step 2: Apply optimization patterns

Apply A → B → D (skip C unless decode introduced a layout transpose
that Phase 4 didn't fix). Between each, re-run Phase 3 gate
(`<model>_inference.py --verify`) and re-measure profile.

#### Pattern A — Multi-launch merge (decode variants)

Stitch decode kernel groups into fused ELFs (mirrors Phase 4 Pattern A
but with GEMV instead of GEMM). Two paths, same decision as Phase 2/4
inheritance vs kernel-first:

| Path | When | What to do |
|---|---|---|
| **Reuse existing fused ELF** | Per-layer decode kernel sequence matches llama bit-for-bit | Import `llama3/multi_launch_builder/{rms_gemv_rope_multi, o_gemv_ffn_multi, lm_head_gemv_multi}` directly. smollm2, llama32_3b, qwen25 all do this. |
| **Build new fused ELF** | Phase 2 chose kernel-first (model has new ops, reordering, or different fused-launch boundaries) | Write decode-specific builders in `<model>/multi_launch/` (qwen3-0.6B prototype). Invoke `merge-multi-launch-kernels`. |

Expected: 10 launches/layer/token → 2-3 launches/layer/token.

**Sub: extern kernel rename for shape-collision.** When multiple GEMV K
values coexist in one fused ELF, they collide on the
`@matvec_vectorized_bf16_bf16` symbol. Compile `mv.cc` with `-D` symbol
renames per group, link them together. Two sizes:

- **2-K (llama3 pattern)**: default `mv.o` for K=2048 + `mv_k8192.o`
  (`-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16` etc.)
  for K=8192 Down. See `_llm_shared/kernel_builder/external_kernels.py`.
- **N-K (qwen3 pattern)**: `n_heads*head_dim != emb_dim` introduces a
  3rd K. Add `compile_mv_og(tile_m=8)` → `mv_og.o` (`og_matvec_*`),
  `compile_mv_dg_qwen3(tile_m=8)` → `mv_dg_qwen3.o` (`dg_matvec_*`),
  route per-launch via `_rename_all_with_externs` allowlists. See
  `qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py` for the working
  3-K example.

**Sub: K-split for K > 8160.** Rule B from `aie2p_hardware_limits.md`:
auto-split GEMV K-DMA caps at outer = 255 → max practical K ≈ 8160.
Above this, pass `down_k_split=N` (where `K % N == 0` AND `N ≤ 255`
AND `K/N ≤ 1023`). Examples: qwen25 K=8960 → `down_k_split=70`;
llama3-8B K=14336 → `down_k_split=56`. Same `k_split` knob is exposed
on `matvec.build_module` directly (default None = back-compat).

#### Pattern B — Static weight BOs (decode amplifies the win)

Decode reuses every weight on every token. Convert weight BOs to
allocated-once with `bo.map()` zero-copy access:

- Allocate all per-layer weight BOs in `prepare_runtime()`
- Use `bo.map()` to expose host-writable views and write once
- On every decode call: pass `static_input_indices=[<weight_indices>]`
  to skip the per-token re-write

Expected: removes per-token BO write of all weights. With 16+ layers
× 7 weight tensors per layer × 100 tokens, this is the dominant
host-side decode overhead pre-Pattern-B.

#### Pattern D — CPU→NPU promotion (LM Head is the headline)

**D1. NPU LM Head GEMV (the big win).** Pre-Phase-5, LM Head is often
on CPU (`logits = hidden @ embed.T`). Replace with NPU GEMV
partitioned across vocab. llama3-1B used 8 partitions, each handling
`vocab/8` rows × `emb_dim` cols, compiled into `lm_head_gemv.elf`.
Llama3 saw **~250 ms → ~14 ms**.

Choosing partitions: pick the largest partition that fits one tile's
L2 budget (Rule D). For vocab=128256, 8 partitions × 16384 rows fits;
for larger vocab, increase partitions.

**Combined-channel constraint at large M (Rule C)**: with LM-head
partitions M ≥ 16384, the B-input shim DMA fires `launch_count ×
(tile_m/m_input)` times per GEMV; combined GEMVs sharing a channel
add up. Stay under 255: set `tile_m = m_input` (inner_loop=1) and
pick `tile_m × herd_m ≥ M / 127`. Example: M=16384, `tile_m=16,
m_input=16, herd_m=8` → `16384/(16*8) × 1 = 128` ✓.

**D2. Other decode CPU→NPU promotion.** If the decode pipeline still
falls back to CPU for any small op (e.g., residual add wrapped in
`np.add`, decode attention if not yet on NPU), promote it using the
standalone harness Phase 1 already validated.

### Step 3: Re-run Phase 3 gate after each pattern

After every applied pattern:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 <model>_inference.py --verify --prompt "The capital of France is"
```

Confirm Phase 3 gate still passes. If regressed, revert and document why.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict) | Same as Phase 4 — Rule A or wrong stitching boundary | Invoke `debug-multi-launch-merge` |
| Extern kernel rename collision (link error, symbol redefined) | Two `.o` files exporting same symbol | Check `-D` mapping uniqueness; each `.o` must export distinct `<group>_matvec_*` names |
| `'aiex.npu.push_queue' op Repeat count exceeds [0:255]` (Pattern A or D) | Rule B (K > 8160) or Rule C (combined channel reads > 255) | For K > 8160 → set `k_split` / `down_k_split`; for large M → set `tile_m == m_input` and grow `tile_m × herd_m` |
| `L2 capacity exceeded` (matvec.py builder assert) | Rule D (`K × herd_m × tile_m × 2 > 512 KiB`) | Reduce `tile_m` (e.g., 8 → 2 for K=8192) |
| Output corruption after static weight BO conversion (correct first call, NaN/garbage on subsequent) | Per-layer BO key collision OR `static_input_indices` wrong | Invoke `debug-bo-corruption` |
| LM Head GEMV NaN / argmax differs from CPU | Partition boundary off-by-one OR vocab shape padding mismatch | Check partition count divides vocab evenly; print partition outputs and concatenate manually to compare |
| Cosine drops after Pattern X | Pattern X assumption violated by this model | Revert Pattern X; check assumption (e.g., decode already seq-first, weights already pre-transposed) |
| ms/token unchanged after Pattern A | Per-call XRT overhead dominates; fusion alone insufficient | Pattern B (static weight BOs) is likely the missing piece — apply it next |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 5 PASS:

- `<model>/docs/development_progress/phase5_decode.md`: per-pattern
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 5, append final ms/token + speedup
  vs Phase 4 baseline
- If a new fused decode ELF was built (kernel-first path of Pattern A),
  surface to Phase 6 for potential promotion to `_llm_shared/` if a
  second deployment validates the same pattern
