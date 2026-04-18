# Phase 4 — Prefill Performance

**Date**: 2026-04-18
**Test**: `llama32_3b_phase4_test.py`
**Setup**: 28-layer NPU prefill, **CPU attention fallback** (NPU FA hangs at our shape — see Pattern 5 finding), CPU LM Head

## Pattern application

| # | Pattern | Status | Source / Notes |
|---|---|---|---|
| 1 | Multi-launch merging          | **INHERITED**       | `rms_gemms_rope` = 6 launches in one ELF; `o_ffn` = 8 launches |
| 2 | Per-layer BO pre-loading      | **APPLIED HERE**    | `preload_prefill_weights(weights, config, cache, seq_len, rope_lut)` — same config-driven helper smollm2 used; drop-in for Llama-3.2-3B |
| 3 | Intermediate buffer reuse     | **INHERITED**       | `intermediate_indices` set per kernel in `run_transformer_block` |
| 4 | Seq-first activation layout   | **INHERITED**       | RoPE + FA accept `(seq, heads*dim)` natively |
| 5 | CPU → NPU op promotion        | **ATTEMPTED-FAILED**| NPU FA at head_dim=128 hangs with `ERT_CMD_STATE_TIMEOUT`; LM Head deferred to Phase 5 (per llama3 pattern) |

**4 of 5 patterns applied or inherited** (gate: ≥3). ✓

## Measurements

### Cold first-prompt vs warm steady-state

| Phase | NPU layers (28) | per-layer | CPU LM Head | Wall total |
|---|---|---|---|---|
| **Cold** (no preload, 1st prompt)             | 16.428 s | 587 ms | 2.024 s | **18.502 s** |
| **Warm** (after preload, avg of 3 runs)       | 13.627 s | 487 ms | 2.041 s | **15.741 s** |

**Pattern 2 gain on first prompt**: 2.80 s NPU-layer reduction (17.1%). Small
relative gain because per-layer time is dominated by CPU attention (~250-300 ms
of the 487 ms), not BO writes. Pattern 2's value here is mostly amortizing the
weight-write cost over the 28-layer × 3 multi-launch ELF call surface.

### Pre-load setup cost

`preload_prefill_weights` for 28 layers: **2.12 s** (one-time, outside timed inference).
Pre-loaded weight footprint: **5,376 MB** in NPU-allocated DRAM-mapped BOs
(28 × 192 MB/layer = `wq + wk + wv + wo + w_gate + w_up + w_down + 2 norms` at emb_dim=3072,
hidden_dim=8192).

### Per-prompt warm runs

| Run | NPU layers | per-layer | LM Head | Wall | top-1 |
|---|---|---|---|---|---|
| 1 | 14.072 s | 503 ms | 2.018 s | 16.139 s | `' the'` |
| 2 | 13.283 s | 474 ms | 2.054 s | 15.421 s | `' the'` |
| 3 | 13.525 s | 483 ms | 2.052 s | 15.663 s | `' the'` |

## Pattern 5 — NPU FlashAttention investigation

Implemented Phase 1's recommended `compile_attn_npu2_split(lqp, lkp, dk, dv)` API
in `_llm_shared/kernel_builder/external_kernels.py:119`. Used the L1-feasible
config that the existing `run_npu2_makefile_peano_llama3_8b.lit` lit test
proves works: `lkp=64, lqp=256, dk=dv=128, dk_chunks=2`. Also monkey-patched
`llama3_prefill._attn_backend_kwargs` so runtime kwargs (`omit_while_true_loop=True`
because no shared buffers when lkp != dk) match the build.

**Compile**: succeeds (79.6 s for `flash_attn.elf`, no errors, all asserts pass).
- `lq=lk=2048`, `lqp=256`, `lkp=64`, `dk=dv=128`, `num_heads=24`, `num_kv_heads=8`, `causal=True`
- `lq % lqp == 0`: 2048 % 256 = 0 ✓
- `lqp/num_q_tiles == lkp` (causal precondition): 256/4 = 64 ✓
- `lk % (lkp * num_cascade_stages) == 0`: 2048 % 256 = 0 ✓
- `dk % lkp == 0`: 128 % 64 = 0 → dk_chunks=2 ✓

**Runtime**: hangs with `RuntimeError: Command failed to complete successfully (ERT_CMD_STATE_TIMEOUT)`.
- BOs allocated successfully (4 BOs for flash_attn)
- Hang on the first invocation (run.wait2() never returns within XRT default timeout)
- Earlier kernels in the layer (rms_gemms_rope) succeed; only flash_attn hangs

**Hypotheses for the hang** (not investigated to root cause):
1. **GQA group_size=3** specifically: smollm2 = MHA (group=1), llama3 = group=4,
   the proven-working llama3-8b lit test = group=4. group=3 (24/8) is
   untested in the seq-first FA kernel; the affine.apply for `kv_head_index =
   q_head_index // gqa_group_size` may have a corner case.
2. **Buffer descriptor / channel exhaustion at lq=lk=2048**: the proven-working
   llama3-8b lit test uses lq=lk=512. Our lq=lk=2048 is 4× larger → 4× more
   chunks per cascade stage and 4× more iteration counts.
3. **L1 budget actually exceeds 64 KB at runtime**: my back-of-envelope gave
   ~50 KB but I didn't account for ping-pong / channel buffer overhead. With
   `omit_pingpong="all"` this should be off but worth verifying.
4. **dk_chunks=2 path may have an integration issue** in seq-first that
   doesn't surface at dk_chunks=1 (the only config the lit tests exercise).

**Triage**: deferring to a follow-up. Options for resolution:
- (a) Bisect by changing one variable at a time: try lq=lk=512 (lit test config), then 1024, then 2048.
- (b) Bisect on (n_heads, n_kv_heads): try with our heads but the lit test sequence size; then with our seq size but lit test heads.
- (c) Try lkp=128 lqp=128 num_q_tiles=8 (Phase 1 noted this still over-budgets L1 but worth verifying the actual failure mode).
- (d) Add a third FA variant: lkp=64, lqp=128, num_q_tiles=2 (smaller per-launch tile; reduces concurrent BO pressure).

This is a real follow-up item but is **not on the Phase 4 critical path**. The
gate "≥3 of 5 patterns + perf measured + no regression" is met with 4/5.

## vs llama3 baseline

| Metric | llama3 (16 layers, GQA, NPU FA) | Llama-3.2-3B (28 layers, GQA, **CPU FA**) | Gap explanation |
|---|---|---|---|
| Per-layer (warm)    | 81 ms | **487 ms** | ~6× — CPU attention dominates each layer (~300 ms numpy) |
| Total NPU prefill (warm) | 1.30 s | **13.6 s** | ~10× total: 1.75× from depth + 6× from CPU attention |
| Wall clock (warm)        | 1.54 s | **15.7 s** | ~10× total |

**With NPU FA working**, the projection (assuming ~30 ms/layer NPU FA at our
config, similar to llama3-8b at head_dim=128): ~250 ms/layer → 7 s NPU prefill
(28 × 250 ms). Still not at parity with llama3 due to depth + width, but a 2×
improvement over the current CPU-attn number. Worth pursuing in a follow-up.

## Correctness regression check

- Cold top-1: `' the'` (id=279) — competitive prompt as documented in Phase 3
- Warm top-1: `' the'` (id=279) on all 3 runs
- **No regression vs Phase 3** ✓

## Items surfaced for Phase 5

- 🔸 **NPU FA hang** (above) — needs root-cause investigation. Highest-impact
  follow-up. ~hours-to-days estimate; not blocking deployment.
- 🔸 **CPU LM Head is 2.0 s of the 15.7 s wall** (~13% of wall). Phase 5 should
  move LM Head to NPU (vocab=128256 → 8×16384 partition scheme is drop-in from
  llama3, per Phase 1 classification). For a 13% wall-clock reduction.
- 🔸 **Memory**: 5.4 GB pre-loaded prefill weights + 6 GB CPU-side BF16 weights ≈
  11 GB. NPU2 system DRAM is 16 GB. Decode adds ~3 GB more for transposed
  weights (per smollm2 Phase 5 measurement, scaled). Total runtime working set
  ~14 GB at peak — tight but feasible. The original Path A budget estimate
  was correct.
- 🔸 **First-prompt latency** is now bounded by preload setup (2.12 s), not
  inference. Production code should call `preload_prefill_weights` from a
  startup hook.

## Phase 4 gate verdict

- ✅ 4 of 5 patterns applied/inherited (gate: ≥3)
- ✅ Prefill latency measured: warm 13.6 s NPU / 15.7 s wall
- ✅ No correctness regression (cold and warm both top-1 `' the'`)

**PASS.** With caveat: absolute perf is poor (~10× slower than llama3) due to
the deferred NPU FA work. Production-grade perf needs the NPU FA hang resolved.
