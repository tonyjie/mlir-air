# Qwen3-4B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen3-4B` inference on AMD NPU2 (AIE2P).

Reference deployments:
- `../qwen3_0_6b/` — kernel-first methodology reference (Q/K Norm,
  3-K matvec rename when q_dim ≠ emb_dim), Option C head-first FA.
- `../qwen3_1_7b/` — second kernel-first deployment (28L, q_dim==emb_dim
  → 2-K rename). 4B brings 36 layers + bigger GQA imbalance.
- `../qwen25_3b/` — same depth (36 layers), padded-hidden recipe at
  hidden 11008→12288. We share the same depth + similar
  hidden=9728 padding question.

## Status

Deployment in progress. See [`TODO.md`](TODO.md) for phase status and
[`docs/development_progress/phase_timing.md`](docs/development_progress/phase_timing.md)
for per-phase wall-clock.

## Model config

36 layers, emb_dim=**2560** (NOT 1024-aligned), n_heads=32,
**head_dim=128**, n_kv_heads=8 (GQA group=**4**), hidden_dim=**9728**
(NOT 1024-aligned), vocab=151936, BF16, **rope_θ=1,000,000**,
**tied embeddings**, **NO QKV bias**, **has Q/K Norm** (per-head).

`q_dim = 32 × 128 = 4096` ≠ emb_dim=2560 → 3-K matvec extern rename
required (mirrors `qwen3_0_6b`).

## Divergences from prior deployments

vs `../qwen3_0_6b/` (kernel-first reference):
1. **36 layers** (vs 28). Deepest in catalog, tied with `qwen25_3b`.
2. **emb_dim=2560 NOT 1024-aligned** → padding probably needed
   (candidates: 3072 = 3×1024, or 4096 = 4×1024 to match q_dim).
3. **hidden_dim=9728 NOT 1024-aligned** → padding probably needed
   (candidates: 10240 = 10×1024). qwen25_3b's 11008→12288 (skip
   11264) recipe is the cautionary tale; 9728 may also need to skip
   the smallest-padding option.
4. **GQA group=4** (vs 0.6B/1.7B's group=2). NEW axis. W1 (NPU
   seq-first FA precision drop) does NOT apply here — hd=128 routes
   through Option C head-first FA (precision-clean).
5. **n_h=32** (vs 16). Bigger but still even ✓ FA OK.
6. Otherwise same as qwen3 family: tied embeddings, rope=1M, NO QKV
   bias, Q/K Norm BEFORE RoPE.

## Inheritance path

**Kernel-first** (mirror `../qwen3_0_6b/multi_launch/` structure):
- New ELF builders in `qwen3_4b/multi_launch/` for the kernel-first
  fused launches (Q/K Norm + RMS + GEMV/GEMM + RoPE).
- 3-K matvec rename (q_dim=4096, kv=1024, hidden_padded). Pattern
  copied from `qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py`.
- Option C wrapper from `_llm_shared/phase_helpers/headfirst_fa.py`
  (head_dim=128 path).
- Q/K Norm via `_llm_shared/phase_helpers/qk_norm.py` (host-side OR
  on-tile; decide in Phase 2 per qwen3_0_6b prototype).

## Watch list

- **emb/hidden padding choice**: hidden=9728 may compile at 10240 but
  hang at runtime (qwen25_3b hit this at 11264 → had to use 12288).
  Default to smallest 1024-aligned; bump only on runtime failure.
- **36-layer compile time**: matches qwen25_3b; expect long compile
  for the prefill multi-launch ELF.
- **GQA group=4**: NEW for kernel-first path (0.6B/1.7B both g=2).
  Confirm Q/K Norm host wrapper handles g=4 fan-out correctly.
- **LM head partition**: vocab=151936; choose partition count so each
  partition fits L2 (Rule D) AND launches ≤255 (Rule C).

## File layout convention

Minimal scaffold + sys.path imports. Qwen3-4B-specific code (produced
by per-phase skills):
- `qwen3_4b_weights.py` — config + HF loader (Q/K Norm, NO QKV bias)
- `qwen3_4b_reference.py` — CPU F32 reference forward
- `qwen3_4b_inference.py` — end-to-end NPU runner (entry for `make run`)
- `qwen3_4b_phaseN_test.py` — per-phase validation scripts
- `multi_launch/` — kernel-first fused ELF builders (mirror qwen3_0_6b)

Imported from `../qwen3_0_6b/`, `../llama3/`, `../_llm_shared/`:
- `llama3_prefill.run_transformer_block` (per-block dispatch)
- `multi_launch_builder.lm_head_gemv` (vocab-partitioned LM head)
- `_llm_shared.phase_helpers.qk_norm.apply_qk_norm`
- `_llm_shared.phase_helpers.headfirst_fa.*` (Option C)
- `_llm_shared.kernel_builder.external_kernels.*`
- `_llm_shared.KernelCache`, `prepare_air_project`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status, active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/phase_timing.md](docs/development_progress/phase_timing.md) | Per-phase wall-clock |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../qwen3_0_6b/CLAUDE.md` | Kernel-first methodology reference |
| `../qwen3_1_7b/CLAUDE.md` | 2nd kernel-first deployment |
| `../qwen25_3b/CLAUDE.md` | Same depth (36L) for compile/perf calibration |
