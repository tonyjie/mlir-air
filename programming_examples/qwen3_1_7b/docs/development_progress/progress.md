# Qwen3-1.7B deployment progress

Second validation of the **kernel-first methodology** developed during
qwen3-0.6B (Phase A → Phase B fusion + host-side opts). Same Qwen3 arch
(Q/K Norm, no QKV bias, GQA group=2, head_dim=128); 2.7× the parameters
(emb_dim 1024→2048, hidden_dim 3072→6144).

## Outcome (PASSED 2026-04-21)

| Phase | Outcome | Key metric |
|---|---|---|
| 0 Bootstrap | PASS | corr=0.99999986 vs HF, top-1 ' Paris' |
| 1 Per-kernel | PASS | 12/12 NPU kernels at 1.7B shapes (cos > 0.99) |
| 2 Single-block | PASS | cos_real=0.9985, per-pos min > 0.98 |
| B Fused decode #1 (rms_attn_gemvs_qknorm_rope, 8 launches) | PASS | cos > 0.9999 on all 8 outputs |
| B Fused decode #2 (o_gemv_ffn_silu, 8 launches, llama3 2-K rename) | PASS | cos > 0.997 on all 8 outputs |
| 3 verify N=8 | PASS | 8/8 top-5, 7/8 exact match |
| 3 sweep | PASS | 6/6 canonical prompts |
| 4 Prefill | PASS | 2.81 s warm @ seq_len=2048 (100 ms/layer) |
| 5 Decode | PASS | 6.73 tok/s (148.6 ms/token) |

End-to-end run sample (`make run N_TOKENS=20`):
> 'The capital of France is Paris. The capital of the United States is
> Washington, D.C. The capital of the United Kingdom'

## What this validates about kernel-first methodology

| Claim from qwen3-0.6B | How 1.7B confirmed it |
|---|---|
| Q/K Norm via heads-as-M `weighted_rms_norm` is portable | Same `compile_qknorm_per_head_gemv` builder reused unchanged at the bigger emb/hidden context — outputs cos > 0.9999 |
| Pre-transpose weights + arg cache + preload give the host-side win | Same patterns ported; decode wall stable at 148 ms/token despite 2.7× weight DMA volume |
| 3-K matvec rename is per-model, only when q_dim ≠ emb_dim | Confirmed: 1.7B has q_dim==emb_dim so it falls back to llama3's 2-K rename (mv.o + dg_matvec); `compile_mv_og`/`compile_mv_dg_qwen3` from 0.6B not needed |
| Phase 1 inheritance-vs-kernel-first decision picks the right path | 1.7B landed on kernel-first (Q/K Norm exists). Same scaffolding, different shape constants, less new code than 0.6B |

## Per-model code added (1.7B)

Only **shape constant changes** vs the 0.6B copy. No new builders, no new
extern kernels. The `_llm_shared/` infra and `llama3/multi_launch_builder/`
ELF builders were reused as-is.

Specific 1.7B-only code paths:
- `qwen3_decode.compile_decode_kernels` calls llama3's
  `o_gemv_ffn_multi.build_o_gemv_ffn_module` directly (NOT the
  qwen3-0.6B `o_gemv_ffn_silu_qwen3` fork — q_dim==emb_dim so the fork's
  3-K rename branch is unnecessary).
- arg layout follows llama3 (wo at arg0, attn_out at arg1).
- LM head GEMV: 19 partitions × 8192 (vs 0.6B's 10 × 16384). At K=2048
  the L2 budget A = herd_m·tile_m·K·2 = 8·16·2048·2 = 524288 = exactly
  the MemTile cap, plus C=256 trips it. Reduced to tile_m=8/herd_m=8
  → A=262144 ✓, but per-partition B-DMA fires = M_part/(tile_m·herd_m)
  = 16384/64 = 256 > 255. Halved M_part to 8192 → fires=128 ✓, padding
  19·8192 − 151936 = 3712 rows.

## Down GEMV K=6144 L2 budget

For the o_gemv_ffn fused ELF: tile_m=8 with K=6144 gives A = 8·8·6144·2
= 786432 B > 512 KB. Reduced to `down_tile_m=2, down_m_input=1` →
A = 8·2·6144·2 = 196608 B ✓. K=6144 ≤ 8160 so K-DMA auto-split fits
without `k_split`. (For Llama-3-8B-class K=14336+ this would need
`down_k_split`; not needed here.)

## Comparison vs. other deployments

| Model | n_layers | emb_dim | hidden_dim | head_dim | Decode tok/s | Per-layer prefill warm |
|---|---|---|---|---|---|---|
| llama3 (1.2 B)    | 16 | 2048 | 8192 | 64  | 10.8 | 81 ms |
| smollm2 (1.7 B)   | 24 | 2048 | 8192 | 64  | 7.3  | 79 ms |
| llama32_3b (3 B)  | 28 | 3072 | 8192 | 128 | 4.7  | ~110 ms |
| qwen25 (1.5 B)    | 28 | 1536 | 8960 | 128 | 4.6  | 85 ms |
| qwen3 (0.6 B)     | 28 | 1024 | 3072 | 128 | 10.5 | 78.6 ms |
| **qwen3 (1.7 B)** | **28** | **2048** | **6144** | **128** | **6.73** | **100 ms** |

1.7B sits between qwen25_1.5B and qwen3-0.6B as expected: 2.7× the FFN
work per layer compared to 0.6B drops decode from 10.5 → 6.73 tok/s
(approximately scales with hidden_dim).

## Methodology sample size = 2

The kernel-first path now has two completed deployments (qwen3-0.6B and
qwen3-1.7B). Promoting to its own skill (Option B from the prior session)
is justified: the methodology generalizes with just shape-parameter swaps.

The Phase 1 *inheritance vs kernel-first decision* bullets added to the
existing skills (`validate-per-kernel-shapes`, `integrate-single-block`,
`optimize-decode-perf`) cover the discovery and routing — no new top-level
skill is strictly required, but the Phase B builders (per-model
`multi_launch/<group>_<model>_test.py` standalone gates) could be
templated as a `phase-b-fuse-and-validate` skill if a third deployment
follows the same pattern.
