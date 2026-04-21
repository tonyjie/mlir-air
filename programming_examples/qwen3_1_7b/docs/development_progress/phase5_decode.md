# Phase 5 — NPU Decode (PASSED 2026-04-20, bottom-up rebuild)

The autonomous loop's first attempt at Phase 5 punted to CPU decode (1.23 s/token)
because the existing fused `rms_gemv_rope` ELF had no host hook for Qwen3's
Q/K Norm step. The user identified the root pattern: we'd been **inheriting
opinionated llama3 stitching plans** instead of testing each leaf kernel
standalone and stitching from scratch per model.

This re-do applies the **bottom-up kernel-first methodology** end-to-end:
test each leaf, then stitch only the pieces we own.

## 5-step build (all PASS)

| Step | Artifact | Result |
|---|---|---|
| 1 | `qwen3_kernel_registry_test.py` — 8 standalone leaf kernels at Qwen3 shapes | 8/8 PASS |
| 2 | `multi_launch/rms_attn_gemvs_qwen3.py` — split 4-launch ELF (RMSNorm + Q/K/V GEMV, NO RoPE) | cos > 0.99997 vs CPU on normed/q/k/v; 0.73 ms NPU run |
| 3 | `o_gemv_ffn_multi.py` extended with `o_in_dim` kwarg (backward-compat) | builds at both llama3 default & Qwen3 shapes; 3-K rename collision pivot → per-leaf approach for now |
| 4 | `multi_launch/lm_head_gemv_qwen3_test.py` — 10 × 16384 partition ELF | 10/10 partitions cos=1.000000; 11.05 ms NPU run |
| 5 | `qwen3_decode.py` — block runtime + `qwen3_inference.py` integration | end-to-end coherent generation; "The capital of France is Paris. The capital of France is also the" |

## Findings captured during the rebuild

1. **matvec backend kwargs are mandatory** (Step 1): the canonical
   `matrix_vector_multiplication/bf16/matvec.py` test uses
   `omit_pingpong=True`, `runtime_loop_tiling_sizes=[4,4]`,
   `use_lock_race_condition_fix=True`, `omit_while_true_loop=False`.
   Without them, GEMV outputs are numerically wrong (random shuffle of
   correct values). Captured in `decode.py::_GEMV_BACKEND_BASE`.
2. **M=16384 GEMV needs larger tile config** (Step 1): default
   tile_m=8/m_input=4/herd_m=8 → 256 outer iters → AIE2P shim
   `repeat_count > 255` hard fail. qwen25 uses `tile_m=16/m_input=16/herd_m=8`
   (= 128 outer iters) and that pattern transfers cleanly here.
3. **Stitched-ELF kernel symbol must match `instance_name`** (Step 2):
   `func.func @rms_attn_gemvs` paired with `instance_name="rms_attn_gemvs"`,
   not `"rms_attn_gemvs_qwen3"`. XRT errors with "Unable to find group idx
   for given kernel" when they don't match.
4. **XRTRunner expects inputs-first / outputs-last arg layout** (Step 2):
   `expanded_inputs = inputs + output_placeholders`. Func args with
   interleaved input/output (e.g., weight, output, weight, output) are
   incompatible — reorganize to inputs first.
5. **Three-way K matvec extern collision** (Step 3): existing
   `o_gemv_ffn_multi` handles 2 K values via `dg_matvec_*` rename + separate
   `mv_k8192.o`. Qwen3 needs THREE K values (1024 Gate/Up, 2048 O, 3072
   Down). Pivoted to per-leaf calls; fusion is a follow-up.

## Performance summary

```
NPU prefill (warm, seq_len=512): 0.59 s   (21.2 ms/layer)
NPU decode (avg, 7 tokens):      0.90 s/token  (1.11 tok/s)
End-to-end demo (8 tokens):      ~7 s wall (prefill + decode)
```

vs. previous CPU-decode interim: 1.23 s/token (0.81 tok/s) → **+37% faster**,
correct. NOT yet competitive with qwen25_1_5b's 216 ms/token (which uses a
fully fused decode ELF) — that's the fusion follow-up's target.

## Comparison against other deployments

| Model | Decode | NPU GEMV decode? |
|---|---|---|
| llama3 (1.2 B)   | 92 ms/token   | YES (fused rms_gemv_rope + o_gemv_ffn) |
| smollm2 (1.7 B)  | 137 ms/token  | YES (fused) |
| llama32_3b (3 B) | ~210 ms/token | YES (fused) |
| qwen25 (1.5 B)   | 216 ms/token  | YES (fused, 3-K rename) |
| **qwen3 (0.6 B)** | **900 ms/token** | **YES (per-leaf ELFs; fusion is follow-up)** |

## Files added in this Phase 5 rebuild

```
qwen3_0_6b/
├── qwen3_kernel_registry_test.py     # Step 1
├── qwen3_decode.py                   # Step 5: NPU decode block + generate
├── qwen3_decode_smoke.py             # Step 5: standalone correctness smoke
└── multi_launch/
    ├── __init__.py
    ├── rms_attn_gemvs_qwen3.py       # Step 2
    └── lm_head_gemv_qwen3_test.py    # Step 4
```

## Files modified (backward-compat extensions)

- `llama3/multi_launch_builder/o_gemv_ffn_multi.py` — added `o_in_dim` kwarg
  (defaults to emb_dim → zero impact on llama3, smollm2, llama32_3b, qwen25)
