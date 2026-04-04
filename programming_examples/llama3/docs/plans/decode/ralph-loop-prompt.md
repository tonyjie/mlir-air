CONTEXT: We have a working LLAMA prefill pipeline at programming_examples/llama3/. Now implementing decode (single-token generation). The plan file has full details.

KEY REFERENCES:
- Existing multi-column GEMV+FFN decode: programming_examples/ffn_swiglu/decode/ffn_decode.py (works at dim=128, num_cols=4)
- Single-core GEMV (all LLAMA shapes pass): programming_examples/matrix_vector_multiplication/bf16/matvec.py
- IRON decode profiling targets: programming_examples/llama3/docs/decode/iron_decode_reference.md
- Prefill pipeline (pattern to follow): programming_examples/llama3/llama3_prefill.py
- Multi-launch builder pattern: programming_examples/llama3/multi_launch_builder/

FINDINGS SO FAR:
- Single-core GEMV works at all LLAMA shapes but is ~8x slower than IRON (needs multi-column)
- ffn_decode.py works at dim=128, num_cols=4. Fails at num_cols=8 (row index limit on NPU2)
- Prefill elementwise kernels (rmsnorm, add, rope, silu) should work at decode sizes with minimal changes

EXECUTE IN ORDER:
1. Scale ffn_decode.py to LLAMA dims (emb=2048, hidden=8192) — test compilation and correctness
2. Build standalone multi-column GEMV kernel for all 5 LLAMA shapes, profile vs IRON
3. Test prefill kernels at decode sizes: weighted_rms_norm(M=1,N=2048), eltwise_add(n=2048), rope_lut(32x64), silu_and_mul(n=8192)
4. Create programming_examples/llama3/llama3_decode.py following the prefill pattern — KV cache + single-token decode loop
5. Verify single-token decode correctness against CPU reference
6. Profile end-to-end and compare vs IRON (target: 132ms standalone kernel, 370ms e2e)

RULES:
- Run from programming_examples/llama3/ or the relevant example directory
- Use build_peano/ as build directory
- Commit progress after each major milestone
- Document findings in programming_examples/llama3/docs/decode/
- If a kernel fails at LLAMA scale, document the error and try smaller scale first
- Follow the coding style in CLAUDE.md (simplicity, no over-engineering)