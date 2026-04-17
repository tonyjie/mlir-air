# SmolLM2-1.7B BF16 on NPU2 (MLIR-AIR)

Deployment of [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
on AMD NPU2 (Strix) via the `deploy-new-llm` skill chain.

**Status**: in progress. See `TODO.md` for phase status and
`docs/development_progress/progress.md` for the phase log.

## Architecture vs. Llama-3.2-1B

SmolLM2-1.7B is a Llama-architecture model — same RMSNorm, SwiGLU MLP,
half-split RoPE, BF16. The reference deployment is `../llama3/`. Divergences:

| Aspect              | Llama-3.2-1B | SmolLM2-1.7B          |
|---------------------|--------------|-----------------------|
| Layers              | 16           | **24**                |
| n_heads / n_kv_heads| 32 / 8 (GQA) | 32 / 32 (**MHA**)     |
| head_dim            | 64           | 64 (same)             |
| hidden_dim          | 8192         | 8192 (same)           |
| vocab_size          | 128256       | **49152**             |
| RoPE θ              | 500000       | **130000**            |
| Embedding tying     | untied       | **tied**              |
| Memory (BF16)       | ~2.5 GB      | ~3.4 GB               |

## Build & run

(populated after Phase 0 produces the model-specific entry point)

```bash
cd programming_examples/smollm2_1_7b
make compile    # TODO
make run        # TODO
```
