# Qwen3-4B DFlash B8 architecture

This document explains the selected implementation delivered in this directory.
The top-level [README](README.md) contains the short build and run procedure.

## Fixed model configuration

| Property | Target | Drafter |
|---|---:|---:|
| Decoder layers | 36 | 5 |
| Attention | Causal | Bidirectional inside the proposed block |
| Hidden width | 2,560 | 2,560 |
| Query / KV heads | 32 / 8 | 32 / 8 |
| Head width | 128 | 128 |
| FFN width | 9,728 | 9,728 |
| Vocabulary | 151,936, tied embedding/head | Target embedding/head |
| NPU rows per call | 8 | 8 |
| Capacity | 4,096 | 4,096 |

The official draft was trained for block size 16. This implementation executes
eight rows because the validated NPU map is B8; using a draft trained at a larger
block size for a shorter inference block is supported by the model.

## Prompt prefill

The default is the same op-by-op NPU prefill as `qwen3_4b_q4nx`, wrapped by
`npu_prefill.py`. It uses eight cached kernel programs: fused RMS/QKV/QK-norm/RoPE,
FlashAttention, O/residual/RMS, gate, up, SwiGLU, down/residual, and the partitioned
LM head. Q4NX weights are decoded to BF16 once and kept in resident prefill BOs.
Embedding lookup, the final one-row RMSNorm, and host orchestration remain on
the CPU.

The shared runner already reads each layer output and roped K/raw V. The adapter
copies outputs after layers 2, 10, 18, 26, and 34 before their transient buffers
are reused, then stacks them as `[real_prompt_length, 5 * 2560]`. It exports
KV as `[36, real_prompt_length, 1024]` and the first greedy token. Padding never
enters the exported KV or taps. These are the same interfaces consumed by the
existing DFlash reset/context-injection loop.

The prefill kernels are separate from the two B8 decode programs. The default
padded prefill shape is 2048 rows; B8 decoding still uses live context within
C4096. `PREFILL=cpu` retains the prior reference path.

## One DFlash round

```text
committed target taps [new positions, 5 x 2560]
        |
        v
CPU feature fusion: five target taps -> hidden_norm -> 2560 features
        |
        v
CPU learned K/V projection for each of five draft layers
        |
        v
upload/replace committed drafter context KV
        |
        v
NPU drafter: [anchor, mask x 7] -> seven proposed tokens
        |
        v
NPU target: [anchor, proposal x 7] -> eight verified distributions + taps
        |
        v
CPU: accept matching prefix, emit correction/bonus, advance committed position
```

The target writes candidate KV in place for all eight rows. Only the accepted
prefix is made live by advancing the host-owned position. A later call overwrites
rejected rows, and causal masks prevent them from affecting earlier positions.
The target hidden-state taps at layer boundaries 2, 10, 18, 26, and 34 become
the next drafter context after CPU projection and BF16 upload.

## NPU program

Each target or drafter call is one host dispatch over a persistent 30-compute-
tile graph. The graph contains all decoder layers and the tied LM head.

- Sixteen projection cores consume Q4NX packed weights. Each packed weight tile
  is transferred once and reused for eight activation rows.
- Eight attention cores form four parallel two-query groups. K/V chunks are
  fetched once per group and shared by four queries; each target row applies its
  own causal boundary. Drafter rows use bidirectional block attention.
- Service tiles perform row-wise RMS normalization, residual updates, RoPE,
  projection gathering, SwiGLU, and output staging.
- A spare compute tile stores the 40-KiB normalized B8 activation and replays it
  to projection phases. This avoids repeatedly applying the RMS scale and norm
  weights when the original single-tile B8 working set does not fit in L1.

Projection arithmetic uses BFP16 `mmul<8,8,8>` with explicit nearest-even
rounding. Activations and KV storage are BF16. Packed model weights use the Q4NX
four-bit format with per-group scale/minimum metadata.

## Selected compile and route choices

`selected_config.py` is the executable definition. Important values are:

```text
batch=8, capacity=4096, dynamic_context=true
normalization_cache=true, normalization_replay=1
attention_layout=fused-shards, attention_query_tile=4
attention_pair_mode=3, attention_optimization=pv
activation_memtile_col=4, gather_mode=packet, token_packets=true
projection_weight_pingpong=true, chain_pingpong=true, l2_pingpong=false
```

The current router does not infer two ownership constraints required by this
floorplan. `route_artifact.py` checks the exact physical ports and current
arbiter assignments, changes only the two expected assignments, reuses the
compiled core ELFs, resumes AIE compilation, and records before/after IR and
binary hashes. A mismatch causes the route step to fail rather than applying an
ambiguous edit.

## Host memory and timing boundary

The runtime allocates host-visible BOs for activations/taps, packed weights,
RMS/RoPE data, logits, and KV. Target and drafter own independent BOs and KV
caches. Packed weights are uploaded once when each model is constructed.

`run.py` reports the selected prompt-prefill backend and its time separately. Its decode timer starts
after target prefill and initial target KV seeding. It includes the initial and
incremental drafter context projections/uploads, both NPU calls per round,
hidden-tap readback, and CPU acceptance. This is the same boundary used by the
recorded NPU performance studies.

## Correctness invariant

For the target's own B8 arithmetic, row `i` of a batched verification must equal
the result obtained by feeding the same proposed tokens sequentially through row
zero. `verify.py` checks the resulting greedy continuation, committed target KV,
and a repeated DFlash run. It also exhaustively checks the pure host
acceptance/bonus/EOS rule.

B1 uses a GEMV path with a different accumulation order. Bitwise B1/B8 equality
is therefore not the delivery's correctness definition. See
[Validation](docs/validation.md) for the tested boundary.
