# Validation and measured scope

## What is guaranteed by the selected checkpoint

The selected target and drafter passed exact logits, hidden-state, and KV-cache
comparisons at 16 histories from position 0 through 4,088, including partial
attention tiles and poisoned unused cache tails. Additional gates covered target
causality and rollback, context-KV injection, 100 alternating target/drafter
full-state replays, repeated execution, and committed speculative loops.

These checks compare against the selected B8 arithmetic. They do not claim
bitwise equality with the separate B1 GEMV implementation, whose accumulation
order differs.

`make verify` is the compact delivery gate. It checks:

1. All artifacts match their manifest hashes and the fixed selected config.
2. Host acceptance, correction/bonus, clipping, and EOS behavior.
3. A DFlash continuation equals sequential row-zero execution of the same B8
   target arithmetic.
4. The committed target KV prefix is exact.
5. A second reset/run produces the same output and non-timing acceptance trace.

## NPU prefill

Live `run`, `chat`, and `verify` now default to the standard Q4NX NPU prefill
with a 2,048-row padded shape. Eight prefill kernel programs were freshly built
using the delivery toolchain. The adapter captures the existing layer outputs
at boundaries 2, 10, 18, 26, and 34 and exports only real prompt KV/tap rows.
The B8 target/drafter kernels and acceptance logic are unchanged.

The focused gate checks repeated NPU prefill K/V/taps and first-token selection,
then exact DFlash continuation against sequential row zero from the **same
NPU-prefilled state**, repeated DFlash output/trace, and committed target KV.
These checks passed for the default Paris chat prompt and a 64-token
speculative-decoding explanation (29 verification rounds). NPU prefill took
5.59 s and 5.47 s respectively; it uses the full padded shape even for these
19- and 17-token prompts.

The shared prefill has a numerical boundary that the gate records explicitly.
Its attention PV matrix product converts V operands to BFP16 with shared
exponents across groups of eight key positions. Adding future positions inside
a partially filled group can change the rounded representation of older V
values in that group, even though their future attention probabilities are
masked. Exact equality between arbitrary prompt-prefix runs is therefore not
claimed for this prefill arithmetic.

In the 19-token probe, appending three tokens left the first layer's K/V exact;
subsequent differences were confined to positions 16–18. Whole-array cosine
similarities were 0.999984 for K, 0.999833 for V, and 0.9999996 for taps. Raw
maximum absolute differences were 1.0, 1.07837, and 3.0 respectively. All earlier
complete eight-position groups were exact. Extending a complete 64-position
attention tile preserved the full K/V/tap prefix exactly. These probes support
the observed block-quantization explanation; they do not replace model accuracy
evaluation or imply bit equality with CPU prefill.

The benchmark below continues to use the original frozen CPU-prefilled states.
It measures the same decode workload as before. A five-prompt, 256-token-cap
smoke run through the delivered benchmark command matched all five historical
output prefixes exactly. The summary command reproduced 34.998886 token/s
from all 80 historical records. A new broad study of live NPU
prefill, output/acceptance differences, and total request latency is separate
from reproducing those published decode numbers.

## Streaming and resident chat

The streaming CLI uses the existing post-acceptance observer. The prefill token
and each committed round are streamed; proposed tokens that fail verification
never enter the text stream. The selected NPU programs and acceptance rules are
unchanged, and existing compiled artifacts can be reused.

The original CPU-prefill frontend update was checked on the device with a 64-token standalone run
and two prompts in one resident chat session. Streamed token IDs matched the
saved non-streaming result exactly; the cached-prefill chat run reproduced the
standalone output. Captured stdout arrived incrementally before final statistics.
The session loaded each model once, ignored blank input, rejected an oversized
request before prefill, continued successfully, and exited with `/quit`.

Host checks cover split UTF-8 characters, whitespace, early EOS, and final
partial characters. Additional local checks used the real Qwen tokenizer on
multilingual text at chunk sizes 1, 2, 7, and 8. The 440 acceptance-rule checks
also pass.

For that CPU-prefill check, chat initialization took 42.16 seconds and retained
8.22 GiB of decoded CPU prefill weights. The two short prompts then took
3.04 and 2.95 seconds for CPU prefill; the same first prompt took 40.47 seconds
in a standalone run that prepared its weights during prefill. These are frontend
smoke measurements, separate from the broader decode benchmark below.

## Broader performance result

The selected pair was measured once on 80 deterministic seed-0 prompts: 16 each
from GSM8K, MATH-500, HumanEval, MBPP, and first-turn MT-Bench. Generation was
greedy, thinking disabled, C4096, with a 2,048-token cap.

| Metric | B1 | Selected DFlash B8 |
|---|---:|---:|
| Pooled decode throughput | 19.311 token/s | 34.999 token/s |
| Pooled speedup | 1.000x | 1.812x |
| Equal-request throughput | 19.503 token/s | 34.256 token/s |

The unweighted mean dataset-level equal-request speedup is 1.864x. The paired,
dataset-stratified prompt-bootstrap interval is 1.696–1.939x for pooled speedup.
Macro mean acceptance length is 4.579; accepted/proposed drafts are 47.53%, and
the loop emits 4.311 tokens per verification round.

The prior four-prompt-per-dataset cohort measured 39.191 DFlash token/s and
2.049x speedup. In the broader cohort, B1 remains stable and DFlash round cost is
slightly lower, while emitted tokens per round fall 13.57%. The broader result
therefore replaces the small cohort as the more representative local estimate.

## Clean reviewer-checkout reproduction

The reviewer branch was rebuilt and measured from an independent clone on
September 15. Both packed model caches reproduced the historical files exactly;
the rebuilt target and drafter each matched all 30 per-core ELF hashes, as well
as the selected configuration, ABI, and transaction headers. AIR was built from
this branch with both reused and newly installed upstream dependencies. Six
focused compiler regressions and the host/device gates passed.

The full one-pass paired study completed in 48.38 minutes:

| Metric | Historical study | Clean reviewer checkout |
|---|---:|---:|
| B1 pooled throughput | 19.3107 token/s | 19.3062 token/s |
| DFlash pooled throughput | 34.9989 token/s | 35.0198 token/s |
| Pooled speedup | 1.8124x | 1.8139x |
| Macro equal-request speedup | 1.8640x | 1.8679x |
| Mean request acceptance length | 4.5788 | 4.5788 |
| Accepted/proposed drafts | 47.5328% | 47.5328% |

All 80 B1 outputs match the historical B1 outputs, and all 80 DFlash outputs
match the historical DFlash outputs. All 8,002 DFlash round records are identical
after removing their timing fields, including proposals, verifier token choices,
acceptance, emitted tokens, and EOS behavior. This comparison is within each
mode; it does not assert B1/B8 output equality.

The benchmark ran at source `7945cfda944c9cf8e3c04359ad43ae18e641e195`.
Later release changes add documentation, the measured summary, and more complete
runtime-version reporting; inference code and the benchmark protocol are unchanged.
See [the measured summary](../benchmark/reproduced_summary.json) and the release's
reproduction-evidence archive for all request records and machine identities.

## Comparison boundaries

- The NPU target and drafter use Q4NX packed weights, BF16 activations, and
  BFP16 matrix arithmetic. The paper uses BF16 models on H200 and block size 16.
- This delivery executes B8. The public draft was trained at B16 and supports a
  shorter inference block, but acceptance is lower than the paper's B16 result.
- Loading, CPU prompt prefill, first-token generation, initial target KV upload,
  and warmup are excluded. CPU learned context projection and drafter KV upload
  are included in decode time.
- The 80-prompt study measures performance and acceptance. It does not score
  mathematical answers, execute generated code tests, include AIME25 or
  LiveCodeBench, or evaluate the second MT-Bench turn.
- Prompt-bootstrap intervals measure prompt variation. The one-pass study does
  not include independent device-repeat variance.
