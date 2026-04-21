# Qwen3-0.6B deployment lessons

# Qwen3-0.6B deployment lessons

## L1 (Phase 2, 2026-04-20): seq_len-specific ELF cache hygiene

**Symptom**: Phase 2 NPU forward returned cos≈0.0 (uncorrelated) for Q, K, V
outputs of `rms_attn_gemms`, even though the kernel logic was correct.

**Root cause**: Phase 1 compiled `rms_attn_gemms.elf` at seq_len=128 for fast
iteration. Phase 2 then ran at seq_len=512 (FA at hd=128 needs lqp=256 minimum,
so seq_len must divide cleanly). The cache loader reused the stale 128-row ELF
to process 512 rows of input, reading uninitialized memory beyond the
hardcoded MLIR func boundaries.

**Fix**: `make clean` (wipes `prefill_kernel_cache/`) before re-running at a
new seq_len. The cache name `rms_attn_gemms` doesn't encode seq_len; if any
shape-relevant parameter changes, all downstream caches must be invalidated.

**How to apply**: When iterating on phase tests, ALWAYS use the same seq_len
for compile and run. If you need a new seq_len, wipe the cache.

## L2 (Phase 2, 2026-04-20): head-first FA wrapper monkey-patch resolution

**Symptom**: Phase 2 NPU forward returned cos=0.48 (partially correct) when
the test script used direct attribute import for `_run_cached`. With the
identical-looking module-attribute call, the same code returned cos=0.998.

**Root cause**: The head-first FA wrapper
(`_llm_shared/phase_helpers/headfirst_fa.py`) patches FA invocation by
rebinding `llama3_prefill._run_cached` to a wrapper function. A
`from llama3_prefill import _run_cached` snapshots the ORIGINAL function
reference at import time, bypassing the runtime patch.

**Fix**: In per-model phase test scripts, import the module
(`import llama3_prefill as _lp`) and call the patched attribute via
`_lp._run_cached(...)`. Same for `_attn_backend_kwargs`. Wrap with thin
local helpers if the verbose access is annoying.

**How to apply**: Whenever you write a new phase test that invokes
`_run_cached("flash_attn", ...)` AND you've called
`install_headfirst_fa_wrapper()`, route the call through the module
attribute, not a snapshotted import.

## L3 (Phase 3, 2026-04-20): canonical_prompts decisive/competitive is LLAMA-calibrated

**Symptom**: Running Phase 3 with the static decisive/competitive split from
`_llm_shared/phase_helpers/canonical_prompts.py`, two of the four "decisive"
prompts had CPU top-1 prob ≤ 0.5 for Qwen3-0.6B (`Water freezes at` p=0.468,
`The largest ocean is the` p=0.220) and got incorrectly held to the strict
top-1 gate even though they were not actually decisive at this model size.

**Root cause**: The DECISIVE/COMPETITIVE buckets in `canonical_prompts.py`
were measured on Llama-3.2-1B (per the file's docstring). Smaller or
different-family models produce different per-prompt probability
distributions; what's decisive for one is competitive for another.

**Fix**: In Phase 3 test scripts, classify each prompt DYNAMICALLY based on
the OBSERVED CPU top-1 probability (compute `cpu_softmax[cpu_top1]` and
threshold at 0.5). The hardcoded `kind` from canonical_prompts is a
human-readable label only.

**How to apply**: New deployments should NOT trust the static bucket — use
the dynamic classifier. The canonical prompt LIST is still useful as a
representative set; only the bucket assignment is family-dependent.
