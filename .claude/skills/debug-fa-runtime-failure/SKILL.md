---
name: debug-fa-runtime-failure
description: Use when NPU FlashAttention hangs (ERT_CMD_STATE_TIMEOUT) or produces NaN at head_dim ≥ 128. Bisects across (n_heads, n_kv_heads, lq=lk, dk) and the .o flag conventions; discriminates the three known root causes (compile-flag mismatch, seq-first dk_chunks bug, true L1 overflow).
---

## Purpose
Captured 2026-04-18 from the llama32_3b Phase 4 NPU-FA debugging session.
NPU FlashAttention failures at head_dim ≥ 128 manifest in three distinct ways
with three distinct root causes. This recipe walks the diagnosis efficiently
instead of bisecting from scratch.

## Knowledge base references (read first)
- `programming_examples/llama32_3b/docs/development_progress/LESSONS.md` Lesson 3 (full root-cause writeup)
- `programming_examples/llama32_3b/docs/development_progress/phase4_prefill.md` (bisect matrix + Option C explanation)
- `programming_examples/flash_attention/kernel_fusion_based/Makefile` (canonical `-D` flag conventions)
- `programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py` vs `.../attn_npu2_seqfirst.py` (the two Python builders)

## Triggers — ALL of these route here:
- `RuntimeError: Command failed to complete successfully (ERT_CMD_STATE_TIMEOUT)` from `cache.load_and_run("flash_attn", ...)` or `XRTRunner.run_test`
- All-NaN output from FA invocation when inputs are well-formed (no NaN/Inf in)
- Compile passes but runtime output is constant garbage (e.g., all `49.0`, `844.0`, growing magnitudes — typical of softmax-not-running)

## Workflow

### Step 1: Identify the failure mode (HANG vs NaN vs garbage)

Run a minimal repro via the standalone harness pattern below at the failing
shape. The output classifies the root cause:

| Symptom | Most-likely root cause | Jump to |
|---|---|---|
| HANG (timeout) at all `dk_chunks > 1` configs but PASS at `dk_chunks = 1` | seq-first `dk_chunks > 1` upstream bug | Step 3 |
| NaN at any config including ones the lit test passes (uniform(0,4) inputs) | `compile_attn_npu2*` flag mismatch (per-launch sizes baked into .o) | Step 2 |
| Garbage non-NaN output (large constant values, no softmax behavior) | `compile_attn_npu2*` flag mismatch — same as NaN, just numerically different | Step 2 |
| HANG at one specific shape but PASS at smaller variants | True L1 overflow at the larger shape | Step 4 |

### Step 2: Verify .o flag conventions (most common — fixes NaN/garbage)

The kernel's `lqp/lkp/dk/dv` defines are **per-tile**, NOT per-launch. The
Makefile's convention is canonical:

```
LQP_TILE := $(shell echo $$(($(LQP) / $(NUM_Q_TILES))))
... -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) \
    -Ddk=$(LKP) -Ddk_full=$(DK) \
    -Ddv=$(LKP) -Ddv_full=$(DV) ...
```

Diff your call against this. The fixed `compile_attn_npu2_split(lqp, lkp,
dk, dv, num_q_tiles=4)` in `_llm_shared/kernel_builder/external_kernels.py`
is correct — derives `lqp_tile = lqp // num_q_tiles` internally.

If your .o was compiled before the fix (commit `6499cae0`-ish), delete the
.o and the cached `flash_attn.elf` and rebuild. Re-run.

### Step 3: Workaround the seq-first `dk_chunks > 1` upstream bug (Option C)

`attn_npu2_seqfirst.py` (the Python builder used by llama3_prefill at
runtime) has an untested `dk_chunks > 1` shim-DMA path that hangs at
runtime. This is upstream — not specific to your model.

Verify via bisect: every `dk_chunks=2` config hangs in seq-first, regardless
of (n_heads/n_kv, lq=lk). The HEAD-first kernel `attn_npu2.py` at the SAME
shape PASSES (e.g., `make run DK=128 DV=128 NUM_HEADS=32 NUM_KV_HEADS=8` →
corr=0.996 PASS).

**Option C — head-first FA + host transposes** (proven on llama32_3b):

1. In `compile_block_kernels`, use `attn_npu2.build_module(...)` instead of
   `attn_npu2_seqfirst.build_module(...)`. Same args, different I/O layout.
2. Monkey-patch `llama3_prefill._run_cached` to intercept `"flash_attn"`
   calls: transpose seq-first inputs to head-first layout, call the
   head-first ELF, transpose output back. **Reusable implementation:**
   `programming_examples/_llm_shared/phase_helpers/headfirst_fa.py` —
   call `install_headfirst_fa_wrapper()` once at module load and
   `compile_headfirst_fa_kernel(cache, seq_len, n_heads, n_kv_heads, head_dim)`
   from your `compile_block_kernels()`. Used by llama32_3b phase tests.
3. Override `_attn_backend_kwargs` to return head-first kwargs:
   `omit_while_true_loop=False`, `runtime_loop_tiling_sizes=[1, 1, 1]` if
   `dv_chunks > 1`, `target_device="npu2"`.

V layout transpose for head-first:
```python
# seq-first (lk, n_kv_heads * dv) → head-first (n_kv_heads * dv_chunks, lk, lkp)
v_hf = np.ascontiguousarray(
    v_seq.reshape(lk, n_kv_heads, dv_chunks, lkp)
         .transpose(1, 2, 0, 3)
         .reshape(n_kv_heads * dv_chunks, lk, lkp)
)
```

Output transpose back is the inverse:
```python
out_packed = results_hf[-1].reshape(n_heads, dv_chunks, lq, lkp)
out_seq = np.ascontiguousarray(
    out_packed.transpose(2, 0, 1, 3).reshape(lq, n_heads * dv)
)
```

Cost: a few ms/layer of host transpose. Gain: NPU FA actually runs
(llama32_3b: 13.6 s CPU-attn → 3.2 s NPU FA = 4.2× warm prefill speedup).

### Step 4: True L1 overflow

If both Step 2 and Step 3 are clean (correct flags, head-first variant) and
the kernel STILL hangs, you're hitting the actual 64 KB per-core L1 limit.
Per-tile budget for FA at `(tile_size_q, lkp, dk_full, dv_full)`:

```
Q tile          : tile_size_q * lkp_per_dk_chunk * 2B
K tile (per dk) : lkp * lkp * 2B            (= 8 KB at lkp=64)
V tile (per dv) : lkp * lkp * 2B            (= 8 KB at lkp=64)
Gp accumulator  : tile_size_q * dv_full * 2B
misc (up,sp,r)  : ~2 KB
```

With `lkp=64`, the per-`dk_chunk` budget is small (~8 KB each) and shared
buffers are off (lkp != dk_full at hd=128). Sum stays well under 64 KB at
typical `tile_size_q ≤ 64`. If you've pushed `tile_size_q` higher to amortize
launch overhead, reduce it.

True L1 overflow is rare for the shapes LLM deployments use. If you suspect
this, drop `lqp` in the Python builder (which reduces `tile_size_q = lqp /
num_q_tiles`) and recompile.

## Reusable bisect harness

Use this template (copied from `/tmp/fa_bisect.py`, adapt configs to your
model). Vary one axis at a time toward the production config — the first
axis that flips PASS → HANG/NaN tells you which dimension is the offender.

```python
def try_config(label, lq, lk, n_heads, n_kv_heads, dk, dv,
               lqp=256, lkp=64, num_q_tiles=4, num_cascade_stages=4,
               causal=True, headfirst=False):
    """Returns 'PASS' / 'HANG' / 'FAIL_NUMERICAL' / 'ERROR: ...'."""
    # ... compile attn_npu2.o via compile_attn_npu2_split with these args
    # ... build module via attn_npu2[_seqfirst].build_module(...)
    # ... random inputs, NumPy reference (causal SDPA), XRTRunner.run_test
    # ... catch TIMEOUT → 'HANG'; numerical mismatch → 'FAIL_NUMERICAL'

# Bisect from a known-good baseline toward production:
results = []
results.append(try_config("baseline_hd64",     lq=512, lk=512, n_heads=32, n_kv_heads=8, dk=64,  dv=64))
results.append(try_config("vary_dk_to_128",    lq=512, lk=512, n_heads=32, n_kv_heads=8, dk=128, dv=128))
results.append(try_config("vary_heads_to_24_8",lq=512, lk=512, n_heads=24, n_kv_heads=8, dk=128, dv=128))
results.append(try_config("vary_seq_to_2048",  lq=2048,lk=2048,n_heads=32, n_kv_heads=8, dk=128, dv=128))
results.append(try_config("production",        lq=2048,lk=2048,n_heads=24, n_kv_heads=8, dk=128, dv=128))
```

## Verification

This recipe is "successful" when the failing FA invocation produces correct
output (per Phase 2's correctness gate at the relevant head_dim threshold)
and runs without hang. Capture the resolution path in
`<model>/docs/development_progress/debug_log.md`.

## Failure mode
If none of Steps 2-4 resolve the failure, this is a new failure mode not
covered by current knowledge — escalate to the user with the bisect matrix
and ask for guidance. Do not silently wrap-fix or further-bisect for hours.

## Update protocol
On successful diagnosis, append a one-paragraph entry to
`<model>/docs/development_progress/debug_log.md` with: date, symptom matrix,
which step resolved it, and the concrete fix (commit hash if applicable).
