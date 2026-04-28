# Qwen2.5-3B deployment — per-phase wall-clock log

Time bookkeeping at PHASE granularity (per user request 2026-04-27).
Each phase: start_ts (epoch s) / end_ts / wall_min /
npu_compile_min / npu_runtime_s / dev_min (≈ wall - compile - runtime,
i.e., agent code-writing + thinking) / notable_events.

For `npu_compile_min` and `npu_runtime_s`, harvest from the per-phase
test script's printed timings (most are stamped already). `dev_min`
is the residual — the cost of writing the per-phase scripts + reading
templates + debugging.

## Baselines

- Deployment session start (deploy-new-llm invoked): **2026-04-27 12:42:13 EDT** (epoch=1777308133)
- Scaffold complete (this file written):              **2026-04-27 12:42:33 EDT** (epoch=1777308153)

Scaffold time (Step 4-6): ~20 s wall (mostly Write tool latency for
6 boilerplate files; near-zero NPU/agent thought).

## Phase log

### Phase 0 — Build CPU Oracle  (PASS, 2026-04-27)

- start_ts:           1777308447  (12:47:27 EDT)
- end_ts:             1777308663  (12:51:03 EDT)
- wall_min:           **3.6**
- npu_compile_min:    0.0   (CPU-only)
- npu_runtime_s:      0.0   (CPU-only)
- hf_download_s:      ~47   (one-time, ~5.9 GB)
- hf_forward_s:       ~5    (model load + 36-layer F32 forward)
- dev_min:            **~2.5**  (mirror qwen25_0_5b's weights+reference, change config defaults only — no debug)
- notable_events:     none — clean pass on first try, all 36 layers cos = 1.0, ' Paris' top-1 match HF

### Phase 1 — Kernel Validation  (PASS, 2026-04-27)

- start_ts:           1777308747  (12:52:27 EDT)
- end_ts:             1777309114  (12:58:34 EDT)
- wall_min:           **6.1**
- npu_compile_min:    ~3.5  (~30s per kernel × ~7 unique compiles)
- npu_runtime_s:      ~10   (each XRTRunner test < 1 s)
- dev_min:            **~2**   (mirror qwen25_0_5b GEMM test, batch CLI calls)
- notable_events:
  - 14/15 PASS standalone (cos 0.994-0.999996); GEMV Down K=11008 deferred to
    Phase 5 (no k_split CLI flag in matvec.py — needs k_split=86 + tile_m=2,
    will use mv_k11008.o per qwen25_1_5b mv_k8960.o pattern)
  - 7 of 14 shapes already covered by llama3 (emb=2048) / llama32_3b (hd=128)
    / qwen25_1_5b (N=256 K/V) — only 7 truly new shapes tested cold
  - W1 risk: same GQA g=8 + n_kv=2 as qwen25_0_5b, but hd=128 here. Paper-
    relevant: tells us if W1 is purely GQA-driven or also hd-modulated

### Phase 2 — Single-Block Validation  (PASS, 2026-04-27)

- start_ts:           1777309355  (13:02:35 EDT)
- end_ts:             1777311228  (13:33:48 EDT)
- wall_min:           **31.2**
- npu_compile_min:    ~12  (multiple compile cycles: 4 successful + 2 BD-blowup compile-fails)
- npu_runtime_s:      ~5
- dev_min:            **~14**  (significant debug — see notable_events)
- notable_events:
  - Initial padded_hidden=11264 + defaults: BD blowup ("Allocator exhausted")
  - 1st fix attempt (herd_m=4 + swiglu_tile_n=704): compiles but runtime HANG
  - 2nd fix (matching herd_m=4 on rms_gemms_rope too): still hangs at seq=2048
  - 3rd fix (seq=512 sanity check): PASS — proves integration is correct, only BD/seq scaling broken
  - **Working config**: padded_hidden=12288 + DEFAULT tile/herd config (no overrides). Mirrors qwen25_1_5b PADDED known-good recipe (which uses 9216 defaults). Cost: 12288/11008=11.6% extra FFN compute (vs 11264's 2.3%).
  - **W1 NOT reproduced**: NPU FA Option C (head-first, hd=128) gives per-pos cos 0.994893 — well above 0.98 gate. qwen25_0_5b's per-pos 0.94 was specific to seq-first FA at GQA-imbalanced shapes. Head-first FA path is precision-clean.

### Phase 3 — Full-Model Validation  (PASS, 2026-04-27)

- start_ts:           1777311307  (13:35:07 EDT)
- end_ts:             1777311660  (13:41:00 EDT)
- wall_min:           **5.9**
- npu_compile_min:    0.0   (kernels cached from Phase 2)
- npu_runtime_s:      ~22   (6 prompts × ~3.65 s NPU prefill 36L)
- cpu_runtime_s:      ~200  (6 prompts × ~33 s CPU reference 36L F32)
- dev_min:            **~2**   (mirror qwen25_0_5b phase3, change shape)
- notable_events:     6/6 top-1 PASS — 4/4 decisive + 2/2 competitive overlap; no NaN; per-layer cos all clean (W1 NOT manifesting at hd=128 Option C as predicted from Phase 2)

### Phase 4 — Prefill Optimization  (PASS, 2026-04-27)

- start_ts:           1777311723  (13:42:03 EDT)
- end_ts:             1777311855  (13:44:15 EDT)
- wall_min:           **2.2**
- npu_compile_min:    0.0   (kernels cached from Phase 2)
- npu_runtime_s:      ~85   (cold 9.2s + preload 3.1s + 5 warm × 5.4s wall)
- dev_min:            **~1**   (mirror qwen25_0_5b phase4)
- notable_events:     5/5 patterns applied/inherited; warm 103 ms/layer (well aligned with predicted ~115 ms based on K-width); Pattern B1 saves 49.7% on cold

### Phase 5 — Decode Optimization  (PASS, 2026-04-27)

- start_ts:           1777311964  (13:46:04 EDT)
- end_ts:             1777312785  (13:59:45 EDT)
- wall_min:           **13.7**
- npu_compile_min:    ~5    (4 successful + 2 failed compile attempts incl. mv_k11008.o)
- npu_runtime_s:      ~30   (4 NPU decode tokens × ~400 ms + LM head preload + cpu-verify)
- dev_min:            **~7**   (Rule C/D iteration: defaults fail → tile_m=16 fails L2 → tile_m=8 m_input=8 + 11 partitions × 13824)
- notable_events:
  - Default tile_m=8 m_input=4 → Rule B fails (Gate/Up M=11008 / 64 × 2 = 344 > 255)
  - tile_m=16 m_input=16 → Rule D L2 fails by exactly C buffer 256B at K=2048
  - **Working config**: tile_m=8 m_input=8 herd_m=8 + 11 × 13824 partitions for LM Head
    (vs 1.5B's 10 × 16384). 11 partitions / 13824 chosen so per-partition launches=216 < 255
    AND L2 fits comfortably at 2048×64×2=256KB.
  - **Stale-cache regression**: 2nd run (after fresh ELF compile, same Python session)
    produced garbage. `make clean` + re-compile fixed. Mirrors qwen25_1_5b's known
    cache-staleness trap.

### Phase 6 — Finalize & Learn  (PASS-with-warnings, 2026-04-27)

- start_ts:           1777312930  (14:02:10 EDT)
- end_ts:             1777314842  (14:34:02 EDT)
- wall_min:           **31.9**
- npu_compile_min:    ~5    (one-time decode kernel compile + multiple recompiles for W2 debug)
- npu_runtime_s:      ~80   (5 make run/verify cycles + W2 fix attempts)
- dev_min:            **~25**  (W2 debug + CPU LM head workaround landing)
- notable_events:
  - **W2 BUG (UNRESOLVED root cause)**: NPU LM Head GEMV via inference.py
    path returns garbage. Phase 5 phase5_test path (CPU prefill seed)
    gives 4/4 NPU/CPU match. Difference: NPU prefill contaminates NPU
    state for subsequent LM head calls.
  - Workaround #1 (failed): reset cache._loaded + re-preload before first
    LM head call. NPU output still garbage.
  - **Workaround #2 (LANDED)**: use CPU LM head for ALL inference.py LM
    head calls (first_token + decode loop). CPU LM head per-call ~20 ms
    (vocab 152k × emb 2048 dot product). **Result: 5/5 NPU/CPU greedy
    match** ✓ Generated text correct: "The capital of France is Paris.
    The capital of"
  - Phase 5 standalone NPU LM head still works (4/4 match) — workaround
    is inference.py-runner-specific, NOT a fundamental NPU LM head bug.

### Phase 7 — Independent Evaluation  (PASS-with-warnings, 2026-04-27)

- start_ts:           1777314958  (14:35:58 EDT)
- end_ts:             1777315875  (14:51:15 EDT)
- wall_min:           **15.3** (subagent wall ~11 min + main agent ~4 min docs)
- npu_compile_min:    0    (cached from Phase 5/6)
- npu_runtime_s:      ~120  (subagent's 2 make run + 1 make verify + 3 phase5_test + 3 adversarial)
- dev_min:            ~4    (main agent docs update)
- notable_events:
  - Subagent verdict: **PASS-with-warnings**
  - Workaround honest (CPU LM head verified real numpy matmul, not hardcoded)
  - make verify 5/5 PASS (cos 0.991, ' Paris' top-1 match)
  - 2/2 reproducibility byte-identical
  - 3/3 adversarial prompts top-1 match (' a', ' de', ' largest')
  - **CRITICAL CORRECTION from evaluator**: Phase 5 standalone NPU LM head
    is **FLAKY** (3 runs: 0/4 → 3/3 → 0/4), NOT "always works" as I had
    claimed. The 11×13824 NPU LM head ELF has a state-dependent bug that
    affects BOTH Phase 5 AND inference.py paths. CPU workaround is the
    only reliable path for production.
  - Documentation drift caught: CLAUDE.md still says hidden=11264 (actual code
    uses 12288); Phase 5 print still says "10×16384" (actual 11×13824).

## Summary table

| Phase | wall_min | npu_compile_min | npu_runtime_s | dev_min | notes |
|---|---:|---:|---:|---:|---|
| Scaffold + Step 0-3 | ~0.3 | 0 | 0 | ~0.3 | 6 boilerplate files (.gitignore, Makefile, README, CLAUDE, TODO, 3 docs) |
| 0: CPU Oracle | 3.6 | 0 | ~5 | ~2.5 | HF download 47 s; mirror qwen25_0_5b weights/reference; clean PASS first try |
| 1: Kernel Validation | 6.1 | ~3.5 | ~10 | ~2 | 14/15 standalone PASS; 7 of 14 shapes already covered by llama3/llama32_3b/qwen25_1_5b |
| 2: Single-Block | 31.2 | ~12 | ~5 | ~14 | **W1-debug heavy**: BD blowup at 11264 → padded 12288 + default tiles works |
| 3: Full-Model | 5.9 | 0 | ~22 | ~2 | 6/6 prompts top-1; mostly CPU reference 36L × 6 (~200 s) |
| 4: Prefill Opt | 2.2 | 0 | ~85 | ~1 | kernels cached; 5/5 patterns; warm 103 ms/layer |
| 5: Decode Opt | 13.7 | ~5 | ~30 | ~7 | **Rule C/D iteration**: tile_m=8 m_input=8 + 11×13824 partition NEW |
| 6: Finalize | 31.9 | ~5 | ~80 | ~25 | **W2-debug heavy**: NPU LM head broken → CPU LM head workaround lands 5/5 PASS |
| 7: Independent Eval | 15.3 | 0 | ~120 | ~4 | subagent 11 min + main agent 4 min docs; refined W2 (Phase 5 also flaky) |
| **Total** | **~110** | **~25** | **~360** | **~58** | ≈ 1.8 hr; ~50% time in W1+W2 debug (Phase 2+5+6+7) |

**Insight from this timing table** (paper-relevant): non-debug phases
(0+1+3+4) total ~17 min — ~15% of total wall. Debug-heavy phases
(2+5+6) total ~77 min — ~70% of total. Reflects that mirror-pattern
deployment (where templates exist) is fast; novel architectural axes
(BD tightness at hidden=11008 / Rule C/D conflict at K=2048+M=11008 /
W2 LM head ELF flakiness) drive cost.
