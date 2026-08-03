# Datasets for SmolVLA on NPU

What SmolVLA was trained on, what this repository actually feeds it, and what
real data is available if we want to change that.

The reason this document exists: our `make verify` gate compares the NPU action
chunk against the CPU model on a **synthetic all-zero batch**. That is the
correct choice for a numerical gate, and this document explains why — but it
also states precisely what that gate can and cannot claim, and what it would
take to close the gap.

Everything about the local data was measured by opening the files, not read off
a model card.

---

## 1. What SmolVLA was trained and evaluated on

**Pretraining (`lerobot/smolvla_base`):** 481 Hugging Face community datasets,
**22.9K trajectories / 10.6M frames**, predominantly the low-cost SO-100 arm.
Community data is heterogeneous — varying embodiments, camera viewpoints, and
non-expert demonstrations — so the authors normalised ambiguous task strings
with Qwen2.5-VL-3B-Instruct and hand-mapped inconsistent camera-viewpoint
naming.

**Published evaluation:**

| Benchmark | Result |
|---|---|
| LIBERO | **87.3%** average success (0.45B, matching or beating 3.3B pi-0) |
| Meta-World | beats diffusion policies and smaller VLAs across difficulty tiers |
| Real SO-100 | **78.3%** across pick-place, stacking, sorting |
| SO-101 transfer | holds up despite training only on SO-100 data |

**The pretraining ablation is the headline number:** without community-data
pretraining, SO-100 success is 51.7%; with it, 78.3% — **+26.6 points**.

The authors' own stated limitation: pretraining is dominated by a single robot
type, and 23K trajectories is far below the ~970K-trajectory regime of models
like OpenVLA.

> Worth noting: the official checkpoint list (metaworld, robocasa, robotwin,
> vlabench, robocerebra, libero, libero_plus — see `model_family.md`) implies
> considerably broader benchmark coverage than the blog post describes.

---

## 2. What this repository actually feeds the model

**A synthetic fixed batch.** Confirmed by reading the code, not by trusting a
summary. Both the oracle path (`smolvla_prefix.py:50-64`) and the inference
path (`smolvla_inference.py:76-88`) call the same `build_batch`:

```python
for k, f in cfg.input_features.items():
    b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
tok = ...tokenizer(["pick up the cube"], padding="max_length",
                   max_length=cfg.tokenizer_max_length, ...)
```

Precisely characterised:

| Input | Value |
|---|---|
| `observation.images.camera1/2/3` | **`torch.zeros`** (3,512,512) each — not noise, exactly zero |
| `observation.state` | **`torch.zeros`** (6,) |
| Language | `"pick up the cube"`, padded to `tokenizer_max_length` = **48** |
| Flow-matching noise | **`torch.zeros`** (`fixed_noise()`), plus `torch.manual_seed(0)` |

The oracle is written to `smolvla_oracle.npz`; the gate compares the (1,50,6)
action chunk.

### Why this is the right choice

Determinism is the entire point. With all-zero images and zero noise, **the
only difference between the NPU and CPU runs is the arithmetic itself** — so
cosine 0.9990 attributes cleanly to BFP16. Feeding real images would introduce
variance we do not want in a numerical gate. This is not a defect.

### The one real consequence

The `"pick up the cube"` prompt is ~4 real tokens padded to 48, which is why
**only 197 of the 241 prefix tokens are real**. The real model excludes that
padding from attention (`make_att_2d_masks`' `pad_2d_masks` term) and freezes
`position_ids` at the last real position. That padding mask is exactly what the
registry FlashAttention cannot express — it is the direct cause of the backbone
and expert falling back to 11 dispatches per layer instead of 1.

So the synthetic prompt is not merely a test fixture; **its padding shape is
load-bearing for the performance story.** A longer real instruction would
change the mask and the dispatch count.

### What the gate can and cannot claim

It proves **port fidelity**: the NPU did not change the model's behaviour. It
does not prove the model is good at the task — the reference is the CPU model
itself, so any error the CPU model makes is faithfully reproduced.

Three further limits, in decreasing order of how cheaply they can be closed:

1. **Single synthetic input.** All-zero images mean SigLIP encodes a constant
   image: all 1024 patch tokens are identical before the positional embedding.
   This does not affect shapes or latency, but it does mean the activation
   dynamic range is unrepresentative. Since the BFP16 bias we characterised is
   driven by **outlier-heavy activations** (post-LayerNorm max/mean-abs ~45),
   **the 0.9990 cosine is not guaranteed to reproduce on real imagery.** This
   is the cheapest gap to close — see §4.
2. **Cosine is scale-blind** on a physical output. It should be reported
   per-dimension with the worst dimension named.
3. **No mapping from action error to task success.** Nobody knows what cosine
   0.9990 costs in success rate. This is a gap in the VLA field generally, not
   an oversight here.

---

## 3. The LIBERO data already on this machine

`~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/` — LeRobot v3.0 format.

### What the metadata declares

```
codebase_version = v3.0     robot_type = panda      fps = 10
total_episodes   = 1693     total_frames = 273465   total_tasks = 40
observation.images.image   [256,256,3]
observation.images.image2  [256,256,3]      <- 2 cameras
observation.state          [8]
action                     [7]
```

40 real LIBERO instructions: *"put the bowl on the plate"*, *"turn on the stove
and put the moka pot on it"*, *"open the middle drawer of the cabinet"*, *"pick
up the black bowl between the plate and the ramekin and place it on the
plate"*, ...

### What is actually on disk — a partial download

Opening the parquet rather than trusting the metadata:

| | Declared | **Present locally** |
|---|---|---|
| Episodes | 1693 | **3** (indices 0, 1, 2) |
| Frames | 273465 | **843** |
| Data files | **69** (`data/chunk-000/file-000..068`) | **1** (`file-000.parquet`, 97 MB) |

The episode metadata (all 1693 rows, 996 KB) is fully present and references 69
data files at ~15 episodes / ~4100 frames each; only the first was fetched.
**Full dataset is therefore roughly 69 x 97 MB ~ 6.5 GB** (estimated by
extrapolation, not measured).

> Correcting an earlier note of mine: I previously described this as "a
> complete real LIBERO dataset already downloaded." The *metadata* is complete;
> the *data* is 3 episodes out of 1693.

### The 843 local frames are genuinely usable

Images are **PNG-encoded inline in the parquet** (`struct<bytes, path>`), no
separate `videos/` directory. Verified by decoding frame 0:

```
observation.images.image :  PNG 256x256 RGB uint8   min 0  max 242  mean 69.2  std 41.0
observation.images.image2:  PNG 256x256 RGB uint8                   mean 68.7  std 32.2
observation.state[0] = [-0.0534, 0.0070, 0.6783, 3.1408, 0.0018, -0.0899, 0.0388, -0.0388]
action[0]            = [0.0161, 0, -0, 0, 0, -0, -1]
```

Three episodes, 214 / 284 / 345 frames, tasks "put the white mug on the left
plate...", "put the white mug on the plate...", "put the yellow and white mug
in the microwave and close it".

**843 real frames with std ~41 is more than enough to answer the question in
§2's limit 1**, which needs tens of frames, not thousands.

### The embodiment mismatch

This dataset is 2-camera / 8-dim state, matching `HuggingFaceVLA/smolvla_libero`
(prefix ~177, 32 layers, expert 480/1280) — **not** the `smolvla_base` we
ported (3-camera / 6-dim state / prefix 241). Two ways around it, §4.

---

## 4. Closing the cheapest gap

The lowest-cost improvement to the gate — no simulator, no robot, no GPU, and
now no download either.

**What:** replace `build_batch`'s `torch.zeros` images with real frames from the
local parquet, keep the noise pinned to zero, run ~20 frames, and report the
**distribution** of NPU-vs-CPU cosine (median and worst) plus **per-dimension**
error with the worst dimension named.

**Handling the embodiment mismatch** — two options:

| Option | What it costs | What it buys |
|---|---|---|
| **(a) Borrow images only.** Feed the two real camera images into base's first two camera slots, third slot zero or a duplicate; keep state zero. | ~10 lines. Nothing recompiles. | Not a physically meaningful observation, but it puts **real natural-image activation statistics** through SigLIP — which is the entire question. |
| **(b) Switch to `HuggingFaceVLA/smolvla_libero`.** | Physically coherent, but prefix 241 -> 177 forces an ELF rebuild, and the 32-layer / 480-width expert invalidates the registry's expert GEMM rows. | A genuinely valid observation. |

**Recommendation: (a).** The question being asked is "does the BFP16 bias grow
under realistic activation dynamic range?" — that needs realistic pixels, not a
coherent robot state. Option (a) answers it for a few lines of code, and closes
limits 1 and 2 from §2 at once.

This still does not measure task quality. That is §5.

---

## 5. Measuring task fidelity — what it would actually take

### How the field does it

| Benchmark | Standard protocol | Episodes |
|---|---|---|
| **LIBERO** | 4 suites x 10 tasks x 50 episodes | **2000** (500 per suite) |
| **CALVIN** | ABC->D, 1000 chained sequences | 1000 |
| **SimplerEnv** | 4 WidowX tasks x 24 episodes | **288** (3 seeds) — cheapest |
| **Meta-World** | tiered by difficulty | varies |

LIBERO's success criterion usually requires the goal predicate to hold for >=10
consecutive timesteps, to avoid spurious credit.

**Publishable levels:** SmolVLA reports LIBERO 87.3% and real SO-100 78.3%;
current SOTA on LIBERO is >95%.

**Cost:** the simulator, not model inference, is the bottleneck, because these
benchmarks run a single environment instance. AI2's `vla-evaluation-harness`
uses episode sharding to take LIBERO from **14 hours to 18 minutes (47x)** and
ships a `smoke_test.yaml`.

> **A caveat the field itself raises:** LIBERO is close to saturated and may be
> measuring the wrong thing. LIBERO-PRO found models scoring >90% on standard
> LIBERO nearly collapse under object-position changes or minor task rewording
> — the high scores largely reflect memorisation on near-identically
> distributed train/test splits. So "we scored 87% on LIBERO" is weaker
> evidence than it looks.

### Options for us, cheapest first

| Tier | What | Cost | What it buys |
|---|---|---|---|
| **0** | **Real-input port fidelity** (§4) | ~1 day, no new infra | Upgrades "0.9990 at one synthetic point" to "port fidelity over a real activation distribution, per-dimension". **Strict superset of the current gate.** |
| **1** | Offline action-matching vs logged actions | low | Compares against *external* ground truth rather than the model itself. **But it does not predict success rate** — SmolVLA's 51.7%->78.3% are rollout numbers with no offline equivalent, and a different-but-valid action is penalised. |
| **2** | SimplerEnv, 288 episodes | simulator setup | Cheapest true task-fidelity number — **but WidowX, while base is SO-100-pretrained**, so the embodiment mismatch makes a poor result unattributable. |
| **2b** | **LIBERO single suite (500 ep) + `HuggingFaceVLA/smolvla_libero`** | simulator setup + ELF rebuild | The only realistic path to a **publishable, comparable** number: "NPU SmolVLA scores X% on LIBERO-Object, CPU scores Y%". |

**Suggested order: do tier 0 now; treat 2b as the target if we ever need to
claim task fidelity. Skip tier 1** — it costs about what tier 0 costs, its
conclusion is easier to attack, and what it measures is softer than 2b's.

---

## 6. Dataset reference

### Directly usable

| Dataset | Size | Format | Note |
|---|---|---|---|
| **`HuggingFaceVLA/libero`** | 1693 ep / 273465 frames / 40 tasks; panda, 10 fps, 2x 256^2 | LeRobot v3.0, PNG-in-parquet | **3 episodes / 843 frames cached locally**; full set ~6.5 GB across 69 files |
| `lerobot/svla_so101_pickplace` | 50 ep / 11939 frames / 1 task; SO-100, 30 fps, 6-DoF | LeRobot v3.0 (v2.1 via `revision=`) | `StreamingLeRobotDataset` needs no download; **closest to base's SO-100 training distribution** |
| `lerobot/libero-assets` | — | — | Simulator assets, needed for rollouts |

### Other LIBERO copies

| Dataset | Downloads | Note |
|---|---|---|
| `physical-intelligence/libero` | 37001 | pi-0 team's copy, most downloaded |
| `yifengzhu-hf/LIBERO-datasets` | 28259 | from the LIBERO authors |
| `Sylvest/libero_plus_lerobot` | 26333 | corresponds to `smolvla_libero_plus` |
| `openvla/modified_libero_rlds` | 11838 | RLDS, as used by OpenVLA |
| `IPEC-COMMUNITY/libero_{goal,spatial,object,90,10}_no_noops_*_lerobot` | ~6-8K each | **split per suite** — the right choice for running one suite |
| `zhouxueyang/LIBERO-Pro`, `RLinf/LIBERO-PRO-assets` | 2546 / 2233 | the robustness variant discussed in §5 |

### Large-scale real-robot corpora (not needed now)

| Dataset | Size | Format | License |
|---|---|---|---|
| Open X-Embodiment | ~1M+ episodes, 22 robots, 527 skills; v1.1 ~8964 GB | RLDS / TFRecord | code Apache-2.0, data CC-BY-4.0 |
| DROID | 76000 demos, 350 h, Franka, 564 scenes | TFDS + HF, LeRobot-compatible | MIT + CC-BY-4.0 |
| BridgeData V2 | 60096 trajectories, WidowX, 24 environments | RLDS | CC-BY-4.0 — use the RAIL Berkeley copy, the OXE bucket one is stale |
| RoboMIND | 55000 trajectories, 4 embodiments, +5000 failure cases | multi-view RGB-D | **unverified** |

---

## 7. Unverified

- The **~6.5 GB** full-dataset size is extrapolated from one 97 MB file x 69,
  not measured.
- Streaming behaviour of `StreamingLeRobotDataset` on this dataset was not
  tested; only local files were opened.
- LIBERO / Meta-World / SO-100 success rates are quoted from the SmolVLA
  paper and blog — **not reproduced here**.
- RoboMIND's license was not confirmed.
- Tier-2b's claim that the ELF rebuild is the main cost assumes the 177-token
  prefix places without new tiling problems. Not attempted.
- Nothing in §4 or §5 has been run. They are proposals.

## Sources

Local data: `~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/` — parquet
schema, row counts, decoded PNG statistics, and episode metadata read directly.
Batch construction: `smolvla_prefix.py:50-64`, `smolvla_inference.py:76-96`.
Padding-mask mechanism: `modeling_smolvla.py` / `make_att_2d_masks`, and this
example's own measurements (see `docs/profile.md`). Training and evaluation
figures: [SmolVLA blog](https://huggingface.co/blog/smolvla),
[arXiv:2506.01844](https://arxiv.org/abs/2506.01844). Benchmark protocols and
costs: [LIBERO-PRO](https://arxiv.org/html/2510.03827v1),
[vla-eval harness](https://arxiv.org/pdf/2603.13966),
[allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness).
Dataset sizes/licenses: [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment),
[LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).
Download counts: HF Hub API, August 2026.
