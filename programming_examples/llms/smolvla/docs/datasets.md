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
summary. Both the oracle path (`smolvla_cpu_baseline.py`) and the inference
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

### Why this copy is not the one to evaluate on

It is 2-camera / 8-dim state, whose keys (`image`, `image2`) match
`HuggingFaceVLA/smolvla_libero` rather than the `smolvla_base` we ported. Since
`base` accepts any camera count (§4.3) that alone is not disqualifying — but a
2-camera source puts the prefix at 177 instead of 241, which is not what the
published numbers were measured at. §4.5 recommends a 3-camera dataset instead.

These 843 frames remain the fastest way to sanity-check image plumbing without
downloading anything.

---

## 4. Evaluating on real data — the plan

> **This section is the handoff.** It is written for whoever implements the
> real-input evaluation. Everything in it was verified by running the model,
> not by reading code — the verification commands are in §4.7 so they can be
> re-run.

### 4.1 What we are and are not doing

**Doing:** a numerical check. Feed real recorded observations instead of zeros,
compare the NPU action chunk against the CPU action chunk, and report how much
worse (or not) the agreement gets.

**Not doing:** measuring whether the model performs the task. That needs a
simulator and closed-loop rollouts (§5). The model does **not** need to have
been trained or finetuned on the dataset we feed it — `smolvla_base` is a
general pretrained model, it will accept any real observation, and the action
quality being poor is irrelevant to a numerical check.

**Not doing either:** inventing inputs. No duplicated camera feeds, no
synthesised pixels. Renaming a dataset's camera key to the key the checkpoint
expects is *not* invention — it wires a real camera to a real slot.

### 4.2 The one number that matters is a delta

Reporting "cosine on real images = X" is close to meaningless on its own —
there is nothing to compare X against. The output of this work is:

```
Δ = cosine(real images) − cosine(all-zero images)
```

which answers the actual question: **does the BFP16 bias grow when the
activations have realistic dynamic range?** Every real-input run therefore
needs a matching all-zero control at the *same camera count* (§4.4).

Report, for each configuration:

| Metric | Why |
|---|---|
| cosine **distribution** — median, worst, P10 | A single mean hides the tail; the worst frame is the risk |
| **per-dimension** error, worst dimension named | Cosine is scale-blind on a physical output — a limitation already recorded in §2 |
| Δ against the all-zero control | The result. Without it there is no baseline |

Keep the noise pinned (`fixed_noise()` → `torch.zeros`). Flow matching starts
from noise; if it varies, CPU and NPU no longer share a starting point and the
comparison collapses. **This is non-negotiable.**

### 4.3 Camera count does not reach the NPU — verified

The NPU port encodes **one image at a time**. Every ELF is compiled for a
single 512x512 image = seq 1024 (`smolvla_vision_npu.py:24-29`, default
`seq_len=1024`), and `SmolVlaVisionRuntime.encode()` loops over however many
images it is handed (`smolvla_runtime.py:194-197`):

```python
out = np.empty((len(arrs), 64, self.cfg.connector_out), np.float32)
for i, a in enumerate(patch_embeds):
    res = run_vit_encoder(a, ...)
```

The prefix is assembled *after* vision, on the CPU, by lerobot. Grepping the
whole example for `241` / `177` / `113` returns nothing — no prefix length is
hardcoded anywhere.

**Consequence: 1, 2 and 3 cameras all work with the ELFs already built.** The
camera count only changes how many times the host loop runs.

Measured end to end with `smolvla_base` (`predict_action_chunk`, real call):

| Cameras fed | SigLIP passes | Visual tokens | **prefix** | Result |
|---|---|---|---|---|
| 1 | 1 | 64 | **113** | chunk (1,50,6), finite |
| 2 | 2 | 128 | **177** | chunk (1,50,6), finite |
| 3 | 3 | 192 | **241** | chunk (1,50,6), finite |

Only a batch with *every* camera missing raises
(`modeling_smolvla.py:411-414`); a partial set is legal.

### 4.4 Step-by-step

Synthetic first, and not merely as a warm-up — it serves two distinct purposes.
It is a **smoke test** for the claim in §4.3 (which was derived by reading the
loop, and is only *proven* by running 1 and 2 cameras on real hardware), and it
produces the **all-zero control** that §4.2's delta is measured against. The
3-camera control already exists: cosine 0.9990.

| Step | What | Download | NPU |
|---|---|---|---|
| **1** | All-zero batch at 1 / 2 / 3 cameras, CPU vs NPU | no | yes |
| 2 | Pick a 3-camera dataset, fetch a few dozen frames | small | no |
| 3 | Real frames, 3 cameras → Δ₃ | — | yes |
| 4 | Same dataset, feed 1 and 2 of its cameras → Δ₁, Δ₂ | — | yes |
| 5 | Optional: a second dataset, to check the conclusion holds | yes | yes |

**Step 4 deliberately reuses one dataset rather than picking a different
dataset per camera count.** Three datasets differ in scene, resolution and
image statistics, so Δ₁/Δ₂/Δ₃ across them would confound camera count with
dataset identity. Feeding a subset of one dataset's cameras makes camera count
the only variable — and is not invention, since the model natively supports it.

### 4.5 Which dataset

No dataset uses the key names `camera1/camera2/camera3` — those are a
normalisation SmolVLA's pretraining applied across its 481 community datasets,
so **`smolvla_base` has no single "native" evaluation dataset**. Any dataset
needs its camera keys mapped. All of the below were verified by pulling
`meta/info.json`:

| Dataset | Cams | Keys | Resolution | Size | state→action |
|---|---|---|---|---|---|
| `lerobot/pusht` | 1 | `image` | 96² | 206 ep | 2→2 |
| `lerobot/berkeley_mvp` | 1 | `hand_image` | 480x640 | 480 ep | 15→8 |
| `lerobot/aloha_sim_insertion_human` | 1 | `top` | 480x640 | 50 ep | 14→14 |
| `lerobot/libero` | 2 | `image`, `image2` | 256² | 1693 ep | 8→7 |
| `lerobot/libero_plus` | 2 | `front`, `wrist` | 256² | **14347 ep / 2.24M fr** | 8→7 |
| `lerobot/austin_buds_dataset` | 2 | `image`, `wrist_image` | 128² | 50 ep | 24→7 |
| **`lerobot/droid_100`** | **3** | `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left` | 180x320 | 100 ep / 32212 fr | 7→7 |
| **`lerobot/abc_130k_v3_smoke`** | **3** | `top`, `left_wrist`, `right_wrist` | 224² | 85 ep / 313094 fr | 14→14 |
| **`lerobot/aloha_mobile_wipe_wine`** | **3** | `cam_high`, `cam_left_wrist`, `cam_right_wrist` | 480x640 | 50 ep / 65000 fr | 14→14 |

**Recommendation: a 3-camera dataset**, for two reasons — prefix stays 241, so
the numbers are directly comparable to the published cosine 0.9990 / 1.19x per
image; and the NPU carries the largest share of the work, so any BFP16 effect
shows up most strongly.

Between the three, **`lerobot/aloha_mobile_wipe_wine`** is the better default:
at 480x640 it is closest to the 512x512 the model resizes everything to, so it
is upscaled least. Low-resolution sources are upscaled more, which smooths away
exactly the outlier structure that drives the BFP16 bias — `droid_100` at
180x320 is enlarged nearly 3x. `droid_100`'s advantage is scene diversity (47
tasks vs 1) and that it is real-world DROID data.

> If a 1-camera cross-check against a *different* dataset is ever wanted,
> `aloha_sim_insertion_human` (1 cam) and `aloha_mobile_wipe_wine` (3 cam) share
> a robot, a resolution and their state/action dims — the most controlled pair
> available.

**All three 3-camera datasets store images as `video` (MP4), not PNG-in-parquet
like the cached LIBERO copy.** Decoding goes through LeRobot's video path;
confirm the decode dependencies are present before committing to one.
**Unverified — no video-backed dataset has been read on this machine.**

### 4.6 Dimensions do not match, and it does not matter

`smolvla_base` expects 3x `[3,256,256]`, state `[6]`, action `[6]`. None of the
3-camera datasets match:

| | base | droid_100 | abc_130k | aloha_wipe |
|---|---|---|---|---|
| Image | 256² | 180x320 | 224² | 480x640 |
| **state** | **6** | **7** | **14** | **14** |
| action | 6 | 7 | 14 | 14 |

Two mechanisms absorb this, and the result was confirmed by running all four
configurations end to end:

- **Resolution** — `resize_imgs_with_padding=(512,512)` sends everything through
  `resize_with_pad` (`modeling_smolvla.py:418-419`). The NPU always sees seq
  1024 regardless of source resolution.
- **State** — `pad_vector(state, max_state_dim=32)` pads any width to 32 before
  `state_proj[960,32]` (`:474-480`, `:570-574`). The device-side tensor is
  identical.

```
droid_100    img180x320 state[ 7] -> OK chunk(1,50,6) finite=True
abc_130k     img224x224 state[14] -> OK chunk(1,50,6) finite=True
aloha_wipe   img480x640 state[14] -> OK chunk(1,50,6) finite=True
base-native  img256x256 state[ 6] -> OK chunk(1,50,6) finite=True
```

⚠️ State is numerically legal but **semantically meaningless** when a 14-dim
Aloha state is fed to a model trained on 6-dim SO-100 state. Irrelevant to a
numerical check; must be stated plainly in any writeup.

### 4.7 Reproducing the claims above

Camera-count support and dimension tolerance (CPU only, no NPU, ~2 min each on
the lerobot venv at `~/Projects/smolvla_playground/.venv`):

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
cfg = p.config
tok = p.model.vlm_with_expert.processor.tokenizer(
    ["put the bowl on the plate"], padding="max_length",
    max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")
noise = torch.zeros((1, cfg.chunk_size, cfg.max_action_dim))

for n in (1, 2, 3):
    b = {f"observation.images.camera{i}": torch.rand(1, 3, 256, 256)
         for i in range(1, n + 1)}
    b["observation.state"] = torch.zeros(1, 6)
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    p.reset()
    with torch.no_grad():
        print(n, p.predict_action_chunk(b, noise=noise).shape)
```

Camera keys of any dataset, without downloading it:

```python
from huggingface_hub import hf_hub_download
import json
d = json.load(open(hf_hub_download("lerobot/droid_100", "meta/info.json",
                                   repo_type="dataset")))
print({k: v["shape"] for k, v in d["features"].items()
       if v.get("dtype") in ("image", "video")})
```

### 4.8 Where to hook in

`build_oracle_batch(policy, prompt)` in `smolvla_inference.py` is the single
place the batch is constructed, and `verify_adapter.py:90` is its only caller
on the gate path. Both the oracle dumper and the inference path go through it.

**Leave the existing gate untouched.** Its value is precisely its degeneracy:
fully deterministic, fast, reproducible — a regression detector. The real-input
evaluation answers a different question (how much headroom the precision has)
and is a characterisation, not a pass/fail. Add it alongside; do not replace.

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
- **§4.3 / §4.6 were verified on CPU only.** `predict_action_chunk` really was
  run at 1/2/3 cameras and at each dataset's native resolution and state width,
  but with `torch.rand` images and **never through the NPU path**. That 1 and 2
  cameras work on device is an inference from the host loop, which is exactly
  what step 1 of §4.4 exists to prove.
- Random images are not natural images. §4.3/§4.6 establish shape
  compatibility, not anything about precision on real data.
- No video-backed (MP4) dataset has been read on this machine; all three
  3-camera candidates are video-backed.
- Dataset sizes and camera keys in §4.5 come from `meta/info.json` only — no
  frames of those datasets have been downloaded or decoded.
- §5 has not been run, and neither has anything past step 1 of §4.4.

## Sources

Local data: `~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/` — parquet
schema, row counts, decoded PNG statistics, and episode metadata read directly.
Batch construction: `smolvla_cpu_baseline.py`, `smolvla_inference.py:76-96`.
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
