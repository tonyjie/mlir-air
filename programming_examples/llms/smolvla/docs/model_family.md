# The SmolVLA model family

Reference notes on what SmolVLA checkpoints exist, how they differ, and what
those differences mean for the NPU port. Written to answer one question: **if we
swap checkpoints, what has to change on our side?**

Short answer: for the eight official checkpoints, nothing. For
`HuggingFaceVLA/smolvla_libero`, the backbone and expert ELFs must be rebuilt.

All numbers below were read from the actual checkpoints (safetensors headers,
`config.json` pulled from the Hub) and from the LeRobot source, not from papers
or model cards. Layer counts are the max `layers.<i>` index found in each
checkpoint's weight names.

---

## 1. SmolVLA is a template, not a fixed architecture

SmolVLA is cut out of the `HuggingFaceTB/SmolVLM2-500M` base by four config
knobs. The whole mechanism is `smolvlm_with_expert.py:88-106`:

```python
# knob 1: truncate the VLM
if num_vlm_layers > 0:
    text_model.layers = text_model.layers[:num_vlm_layers]
self.num_vlm_layers = len(text_model.layers)

# knob 2: expert width = VLM width x multiplier
lm_expert_config.hidden_size       = int(960 * expert_width_multiplier)
lm_expert_config.intermediate_size = get_intermediate_size(...)

# knob 3: expert depth (defaults to the VLM's)
lm_expert_config.num_hidden_layers = self.num_vlm_layers
if num_expert_layers > 0:
    assert num_vlm_layers % num_expert_layers == 0
    lm_expert_config.num_hidden_layers = num_expert_layers
```

| Knob | Meaning | Shape it controls |
|---|---|---|
| `num_vlm_layers` | **>0 = keep first N layers; <=0 = keep all 32** | backbone depth (= expert depth by default) |
| `expert_width_multiplier` | expert hidden = 960 x mult | **every expert GEMM width** |
| `num_expert_layers` | expert depth (<=0 = same as VLM) | expert dispatch count |
| camera count + `empty_cameras` | visual token count | **backbone prefix sequence length** |

> ### `num_vlm_layers: 0` does not mean zero layers
>
> It means *do not truncate* — use all 32 SmolLM2 layers. This is the single
> most confusing thing about the family, and it is the reason two checkpoints
> both named "smolvla_libero" have 16 and 32 layers respectively. See §4.

The three stages differ in how mutable they are:

| Stage | Origin | Varies across the family? |
|---|---|---|
| **(1) SigLIP vision** | from SmolVLM2, `freeze_vision_encoder=True` | **Never.** Always 12 layers / 768 / 3072 / seq 1024 |
| **(2) SmolLM2 backbone** | SmolVLM2's `text_model`, truncated | Depth varies (16 or 32). **Width is always 960 / 2560** |
| **(3) Action expert** | **initialised from scratch** | Depth *and* width both vary |

The vision encoder being frozen is why it is identical in every checkpoint —
not just the same shape, but the same weight values.

---

## 2. The base model, annotated

```
input: 3 cameras (3,512,512) + state (6-dim) + text "pick up the cube"
  |
  +-- (1) SigLIP ViT, run once per camera
  |      Conv2d patch16 -> 1024 tokens -> 12 layers
  |      hidden 768, MLP 3072, 12 MHA x 64, bidirectional / no mask
  |      -> (1024, 768) per camera
  |
  +-- (1b) Connector
  |      pixel-shuffle x4 (pure reshape) -> (64, 12288)
  |      -> Linear[960,12288] -> (64, 960)          = 64 visual tokens/camera
  |
  +-- (2) prefix = 3x64 visual + 48 language + 1 state = 241 tokens
  |      state: pad 6->32 -> state_proj[960,32] -> (1,960)
  |
  +-- (3) SmolLM2 backbone, 16 layers
  |      hidden 960, MLP 2560, GQA 15/5, head_dim 64
  |      non-causal prefix-LM mask
  |      -> per-layer KV cache (not a hidden state)
  |
  +-- (4) Action expert, 16 layers, x10 denoise steps
         hidden 720 (= 960 x 0.75), MLP 2048, q_proj[960,720]
         action: pad 6->32 -> action_in_proj[720,32] -> (50,720)
         layers alternate (self_attn_every_n_layers=2):
           even layer -> self-attention among the 50 action tokens
           odd  layer -> cross-attention, K/V from the backbone's 241-token cache
         -> action_out_proj[32,720] -> (50,32) -> sliced back to (50,6)
```

`self_attn_every_n_layers = 2` is where the even/odd alternation comes from. It
is 2 in all nine checkpoints.

---

## 3. Every checkpoint, measured

Layer counts and tensor shapes read from each `model.safetensors` header.

### Group A — the eight official checkpoints: identical architecture

All are 450.0M params, `vlm=16`, `vision=12`, `expert=16`, expert hidden 720,
expert MLP 2048, `state_proj=[960,32]`, `action_in_proj=[720,32]`.

| Checkpoint | Downloads | Cameras | **prefix seq** | Robot (state->action) | Trained/eval'd on |
|---|---|---|---|---|---|
| **`lerobot/smolvla_base`** | 69408 | 3 | **241** | SO-100 (6->6) | 481 community datasets — **what we ported** |
| `lerobot/smolvla_libero` | 17282 | 3 | 241 | Franka (6->7) | LIBERO |
| `lerobot/smolvla_libero_plus` | 501 | **5** (3 real + 2 empty) | **369** | Franka (6->7) | LIBERO-Plus |
| `lerobot/smolvla_metaworld` | 839 | 3 | 241 | (6->4) | Meta-World |
| `lerobot/smolvla_robocasa` | 532 | 3 | 241 | mobile manip (6->12) | RoboCasa |
| `lerobot/smolvla_robotwin` | 327 | 3 | 241 | **bimanual** (6->14) | RoboTwin |
| `lerobot/smolvla_vlabench` | 260 | 3 | 241 | (6->7) | VLABench |
| `lerobot/smolvla_robocerebra` | 188 | **4** (3 + 1 empty) | **305** | (6->7) | RoboCerebra |

Only `smolvla_base` is genuinely pretrained; the other seven are finetunes of
it (Hub tag `base_model:finetune:lerobot/smolvla_base`). Because the vision
encoder is frozen, **their SigLIP weights are bit-identical to base's** —
finetuning only moved the backbone and expert.

**The only NPU-relevant difference among these eight is the prefix sequence
length: 241 / 305 / 369.**

### Group B — `HuggingFaceVLA/smolvla_libero`: a different architecture

| | base | HFVLA/libero |
|---|---|---|
| Params | 450.0M | **604.9M** |
| `num_vlm_layers` in config | 16 | **0** (= no truncation) |
| **backbone layers** | 16 | **32** |
| **expert layers** | 16 | **32** |
| `expert_width_multiplier` | 0.75 | **0.5** |
| **expert hidden** | **720** | **480** |
| **expert MLP** | **2048** | **1280** |
| expert `q_proj` | [960, **720**] | [960, **480**] |
| `action_in_proj` | [**720**, 32] | [**480**, 32] |
| Cameras | 3 | **2** |
| **prefix seq** | **241** | **~177** |
| backbone hidden / MLP | 960 / 2560 | **960 / 2560 (same)** |
| vision | 12 layers, 768/3072 | **identical** |
| VLM base | SmolVLM2-500M-**Video**-Instruct | SmolVLM2-500M-Instruct |

This is the checkpoint with real LIBERO ground truth — the one
`eval_vs_groundtruth.py` in the playground measured at 85.9% action-MSE
improvement. Its dataset is already on this machine (see §6).

### Group C — community

`nvidia/smolvla-arena-gr1-microwave`: architecture matches base (16 layers,
multiplier 0.75) but **1 camera** -> prefix ~113. GR1 humanoid, state 54 ->
action 36.

---

## 4. Two checkpoints named "smolvla_libero"

This trips people up, so stating it plainly:

| | `lerobot/smolvla_libero` | `HuggingFaceVLA/smolvla_libero` |
|---|---|---|
| Publisher | LeRobot official | HuggingFaceVLA org |
| Params | 450.0M | **604.9M** |
| VLM layers | **16** | **32** |
| `num_vlm_layers` in config | `16` | `0` |

Both are real; they are different models. The official one is architecturally
identical to base. The HuggingFaceVLA one is the 32-layer variant.

> The HuggingFaceVLA checkpoint also carries a `finetune of smolvla_base` Hub
> tag, which cannot be literally true — base has only 16 backbone layers, so no
> finetune of it yields 32. It most likely reloaded all 32 layers from
> SmolVLM2-500M and trained the expert from scratch. **Unverified**; the tag is
> probably just imprecise.

---

## 5. `state -> action` does not reach the NPU

The per-robot state/action dimensions in the table above are useful for
identifying which robot a checkpoint targets, and for nothing else. Every
state and action vector is padded to a fixed 32 before it touches the network
(`modeling_smolvla.py:474-480`):

```python
state   = pad_vector(state,   self.config.max_state_dim)    # max_state_dim  = 32
actions = pad_vector(actions, self.config.max_action_dim)   # max_action_dim = 32
```

and the projections are sized from that constant, not from the real dimension
(`:570-574`):

```python
self.state_proj      = nn.Linear(max_state_dim,   960)   # 32 -> 960
self.action_in_proj  = nn.Linear(max_action_dim,  720)   # 32 -> 720
self.action_out_proj = nn.Linear(720, max_action_dim)    # 720 -> 32
```

Confirmed against the weights: `state_proj` is `[960, 32]` in all nine
checkpoints. The output is sliced back at `:296`
(`actions = actions[:, :, :original_action_dim]`).

So RoboTwin's 14-dim bimanual action and Meta-World's 4-dim action produce
**identical tensor shapes on device**.

### Empty cameras still run a full SigLIP forward

`modeling_smolvla.py:434-438`:

```python
for num_empty_cameras in range(len(missing_img_keys)):
    if num_empty_cameras >= self.config.empty_cameras: break
    img  = torch.ones_like(img) * -1     # a fake all -1 image
    mask = torch.zeros_like(mask)        # masked out of attention
    images.append(img)
```

The image is masked out of attention, but all 12 ViT layers still execute on
it. So `libero_plus` runs vision **five** times per inference, not three, and
`robocerebra` four.

This matters to us: vision is the only stage we win on, so a checkpoint with
more cameras raises the share of the workload that runs on the NPU. See §7.

---

## 6. Datasets

Covered in depth in **[`datasets.md`](datasets.md)**: what SmolVLA was trained
and evaluated on, what this repository actually feeds the model (a synthetic
all-zero batch) and why that is right for a numerical gate, what the gate can
and cannot claim, the LIBERO data cached on this machine, and the options for
measuring task fidelity.

The two facts from there that bear on the family:

- **Pretraining:** 481 community datasets, 22.9K trajectories / 10.6M frames,
  mostly SO-100. Published results: LIBERO 87.3%, real SO-100 78.3%; the
  pretraining ablation is 51.7% -> 78.3%.
- **The cached LIBERO data is 2-camera / 8-dim state**, which matches
  `HuggingFaceVLA/smolvla_libero` (prefix ~177), not the `smolvla_base` we
  ported (prefix 241). Choosing between them is an architecture decision, not
  just a data one — see `datasets.md` §4.

---

## 7. What this means for the port

| | vision (NPU, 1.19x/image) | backbone (CPU) | expert (CPU) |
|---|---|---|---|
| **base** | 12 layers 768/3072, seq 1024, **x3** | 16 layers, seq 241 | 16 layers, 720/2048 |
| 7 official finetunes | **identical**, x3 (libero_plus x5, robocerebra x4) | 16 layers, seq 241/305/369 | **identical** |
| HFVLA/libero | **identical**, x2 | **32 layers**, seq 177 | **32 layers, 480/1280** |

**1. The vision work transfers to the whole family at zero cost.** 12 layers /
768 / 3072 / seq 1024 / 12 MHA is constant across all nine checkpoints, and
across the eight official ones the weights are identical too. The fused ELFs do
not change by a byte.

**2. No family member changes the backbone/expert verdict.** Backbone width is
permanently 960 / 2560 and prefix seq tops out at 369 — far from the 1024 where
we measured a win. Expert width only ever shrinks (720 -> 480). Using the
single-layer GEMM FLOP yardstick against our measured points (vision fc2
4.83 GFLOP = win, backbone 1.26 = lose, expert 0.15 = lose badly):

| Variant | vision | backbone | expert |
|---|---|---|---|
| base (seq 241) | 4.83 G (win) | 1.26 G | 0.15 G |
| libero_plus (seq 369) | 4.83 G (win) | 1.81 G (+44%) | 0.15 G |
| HFVLA/libero (seq 177) | 4.83 G (win) | 0.87 G (worse) | **0.06 G (worse)** |

**3. `smolvla_libero_plus` is the family's best showcase target.** Same
architecture, so no recompilation, but five vision forwards per inference
instead of three — the stage we win on carries a larger share of the workload,
which should yield a better end-to-end speedup than base's 1.07x.

**4. Correction to a prior note.** An earlier record said the LIBERO checkpoint
reuses base's kernels and fused ELFs for free. That holds for
`lerobot/smolvla_libero` (16 layers, seq 241 — genuinely zero change). It does
**not** hold for `HuggingFaceVLA/smolvla_libero`: the backbone *width* matches,
but seq 241 -> 177 forces an ELF rebuild, and the expert going from 720/2048 to
480/1280 invalidates all 15 expert GEMM rows in the kernel registry.

**5. Expert width is a training-time choice, not an architectural constraint.**
`expert_width_multiplier` is a config knob. Finetuning with `2.0` would give an
expert of hidden 1920 / MLP ~5120, which would move it toward the regime where
the NPU wins. No such checkpoint exists, and this requires training rather than
inference — recorded because it shows "the expert is too small" is a design
decision, not a property of the model.

---

## 8. Unverified

- Prefix lengths 305 / 369 / 177 / 113 are computed as `64 x n_cams + 48 + 1`
  and **not dumped from a real run**. Only base's 241 is verified.
- Only base has been run end to end here. The other seven official checkpoints
  were inspected at the weight-header level, never loaded or executed.
- Relative LIBERO success rates of `lerobot/smolvla_libero` vs
  `HuggingFaceVLA/smolvla_libero` are unknown.
- `nvidia/smolvla-arena-gr1-microwave` has state 54 > `max_state_dim` 32; what
  `pad_vector` does with an over-long vector was not checked.
- The `finetune of smolvla_base` tag on the HuggingFaceVLA checkpoint appears
  inconsistent with its 32 layers (see §4).

## Sources

Checkpoint shapes and layer counts: safetensors headers of the nine
checkpoints. Config values: `config.json` pulled from the Hub per repo.
Mechanisms: `smolvlm_with_expert.py:88-106` (family knobs),
`modeling_smolvla.py:434-438` (empty cameras), `:474-480` and `:570-574`
(state/action padding), `:296` (output slicing). Batch construction:
`smolvla_prefix.py:50-64`, `smolvla_inference.py:76-96`. Local dataset:
`~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/meta/`. Training/eval
figures: [SmolVLA blog](https://huggingface.co/blog/smolvla),
[arXiv:2506.01844](https://arxiv.org/abs/2506.01844). Download counts: HF Hub
API, August 2026.
