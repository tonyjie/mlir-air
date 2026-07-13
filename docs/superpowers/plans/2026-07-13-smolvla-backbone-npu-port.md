# SmolVLA Backbone → NPU2 Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port SmolVLA's 16-layer SmolLM2-360M language backbone (`lerobot/smolvla_base`) onto AMD NPU2 as a hybrid CPU/NPU pipeline, numerically verified against the CPU baseline.

**Architecture:** Thin re-parameterization of the existing `smollm2_1_7b` example (same SmolLM2 family) with two real deltas — GQA (15q/5kv vs sibling's MHA) and **non-causal prefix-mask attention** (one net-new registry kernel). Vision/connector/action-expert/denoise stay on CPU; only the backbone prefill runs on NPU. Correctness is dual-track: per-layer cosine (diagnosis) + a new regression PASS/FAIL gate on the backbone hidden output and the end-to-end action chunk.

**Tech Stack:** mlir-air, kernel_registry, `programming_examples/llms/shared/` builders+infra, `programming_examples/llms/verify/` subsystem, LeRobot v0.5.0 CPU oracle (in `~/Projects/smolvla_playground/`).

---

## Reference facts (measured from the real checkpoint — do not re-derive)

- **Backbone config:** hidden=960, intermediate=2560, n_heads=15, n_kv_heads=5 (GQA group=3), head_dim=64, n_layers=16, rope_theta=100000, rope_interleaved=False (half-split), rms_norm_eps=1e-5, vocab=49280, **no QK-norm, no bias**.
- **Prefix seq = 241** = 3×64 visual + 48 language + 1 state.
- **Attention is non-causal** (prefix-LM mask): image+language bidirectional, state token starts a new block (`modeling_smolvla.py:101-131`).
- **Prefill-only.** No decode, no lm_head, no token sampling. Output = per-layer KV cache + final-norm hidden.
- **Closest sibling:** `programming_examples/llms/smollm2_1_7b/` (same family). Deltas vs sibling: n_layers 24→16, emb 2048→960, MHA→GQA(15/5), hidden 8192→2560, rope 130000→100000, causal→non-causal.
- **CPU oracle:** `~/Projects/smolvla_playground/.venv` + `lerobot/smolvla_base`. NPU env: main-repo shared toolchain (already rebuilt this session).

## File Structure

Create under `programming_examples/llms/smolvla/`:
- `smolvla_backbone_weights.py` — HF loader for the 16-layer backbone (fork of `smollm2_1_7b_weights.py`, GQA config).
- `smolvla_cpu_helpers.py` — NumPy helpers: `rms_norm`, `noncausal_attention_reference`, `build_prefix_mask`.
- `smolvla_prefix.py` — CPU-side prefix assembly + oracle hooks: dumps the (241,960) prefix embedding and per-layer hidden tensors from the real CPU model.
- `smolvla_backbone_prefill.py` — NPU backbone prefill pipeline (fork of the sibling's prefill; GQA + non-causal attn).
- `smolvla_inference.py` — top-level hybrid forward: CPU prefix → NPU backbone → CPU action expert/denoise → (50,6) chunk.
- `verify_adapter.py` — regression-oriented adapter (new gate, not token-set).
- `Makefile`, `README.md`, `ARCHITECTURE.md`, `requirements.txt`.
- `docs/` — per-model progress tracker.

Modify:
- `programming_examples/llms/verify/comparators.py` — add a regression gate comparator.
- `programming_examples/kernel_registry/` — add the non-causal masked-attention kernel (via add-kernel) + record tested shapes.

---

## Phase 0 — CPU reference + oracle hooks

### Task 0.1: Scaffold the smolvla example dir + requirements

**Files:**
- Create: `programming_examples/llms/smolvla/requirements.txt`
- Create: `programming_examples/llms/smolvla/docs/PROGRESS.md`

- [ ] **Step 1: Create requirements.txt**

```
# SmolVLA backbone NPU2 port — mirrors sibling smollm2_1_7b deps.
# The CPU oracle (lerobot) lives in ~/Projects/smolvla_playground/.venv and is
# NOT installed here; this file covers the mlir-air-side verify deps only.
transformers>=5.3.0
numpy
ml_dtypes
safetensors
huggingface_hub
```

- [ ] **Step 2: Create docs/PROGRESS.md**

```markdown
# SmolVLA Backbone Port — Progress

Target: lerobot/smolvla_base 16-layer SmolLM2-360M backbone on NPU2.
Spec: docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md

## Phase status
- [ ] Phase 0: CPU reference + oracle hooks
- [ ] Phase 1: kernel validation (7 existing shapes + non-causal attn)
- [ ] Phase 2: single-block validation
- [ ] Phase 3: full-backbone + end-to-end action-chunk gate

## Tested (kernel, shape) — filled in Phase 1
```

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llms/smolvla/requirements.txt programming_examples/llms/smolvla/docs/PROGRESS.md
git commit -m "[smolvla] scaffold example dir + progress tracker"
```

### Task 0.2: Backbone weight loader

**Files:**
- Create: `programming_examples/llms/smolvla/smolvla_backbone_weights.py`
- Reference: `programming_examples/llms/smollm2_1_7b/smollm2_1_7b_weights.py`

- [ ] **Step 1: Fork the sibling loader, set GQA config**

Copy `smollm2_1_7b_weights.py` structure. Change the config dataclass to:

```python
@dataclass
class SmolVLABackboneConfig:
    n_layers: int = 16
    emb_dim: int = 960
    n_heads: int = 15
    head_dim: int = 64
    n_kv_heads: int = 5          # GQA group = 3 (sibling was MHA 32/32)
    hidden_dim: int = 2560
    vocab_size: int = 49280
    rope_base: float = 100000.0
    rms_norm_eps: float = 1e-5
    dtype: Any = bfloat16
```

The `LayerWeights`/`LlamaWeights` dataclasses and `_HF_LAYER_MAP` carry over unchanged (SmolVLA backbone = Llama layer topology). `wk`/`wv` are now `(960, 320)` (GQA-narrow), not square.

**Name the entry point `load_backbone_weights(model_name_or_path, config=None, dtype=bfloat16) -> LlamaWeights`** (rename the sibling's `load_weights`). Every later task imports this exact name. Keep `synthetic_weights` and `generate_rope_lut` from the sibling.

- [ ] **Step 2: Point the HF loader at the nested VLM text_model weights**

SmolVLA's backbone weights live inside the `lerobot/smolvla_base` checkpoint under the VLM submodule. Add a key-prefix resolver so `load_weights` maps `model.vlm_with_expert...text_model.layers.{i}.*` → `LayerWeights` fields, and **only loads layers 0..15**.

```python
_SMOLVLA_TEXT_PREFIX = "model.vlm_with_expert.get_vlm_model.text_model."  # verify exact prefix from safetensors keys in Step 3
```

- [ ] **Step 3: Verify exact safetensor key names against the checkpoint**

Run: `python -c "from safetensors import safe_open; import glob; f=glob.glob('/home/jiajli/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/*/model*.safetensors')[0]; k=list(safe_open(f,'pt').keys()); print([x for x in k if 'text_model.layers.0.' in x])"`
Expected: prints the 9 per-layer weight keys (q/k/v/o_proj, gate/up/down_proj, input/post_attention_layernorm) with their real prefix. Fix `_SMOLVLA_TEXT_PREFIX` to match.

- [ ] **Step 4: Smoke-test the loader**

Run: `python programming_examples/llms/smolvla/smolvla_backbone_weights.py`
Expected: loads 16 layers, prints shapes; assert `wq.shape==(960,960)`, `wk.shape==(960,320)`, `w_gate.shape==(960,2560)`.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_backbone_weights.py
git commit -m "[smolvla] backbone HF weight loader (16-layer, GQA 15/5)"
```

### Task 0.3: CPU helpers (rms_norm, non-causal attention, prefix mask)

**Files:**
- Create: `programming_examples/llms/smolvla/smolvla_cpu_helpers.py`
- Reference: `programming_examples/llms/smollm2_1_7b/smollm2_1_7b_cpu_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `programming_examples/llms/smolvla/test_cpu_helpers.py`:

```python
import numpy as np
from ml_dtypes import bfloat16
from smolvla_cpu_helpers import rms_norm, build_prefix_mask, noncausal_attention_reference

def test_prefix_mask_shape_and_bidirectional():
    # 240 bidirectional prefix tokens + 1 state token = 241
    m = build_prefix_mask(n_prefix=240, n_state=1)          # additive mask (241,241)
    assert m.shape == (241, 241)
    # prefix block is fully bidirectional -> all zeros in [0:240,0:240]
    assert np.all(m[:240, :240] == 0.0)
    # state token (row 240) can attend to everything before it (no -inf in its row up to 241)
    assert np.all(m[240, :241] == 0.0)
    # prefix tokens CANNOT attend to the state token (col 240) -> -inf
    assert np.all(np.isneginf(m[:240, 240]))

def test_noncausal_attention_matches_manual():
    rng = np.random.default_rng(0)
    L, H, Hkv, D = 8, 2, 1, 4
    q = rng.standard_normal((L, H, D)).astype(np.float32)
    k = rng.standard_normal((L, Hkv, D)).astype(np.float32)
    v = rng.standard_normal((L, Hkv, D)).astype(np.float32)
    mask = np.zeros((L, L), np.float32)                     # full bidirectional
    out = noncausal_attention_reference(q, k, v, mask, n_heads=H, n_kv_heads=Hkv)
    assert out.shape == (L, H, D)
    # head 0 vs manual softmax(QK^T/sqrt(D))V with kv-head 0
    s = (q[:,0,:] @ k[:,0,:].T) / np.sqrt(D)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(out[:,0,:], p @ v[:,0,:], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd programming_examples/llms/smolvla && python -m pytest test_cpu_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: smolvla_cpu_helpers`.

- [ ] **Step 3: Implement the helpers**

```python
"""CPU NumPy helpers for the SmolVLA backbone port (F32 reference math)."""
import numpy as np

def rms_norm(x, weight, eps=1e-5):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps) * weight.astype(np.float32))

def build_prefix_mask(n_prefix, n_state=1):
    """Additive attention mask for SmolVLA's prefix-LM pattern.
    Prefix tokens (image+language) attend bidirectionally among themselves but
    NOT to the state token; the state token attends to everything before it.
    Mirrors make_att_2d_masks (modeling_smolvla.py:101-131)."""
    n = n_prefix + n_state
    mask = np.zeros((n, n), np.float32)
    # prefix rows cannot see the state column(s)
    mask[:n_prefix, n_prefix:] = -np.inf
    return mask

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def noncausal_attention_reference(q, k, v, mask, n_heads, n_kv_heads):
    """q:(L,H,D) k,v:(L,Hkv,D) mask:(L,L) additive. GQA repeat, F32."""
    L, H, D = q.shape
    group = n_heads // n_kv_heads
    out = np.empty((L, H, D), np.float32)
    scale = 1.0 / np.sqrt(D)
    for h in range(H):
        kv = h // group
        s = (q[:, h, :].astype(np.float32) @ k[:, kv, :].astype(np.float32).T) * scale
        s = s + mask
        p = _softmax(s, axis=-1)
        out[:, h, :] = p @ v[:, kv, :].astype(np.float32)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd programming_examples/llms/smolvla && python -m pytest test_cpu_helpers.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_cpu_helpers.py programming_examples/llms/smolvla/test_cpu_helpers.py
git commit -m "[smolvla] CPU helpers: rms_norm, prefix mask, non-causal attention ref"
```

### Task 0.4: Prefix assembly + oracle hooks (the CPU→NPU cut point)

**Files:**
- Create: `programming_examples/llms/smolvla/smolvla_prefix.py`
- Oracle env: `~/Projects/smolvla_playground/.venv`

This task produces the **ground-truth tensors** every later phase compares against: the (241,960) assembled prefix embedding fed to backbone layer 0, and the per-layer hidden outputs. It runs in the lerobot venv (has the model); it writes `.npz` fixtures the mlir-air side loads.

- [ ] **Step 1: Write the oracle dumper**

```python
"""Dump SmolVLA backbone oracle fixtures from the real CPU model.
Run with the lerobot venv:  ~/Projects/smolvla_playground/.venv/bin/python
Writes smolvla_oracle.npz with: prefix_embed (241,960), per-layer hidden
(16 × (241,960)), final_norm_hidden (241,960), and the (50,6) action chunk."""
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_ID = "lerobot/smolvla_base"
OUT = "smolvla_oracle.npz"

def build_batch(policy):
    cfg = policy.config; b = {}
    for k, f in cfg.input_features.items():
        b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        ["pick up the cube"], padding="max_length",
        max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b

def main():
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).eval()
    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    caps = {}
    caps["layers"] = []
    h0 = tm.layers[0].register_forward_pre_hook(
        lambda m, args: caps.__setitem__("prefix_embed", args[0].detach().float().numpy()[0]))
    hooks = [h0]
    for i, layer in enumerate(tm.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, i_, o, idx=i: caps["layers"].append(
                (idx, (o[0] if isinstance(o, tuple) else o).detach().float().numpy()[0]))))
    hn = tm.norm.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("final_norm_hidden", o.detach().float().numpy()[0]))
    hooks.append(hn)
    batch = build_batch(policy); policy.reset()
    with torch.no_grad():
        action = policy.select_action(batch)
    for h in hooks: h.remove()
    layer_hidden = np.stack([h for _, h in sorted(caps["layers"])[:16]])
    np.savez(OUT, prefix_embed=caps["prefix_embed"],
             layer_hidden=layer_hidden,
             final_norm_hidden=caps["final_norm_hidden"],
             action=action.numpy())
    print(f"[oracle] wrote {OUT}: prefix{caps['prefix_embed'].shape} "
          f"layers{layer_hidden.shape} action{tuple(action.shape)}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the oracle fixture**

Run: `cd programming_examples/llms/smolvla && ~/Projects/smolvla_playground/.venv/bin/python smolvla_prefix.py`
Expected: `[oracle] wrote smolvla_oracle.npz: prefix(241, 960) layers(16, 241, 960) action(1, 6)`
Note: `smolvla_oracle.npz` is a fixture — add to a local `.gitignore` for the dir (do NOT commit the binary; it regenerates deterministically).

- [ ] **Step 3: Add fixture gitignore + commit the dumper**

```bash
echo "smolvla_oracle.npz" > programming_examples/llms/smolvla/.gitignore
git add programming_examples/llms/smolvla/smolvla_prefix.py programming_examples/llms/smolvla/.gitignore
git commit -m "[smolvla] oracle dumper: prefix embed + per-layer hidden fixtures"
```

### Task 0.5: CPU backbone reference (validates helpers reproduce the oracle)

**Files:**
- Modify: `programming_examples/llms/smolvla/smolvla_cpu_helpers.py` (add `cpu_backbone_forward`)
- Test: `programming_examples/llms/smolvla/test_cpu_backbone.py`

This proves our NumPy backbone math matches the real model BEFORE any NPU work — the CPU reference is itself verified against the oracle.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from smolvla_backbone_weights import load_backbone_weights, SmolVLABackboneConfig
from smolvla_cpu_helpers import cpu_backbone_forward, build_prefix_mask

def test_cpu_backbone_matches_oracle():
    o = np.load("smolvla_oracle.npz")
    cfg = SmolVLABackboneConfig()
    w = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    mask = build_prefix_mask(n_prefix=240, n_state=1)
    hidden = cpu_backbone_forward(o["prefix_embed"], w, cfg, mask, rope_base=cfg.rope_base)
    # final-norm hidden must match the oracle within f32 round-off
    cos = np.sum(hidden * o["final_norm_hidden"]) / (
        np.linalg.norm(hidden) * np.linalg.norm(o["final_norm_hidden"]))
    assert cos > 0.9999, f"cosine {cos} too low — CPU backbone diverges from oracle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd programming_examples/llms/smolvla && python -m pytest test_cpu_backbone.py -v`
Expected: FAIL — `cpu_backbone_forward` not defined (and `load_backbone_weights` name must exist from Task 0.2).

- [ ] **Step 3: Implement cpu_backbone_forward**

Add to `smolvla_cpu_helpers.py`. Full 16-layer forward using rms_norm + half-split RoPE + `noncausal_attention_reference` + SwiGLU, matching Llama layer order:

```python
def _rope_half_split(x, positions, base, head_dim):
    # x:(L,H,D). Non-interleaved (half-split) RoPE.
    half = head_dim // 2
    inv = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    ang = positions[:, None].astype(np.float32) * inv[None, :]   # (L,half)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)[:, None, :]
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], -1)
    return x * cos + rot * sin

def cpu_backbone_forward(prefix_embed, w, cfg, mask, rope_base):
    x = prefix_embed.astype(np.float32)                # (241,960)
    L = x.shape[0]; pos = np.arange(L)
    for lw in w.layers:
        h = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps)
        q = (h @ lw.wq.astype(np.float32)).reshape(L, cfg.n_heads, cfg.head_dim)
        k = (h @ lw.wk.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ lw.wv.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        q = _rope_half_split(q, pos, rope_base, cfg.head_dim)
        k = _rope_half_split(k, pos, rope_base, cfg.head_dim)
        a = noncausal_attention_reference(q, k, v, mask, cfg.n_heads, cfg.n_kv_heads)
        a = a.reshape(L, cfg.n_heads * cfg.head_dim) @ lw.wo.astype(np.float32)
        x = x + a
        h = rms_norm(x, lw.ffn_norm, cfg.rms_norm_eps)
        g = h @ lw.w_gate.astype(np.float32); u = h @ lw.w_up.astype(np.float32)
        silu = g / (1.0 + np.exp(-g))
        x = x + (silu * u) @ lw.w_down.astype(np.float32)
    return rms_norm(x, w.final_norm, cfg.rms_norm_eps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd programming_examples/llms/smolvla && python -m pytest test_cpu_backbone.py -v`
Expected: PASS (cosine > 0.9999). If it fails, the divergence localizes the bug (RoPE base? mask? GQA grouping?) before any NPU time is spent.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_cpu_helpers.py programming_examples/llms/smolvla/test_cpu_backbone.py
git commit -m "[smolvla] CPU backbone forward verified against oracle (cosine>0.9999)"
```

**Phase 0 gate:** `test_cpu_backbone.py` PASSES — the NumPy backbone reproduces the real model. Update `docs/PROGRESS.md` Phase 0 checkbox.

---

## Phase 1 — kernel validation

Every leaf kernel × the SmolVLA shape, verified on real NPU2. The 7 existing registry kernels need re-validation at new shapes; the non-causal attention is net-new. **All NPU runs use the lock:** `flock -x -w 1800 /tmp/mlir-air-npu.lock <cmd>`.

### Task 1.1: Validate the 7 existing registry kernels at SmolVLA shapes

**Files:**
- Modify: `programming_examples/kernel_registry/details/*.md` (add tested-shape rows)
- Modify: `programming_examples/llms/smolvla/docs/PROGRESS.md`

Shapes to validate (M=seq=241, pad to 256 where a kernel needs power-of-two):

| Kernel | Shape | Registry harness |
|---|---|---|
| RMSNorm | 256×960 | `kernel_registry` RMSNorm harness |
| GEMM (q/o_proj) | 256×960×960 | GEMM bf16-out |
| GEMM (k/v_proj) | 256×960×320 | GEMM bf16-out |
| GEMM (gate/up) | 256×960×2560 | GEMM bf16-out |
| GEMM (down) | 256×2560×960 | GEMM bf16-out |
| RoPE (half-split) | 256×64, θ=100000 | RoPE harness |
| SiLU-and-Mul | 256×2560 | SiLU harness |
| EltwiseAdd | 256×960 | EltwiseAdd harness |

- [ ] **Step 1: For each shape, run the registry harness's `make run`**

For each kernel dir under `programming_examples/kernel_registry/`, invoke its standalone harness at the SmolVLA shape (parameterized via the harness's shape args — see each `details/<K>.md` "reproduce" section). Example (RMSNorm):
Run: `flock -x -w 1800 /tmp/mlir-air-npu.lock make -C <rmsnorm harness dir> run M=256 N=960`
Expected: harness prints `PASS` (element-wise `np.isclose` at the kernel's rtol/atol vs FP32 ref).

- [ ] **Step 2: Record each PASS as a tested-shape row**

For each PASS, append a row to the kernel's `details/<K>_bf16.md` tested-shapes table with `Used by = smolvla` and the measured `mean_rel_L1`.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/kernel_registry/details/
git commit -m "[kernel_registry] record SmolVLA backbone tested shapes (7 kernels @ seq=256)"
```

### Task 1.2: Non-causal masked attention — compose GEMM + full-row masked softmax (approach B)

**DECISION (2026-07-13, supersedes the original add-kernel plan):** The original plan
was to fork the causal FlashAttention kernel. Scouting the FA kernel
(`flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` + `attn_npu2.cc`)
found: (1) causality is device-side, computed per-block from indices, with NO
host-mask infrastructure; (2) softmax is ONLINE/flash (streamed per K-chunk), so
an arbitrary mask must be applied per (q_block,kv_block) tile matching G's 8×8
column-major layout — invasive and error-prone; (3) it needs a 3rd DMA stream
(tight budget) and GQA 15/5 forces hpu=1 (half-array). Meanwhile seq=256 is tiny
(QKᵀ=256×256≈128KB fp32, fits L2) so the flash online-softmax buys nothing.

**Approach B — compose already-verified pieces, no new fused AIE kernel:**
Attention per head = `S = Q@Kᵀ * (1/√64)` → `S += host_mask` → full-row softmax →
`O = P@V`. The two matmuls are registry GEMM (same family as Task 1.1); the only
new device piece is a **full-row masked softmax** (mask-agnostic: materialize the
256-wide score row, add the host additive mask, softmax). This is forked from
`programming_examples/softmax/` (which does a plain per-row softmax already), NOT
from FlashAttention.

**Files:**
- Create: `programming_examples/masked_softmax/` — fork of `programming_examples/softmax/` (`softmax.py`, `softmax.cc`, `Makefile`, an npu2 lit) adding a second L3 input (the additive mask) that is added to the row before the existing softmax.
- Test: the fork's own harness compares against an FP32 masked-softmax reference.

- [ ] **Step 1: Fork the softmax example, add a mask input**

Copy `programming_examples/softmax/` → `programming_examples/masked_softmax/`. The
existing kernel does per-row softmax over `n` (tile_n per core). Add a second L3
memref `mask` of the same shape as the input; in the herd body, DMA the mask tile
alongside the data tile and do `row = row + mask_tile` before calling the softmax
routine (either add a `masked_softmax_bf16` variant in the `.cc`, or add elementwise
in-MLIR before the existing `softmax_bf16` call). Keep the row-wise softmax
numerics from the sibling (LUT-based exp). The additive mask uses 0 / large-negative
(match `torch.finfo(bf16).min`, NOT -inf, to avoid NaN on fully-masked rows — the
real model uses `torch.where(...finfo.min)`; a fully-masked row yields a uniform row).

- [ ] **Step 2: Write the masked-softmax test (TDD)**

The harness builds a random `(rows, 256)` score matrix + a random 0/finfo.min mask,
runs the NPU kernel, and compares to an FP32 reference:
`ref = softmax(scores + mask, axis=-1)` (with the finfo.min → uniform-row semantics).
PASS at bf16 softmax tolerance (rtol 1.6e-2 / atol ~3e-2, matching the registry
softmax/attention tier). Run:
`flock -x -w 1800 /tmp/mlir-air-npu.lock make -C programming_examples/masked_softmax run`
Expected: harness prints PASS.

- [ ] **Step 3: Assemble the attention path in the prefill module (deferred to Task 2.1)**

The full per-head attention (GEMM QKᵀ → masked_softmax → GEMM PV, GQA 15/5 host-side
head loop) is WIRED in Task 2.1's `smolvla_backbone_prefill.py`, reusing the
registry GEMM (Task 1.1 shapes: 256×64×256 for QKᵀ needs a new 256×64×256 / 256×256×64
GEMM shape — validate these two additional GEMM shapes here in Step 3 the same way
Task 1.1 did, recording rows). GQA repeat is host-side (q-head h uses kv-head h//3).

- [ ] **Step 4: Record tested shapes + commit**

Record the masked_softmax (256-wide) and the two attention GEMM shapes
(256×64×256, 256×256×64) as registry tested rows (masked_softmax gets its own
detail page or a note; the GEMM shapes append to `GEMM_bf16_in_bf16_out.md`).

```bash
git add programming_examples/masked_softmax/ programming_examples/kernel_registry/ programming_examples/llms/smolvla/docs/PROGRESS.md
git commit -m "[smolvla] non-causal attention via GEMM + masked softmax (approach B)"
```

**Phase 1 gate:** the 7 registry kernels (Task 1.1) + masked_softmax + the two
attention GEMM shapes all PASS at SmolVLA shapes on NPU2. Update `docs/PROGRESS.md`.

---

## Phase 2 — single-block validation

Wire the verified kernels into ONE backbone layer on NPU; per-layer cosine vs the oracle at layer 0.

### Task 2.1: NPU single-layer prefill

**Files:**
- Create: `programming_examples/llms/smolvla/smolvla_backbone_prefill.py`
- Reference: `programming_examples/llms/smollm2_1_7b/smollm2_1_7b_inference.py` + `shared/builders/rms_qkv_bias_rope_multi.py`, `shared/builders/o_ffn_multi.py`

- [ ] **Step 1: Fork the sibling prefill; parameterize for GQA + non-causal attn**

Implement `compile_all_kernels(cache, config, seq_len=256)` and `run_transformer_block(x_bf16, layer_weights, rope_lut_bf16, config, cache, mask, layer_idx=0)`. Reuse the shared multi-launch builders for the RMS→QKV→RoPE and O→FFN groups (they already accept GQA kv-dim via config); the attention call dispatches the **new masked-attn kernel** with the host-supplied `mask`, replacing the sibling's causal flash-attn / cpu_attn call.

```python
def run_transformer_block(x_bf16, lw, rope_lut_bf16, config, cache, mask, layer_idx=0):
    # 1) fused RMS + q/k/v proj + RoPE  (shared rms_qkv_rope multi-launch ELF)
    q, k, v = cache.load_and_run("rms_qkv_rope", ...,
                                 static_input_indices=..., bo_key=f"L{layer_idx}")
    # 2) non-causal masked attention (NEW kernel), mask is a host input
    attn = cache.load_and_run("masked_attn", ..., mask, output_indices=...)
    # 3) fused O-proj + residual + FFN  (shared o_ffn multi-launch ELF)
    out = cache.load_and_run("o_ffn", attn, x_bf16, ..., bo_key=f"L{layer_idx}")
    return out
```

- [ ] **Step 2: Write the single-layer test**

Create `programming_examples/llms/smolvla/test_single_block.py`:

```python
import numpy as np
from ml_dtypes import bfloat16
from smolvla_backbone_weights import load_backbone_weights, SmolVLABackboneConfig
from smolvla_cpu_helpers import build_prefix_mask
from smolvla_backbone_prefill import compile_all_kernels, run_transformer_block
from shared.infra.cache import KernelCache  # sibling's cache infra

def test_layer0_cosine_vs_oracle():
    o = np.load("smolvla_oracle.npz")
    cfg = SmolVLABackboneConfig()
    w = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    mask = build_prefix_mask(240, 1)
    cache = KernelCache(...)                      # per sibling setup
    compile_all_kernels(cache, cfg, seq_len=256)
    x = o["prefix_embed"].astype(bfloat16)        # pad 241->256 inside run_transformer_block
    out = run_transformer_block(x, w.layers[0], rope_lut(cfg), cfg, cache, mask, layer_idx=0)
    out = np.asarray(out, np.float32)[:241]
    ref = o["layer_hidden"][0]                     # oracle layer-0 output
    cos = np.sum(out*ref)/(np.linalg.norm(out)*np.linalg.norm(ref))
    assert cos > 0.99, f"layer-0 cosine {cos} below 0.99"
```

- [ ] **Step 3: Run on NPU (compile then run)**

Run: `flock -x -w 1800 /tmp/mlir-air-npu.lock python -m pytest programming_examples/llms/smolvla/test_single_block.py -v`
Expected: PASS, cosine > 0.99. A low cosine localizes the integration bug (layout mismatch, GQA repeat, RoPE, mask injection) — debug at the boundary, not the whole model.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_backbone_prefill.py programming_examples/llms/smolvla/test_single_block.py
git commit -m "[smolvla] NPU single-block prefill; layer-0 cosine>0.99 vs oracle"
```

**Phase 2 gate:** layer-0 cosine > 0.99 on NPU2. Update `docs/PROGRESS.md`.

---

## Phase 3 — full-backbone + end-to-end gate

### Task 3.1: Full 16-layer NPU backbone + per-layer diagnosis

**Files:**
- Modify: `programming_examples/llms/smolvla/smolvla_backbone_prefill.py` (add `run_backbone_prefill` looping 16 layers)
- Test: `programming_examples/llms/smolvla/test_full_backbone.py`

- [ ] **Step 1: Add the 16-layer loop**

```python
def run_backbone_prefill(prefix_embed_bf16, weights, config, cache, mask, rope_lut):
    x = prefix_embed_bf16
    per_layer = []
    for i in range(config.n_layers):
        x = run_transformer_block(x, weights.layers[i], rope_lut, config, cache, mask, i)
        per_layer.append(np.asarray(x, np.float32)[:241])
    final = rms_norm_npu(x, weights.final_norm, cache)   # final norm on NPU
    return np.asarray(final, np.float32)[:241], per_layer
```

- [ ] **Step 2: Write the full-backbone test (per-layer cosine + final gate)**

```python
def test_full_backbone_vs_oracle():
    o = np.load("smolvla_oracle.npz")
    ...
    final, per_layer = run_backbone_prefill(prefix, w, cfg, cache, mask, rope_lut)
    # diagnosis: every layer cosine
    for i, h in enumerate(per_layer):
        ref = o["layer_hidden"][i]
        cos = np.sum(h*ref)/(np.linalg.norm(h)*np.linalg.norm(ref))
        print(f"layer {i}: cos={cos:.5f}")
        assert cos > 0.98, f"layer {i} cosine {cos} — drift"
    # GATE: final-norm hidden
    fref = o["final_norm_hidden"]
    fcos = np.sum(final*fref)/(np.linalg.norm(final)*np.linalg.norm(fref))
    assert fcos > 0.99, f"final hidden cosine {fcos} below gate"
```

- [ ] **Step 3: Run on NPU**

Run: `flock -x -w 1800 /tmp/mlir-air-npu.lock python -m pytest programming_examples/llms/smolvla/test_full_backbone.py -v -s`
Expected: PASS; printed per-layer cosines stay > 0.98, final > 0.99. Rising drift across layers points to accumulation/precision; a sudden drop at layer k localizes a layer-indexed weight-load bug.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_backbone_prefill.py programming_examples/llms/smolvla/test_full_backbone.py
git commit -m "[smolvla] full 16-layer NPU backbone; per-layer cosine + final gate PASS"
```

### Task 3.2: Add a regression PASS/FAIL comparator to the verify subsystem

**Files:**
- Modify: `programming_examples/llms/verify/comparators.py`
- Test: `programming_examples/llms/verify/test_regression_gate.py`

The verify subsystem's gate (`compute_topk_set_check`) is token-only. SmolVLA needs a continuous-output gate. Add one that reuses the existing generic `per_position_cosine`/`error_metrics`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from comparators import regression_gate

def test_regression_gate_pass_and_fail():
    a = np.random.default_rng(0).standard_normal((50, 6)).astype(np.float32)
    assert regression_gate(a, a.copy(), cos_min=0.99, mse_max=1e-3)["passed"] is True
    b = a + 5.0
    assert regression_gate(a, b, cos_min=0.99, mse_max=1e-3)["passed"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd programming_examples/llms/verify && python -m pytest test_regression_gate.py -v`
Expected: FAIL — `regression_gate` not defined.

- [ ] **Step 3: Implement regression_gate**

Add to `comparators.py`:

```python
def regression_gate(npu, ref, cos_min=0.99, mse_max=1e-3):
    """PASS/FAIL gate for continuous-output models (SmolVLA action chunks,
    backbone hidden). Reuses per_position_cosine + MSE. Returns a dict with
    passed/cosine/mse for reporting."""
    npu = np.asarray(npu, np.float32); ref = np.asarray(ref, np.float32)
    a = npu.reshape(-1, npu.shape[-1]); b = ref.reshape(-1, ref.shape[-1])
    cos = float(np.nanmedian(per_position_cosine(a, b)))
    mse = float(np.mean((a - b) ** 2))
    return {"passed": bool(cos >= cos_min and mse <= mse_max),
            "cosine": cos, "mse": mse, "cos_min": cos_min, "mse_max": mse_max}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd programming_examples/llms/verify && python -m pytest test_regression_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llms/verify/comparators.py programming_examples/llms/verify/test_regression_gate.py
git commit -m "[verify] add regression_gate comparator for continuous-output models"
```

### Task 3.3: End-to-end hybrid inference + action-chunk gate

**Files:**
- Create: `programming_examples/llms/smolvla/smolvla_inference.py`
- Create: `programming_examples/llms/smolvla/verify_adapter.py`
- Test: `programming_examples/llms/smolvla/test_e2e.py`

The end-to-end gate: run the full hybrid pipeline (CPU prefix → NPU backbone → CPU action expert/denoise) and compare the (50,6) action chunk to the pure-CPU oracle chunk.

- [ ] **Step 1: Implement the hybrid forward**

`smolvla_inference.py` — `run_hybrid_forward(batch)`:
1. Use the lerobot CPU model for vision→connector→prefix assembly (CPU) up to the (241,960) prefix embed — i.e. everything the oracle dumper hooked BEFORE layer 0.
2. Run `run_backbone_prefill` on NPU → produces the per-layer KV cache / hidden the action expert consumes.
3. Feed NPU backbone output into the lerobot CPU action-expert + 10-step denoise → (50,6) chunk.

Because the action expert consumes the backbone's per-layer KV cache (fused loop, `smolvlm_with_expert.py:403-498`), Step 2 must expose the same per-layer K/V the CPU model would have cached. Dump K/V per layer from `run_backbone_prefill` and inject into the CPU expert's `past_key_values`.

- [ ] **Step 2: Write the e2e test**

```python
import numpy as np
from smolvla_inference import run_hybrid_forward, build_oracle_batch
from verify.comparators import regression_gate

def test_e2e_action_chunk_vs_cpu():
    o = np.load("smolvla_oracle.npz")
    batch = build_oracle_batch()          # same synthetic batch as the oracle dumper
    chunk = run_hybrid_forward(batch)     # (1,50,6) or (1,6) select_action head
    ref = o["action"]
    g = regression_gate(chunk, ref, cos_min=0.99, mse_max=1e-3)
    assert g["passed"], f"e2e action gate FAIL: {g}"
```

- [ ] **Step 3: Run on NPU**

Run: `flock -x -w 1800 /tmp/mlir-air-npu.lock python -m pytest programming_examples/llms/smolvla/test_e2e.py -v -s`
Expected: PASS — hybrid action chunk matches the pure-CPU oracle within the gate. NOTE: if the gate's `mse_max`/`cos_min` need tuning for the bf16 backbone, record the observed values first, then set thresholds from the measured clean-run distribution (do NOT loosen blindly).

- [ ] **Step 4: Write verify_adapter.py (regression-oriented)**

Implement a `verify_adapter.py` exposing `DEFAULT_MODEL="lerobot/smolvla_base"`, `build_config()`, and a `build_runner(...)` whose gate calls `regression_gate` on the action chunk (not `compute_topk_set_check`). This lets `make verify` drive the same e2e comparison.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llms/smolvla/smolvla_inference.py programming_examples/llms/smolvla/verify_adapter.py programming_examples/llms/smolvla/test_e2e.py
git commit -m "[smolvla] end-to-end hybrid inference; action-chunk gate PASS vs CPU oracle"
```

### Task 3.4: Makefile, README, ARCHITECTURE + finalize

**Files:**
- Create: `programming_examples/llms/smolvla/Makefile`, `README.md`, `ARCHITECTURE.md`
- Reference: `programming_examples/llms/smollm2_1_7b/Makefile`

- [ ] **Step 1: Write the Makefile (fork sibling, regression gate)**

Targets: `compile` (build all kernels), `run` (hybrid forward once), `verify` (e2e action gate via verify_adapter), `diagnosis` (per-layer backbone cosine), `profile`, `clean`. `verify`/`diagnosis`/`run` NPU targets wrap `flock -x -w 1800 /tmp/mlir-air-npu.lock`.

- [ ] **Step 2: Run the full verify gate**

Run: `flock -x -w 1800 /tmp/mlir-air-npu.lock make -C programming_examples/llms/smolvla verify`
Expected: prints the action-chunk `regression_gate` result with `passed=True`.

- [ ] **Step 3: Write README.md + ARCHITECTURE.md**

README: what this is (backbone-only hybrid port), how to run, the CPU/NPU split diagram, correctness gates. ARCHITECTURE.md (NOT CLAUDE.md — top-level .gitignore excludes CLAUDE.md): the backbone kernel mapping + non-causal attn design.

- [ ] **Step 4: Update PROGRESS.md (all phases checked) + commit**

```bash
git add programming_examples/llms/smolvla/Makefile programming_examples/llms/smolvla/README.md programming_examples/llms/smolvla/ARCHITECTURE.md programming_examples/llms/smolvla/docs/PROGRESS.md
git commit -m "[smolvla] Makefile + docs; backbone port A1 milestone complete"
```

**Phase 3 gate (A1 done):** `make verify` PASSES — the hybrid pipeline's action chunk matches the pure-CPU oracle, with the 16-layer backbone running on NPU2.

---

## Self-Review Notes

- **Spec coverage:** Phase 0 (CPU ref + oracle) ✓; Phase 1 (7 kernels + non-causal attn) ✓; Phase 2 (single block) ✓; Phase 3 (full backbone + dual-track gate) ✓; regression gate for continuous output ✓ (spec §5); CLAUDE.md→ARCHITECTURE.md ✓ (spec §7.5); Allo = reference only ✓ (never imported).
- **Out of scope (per spec §1, §7):** A2 action-expert-on-NPU, A3 vision-on-NPU, Phase 4-5 optimization — not in this plan.
- **Risk hooks:** prefix-assembly fidelity (spec §7.3) handled by Task 0.4 dumping the exact pre-layer-0 tensor; `num_expert_layers=0` (spec §7.4) irrelevant to A1 (expert stays CPU).
- **Threshold caveat:** cos/MSE gate thresholds (0.99 / 1e-3) are starting points; Task 3.3 Step 3 measures the clean-run distribution before locking them.
