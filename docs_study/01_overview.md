# 01 — 这批 example 是什么

## 不是一个 example,是三块互相咬合的东西

从 `programming_examples/llms/` 下看到 `llama32_1b_q4nx` / `llama32_3b_q4nx` 两个新目录,
容易以为是"又两个模型例子"。实际上真正的新东西在别处:

```
programming_examples/
├── fused_decode/                       ← 真正的新东西(~6k 行)
│   ├── fused_decode.py                 一个 dispatch = N 层 + LM head 的 AIR module
│   ├── decode_insts_gen.py             host 端每 token 指令流打补丁器
│   ├── proj_qmm_pack.py                Q4NX block packer
│   ├── kernels/                        Peano kernel(proj_qmm/rms_residual/glu/rope/attn_qk/attn_kv)
│   └── models/                         每模型 dim 头文件
├── flash_attention/kernel_fusion_based/
│   └── attn_npu2_temporal_causal.py    prefill 侧新 FA 变体(单 dispatch temporal)
└── llms/
    ├── llama32_1b_q4nx/                e2e 编排:prefill 填 KV → fused decode 自回归 → chatbot
    └── llama32_3b_q4nx/                (见下方"名字容易骗人")
```

`fused_decode/` 是一个**独立 example**,不属于 `llms/`。`llms/llama32_1b_q4nx/` 通过
跨目录 import 消费它(`_DEC` 指向 `../../fused_decode`)。

## 三块的分工

```
  user turn ─► chat template ─► [ 批量 PREFILL ]──填充──► 共享 per-layer KV cache
                              (llama32_1b_q4nx_prefill)              │
                                                                     ▼
  reply ◄─ detok ◄─ sampler ◄─ host embed ◄─ [ 每 token 融合 DECODE ]
     (streaming)                              (fused_decode,读+追加 KV)
                                              一次 dispatch = 16 层 + LM head
```

- **Prefill**:op-by-op,复用 `llms/shared/` 的 stitcher(`rms_gemms_rope` · `flash_attn` ·
  `o_ffn`)+ 8-partition LM head GEMV。权重 BO 常驻。产出 per-layer KV cache。
- **Decode**:整块手写的巨型 AIR module,一次 dispatch 跑完所有层。
- **Host**:embedding、final RMSNorm、sampler、chat template、EOS、streaming。

注意 prefill 和 decode 用的是**两套完全不同的权重打包**:prefill 用 Q4NX→bf16 host
dequant 喂 GEMM;decode 用 re-quantize 后的 q4k-cascade 流(`q4nx_requant.py`)。同一份
`model.q4nx` 出发,两条路径。

## 名字容易骗人的地方

**`llama32_3b_q4nx` 在最初的 PR #1766 里跟 fused decode 毫无关系** —— 它只是换了权重来源,
pipeline 原样复用 bf16 `llama32_3b`(CPU attention,~5 tok/s)。直到 **PR #1776**(即当前
HEAD `85f638d8`)才补上 "fully on-NPU decode + NPU prefill with KV handoff"。所以读这个
目录时要看清版本。

**`attn_npu2_temporal_causal.py` 的 `causal_skip` 是被强制关掉的**:

```python
causal_skip = False   # 函数体里硬写死,不管调用方传什么
```

注释说明了原因:skip 只 guard compute,而 K/V 的 channel get 是无条件的,超过约 8 个
K-block 就会 get↔compute 失步;真正的 skip 要连 streaming 一起跳,那是 conditional-channel
deadlock。**实际省下来的是 DMA 三角**(K/V put 用 IV-dependent size,wrap-and-stride 折不动
→ decline → 展开成 per-round 常量 BD),不是 compute skip。真正的收益点是
`num_cascade_stages=1` + herd row 映射到 GQA group 内的 q-head:填满 32 个 tile(原来 8),
launch 数降 4×。

## 上游演进时间线

| Commit | PR | 内容 |
|---|---|---|
| `5caadbfb` | #1764 | 主体:fused_decode + llama32_1b_q4nx + temporal-causal FA |
| `73d69cf4` | #1766 | Llama-3.2-3B Q4NX(此时仍是 CPU attention 复用 bf16 路径) |
| `9dcd20b6` | #1768 | 把 llama32_1b_q4nx 对齐共享 LLM driver contract(verify_adapter 等) |
| `a88b180d` | #1769 | CI:提供 pre-23 llvm-link 让 benchmark 真的跑起来 |
| `22f2d9c4` | #1775 | flash_attention:从 SCRIPT 推导 DK_TILE/DV_TILE |
| `b2959c9a` | #1772 | Gemma3-4B Q4NX(fused decode + 批量 NPU prefill) |
| `d1287dce` | #1777 | Qwen2.5-3B Q4_0(新增 `fused_decode_qwen.py`,2153 行) |
| `46f0ba2c` | #1778 | mlir-aie bump,LLVM 23 → 24 |
| `85f638d8` | #1776 | Llama-3.2-3B Q4NX 全 NPU decode + KV handoff |

作者均为 Erwei Wang,全部标注 Claude(4.7 / 5)为 co-author。

## HEAD 相对首个 PR 的两个重要变化

1. **模型参数化**:`fused_decode.py` 顶部新增 `_MODELS` 表,由 `DECODE_MODEL` 环境变量选择,
   目前含 `llama-3.2-1b` / `gemma3-4b` / `llama-3.2-3b`(Qwen 走独立的
   `fused_decode_qwen.py`)。注释声明 llama 条目"byte-identical"复现原先硬编码值。
   表里带了不少踩坑知识,例如 3B 条目写明 `K/PAYLOAD` 必须整除 `VOCAB_RNDS`,否则
   vocab wave 会 **deadlock**,所以 3B 必须用 `VOCAB_CHUNK_I2=9` 而非 1B 的 14。

2. **`ATTN_HOT` 语义翻转**:首个 PR 里是 `always_inline`(热路径内联),HEAD 改成 Peano 下
   **`noinline`**:

   ```c
   #if defined(__chess__)
   #define ATTN_HOT inline __attribute__((always_inline))
   #else
   #define ATTN_HOT __attribute__((noinline))
   #endif
   ```

   原因写在注释里:Peano 会 spill online-softmax 的多个 BFP16 矩阵累加器,而
   cross-regfile spill/reload 缺陷会**破坏或死锁**内联后的 attention。所以要让每个热
   helper 保持 bounded-register 函数。Chess 不受影响。

   这条对我们有直接价值——它是 llvm-aie 寄存器分配缺陷的一个明确记录点。
