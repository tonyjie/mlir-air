# 06 — 载荷性约束与代码卫生

这套 module 性能很好,但脆弱性也是**明写在代码里**的。这些约束大多带着"不这么做会怎么死"
的注释,可读性很好 —— 但也说明它离自动化 placement 还很远。

## 载荷性约束(改了就坏)

### 1. per-kernel `-O` 不能统一

Makefile 里明确分组:

```make
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # 必须 -O2
ATTN_LL_KERNELS := attn_qk attn_kv                  # 必须 -O1
```

- `rope.cc` 在 **-O1 会 miscompile**(Peano loop-metadata)→ device **挂死**
  (6143 ms `ERT_CMD_STATE_TIMEOUT`)。
- `attn_qk` / `attn_kv` 在 **-O2 有 do-while 死锁**,必须留在 -O1。

也就是说 -O1 和 -O2 各自会打死一半 kernel,没有一个统一档位可用。

### 2. `llvm-link` 版本

inline-attn merge 会 shell out 到**外部** `llvm-link`(从 PATH 解析)。

- 首个 PR 的要求:**必须 < LLVM 23**。≥23 的 `llvm-link` 会把 `llvm.lifetime` intrinsic
  改写成 no-size 形式,Peano `opt`(当时 LLVM 21)拒收 → `Broken module found`。
- HEAD 的 preflight 已改成 **"必须匹配 Peano 的 LLVM major"**,更准确。

Makefile 有 preflight 会提前 abort。CI 为此专门下载了一个 pre-23 的 llvm-link(PR #1769)。

### 3. 块循环必须单缓冲

`air.disable_ping_pong` 是**无条件**加的。ping-pong 的 unroll-by-2 余数会在 3-buffer ring
上读错 buffer → KV 错位。**症状:首 token 正确,之后全是垃圾。**

### 4. `ATTN_HOT` 在 Peano 下必须 noinline(HEAD 的修正)

首个 PR 是 `always_inline`,HEAD 翻转成:

```c
#if defined(__chess__)
#define ATTN_HOT inline __attribute__((always_inline))
#else
#define ATTN_HOT __attribute__((noinline))
#endif
```

原因:Peano spill online-softmax 的多个 BFP16 矩阵累加器,cross-regfile spill/reload 缺陷
会**破坏 / 死锁**内联后的 attention。要让每个热 helper 保持 bounded-register 函数。

`attn_fv` 里还有一处更细的重构,专门绕开同一族缺陷(注释称为 "bug B",
`AIESpillSlotOptimization` 家族):把原本 `j+=4`(4 个 mmul 累加器 + 64 宽 CORRECT 向量同时
live)拆成两趟 —— 先纯向量做 online-softmax 修正(让 CORRECT 提前 dead),再用**恰好一个**
零压力累加器做 mac。

### 5. Peano 的 `.bss` / `.data` 陷阱

kernel 里有一批注释明确的 workaround:

```c
// CDO loader 不清零 .bss;ping/pong 选择子和 scratch 必须放 .data 才真的被清零
__attribute__((section(".data"))) static bool is_ql_ping = false;

// Peano 对持久化 .data bool 的 read-modify-write 会 miscompile
// → 状态放 stack local,只在进出口同步回 .data
bool ql = is_ql_ping;  ...  is_ql_ping = ql;

// static-init guard 在未清零的 .bss 里,构造函数可能被跳过 → 故意声明成非 static
const aie::vector<bf16,16> neg_inf = aie::broadcast<bf16,16>(-0x1.FEp127f);
```

这三条对我们自己写 decode kernel 有直接参考价值。

### 6. 手动 pin 遍布全文

`air.shim_col`、`air.tile_dma_channel`、packet vs circuit 的选择,几乎每一处都带一段
"不这么 pin 会怎么死"的注释。举几个:

- **rope 的 Q 必须 pin 到 MM2S0**,否则 placer 把 packet append 放 MM2S0(先分配),把 Q
  挤到 MM2S1 → 前端死锁。
- **rms 的 `xnorm` pin 到 MM2S1、`layerOut` pin 到 MM2S0**,否则加入 append 通道后 placer
  会把两者都塞进 MM2S0(dual-fan packet),`layerOut` 从 circuit 翻成 packet → 死锁。
- **`toAttnKV` 必须是 packet**:一条 circuit 通道点对点,喂 2 个列 memtile 会死锁。
- **rope k/v 每列一条通道**:单通道喂 2 列的 memtile,FIFO 会交错两列的 get,一列阻塞另一列。
- **`VOCAB_CHUNK_I2` 必须匹配模型**:3B 条目注释说明 `K/PAYLOAD` 必须整除 `VOCAB_RNDS`,
  否则 round 数向下取整 → xnorm 广播不足 + logit drain 过短 → vocab wave **死锁**。
  1B 用 14,3B 必须用 9,Gemma3-4B 必须用 5。

### 7. `preserve_shim_dma_order` 与 pacing 的相互作用

见 [04](04_kv_cache_layout.md):融合 launch 对每个 preserve 通道按 wave 做 depth-2 pacing,
每 wave 只有 1 个 task 时退化成 fence → K 串行在 V 前 → 大 L 死锁。必须用
`air.shim_feed_no_pace` 退出 pacing。

## 代码卫生问题

读的时候要有心理准备,这批代码带着明显的开发期痕迹。

### 顶部 60 行注释已经过时且自相矛盾

`fused_decode.py` 开头的 "HOW TO RUN" 引用的路径**全都不存在**:
`programming_examples/attn/q4nx_decode.py`、`bringup/build_gen.sh`、
`q4nx_decode_BACKUP_stage2_circuit_golden.py`。

更严重的是里面的 "SCOPE / LIMITS" 段落:

> this is a single-decode-LAYER dataflow+ABI prototype, NOT a deployable model ...
> ONE layer (not 16), no LM head / embedding / sampling / tokenizer

这与实际状态(16 层 + LM head + 完整 chatbot,已跑通 verify 门)**完全矛盾**,是彻底
过时的段落。新读者极易被误导。

### "the reference" 指代丢失

全文近百处 "the reference" / "the reference harness",指的是 FastFlowLM 的 C++ 栈
(`npu_app` / `decoding_layer` / `llama_npu.cpp` / `npu_sequence` / `_receive_kv_cache` /
`_move_kv_cache`)。看起来是 sed 批量替换掉具体产品名留下的,**指代对象丢了**,读起来
相当费劲 —— 尤其像 "the reference's tile_2_2 DMA0" 这种句子,不知道 tile 编号出自哪。

### 死配置分支

`KV_REGION` / `KV_SPLIT` / `UNIFIED` / `MULTIBLK` 都硬编码 `True`,但保留了 flag 形式和
`if` 分支。更糟的是 `_emit_append` / `_emit_readback` 的非 region 分支是**静默 no-op**
(函数直接走到底 return None)而不是报错:

```python
def _emit_append(_kbase=_kbase):
    if KV_REGION:
        ...
        return
    # ← 这里什么都没有。KV_REGION=False 时静默不发 append
```

如果有人把 flag 改回 False,会得到一个不追加 KV 却不报错的 decode。建议改成
`raise NotImplementedError`。

### 环境变量作为隐式契约

`VOCAB_CHUNK_I2` / `DECODE_MODEL` / `DECODE_GOLDEN_L` / `NLAYERS` / `LM_HEAD` 等一堆开关
通过环境变量传递,而且**driver 必须设对**(见上面的 vocab 死锁)。`_MODELS` 表里用注释
写着 "The driver MUST set VOCAB_CHUNK_I2=9 (env) to match this UNI_LM" —— 这种耦合更适合
放进表里由代码强制,而不是靠注释约定。

### 命名与实际不符

- `attn_npu2_temporal_causal.py` 的 `causal_skip` 被函数体强制关掉(见 [01](01_overview.md))。
- `DECODE_GOLDEN` 已经退化成一个纯布尔开关(启用 post-attention-RMS 路径),名字还留着
  "golden dump 目录"的语义;Makefile 里写 `DECODE_GOLDEN=1` 并注释说明"这是布尔量,
  编译期不读任何权重文件"。

## 小结

这些约束**不是**代码质量差,大多是硬件/编译器现实的忠实记录 —— 而且记录得相当详尽,
是难得的一手资料。真正需要注意的只有两类:(1) 过时且自相矛盾的顶部注释;(2) 静默 no-op
的死分支。前者误导读者,后者在未来会咬人。
