# 07 — 可借鉴与不可借鉴

## 与我们现有方法论的关系

这条路线和我们的 kernel-first 流程(kernel registry → 逐 shape 验证 → 拼装 → 优化)
基本是**正交甚至相反**的:

| | 我们的做法 | 这批代码 |
|---|---|---|
| 组织方式 | 复用 `llms/shared/builders`,kernel 进 registry | 一整块手写巨型 AIR module |
| 验证粒度 | 每个 (kernel, shape) 独立门 | 只有 e2e 门(Paris / top-k) |
| 复用性 | 跨模型复用 builder | 按模型手工加 `_MODELS` 表项 |
| 优化路径 | 分阶段 skill(merge → BO reuse → layout) | 一次性写死最优形态 |

对 `llms/shared/` 的改动仅有 `o_ffn_multi.py` 加了个 `herd_m` 参数。也就是说
**fused decode 完全没有进我们的复用体系**。

但它在某些维度上走得更远,值得挑着学。

## 值得借鉴的(按价值排序)

### 1. 运行时参数从"编译期常量"降级为"指令流补丁"⭐⭐⭐

见 [03](03_one_xclbin_any_L.md)。模式是通用的:

1. 把变量在编译期钉在**最大值**;
2. kernel 内部按运行时值**跳过**多余工作(注意跳过条件必须在配对的 producer/consumer
   之间完全一致);
3. host 端用**两点标定 + 线性插值**合成指令流;
4. 用**整除性检查**做线性假设的自检,不满足就拒绝该模板而不是给错结果。

我们的 decode 每换 context 长度都要重编,这套办法可以直接套。即使不搬整个 fused decode,
单把这个模式用在我们自己的 runtime 参数上也成立。

**注意**:验证一定要用**没参与拟合的点**。上游自带的 self-check 是循环论证。

### 2. 晚展开的 N-wave unified launch ⭐⭐⭐

```python
for _iv in for_(idx(0), idx(UNI_WAVES), idx(1)):
    launch(operands=list(_fa) + [_iv], attributes={"air.preserve_shim_dma_order": ...})(body)
```

device 只 emit 一次,per-layer offset 由归纳变量缩放,`scf.for` 在 **airrt-to-npu 才展开**。
结果:`air-to-aie` 看到的 op count 是**常数**,aie.device/CDO 与单层构建逐字节相同,
只有指令流变长。

这比我们 `opt-merge-multi-launch-kernels` 走得更远 —— 那个 skill 是把 N 个 launch 融成
multi-launch ELF(op count 仍随 N 增长),这里是**一个 launch 跑 N 个 wave**。对层数多的
模型(28/34 层)这是能不能编译得动的分水岭。

### 3. region-major(quadrant)KV 布局 ⭐⭐

见 [04](04_kv_cache_layout.md)。核心洞察:**按消费者的读取模式来组织 DDR 布局**,而不是按
生产顺序。K 和 V 分别连续 → readback 塌成 4 个连续 BD;代价转移到 append 侧的少量 strided 写。

配套的 `air.shim_feed_no_pace` fire-and-free 也要一起理解,否则会撞上 pacing 退化成 fence
的死锁。

### 4. Peano 的 `.bss`/`.data` 与寄存器压力陷阱 ⭐⭐

见 [06](06_fragility_and_gotchas.md) 第 4、5 条。这几条是硬知识:

- CDO loader 不清零 `.bss` → 持久状态放 `.data`;
- Peano 对 `.data` bool 的 RMW 会 miscompile → 状态放 stack local,进出口同步;
- static-init guard 在未清零 `.bss` 里可能跳过构造 → 常量声明成非 static;
- online-softmax 的多累加器 + 宽向量同时 live 会触发 spill/reload 缺陷 → 拆成两趟,
  让中间量提前 dead;热 helper 在 Peano 下用 **noinline** 而非 always_inline。

### 5. 一个 kernel 两种 MODE(RTP-gated)⭐

LM head 复用 proj 核,靠 RTP `IS_ATTN` 切换,零新增 flow。我们如果要在同一个 ELF 里塞
prefill/decode 两种形态,这是个可参考的组织方式。

## 不建议借鉴的

### 手工 pin 一切

`air.shim_col` / `air.tile_dma_channel` / packet-vs-circuit 的逐条手动指定,是在跟
placer 搏斗。可读性好(每条都有"不这么做会怎么死"),但**不可移植、不可复用**:换个
模型尺寸就要重新调一遍。我们的 builder 体系不应往这个方向走。

正确的读法是:**把这些注释当成 AIR placer 的 bug report 清单** —— 每一条手动 pin 都
对应一个 placer 本该自动做对却做错的场景。

### 巨型单文件

2777 行(HEAD 更多)且带大量过时注释。加一个模型要在 `_MODELS` 表里手工推导 `I2P`/`J2P`
并保证一堆整除关系(否则死锁)。Qwen 干脆另开了一个 2153 行的
`fused_decode_qwen.py` —— 这说明参数化表已经撑不住了。

### 只有 e2e 门

没有 per-kernel 的数值门。一旦 e2e 挂了,定位面是整个 module。我们的 registry + 逐 shape
验证在这点上明显更好,不应为了追这套设计而放弃。

## 具体的下一步建议

1. **优先试第 1 条(指令流插值)**。它独立于 fused decode 的其余部分,可以单独用在我们
   现有的 decode 路径上,收益是消除 per-L 重编。
2. **第 2 条(晚展开 N-wave launch)在层数多的模型上验证**。我们之前遇到过层数一多
   air-to-aie 就吃不消的情况,这可能是解法。
3. **把第 4 条(Peano 陷阱)写进 kernel 编写规范**。这是纯增量知识,零风险。
4. **不要**把 fused decode 整体搬进我们的 registry —— 它的抽象层级和复用假设跟 registry
   不兼容。

## 一个观察

这批代码的注释密度极高,而且大量注释是"**我试过 X,它以这种方式失败,所以用 Y**"的形式。
从工程记录的角度这是很好的实践 —— 大部分性能相关的决策都能追溯到一次具体的失败。
问题只在于**没有人回头清理**过时的部分(顶部 60 行、"the reference" 指代、死分支),
导致新读者要自己分辨哪些还成立。

如果我们要产出类似规模的代码,值得把"决策注释"和"当前状态描述"分开:前者可以留档,
后者必须随代码更新。
