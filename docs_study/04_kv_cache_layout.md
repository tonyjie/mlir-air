# 04 — region-major KV cache 与 fire-and-free readback

decode 的 KV cache 在 DDR 里,每 token 都要:(1) 追加这个 token 的 K/V;(2) 把整个
cache 读回来做 attention。布局选择直接决定了 shim DMA 的效率,这里是这批代码里
第二值得学的部分。

## 两种布局

### 交错(per-token interleaved)—— 被否决的方案

```
[tok0: K(512) | V(512)][tok1: K | V][tok2: K | V] ...
```

直觉上很自然,追加时只要写一个连续的 1024。但**读回时是灾难**:attention 的 qk 核只要
K,kv 核只要 V。要读出所有 token 的 K,就得跳过每个 token 的 V —— 这是
**strided-with-holes**,不可合并(non-coalescible)。注释给了实测代价:

> ~1 shim task/token (~4100 @L2k)

### region-major(quadrant)—— 实际采用

```
[ K_grp0 | K_grp1 | V_grp0 | V_grp1 ]     每个 region 连续 ATTN_MAXL * REGION_W
```

每个 group 的 K(或 V)在 DDR 里是**一整段连续**。于是整个 readback 塌成
**4 个连续 BD**,在两条独立通道(`inKV_K` / `inKV_V`)上并发流式推送。代价转移到 append 侧:
现在要把一个 token 的 K/V **散射**进 4 个 region,但那是每 token 常数次的少量 strided 写。

用实测数字表述(Llama-3.2-1B @ ATTN_MAXL=2048):

| 量 | 值 |
|---|---|
| `NGRP`(group 数) | 2 |
| `REGION_W`(每 group 每 token 宽度) | 256 |
| `REGION_STRIDE`(一个 region 跨度) | 524,288 elem |
| `KVSZ_TOK`(每 token K++V) | 1024 elem |
| 每层 cache | 2,097,152 elem |
| 16 层合计 | **67.1 MB** |

偏移计算就两行:

```python
def _kreg_off(gi):  return gi * REGION_STRIDE                 # group gi 的 K region
def _vreg_off(gi):  return (NGRP + gi) * REGION_STRIDE        # group gi 的 V region
```

## append → readback 的 RAW 定序

同一个 token 内,append 必须先于 readback 完成(读的是刚写的槽位 L-1)。这是跨 DMA 的
RAW 依赖,靠两个属性表达:

```python
_apkG.operation.attributes["air.append_barrier"] = UnitAttr.get()   # 打在 append 上
_pk.operation.attributes["air.await_appends"]   = UnitAttr.get()   # 打在首个 readback 上
```

`AIRRtToNpu` 会把 append 的完成等待移到被标记的 readback 之前。

append 由 **rope 核**在片上发起(roped-K / raw-V 直接 S2MM 到 DDR),不经过 host。
对应 FastFlowLM 的 `_receive_kv_cache`;readback 对应 `_move_kv_cache`。

## fire-and-free:绕开 launch 的 depth-2 pacing

这里有个很微妙的死锁,注释记录得很详细,值得完整理解:

融合的 N-wave launch 会对每个 `preserve_shim_dma_order` 通道**按 wave** 做 depth-2 pacing
(`synthesizeDoubleBufferedAwaits`)。如果一个通道每 wave 只有 1 个 task,它会退化成
**fence**(start 后立即 inline await)—— 结果 K 被完全串行化在 V 之前。而 qk→score→kv
是流水线:K 那个 128-block 的 BD 在 V 还没开始时**排不空** depth-2 ring → **死锁**(大 L 时)。

解法是给每个 readback put 打:

```python
_pk.operation.attributes["air.shim_feed_no_pace"] = UnitAttr.get()
```

这让 AIR 把它排除在 depth-2 pacing 之外,lower 成 **fire-and-free** MM2S feed —— 只由
memtile 的 ring lock 提供背压,K/V 在两条独立通道上真正并发。这正是 FastFlowLM 那 4 个
`npu_dma_memcpy_nd` 的行为。

顺带一提,`air.shim_feed_no_pace` 这个属性本身就是为这个场景加进 AIR 的
(commit `8fca978a`, PR #1754)。

`DECODE_KV_RB_NRB` 控制把每个 region 切成几个 chunk;默认 `NRB=1`,即
**2×NGRP = 4 个整 region 连续传输**,与参考实现一致。注释解释了 NRB>1 存在的历史原因
(在还没有 `shim_feed_no_pace` 时,用"chunk 数 > ring depth"来强制 batching 而非 fence)。

## host 侧的配合:seed + 常驻

prefill 产出的 K/V 要按同样的 region-major 布局摆进 device BO:

```python
for Lyr in range(16):
    for g in range(NG):
        self.KV[Lyr, g*RS : g*RS + P*RW].reshape(P, RW)[:] = fk[Lyr, :P, g*RW:(g+1)*RW]
        self.KV[Lyr, (NG+g)*RS : (NG+g)*RS + P*RW].reshape(P, RW)[:] = fv[Lyr, :P, g*RW:(g+1)*RW]
```

关键优化在注释里:KV BO **只上传一次**(seed 阶段,且刻意放在计时之外),之后
device-resident,kernel 每 token 在片上追加到槽位 L-1。**不能每 token 重新打包 + sync**
那 67 MB —— 注释给了实测代价:

> 那个 host copy 主导了 per-token 时间,把 chatbot 从 ~43 tok/s 压到 ~24 tok/s

这条和我们已知的 `opt-buffer-object-reuse`(B1/B2)是同一类优化,但这里是 decode 场景下
最极端的一例:buffer 大(67MB)、复用频率高(每 token)。

## 单缓冲是必须的

块循环**无条件**关掉 ping-pong:

```python
_blk.owner.owner.attributes["air.disable_ping_pong"] = UnitAttr.get()
```

原因:ping-pong 会把循环 unroll-by-2 + 留 1 个余数,而余数迭代在 3-buffer 的 toK/toV ring
上**读错 buffer**(与 DMA 轮转错位)→ KV 错位。症状很有欺骗性:**首 token 正确,后续
全是垃圾**(因为 L=1 时只有一块,没有余数问题)。

这是个值得记住的调试指纹:*"第一个 token 对、之后胡言乱语"* → 先怀疑 ring 对齐 / 缓冲轮转,
而不是数值精度。
