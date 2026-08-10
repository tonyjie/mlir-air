# 02 — fused decode 超算子的设计

一次 XRT dispatch 跑完 16 层 decoder + LM head。本文按"空间上怎么摆"和"时间上怎么走"
两条线拆解。

以下所有数字是 Llama-3.2-1B @ `ATTN_MAXL=2048`,从 `fused_decode.py` 实际 import 出来的。

## 空间:tile 怎么分配

NPU2 是 8 列 × (1 shim + 1 memtile + 4 compute) 的阵列。这个 module 的占用:

| 角色 | 物理位置 | 数量 |
|---|---|---|
| proj 核(GEMV) | cols 0,1,6,7 × rows 2..5 | 16 |
| rms 核(norm + residual + logits 转发) | col 2, row 2 | 1 |
| rope 核 | col 2, row 3 | 1 |
| attention CU(qk + kv 成对) | cols 3,4 × rows 2..5 | 8(= 4 CU) |
| GLU 核 | col 5, row 3 | 1 |
| memtile | col 0/6(group)、col 1(main)、col 2(X)、col 3/4(KV)、col 5(down) | — |

16 个 proj 核组成 **8 个 cascade pair**:每列两对,lead 在 row 2/4,partner 在 row 3/5。
一对共享 lead tile 上的两个 L1 y-buffer(`memref<80>` = 16 header 区 + 2×32 payload)。
lead 写 packet id 到 header 并发 2 行的包(offset 14, size 66),partner 把自己那行
**跨 tile 写进同一个 lead buffer**。

## 时间:4 个 phase 复用同一批核

proj 核不是"每个算子一批核",而是同一批核跑 4 个 phase,靠 `scf.index_switch` 选参数:

| phase | 内容 | packet id | I2(行对迭代) | J2(列块对) | K |
|---|---|---|---|---|---|
| 0 | QKV proj | 1 | 3 | 4 | 2048 |
| 1 | o-proj | 4 | 2 | 4 | 2048 |
| 2 | gate-up | 8 | 16 | 4 | 2048 |
| 3 | down | 4 | 2 | 16 | 8192 |

注意 phase 3 的 K 是 8192(INTERMEDIATE),其余是 2048(MODEL_DIM)。weight memtile fan 和
X memtile 是 **phase 无关的扁平流**(一条连续 ring),只有 compute core 带 phase 结构 ——
所以 runtime 把各 phase 的权重 slab 直接拼接,fan 循环跑总步数即可。

### 数据在核间怎么流

```
host X ──@rmsX──► rms核 ── rmsnorm ──@xnorm──► X memtile ──@inX(broadcast)──► 16 proj核
                                                                                  │
                              ┌───────────────────────────────────────────────────┘
                              ▼ (packet id 写在 header 里)
                 group memtile(非对称 gather, 258) ─► main memtile(菊花链, 514)
                              │
                              ▼ 一个 egress 按 id 分流(switchbox 路由 + strip header)
              ┌───────────────┼────────────────────────┐
        id1 (QKV)        id8 (gate-up)            id4 (o-proj / down)
              ▼                ▼                        ▼
           rope核          GLU核(直达,无 relay)      rms核(residual1 / residual2)
              │                │
       ┌──────┴──────┐         └── silu(gate)*up ─► down memtile(8192)
       ▼             ▼                                   │
   Q → q memtile  K/V → append 到 DDR KV cache            └─ 片上回喂成 ph3 的 X
       │             │                                       (NOT 走 host)
       ▼             ▼
   4 个 attention CU ◄── KV readback ──┘
       │
       └── attn o ─► o memtile ─► 作为 ph1(o-proj)的 X(闭环)
```

几个设计点值得单独说:

**packet-id demux 而不是每个目的地一条通道。** kernel 自己往 wire header 写 id
(`proj_qmm_flush_hdr` → id@14),下游 switchbox 按 id 路由,每个 dest 再 strip 掉 header
拿到纯 payload(512)。好处是 proj 侧只有**一条** egress flow,不用为 4 个 phase 开 4 套
通道。`KIDP=[1,4,8,4]` 里 o-proj 和 down 复用 id4——因为它们的消费者都是 rms 核。

**GLU 输出片上回喂。** gate-up 出 16384,demux 直达 GLU tile(`keep_pkt_header=false`,无
relay),`silu(gate)*up` 出 8192,存进 down memtile,然后由 down_buffer **重新广播** 4 次
(`DOWN_REFEED`)进 `@xnorm` 通道作为 phase 3 的 X。**完全不落 host**。

**loop-close:o-proj 的 X 是 attention 的输出。** `@xnorm` 是一条**汇聚(convergent)**通道,
按 phase 时序先后承载 4 个来源:ph0 rmsnorm(input) → ph1 attn-o → ph2 rmsnorm(x+oproj) →
ph3 GLU。由**一个** get 循环读取。注释强调必须是一个循环:两个 feed 循环会 lower 成两个
`repeat_count` task,那是 stale-rebroadcast 死锁。

**residual 在 rms 核上做两次**,layer 输出 in-place 写回 arg0(hidden BO):

```
residual1 = input + o-proj-out  → h
residual2 = h + down-out        → layer output → 写回 arg0[0]
```

所以第 N 层的输出就是第 N+1 层的输入,**同一个 BO**,靠 `air.preserve_shim_dma_order`
保证程序序。这是从 FastFlowLM 参考实现继承的 chaining ABI。

## X 的 re-feed 机制(两种)

一个 token 的 X 要被反复读:每个核每个 phase 出 `I2*2` 个 row-block,每块都要读一遍完整的 K。
`REFEED=[6,4,32,4]`,合计 46 次。实现上有两种机制:

- **机制 1(生产者侧)**:rms 核把输出锁 release N 次 —— `air.refeed_count` 加在**通道**上
  (ph0 用 channel-level count=6),或加在**单次 emission** 上(ph2 用 per-put override=32)。
- **机制 2(memtile 侧)**:down_buffer 用自己 alloc 上的 counting-lock 重播。注释特别指出
  memtile producer **不能**再乘 channel 的 refeed count,AIRToAIE 会跳过。

## unified launch:25 个 wave 一次发射

最外层不是 16 次 launch,而是**一个** launch 包在 `scf.for(0, 25)` 里:

```python
for _iv in for_(idx(UNI_WAVE_LO), idx(UNI_WAVE_HI), idx(1)):
    launch(sizes=[1,1], operands=list(_fa) + [_iv],
           attributes={"air.preserve_shim_dma_order": UnitAttr.get()})(launch_body)
```

`UNI_DEC=16`(decode wave)+ `UNI_LM=9`(LM head vocab chunk)= `UNI_WAVES=25`。
每个 wave 由 `arm = (iv < UNI_DEC) ? 1 : 0` 驱动:

- herd 内部的 `index_switch` 选 decode 分支还是 vocab 分支(靠 RTP `IS_ATTN`);
- launch scope 的 `index_switch` 选对应的 host feed。

**关键点:device(segment/herd)只 emit 一次**,per-layer DDR offset 由归纳变量
`a_iv` 缩放,而且这个 `scf.for` 是**晚展开**的(在 airrt-to-npu 才 unroll)。所以
`air-to-aie` 看到的 op count 是**常数**,不随层数增长——否则 16 层展开会直接把编译器压垮。
aie.device(→ xclbin/CDO)与单层构建**逐字节相同**,只有 runtime 指令流变长。

这是整套设计里可复用性最高的手法之一,比单纯把 N 个 launch 融成 multi-launch ELF 走得更远。

## LM head 也是同一批 proj 核

LM head 不是独立 kernel,而是同一批 proj 核 + rms 核的一个 **RTP-gated MODE**
(`IS_ATTN=0`)。vocab GEMV 在结构上就是 QKV phase(同样的 `proj_qmm_acc256`,K=MODEL_DIM),
只是把 I2 放大到覆盖 vocab 行,并用 `RMS_DEST`(id4)发出——完全复用现有路由,proj 侧
零新增 flow。rms 核在 mode 0 下做 final rmsnorm 并把收回的 vocab chunk 转发出 shim 当 logits。

**为什么要切成 9 个 chunk**:单次全 vocab dispatch 不可构建 —— 8064 个 launch inW put 会
压垮 air-to-aie,per-round drain 会耗尽 shim BD ID。`VOCAB_I2=14` → 每 chunk 448 row-block,
4032/448 = 9 次 dispatch。

## 关键几何数字(实测 import 得到)

| 量 | 值 |
|---|---|
| `K` / `M` | 2048 / 3072 |
| proj 核 | 4×4 = 16,8 cascade pair |
| `ATTN_ROUNDS` / `ATTN_MAXL` | 128 / 2048 |
| attention CU | 4(每 CU 8 q-head + 2 kv-head) |
| `KVSZ_TOK`(每 token K++V) | 1024 elem |
| KV cache 每层 | 2,097,152 elem → 16 层共 **67.1 MB** |
| 权重每层 | 19,005,440 elem → **38.0 MB** |
| `UNI_DEC` / `UNI_LM` / `UNI_WAVES` | 16 / 9 / 25 |
| `PAYLOAD`(每 egress round) | 512 |
| `GLU_OUT` | 8192 |
