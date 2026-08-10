# 03 — 一个 xclbin 服务任意 context length

这是整批代码里**最值得学的一招**,也是我唯一做了独立验证的主张。

## 问题

自回归 decode 每生成一个 token,KV context 长度 `L` 就 +1。attention 要读 `ceil(L/16)` 个
KV block。常规做法有三种,都不好:

1. **每个 L 编一份 xclbin** —— 2048 个 build,不可能。
2. **分档 window**(L≤256 一份、≤512 一份…)—— 切换时要重新 load xclbin,而且档内仍在浪费。
3. **让块数变成 runtime 值** —— AIR 里 `scf.for` 上界如果是 runtime value,能活到 core codegen
   (`unrollSCFFors` 只展开全常量循环),但 shim 侧推多少 KV 仍然要跟 core 消费量对齐。

## 解法:编译期固定块数 + kernel 内跳过 + host 端插值

### 第一步:块数固定为最大值,kernel 自己跳过

xclbin 在 `ATTN_MAXL=2048` 编译一次,attention 是**编译期常量** 128 块的循环:

```python
_nblk_qk = idx(ATTN_ROUNDS)      # 128,编译期常量
for _blk in for_(idx(0), _nblk_qk, idx(1)):
    _blk.owner.owner.attributes["air.disable_ping_pong"] = UnitAttr.get()
    ...
```

kernel 里判断当前块是否完全超出 `L`,是就直接返回:

```c
// attn_qk.cc
int rem = L - blk * 16;
if (rem <= 0) return;              // 全 masked 块:跳过
rem = (rem < 16) ? rem : 16;
aie::mask<16> mask = aie::le(idx, rem);   // 最后一块做部分 mask
```

```c
// attn_kv.cc — 必须成对跳过
if (blk == 0) { /* reset y_state, l_state */ }
if (L - blk * 16 <= 0) return;     // qk 没产 s_block,kv 就不能消费
```

**为什么必须"跳过"而不是"算了再 mask 成 -inf"**:注释说得很清楚 —— 对全 masked 块跑
`_attn_qk` 会把 -inf 喂进 online softmax 的 max/rescale,**污染 m/c 状态**。数学上
"贡献 exp(-inf)=0" 只对已经 normalize 的形式成立,对 running-max 的增量形式不成立。

注意 qk 和 kv 的跳过条件必须**完全一致**,否则一边产一边不消费,ping-pong ring 立刻错位。

### 第二步:host 端把 L 相关的指令词插出来

剩下的问题是 runtime 指令流(insts.bin)里有一些 L 相关的字:RTP-L 的值、KV-append 的
字节偏移。作者的观察是:**这些字是 L 的线性函数**。

于是编**两个同 ATTN_MAXL 的 build**(L=2048 和 L=2047)求差得斜率:

```python
d = ref.astype(np.int64) - base.astype(np.int64)
dL = ref_L - base_L
if (d % dL == 0).all():            # 整除性检查 = 线性假设的自检
    slope = (d // dL).astype(np.int64)
```

任意 L 直接插值:

```python
out[ld] = base[ld] + (L - base_L) * slope[ld]
```

### 第三步:每 token 只改动那几百个字

`FusedDecoder.dispatch()` 里,insts BO 只在第一次全量写入,之后每 token 只覆盖并
sync `[lo:hi]` 区间。

## 实测数据(本机)

```
template ATTN_MAXL=2048: base_L=2047 builds=[2047, 2048] L-dep_words=264 (active)
指令流总长: 29264 words (117056 bytes)
L 相关词:  264 words (0.9%)
斜率种类:  只有两种 —— slope=1 (200 words), slope=512 (64 words)
分布区间:  [14, 28658]
```

两种斜率的含义很清楚:`slope=1` 是 RTP-L 计数本身;`slope=512` 是 KV-append 的字节偏移
(512 = `REGION_W`(256) × 2 bytes)。

## 独立验证:这个主张是真的

自带的 `python3 decode_insts_gen.py` self-check 有个方法论问题 —— 它**只验证用来拟合的
那两个点**(2047/2048),属于循环论证:

```
  ATTN_MAXL=2048 L=2047 byte-exact=True     ← 拟合点
  ATTN_MAXL=2048 L=2048 byte-exact=True     ← 拟合点
ALL BYTE-EXACT
```

所以我在同一个 ATTN_MAXL 窗口内**原生编译了第三个点 L=2033**(从未参与拟合),再与
生成器的合成结果逐字节对比:

```
native words 29264 | gen words 29264
INDEPENDENT CHECK L=2033: BYTE-EXACT
```

**主张成立。** 29264 个词全部相同,包括那 264 个 L 相关词。

复现方法见 [05_reproduction_log.md](05_reproduction_log.md)。注意原生构建必须在
`fused_decode/` 源目录里跑 —— inline-attn merge 要求 `attn_qk.ll` / `attn_kv.ll` 在
工作目录,换目录会得到 `llvm-link: No such file or directory: 'air_project/attn_kv.ll'`。

## 这一招的边界

- **只在同一个 `ATTN_MAXL` 窗口内有效**。窗口由 `attn_maxl_of(L) = 16*ceil(L/16)` 决定;
  跨窗口的两个 build 指令流长度不同,不能互相插值。生成器按 ATTN_MAXL 分组管理模板。
- **需要两个 build 标定**。`select()` 会拒绝只有一个 build 的组(`slope is None`)。
- **有隐含的线性假设**,由 `(d % dL == 0).all()` 这个整除性检查兜底;不满足就 `slope=None`,
  该模板不可用,而不是给出错误结果。这个设计是稳妥的。
- **代价:大 L 时在浪费**。L=100 时仍然跑满 128 块循环的控制流(虽然 kernel 立刻 return,
  shim 侧的 readback 由 `RB_ROUNDS` 控制推送量)。这是"一个 xclbin 通吃"换来的。

## 为什么这招对我们有用

我们现在的 decode 路径每换一个 context 长度就要重新走一遍编译。这套办法把"L"从
**编译期常量**降级成**指令流里的几百个字**,而且降级过程是可验证的(byte-exact vs 原生
构建)。即便不照搬整个 fused decode,单独把"两点标定 + 线性插值 + 整除性自检"这个模式
用在我们自己的 runtime 参数上,也是可行的。
