# SmolVLA 代码导读 — 给第一次看这份代码的人

假设你没读过这里的任何一行代码。这份文档从最上层开始，回答四个问题：

1. SmolVLA 在做什么？
2. CPU baseline 在哪，怎么跑？
3. 我们的 NPU 实现在哪，改了什么？
4. 代码按什么顺序读？

技术细节在 [`explain.md`](explain.md)，实测数字在 [`profile.md`](profile.md)，
命令清单在 [`usage.md`](usage.md)。这份是入口。

---

## 1. SmolVLA 在做什么

一个**机器人策略模型**：给它几张相机图 + 一句话指令 + 机器人当前状态，
它输出接下来 50 步该怎么动。

```
3 张相机图 (3,512,512)  ─┐
"pick up the cube"       ─┼──→  SmolVLA  ──→  action chunk (50, 6)
机器人关节状态            ─┘                    未来 50 步的动作
```

内部分三段，**这个划分是理解全部内容的基础**：

| 段 | 做什么 | 关键 shape | 跑几次 |
|---|---|---|---|
| **① SigLIP 视觉编码器** | 图 → 视觉 token | seq **1024** | 每张图 1 次，**共 3 次** |
| **② SmolLM2-360M backbone** | 视觉+语言+状态 → 前缀表示 | seq 256 | **1 次** |
| **③ Action expert** | 前缀 → 动作，流匹配去噪 | seq 50 | **10 次**（去噪步） |

**这个 PR 只把 ① 搬到了 NPU。②③ 原样跑在 CPU 上。**

为什么？见第 6 节——简单说：① 的 shape 填得满 NPU 阵列，②③ 填不满。

---

## 2. CPU baseline 在哪 —— 它不在这个 repo 里

这是最容易困惑的一点。

**CPU baseline 就是官方的 lerobot 包本身**，是 pip 装的第三方库，
不是我们写的代码：

```
/home/jiajli/Projects/smolvla_playground/lerobot/src/lerobot/policies/smolvla/
    modeling_smolvla.py      ← 903 行，官方实现，我们一个字没改
```

我们的代码里三处都是直接加载它：

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
```

**所以"我们的 CPU baseline 在哪"这个问题的答案是：没有"我们的" CPU
baseline，只有官方的。** 这是有意的设计——参照物是官方模型，不是我
重写的一份，否则对比就没有意义了。

### 跑纯 CPU baseline

```bash
cd programming_examples/llms/smolvla
make oracle      # 用纯 CPU 跑一遍，把结果存进 smolvla_oracle.npz
```

它固定噪声（去噪过程本来是随机的，固定了才可复现），跑完整的官方模
型，存下 `action_chunk` (1,50,6)。这就是"正确答案"。

---

## 3. 我们改了什么 —— 只有一个函数

整个 NPU 集成，机制上只有一件事：

```
                原版 lerobot 完整跑一遍推理
                            │
    ┌───────────────────────┴────────────────────────┐
    │  只在 embed_image 这一个函数上做临时替换        │
    │      原版：CPU 跑 SigLIP                        │
    │      我们：直接返回 NPU 已经算好的结果          │
    │  算完立刻恢复（finally 保证，异常也恢复）       │
    └────────────────────────────────────────────────┘
                            │
              backbone / action expert / prefix 组装 /
              mask / 采样 / 去噪循环 —— 全是 lerobot
              自己的代码，一行没动
```

代码在 `smolvla_inference.py`，核心就这几行：

```python
vwe.embed_image = _npu_embed_image     # 换掉
try:
    return orig_embed_prefix(*a, **kw)  # 调原版 lerobot
finally:
    vwe.embed_image = orig_embed_image  # 换回来
```

**这就是为什么 `make verify` 有说服力**：比的是同一个官方模型，换掉
SigLIP 前后的差别。不是"我的实现 vs 我的另一个实现"。

### 跑 NPU 版本

```bash
make compile   # 编译所有 vision ELF（不碰 NPU，不下载模型）
make verify    # 【关键】NPU 版 vs 纯 CPU 版，输出 PASS/FAIL
make run       # 跑一次完整推理，打印 action chunk
make profile   # 分段墙钟：NPU vision vs 纯 CPU
```

`make verify` 当前输出：

```
cosine   = 0.998996     (门槛 0.99)
nmse     = 0.003023     (门槛 0.04)
[verify] PASS
```

> 每条 make 目标**自带 flock**，外面不要再套 `flock`，会自死锁。

---

## 4. 代码怎么读 —— 7 个 Python 文件，按数据流

按 `git diff` 的字母序读会很痛苦。按数据流读：

```
        图像
          │
          ▼
 ① smolvla_inference.py   210 行  ★★★ 接缝，唯一碰 lerobot 的地方
          │
          ▼
 ② smolvla_runtime.py     354 行  ★★  进程级单例
          │                            （权重/ELF/XRT 只建一次）
          ├─→ smolvla_vision_weights.py   345  HF 权重 → NPU 布局
          ├─→ smolvla_vision_builders.py  514  18 个 launch → 3 个 ELF
          └─→ smolvla_vision_npu.py       955  逐层 dispatch（最大）
                    │
                    └─→ smolvla_cpu_helpers.py  95  故意留在 host 的部分
          │
          ▼
 ③ verify_adapter.py      131 行  ★★★ gate
 ④ smolvla_prefix.py      233 行  ★   生成 CPU oracle
```

### 只想花 30 分钟？读这三个，约 700 行

**① `smolvla_inference.py`（210 行）** — 接缝。读完你就懂了整个集成机制。
重点看 `_wrapped_embed_prefix`，以及 `served["i"]` 这个计数器（它假设
lerobot 请求图像的顺序和 runtime 编码的顺序一致，这个假设目前没有断言
钉住，是我认为最该被质疑的一处）。

**② `smolvla_runtime.py`（354 行）** — 为什么要单例：权重加载、ELF 载入、
XRT context 创建加起来约 585 ms，每次推理都做一遍就全亏光了。所以做成
进程级的，建一次反复用。

**③ `verify_adapter.py`（131 行）** — gate 本身。它只比一样东西：
`oracle["action_chunk"]`，也就是机器人真正会执行的那个张量。
建议你**亲自跑一次 `make verify --cpu-vision`**：拿未改的模型跑自己的
baseline，应该正好是 1.0。如果不是，说明 harness 本身有问题。

### 剩下四个，抽查即可（`make verify` 兜底）

| 文件 | 干什么 | 看的时候注意 |
|---|---|---|
| `smolvla_vision_weights.py` 345 | HF checkpoint → NPU 布局 | 每个 Linear 有没有转置（HF 存 `[out,in]`，GEMM 要 `[in,out]`） |
| `smolvla_vision_builders.py` 514 | 拼 3 个融合 ELF | `_force_tile_n_suffix`，见第 7 节 |
| `smolvla_vision_npu.py` 955 | 逐层 dispatch | 最大的文件，里面有个 attention backend 的 A/B 开关（`"cpu"` 走主机），是它偏大的原因之一 |
| `smolvla_cpu_helpers.py` 95 | 3 个留在 host 的函数 | 见下 |

### 为什么有两步故意留在 CPU

| 步骤 | 原因 |
|---|---|
| im2col patch embed | layer loop **之前**的一次性 reshape，不是热路径 |
| pixel-shuffle | 纯 space-to-depth，**零算术**，已验证和 HF bit-exact |

connector 真正的计算（64×12288×960 投影）**在 NPU 上**。如果你觉得这两
条理由站不住，那是该 push back 的地方。

---

## 5. 性能从哪来 —— 融合，不是省驱动开销

一层 SigLIP 是 18 次 kernel launch。如果老老实实发 18 个程序：

```
每张图 121 次 dispatch  →  368 ms
```

`smolvla_vision_builders.py` 把它们拼成 **3 个多 launch ELF**：

| ELF | launch 数 | 内容 |
|---|---|---|
| `vit_ln_qkv` | 7 | LayerNorm + Q/K/V GEMM + 3 个片上 bias-add |
| `flash_attn` | 1 | registry 的 FlashAttention，原样用 |
| `vit_o_ffn` | 10 | O GEMM + bias + residual + LN + fc1 + bias + GELU + fc2 + bias + residual |

```
每张图 38 次 dispatch  →  141.6 ms     (2.6×)
```

**但收益主要不是省驱动开销。** 是把 bias-add 和 residual 挪到片上，消掉
了每次操作的 bf16 → f32 → bf16 主机往返。主机侧空隙从 212 ms 降到 4 ms。

**精度还顺带变好了**（编码器输出 cosine 0.945 → 0.9906），因为那个往返
本来在反复重量化每个中间结果。

---

## 6. 为什么只有 vision 上 NPU

三个 stage **全都 port 过、也都验证通过了**。只有 ① 更快：

| Stage | seq · 频次 | CPU | NPU | 归属 |
|---|---|---|---|---|
| ① SigLIP + connector | **1024** · ×3 | 546 ms | **465 ms** | **NPU，1.19×** |
| ② backbone | 256 · ×1 | **77 ms** | 229 ms | CPU |
| ③ action expert | 50 · **×10** | **285 ms** | ~4× CPU | CPU |

原因是 shape 填不填得满 8×4 阵列。**同一个 matmul kernel** 实测 GFLOP/s：

```
seq 1024  →  3798–5790      填满
seq  256  →  2046–3366      一半
seq   50  →    25–1163      padding 到 64，只能占半个阵列
```

另外两个原因：每次 launch 有约 85 µs 固定开销；registry 的 FlashAttention
**不施加 mask**，而 ②③ 都需要真 mask，只能退化成每层 11 次 dispatch 的
分解注意力。

**②③ 的 NPU 实现不在这个 PR 里。** 它们存在、也过了各自的门槛，但收进来
会让 diff 翻倍，还要拖进一个只有它们需要的 `masked_softmax` example。
可以单独提。

---

## 7. 审核时该盯的地方

按风险排序：

**① 接缝有没有偷改 lerobot 语义**（`smolvla_inference.py`）
假装你是 lerobot 维护者，问："这个 wrapper 改变了我任何可观测的行为吗？"
答案应该是"只有 embed_image 的返回值来源，别的都没动"。

**② gate 够不够严**（`verify_adapter.py`）
cosine 实测 0.9990 / 门槛 0.99，余量约 10×。nmse 实测 0.0030 / 门槛 0.04，
余量约 13×。SmolVLA 输出连续动作而非 token，所以用 `regression_gate`，
不是兄弟模型的 top-k token-set gate。

**③ 一个已知隐患，本 PR 不修**
`compile_gemm_mm` 把 `DIM_M`/`DIM_N`/`DIM_K` 编译进 `mm.o`，但共享 helper
只用 `tile_n` 命名这个 object。vision 在 seq=1024 时会解析出两个不同的
`tile_n`（96 和 128），所以 `_force_tile_n_suffix` 强制用带 tile_n 的名字
——第一次没做时 cosine 掉到 0.07。

命名没考虑 `DIM_K`，本 PR 新增的 registry 行里有 `tile_k_l1=48` 的 shape，
**扩大了暴露面**。已在 PR 描述里写明。

> 我尝试修过一次，**改坏了 qwen25_0_5b**（verify PASS 2/2 → FAIL 0/2），
> 已撤回。失败原因值得记住：我用 `make compile` 当验收标准，而编译通过
> 只证明 linker 找到了 object，**不证明它的符号含义和 IR 一致**。而且没
> 留改动前的 baseline，分不清是"发现旧 bug"还是"引入新 bug"。重修时
> 必须先录 per-model `make verify` baseline，并用 verify 当 gate。

**④ PR 边界外的改动**（影响其他 9 个模型）

| 路径 | 改了什么 | 风险 |
|---|---|---|
| `shared/gemm_builder.py` | 加 `disambiguate_by_tile_n` | 加性；一个 method 下 tile_n 统一时是 no-op（现有 caller 全部如此） |
| `shared/o_ffn_multi.py` | 改成按 shape 查 registry | 需确认兄弟模型仍能跑 |
| `verify/comparators.py` | 加连续输出 gate | 加性 |
| `kernel_registry/` | 19 个新 GEMM shape + LayerNorm/GELU 详情页 | 其中 15 行是 action-expert 的 shape，而本 PR 不含 expert 代码 |

最后一条是有意保留的：registry 的用途是记录所有真机验证过的
(kernel, shape)，删了以后还得重测。但这该由你拍板。

---

## 8. 五分钟上手

```bash
cd programming_examples/llms/smolvla

make help      # 目标清单
make compile   # 建所有 ELF（不碰 NPU、不下载）
make oracle    # 纯 CPU baseline
make verify    # 门禁
make profile   # NPU vs CPU 分段墙钟
```

所有数字都在 **CPU governor + EPP `performance`、NPU `pmode=Turbo`、
机器空闲、配置交错测量** 的条件下取得。`balanced` 模式下读数差别很大，
而且 CPU baseline 比 NPU 段变化更多——**变的是比值，不只是绝对值**。
