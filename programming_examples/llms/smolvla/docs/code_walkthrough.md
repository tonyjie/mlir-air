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

**CPU baseline 就是官方的 lerobot 包本身**，是 pip 装的第三方库
（`requirements.txt` 里声明），不是我们写的代码：

```
<site-packages>/lerobot/policies/smolvla/
    modeling_smolvla.py      ← 官方实现，我们一个字没改
```

想找到它：

```bash
python3 -c "import lerobot.policies.smolvla.modeling_smolvla as m; print(m.__file__)"
```

（`requirements.txt` 装好后，这在 mlir-air 环境里直接可用。）

这和兄弟模型的做法一致——`llama32_1b` 的 CPU 参照是 HF `transformers`
里的 Llama 实现，同样是 pip 依赖，同样没有 vendor 进本仓库。

我们的代码里四处都是直接加载它 —— `smolvla_inference.py` 两处、
`verify_adapter.py`、`smolvla_cpu_baseline.py`：

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

## 2b. 一次 CPU 推理具体在跑什么

下面每个 shape 都是**在真机上 instrument 一次真实推理测出来的**，不是从
代码推的。看懂这一节，第 3 节的替换就是显然的。

### 入口

```
predict_action_chunk(batch, noise)
  ├─ _prepare_batch(batch)                     归一化
  └─ _get_action_chunk
       └─ VLAFlowMatching.sample_actions(...)  ← 真正的流程在这
```

### 输入

```
observation.images.camera1/2/3     (1, 3, 256, 256) f32   3 台相机
observation.language.tokens        (1, 48)          i64   补齐到 48
observation.language.attention_mask(1, 48)          bool
observation.state                  (1, 6)           f32   机器人关节
```

### 中间步骤

```
① embed_prefix ─────────────────────────────────────────────────
     embed_image (1,3,512,512) → (1,64,960)   ×3   ← SigLIP，本 PR 换这里
        （注意 512：lerobot 预处理把 256 resize_with_pad 到 512）
     img_emb *= sqrt(960)                          ← 缩放在替换点之后！
     + 语言 token + state，拼接
   ⇒ prefix_embs   (1, 241, 960)
     pad_masks     (1, 241)   其中真 token = 197  ← 见下面「241 从哪来」
     position_ids  = cumsum(pad_masks) - 1        ← padding 不推进 position

② vlm_with_expert.forward(use_cache=True) ─────────────────────
     attn_mask (1, 241, 241)  非因果，但 state 前有一道块边界（见下）
   ⇒ DynamicCache: 16 层 × k (1, 5, 241, 64)      ← GQA 5 个 KV 头
     只要这份 KV，输出直接丢掉（源码里就是 `_,`）

③ euler_integrate(denoise_step, ×10) ──────────────────────────
     每一步：
       embed_suffix(x_t)  (1,50,32) → (1,50,720)  ← 动作+时间步，expert 宽度 720
       attn_mask (1, 50, 291)                     ← 291 = 241 prefix + 50 suffix
       vlm_with_expert.forward(past_key_values=…)
       past_key_values.crop(241)                  ← 裁掉本步追加的 suffix K/V
       action_out_proj → v_t                      速度场
     x ← x + Δt·v_t
```

实测调用次数：`embed_image` **3 次**，backbone forward **1 次**，
denoise forward **10 次**。

### 输出

```
action_chunk (1, 50, 6) f32     未来 50 步、每步 6 个关节
```

### 241 从哪来，197 又从哪来

prefix 是三部分拼接：

```
  3 台相机 × 64 token  =  192      全部是真的
  语言槽               =   48      本例只有 4 个真，44 个 padding
  state                =    1      真的
  ──────────────────────────────
                          241      真 token = 192 + 4 + 1 = 197
```

**44 个 padding 全部来自语言。** tokenizer 是这样调的：

```python
padding="max_length", max_length=cfg.tokenizer_max_length   # 48
```

不管指令多长都补齐到 48 个槽，这样张量形状固定、不随 prompt 变。本例的
`"pick up the cube"` 只 tokenize 成 4 个 id（`[18188, 614, 260, 20636]`）。

所以 **197 不是常数**，是这个 prompt 的属性：换一句更长的指令，真 token 变
多、padding 变少。图像和 state 永远是真的，只有语言段会有 padding。

### 三种模态，三条编码路径

| | 输入 | 怎么编码 | 输出 |
|---|---|---|---|
| **图像** | (3,512,512) ×3 | **SigLIP ViT 12 层 + connector** | 各 64×960 |
| **语言** | 48 个 token id | `embed_language_tokens` —— **查 embedding 表** | 48×960 |
| **state** | (1,6)，补到 32 | `state_proj` —— **一层 `Linear(32→960)`** | 1×960 |

计算量差几个数量级：图像是一整个 12 层 Transformer（每张图 1024 个 patch
token 跑完再压成 64）；语言就是查表，**没有任何 Transformer**；state 字面
意义上只有一层全连接。

**这直接支撑了「只搬 vision」这个决定**——另外两条路径的算力可以忽略。

两个容易漏的细节：

```python
img_emb  = img_emb  * sqrt(960)        # 图像：有缩放
lang_emb = lang_emb * math.sqrt(960)   # 语言：有缩放
state_emb = self.state_proj(state)     # state：没有，直接 append
```

### prefix 和 suffix，以及真实的注意力结构

模型内部只有**一条序列**，由两段拼成：

```
[  prefix 241 token  |  suffix 50 token  ]
    观测，算一次就固定      正在去噪的动作，每步都变
```

`att_masks` 里的 `1` 标记块边界，而它出现了**两次**——所以不是简单的
「prefix 内部双向」，是三段递进：

```python
att_masks += [0] * num_img_embs      # 图像
att_masks += [0] * num_lang_embs     # 语言
att_masks += [1] * states_seq_len    # state ← 边界一
att_masks += [1] * chunk_size        # suffix ← 边界二（在 embed_suffix 里）
```

```
[图像 + 语言]  ←→  互相双向
    state       →   能看图像和语言，图像/语言看不见它
    suffix      →   能看前面全部，前面全部看不见它
```

**第二道边界正是 prefix 的 KV 能缓存的原因**——prefix 看不见 suffix，所以
prefix 的 K/V 与 suffix 无关，backbone 算一次、10 步去噪反复读同一份。

### 这对换 NPU 意味着什么

三段的边界恰好都是干净的函数边界，所以每一段都能独立替换，不用改
lerobot 一行代码：

| 段 | 边界函数 | 每次推理调用 | seq |
|---|---|---|---|
| ① SigLIP | `embed_image` | **3** | 1024 |
| ② backbone | `vlm_with_expert.forward` | **1** | 241→256 |
| ③ expert | `denoise_step` | **10** | 50 |

调用次数和 seq 直接决定了哪一段值得搬：① 形状填得满阵列、启动开销摊得
薄；③ 只有 50 个 token 却要跑 10 遍，两头都吃亏。详见第 6 节。

> **`VLAFlowMatching.forward()` 是训练路径，推理不走它。** 它返回
> `F.mse_loss(...)`，且 `use_cache=False`、prefix 和 suffix 一起喂。读代码
> 时容易和 `sample_actions` 混。

> **lerobot 0.6.1 起 `past_key_values` 是 `DynamicCache`，不能 `pkv[i]`
> 下标访问。** 0.5.0 的写法在新版会 `TypeError`。要重做 backbone 移植
> （它靠覆写 KV cache）的话，这是第一个会踩到的地方。

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
 ① smolvla_inference.py   274 行  ★★★ 接缝 + 唯一的 CLI 入口
          │                            （--compile-only 也在这里）
          ▼
 ② smolvla_runtime.py     243 行  ★★  进程级单例
          │                            （权重/ELF/XRT 只建一次）
          ├─→ smolvla_vision_weights.py   345  HF 权重 → NPU 布局
          ├─→ smolvla_vision_builders.py  514  18 个 launch → 3 个 ELF
          └─→ smolvla_vision_npu.py       919  逐层 dispatch（最大）
                    │
                    └─→ smolvla_cpu_helpers.py  95  故意留在 host 的部分
          │
          ▼
 ③ verify_adapter.py       131 行  ★★★ gate
 ④ smolvla_cpu_baseline.py  54 行  ★   生成 CPU oracle
```

`smolvla_runtime.py` 从上到下就是它实现的生命周期，可以顺着读：

```
1. ensure_kernels       编译 ELF，或复用磁盘上的 cache
2. VisionRuntime        权重 + ELF + XRT context，只建一次
   .encode(images)      → (N, 64, 960)
3. get_vision_runtime   让第 2 步每进程只发生一次的单例
```

（主机 BLAS 线程钳制那 125 行 ctypes 已经挪去
`shared/infra/thread_limits.py`，它和 vision 无关。）

---

## 4b. 要把另一个 stage 搬上 NPU 该怎么开始

三个 stage 都 port 过。它们的接缝是**三种不同的机制**，不是同一个模式的
三份拷贝——知道这点，将来做 backbone 或 expert 时不会走弯路。

```
lerobot 的推理流程                     替换点
────────────────────────────────────────────────────────
embed_prefix(images, lang, state)
   │
   ├─ embed_image(img) ×3       ←──① Vision：换掉这个函数，
   │                                  直接返回 NPU 算好的 (64,960)
   ▼
vlm_with_expert.forward(fill_kv_cache=True)
   │  CPU 跑完 16 层，产出 KV cache
   │                            ←──② Backbone：不拦截，等它算完，
   │                                  再整个覆写 pkv[i]["key_states"]
   ▼
denoise_step(...) ×10           ←──③ Expert：换掉整个函数，
   │                                  10 步去噪全在 NPU
   ▼
action_out_proj → action chunk
```

| Stage | 换什么 | 时机 |
|---|---|---|
| Vision | `vwe.embed_image` | **之前** —— 换掉输入的生产者 |
| Backbone | `vwe.forward` | **之后** —— 先让 CPU 算，再覆写结果 |
| Expert | `m.denoise_step` | **整体** —— 换掉一整步 |

② 的代价值得注意：它是"先在 CPU 上算一遍再扔掉"。这是 backbone 在 NPU 上
不划算的原因之一，但不是全部（主要还是 seq=256 填不满阵列，见第 6 节）。

共同点只有一条：**都用 `try/finally` 恢复**，任何异常都不会污染后续推理。

### 两个值得先知道的坑

**坑 1：hook 挂在 decoder layer 上不会触发。** lerobot 不调用整个
`LlamaDecoderLayer`，而是手动调它的子模块（`input_layernorm`、
`self_attn.{q,k,v,o}_proj`、`mlp`）再自己做残差加。所以
`layers[i].register_forward_hook(...)` **永远不触发**，你会拿到空 list。
正确做法是挂三个子模块，再按它的顺序重建：
`hidden_in + o_proj_out + mlp_out`。

**坑 2：不要用合成输入推导结构。** 前缀有 241 个 token，但**只有 197 个是
真的**——语言部分按 `max_length=48` 补齐，而 prompt 只有 4 个真 token。
这决定了 attention mask 和 RoPE 的起始 position。我当初用合成的
`ones(241)` 推导，得出了错误的架构结论；从真模型 dump 才发现。
**结构性事实必须从真模型读，不能靠推。**

`smolvla_cpu_baseline.py` 现在只 dump 门禁需要的 `action_chunk`。backbone
和 expert 需要的逐层 hidden、KV cache、pad mask、position ids 的抓取代码，
连同它们的 NPU 实现，都在 **`smolvla` 分支**上（100 个文件的完整研究树）。

### 只想花 30 分钟？读这三个，约 700 行

**① `smolvla_inference.py`（274 行）** — 接缝。读完你就懂了整个集成机制。

重点看 `_wrapped_embed_prefix`。它是**双层**替换，看起来比必要的复杂，
但外层是为了**性能**而不是正确性：所有图必须在一次 runtime 调用里编码完，
好让主机线程钳制覆盖整段 dispatch 循环。拆成单层（每张图各自编码）确实更
简洁、门禁也照样过，但实测 818 → 920 ms，而纯 CPU 是 913 ms——整个收益
没了。代码里记了这个测量，就是为了让下一个想"顺手简化"的人先重测。

里面那个 `served["i"]` 计数器假设 lerobot 按顺序消费 `images`。这是对别人
循环的假设，所以**用断言钉住了**：越界会报错，少消费一张也会报错（意味着
某个相机静默退回了 CPU）。

**② `smolvla_runtime.py`（243 行）** — 为什么要单例：权重加载、ELF 载入、
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
| `smolvla_vision_npu.py` 919 | 逐层 dispatch | 最大的文件，里面有个 attention backend 的 A/B 开关（`"cpu"` 走主机），是它偏大的原因之一 |
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
**门禁的输入是合成的**：三张相机图全零、state 全零、指令写死
`"pick up the cube"`、噪声全零。噪声固定是必要的（否则不可复现），但图像
和 state 全零是真实的局限，报余量时必须把它算进去。

| 输入 | cosine | nmse | 到门槛的余量 |
|---|---|---|---|
| 全零（门禁实际跑的） | 0.998996 | 0.003023 | cos 10× / nmse 13× |
| 固定 seed 随机图像+state | 0.996201 | 0.009976 | **cos 2.6× / nmse 4×** |

**按后者判断。** 退化输入上量出来的 10×/13× 高估了余量；非退化输入实测
约 2.6–4×，仍然过，但这是审核时该拿的数字。

还有一个后果值得知道：**三张图全零意味着三个编码结果完全相同**，所以门禁
测不出相机顺序错乱——`smolvla_inference.py` 里那两个断言在做门禁做不到的
事。让门禁改用固定 seed 的随机图像可以覆盖这一点，但会让 oracle 和所有
门禁数字重新生成，本 PR 没做。

SmolVLA 输出连续动作而非 token，所以用 `regression_gate`，
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
