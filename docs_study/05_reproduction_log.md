# 05 — 复现日志与实测数据

本机环境:AMD Strix (NPU2/AIE2P),kernel 6.17.0-14,Peano = llvm-aie LLVM 21,
本地 mlir-air install 构建于 2026-08-05。研读基线 `85f638d8`。

**未开 turbo。** 上游 README 的 ~50 tok/s 是 turbo 下的数字。

## 0. 前置检查

```bash
which llvm-link && llvm-link --version    # 必须与 Peano 的 LLVM major 一致(本机 21)
echo $PEANO_INSTALL_DIR
ls /dev/accel/                            # accel0
```

本机 `llvm-link` 来自 `/home/jiajli/apps/mlir-air/llvm-link-bin/`(LLVM 21),与 Peano 一致。

> 注意上游措辞已变:首个 PR 的 README 说"必须 < LLVM 23",HEAD 的 Makefile preflight
> 已改成 **"必须匹配 Peano 的 LLVM major"**。别按旧 README 去找 LLVM 20。

## 1. 环境兼容性(重要发现)

本地 install 是 2026-08-05(≈`a88b180d`),而 HEAD 已经过了 mlir-aie LLVM 23→24 的 bump
(`46f0ba2c`)。担心编不了,先做了核对:

```bash
git diff a88b180d..85f638d8 --stat -- mlir/ python/air/
# mlir/ 侧对 fused_decode 零改动
comm -13 <(git show a88b180d:...fused_decode.py | grep -o 'air\.[a-z_]*' | sort -u) \
         <(git show 85f638d8:...fused_decode.py | grep -o 'air\.[a-z_]*' | sort -u)
# 空 —— HEAD 没有引入任何新的 air.* 属性
```

**结论:旧 install 能直接编 HEAD 的 fused_decode。** 用一次 dry-run 确认:

```bash
cd programming_examples/fused_decode
DECODE_GOLDEN=1 DECODE_GOLDEN_L=32 NLAYERS=1 LM_HEAD=0 VOCAB_CHUNK_I2=14 \
  python3 -c "import fused_decode as fd; fd.build_module()"
# → 1687 行 IR,OK
```

## 2. 编译 decode 模板

```bash
cd programming_examples/fused_decode
make compile-decode
```

产出 `decode_L2048.{xclbin,insts.bin}` + `decode_L2047.{xclbin,insts.bin}`
(各 161 KB / 117 KB)。

**实际耗时 < 3 分钟**,README 说 ~15 min —— 差得比较多,可能是文档写于更早的编译器版本。

`rope.cc` 会出一条无害 warning:`Loop iteration count metadata (8) is inconsistent with
loop condition`。

## 3. 验证指令流插值(核心主张)

自带 self-check(注意它只验证拟合点,是循环论证):

```bash
python3 decode_insts_gen.py
# template ATTN_MAXL=2048: base_L=2047 builds=[2047, 2048] L-dep_words=264 (active)
#   L=2047 byte-exact=True / L=2048 byte-exact=True  ← 都是拟合点
```

**独立验证**(原生编译第三个点,从未参与拟合):

```bash
# 必须在 fused_decode/ 源目录里跑:inline-attn merge 要求 attn_qk.ll / attn_kv.ll 在 cwd
cd programming_examples/fused_decode
VOCAB_CHUNK_I2=14 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 DECODE_GOLDEN_L=2033 \
  python3 fused_decode.py

python3 -c "
import numpy as np
from decode_insts_gen import DecodeInstsGen
g = DecodeInstsGen('.'); g.select(max_L=2048)
nat = np.fromfile('decode.insts.bin', dtype=np.uint32)
gen = g.insts_for_L(2033)
print((nat==gen).all(), nat.size)"
# → True 29264
```

**结果:byte-exact。** 29264 词全等。

> 坑:我第一次在 `$CLAUDE_JOB_DIR/tmp` 下跑,失败于
> `llvm-link: No such file or directory: 'air_project/attn_kv.ll'`。必须在源目录。

## 4. 编译 prefill

```bash
cd programming_examples/llms/llama32_1b_q4nx
make compile          # ~5 min
```

编出 3 个 stitcher + LM head GEMV:

| kernel | 耗时 |
|---|---|
| `rms_gemms_rope` | 108.7 s |
| `o_ffn` | 167.4 s |
| `flash_attn`(TEMPORAL_CAUSAL_SKIP=1) | 11.1 s |
| `lm_head_gemv`(8-partition) | 13.1 s |

## 5. 权重

```bash
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('FastFlowLM/Llama-3.2-1B-NPU2','model.q4nx',
      revision='d0c7f84ac9c5cf796db0fc8255afac42592d9db3'))"
```

1297.8 MB,**公开 repo,不需要 HF_TOKEN**(但 `make verify` 需要 token 拿 meta-llama 的
bf16 参考)。revision 是 pin 死的 —— 上游 Hub bundle 会被重新打包成新 block 布局,不 pin
会 reshape 失败。

decode 侧的 q4k-cascade requant cache 首次运行自动派生,缓存到 `~/.cache/q4nx/requant.npz`。

## 6. 跑 NPU(全部加锁)

按项目约定,每条碰 NPU 的命令都包 `flock`:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify-paris
flock -x -w 1800 /tmp/mlir-air-npu.lock make gen
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile N_TOKENS=128 PROMPT="What is the capital of France?"
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

## 实测结果

### `make verify-paris` — PASS
```
[q4nx_prefill] first-token argmax=12366 (expect 12366 ' Paris')
[q4nx_prefill] *** PARIS ***
```

### `make gen` — PASS
```
[prefill] ctx=6 first_token=12366
Time to first token (TTFT): 12.06s          ← 冷启动(含权重加载)
Generated 12 tokens in 0.25s (48.06 tok/s)
[inference] TEXT: ' Paris. The capital of Germany is Berlin. The capital of Italy'
*** PARIS ***
```

### `make profile N_TOKENS=128` — 复现上游数字
```
Time to first token (TTFT): 11.92s
Warm time to first token (TTFT): 0.936s      ← 上游 README: ~0.93s ✓
Generated 128 tokens in 2.75s (46.55 tok/s)
[profile] decode 21.5 ms/token, prefill TTFT 11.92s, P=42
```

输出连贯:`"The capital of France is Paris. ... France is a country located in Western
Europe, known for its rich history, art, fashion, and cuisine. ..."`

### `make verify` — PASS(正式正确性门)
```
[verify] top-k token gate: 2 prompts × 32 tokens, k=5
[verify] Summary: {'n_layer_records': 0, 'topk_passed': 2, 'topk_failed': 0}
[verify] PASS
```

NPU q4nx(prefill + fused decode)对 HF bf16 参考(`meta-llama/Llama-3.2-1B-Instruct`)
做 top-k token-set inclusion,2/2 通过。

## 汇总对比

| 指标 | 上游声称 | 本机实测 | |
|---|---|---|---|
| Warm TTFT @2048 | ~0.93 s | **0.936 s** | ✓ |
| Decode 吞吐 | ~50 tok/s (turbo) | **46.55 tok/s** (无 turbo) | ✓ |
| Paris 首 token | 12366 | **12366** | ✓ |
| verify top-k | PASS | **PASS 2/0** | ✓ |
| insts 插值 byte-exact | 声称 | **独立验证通过 (L=2033)** | ✓ |
| `make compile-decode` 耗时 | ~15 min | **< 3 min** | ✗(文档偏保守) |

**上游的所有性能与正确性主张均复现成功。**

## 复现时踩到的坑

1. **原生 build 必须在 `fused_decode/` 源目录** —— 否则 inline-attn 的 llvm-link 找不到
   `attn_*.ll`。
2. **`make verify-paris` 用独立缓存目录**(`build_peano/q4nx_kernel_cache_2048/` vs
   `make compile` 的 `_q4nx_cache_seq2048/`),会**重编一遍全部 kernel**(~5 min)。想省
   时间的话先跑 `verify-paris` 再跑别的。
3. **`make gen` 报的 TTFT 是冷启动**(12.06s,含 1856 MB 权重 pre-load)。真正可比的
   warm TTFT 只有 `make profile` 才打印。
4. `torchao` 的几条 `Failed to load ... .so` 是无害噪声,不影响 verify 结果。
