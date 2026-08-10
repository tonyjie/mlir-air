# fused_decode 学习笔记

对 upstream `Xilinx/mlir-air` 新增的 Q4NX / fused-decode 系列 example 的研读记录。
所有性能与正确性数字都是在本机 NPU2 (AMD Strix, AIE2P) 上**实测复现**的,不是从
上游 README 抄来的。

基线 commit:`85f638d8`(= 研读时的 `upstream/main`)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01_overview.md](01_overview.md) | 这批 example 是什么、彼此怎么咬合、上游演进时间线 |
| [02_fused_decode_design.md](02_fused_decode_design.md) | fused decode 超算子的设计:4-phase proj 复用、packet-id demux、片上 GLU 回喂、unified launch waves |
| [03_one_xclbin_any_L.md](03_one_xclbin_any_L.md) | **核心创新**:一个 xclbin 服务任意 context length(masked-block skip + host 指令流插值),含独立验证 |
| [04_kv_cache_layout.md](04_kv_cache_layout.md) | region-major(quadrant)KV cache 布局与 fire-and-free readback |
| [05_reproduction_log.md](05_reproduction_log.md) | 完整复现步骤 + 实测数据 + 踩到的坑 |
| [06_fragility_and_gotchas.md](06_fragility_and_gotchas.md) | 载荷性约束(per-kernel -O、llvm-link 版本、disable_ping_pong 等)与代码卫生问题 |
| [07_takeaways.md](07_takeaways.md) | 对我们自己 LLM 部署流程的可借鉴点与不可借鉴点 |

## 一句话总结

用**一次 XRT dispatch 跑完 16 层 + LM head**,并且靠"编译期固定最大块数 + kernel 内跳过
全 masked 块 + host 端对指令流做线性插值"让**单个 xclbin 覆盖 [1, 2048] 全部 context
length**,实测 decode 46.55 tok/s、warm TTFT 0.936s。

## 实测结果速览(本机复现)

| 门 | 结果 |
|---|---|
| `make verify-paris` | PASS — prefill 首 token argmax **12366** (" Paris") |
| `make gen` | PASS — `" Paris. The capital of Germany is Berlin. The capital of Italy"`,48.06 tok/s |
| `make profile` | warm TTFT **0.936 s**;decode **46.55 tok/s** (21.5 ms/token, 128 tokens) |
| `make verify` | **PASS** — top-k(k=5) token-set inclusion vs HF bf16,2 passed / 0 failed |
| 指令流插值独立验证 | **byte-exact** — L=2033 原生编译 vs 合成,29264 词全等 |

未开 turbo(`xrt-smi configure --pmode turbo`),上游 README 在 turbo 下报 ~50 tok/s,
本机 46.55 tok/s 与之一致。
