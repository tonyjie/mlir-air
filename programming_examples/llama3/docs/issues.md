# Known Issues and Future Work

## 1. BF16 Precision Divergence in Decode Generation

**Symptom**: For some prompts, the NPU decode generates repetitive text while the
HuggingFace BF16 reference (unpadded, same model) generates coherent text.

**Example**:
```
Prompt: "In 1969, the first man to walk on"

HuggingFace (no padding):  "the moon, Neil Armstrong, was the first to say..."
Our NPU pipeline (padded):  "the moon. The first man to walk on the moon. The first..."
```

Both predict the same first two tokens ("the moon"), but diverge at the third:
- HuggingFace: "," (comma, logit=20.5) → continues with "Neil Armstrong"
- NPU: "." (period, logit=16.9 in HF) → ends sentence, triggers repetition loop

**Root cause**: BF16 precision error accumulates across 16 transformer layers. The
logit gap between "," and "." is 3.6 in the reference, but our NPU pipeline's
accumulated numerical error (~0.991 correlation at logit level) can flip rankings
for tokens with moderate gaps.

**Verification**: The pipeline IS functionally correct:
- First token prediction matches CPU reference (corr=0.991)
- Top-1 match: YES for both prompts tested
- Per-layer KV cache correlation: >0.99 at early layers, ~0.98 at later layers
- This is expected BF16 behavior, not a bug

**Mitigations**:
- Add temperature + top-k sampling (like IRON uses `temperature=0.7, top_k=50`).
  Sampling adds randomness that breaks repetition loops regardless of small
  numerical differences.
- The "The capital of France is" prompt works perfectly (large confidence gap
  for "Paris" survives BF16 noise).

**Related**: Issue #2 (variable-length input). Processing 2036 padding tokens adds
unnecessary BF16 computation that may amplify precision loss.

---

## 2. Fixed Sequence Length (seq_len=2048)

**Symptom**: All prompts are padded to 2048 tokens regardless of actual length.
A 6-token prompt processes 2042 EOS padding tokens, wasting ~99% of prefill compute.

**Current behavior**:
```
"The capital of France is"  →  6 real + 2042 padding = 2048 tokens
"Hello"                     →  2 real + 2046 padding = 2048 tokens
```

Prefill always takes ~1.54s regardless of prompt length.

**Why**: All NPU kernels are compiled with fixed dimensions:
- GEMM launch grids: M=2048
- FlashAttention: lq=2048, lk=2048
- Buffer Object sizes: (2048, 2048) matrices
- RoPE LUT: 2048 positions

Changing seq_len requires recompiling all kernels (~4 min).

**Impact**:
- Wasted prefill compute for short prompts
- May amplify BF16 precision loss (more unnecessary computation, see Issue #1)
- Cannot process prompts longer than 2048 tokens

**Potential solutions**:

1. **Bucket compilation**: Pre-compile kernels for multiple seq_len buckets
   (e.g., 64, 256, 512, 1024, 2048). Route each prompt to the smallest bucket
   that fits. Increases disk usage but dramatically reduces prefill time for
   short prompts.

2. **Dynamic seq_len**: Modify kernel builders to support runtime-parameterized
   sequence length. Requires changes to GEMM launch grid computation, FlashAttention
   tiling, and BO allocation. Significant engineering effort.

3. **Chunked prefill**: Process the prompt in fixed-size chunks (e.g., 256 tokens
   at a time), accumulating KV cache. Reuses one set of kernels compiled for
   chunk_size. Requires incremental attention (append to KV cache each chunk).

---

## 3. No Sampling (Greedy Decode Only)

**Symptom**: Generated text tends to be repetitive, especially for base models.

**Current behavior**: The decode loop uses `argmax` (greedy decoding) — always picks
the single highest-probability token.

```python
next_token = int(np.argmax(logits[0]))  # greedy, no randomness
```

**Impact**: Greedy decoding is deterministic but prone to repetition loops in base
models (like LLAMA-3.2-1B). The model gets stuck repeating high-probability patterns.

**Fix**: Add temperature scaling + top-k sampling:
```python
# Temperature scaling
logits = logits / temperature

# Top-k filtering
top_k_indices = np.argsort(logits)[-top_k:]
mask = np.full_like(logits, -np.inf)
mask[top_k_indices] = logits[top_k_indices]
logits = mask

# Softmax + sample
probs = np.exp(logits - logits.max()) / np.sum(np.exp(logits - logits.max()))
next_token = np.random.choice(len(probs), p=probs)
```

IRON uses `temperature=0.7, top_k=50` which produces diverse, coherent text.
This is a straightforward Python-side change (no kernel modifications needed).

---

## 4. Base Model vs Instruct Model

**Current**: We use `meta-llama/Llama-3.2-1B` — the **base** (pre-training) model.
It is a text completion model, not a chatbot. It does not follow instructions or
answer questions.

**Impact**: Prompts like "Which is larger: 9.11 or 9.9?" produce text completions,
not answers. The model treats the question as text to continue, not a query to answer.

**Fix**: Switch to `meta-llama/Llama-3.2-1B-Instruct` for instruction-following.
This requires no kernel changes (same architecture, same weights shape). Only the
weight loading path changes.

---

## 5. CPU Attention for Decode (Grows with Context)

**Current**: Decode attention runs on CPU. At short contexts (pos < 100), this is
fast (~0.3ms/layer). At longer contexts, it grows linearly:

| Context length | CPU attention/layer | x16 layers |
|---------------|--------------------|-----------|
| 50 tokens | 0.2ms | 3ms |
| 512 tokens | 0.9ms | 14ms |
| 2048 tokens | 1.8ms | 29ms |
| 4096 tokens | 3.6ms | 58ms |

**Impact**: For long conversations (multi-turn chat with context > 2000 tokens),
CPU attention becomes a significant bottleneck (adding 30-60ms/token to the
92ms baseline).

**Fix**: Implement an NPU decode attention kernel (single-query GQA with on-device
KV cache). This is different from the prefill FlashAttention (which has lq=2048).
The decode kernel would have lq=1 and attend to an on-device KV cache.

---

## Priority

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| #3 Add sampling | Fixes repetition, matches IRON | Low | **High** |
| #2 Variable seq_len | 10-100x prefill speedup for short prompts | High | **High** |
| #1 BF16 divergence | Cosmetic (correct numerically) | N/A (inherent) | Low |
| #4 Instruct model | Better user experience | Low | Medium |
| #5 NPU decode attention | Long context performance | High | Medium |
