# LLAMA-3 Interactive Chat Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a REPL mode to `llama3_inference.py` that runs the heavy one-time setup (kernel cache load, weights, `prepare_runtime`) once, then loops on user-typed prompts with token-by-token streaming output.

**Architecture:** Refactor today's `__main__` block into `build_session()` and `run_once()` helpers, add an `on_token` streaming callback to `generate()`, and add a `repl_loop()` driven by a new `--interactive` flag. New `make chat` target wires it up. Each turn is independent — no chat memory.

**Tech Stack:** Python 3, MLIR-AIR XRT backend, Hugging Face `transformers` tokenizer, existing `KernelCache` infrastructure.

**Spec:** [docs/superpowers/specs/2026-04-22-llama3-interactive-chat-design.md](../specs/2026-04-22-llama3-interactive-chat-design.md)

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `programming_examples/llama3/llama3_inference.py` | Modify | Add `_StreamState`, `_delta_text`, `build_session`, `run_once`, `repl_loop`; modify `generate()` signature; restructure `__main__`. |
| `programming_examples/llama3/Makefile` | Modify | Add `chat` target; update `.PHONY` and help text. |
| `programming_examples/llama3/test_stream_helpers.py` | Create | Unit tests for `_delta_text` (pure-Python, no NPU). |

No new modules; everything lives in `llama3_inference.py` because the helpers are tightly coupled to its existing `generate()` and `__main__` and won't be reused.

---

## Background for Engineer

You are working in an MLIR-AIR repo. The relevant directory is `programming_examples/llama3/`. Read these first if you have not already:
- `programming_examples/llama3/CLAUDE.md` — architectural overview.
- `programming_examples/llama3/llama3_inference.py:676-809` — the existing `__main__` block you'll refactor.
- `programming_examples/llama3/llama3_inference.py:546-669` — `generate()`.

The environment is auto-loaded by a SessionStart hook — just run `make` commands directly from `programming_examples/llama3/`. No manual venv activation needed.

---

## Task 1: Add streaming helpers (`_StreamState`, `_delta_text`) with unit tests

**Files:**
- Modify: `programming_examples/llama3/llama3_inference.py` (add after the imports block, before the `_LM_GEMV_BACKEND` constant)
- Create: `programming_examples/llama3/test_stream_helpers.py`

These helpers implement the BPE-safe incremental-decode pattern: keep a `printed_len` counter, decode the full id list each call, return only the new suffix. This is the only piece worth unit-testing — everything else hits the NPU.

- [ ] **Step 1: Write the failing test**

Create `programming_examples/llama3/test_stream_helpers.py`:

```python
"""Unit tests for the streaming-decode helpers in llama3_inference.py.

These tests don't touch the NPU — they verify the pure-Python
incremental-decode logic that handles BPE merges correctly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llama3_inference import _StreamState, _delta_text


class _FakeTokenizer:
    """Minimal tokenizer stub: maps token IDs to strings via a table."""

    def __init__(self, table):
        self._table = table  # dict[int, str]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._table.get(i, "") for i in ids)


def test_delta_text_emits_only_new_suffix():
    tok = _FakeTokenizer({1: "Hello", 2: " ", 3: "world"})
    state = _StreamState()
    assert _delta_text(tok, [1], state) == "Hello"
    assert _delta_text(tok, [1, 2], state) == " "
    assert _delta_text(tok, [1, 2, 3], state) == "world"


def test_delta_text_empty_when_token_adds_nothing():
    """Some BPE tokens decode to '' in isolation — we must still advance state cleanly."""
    tok = _FakeTokenizer({1: "Hello", 2: ""})
    state = _StreamState()
    assert _delta_text(tok, [1], state) == "Hello"
    assert _delta_text(tok, [1, 2], state) == ""
    assert state.printed_len == len("Hello")


def test_delta_text_handles_growing_decode():
    """If a later token causes the decoder to emit *more* characters than
    the sum of per-token decodes (e.g. a multi-byte char completing), the
    delta should still be just the new suffix."""
    # Simulate: id 1 decodes to "a", id 2 alone decodes to "", but [1,2]
    # decodes to "ab" — the second token completes a multi-byte sequence.
    class _MergingTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            if ids == [1]:
                return "a"
            if ids == [1, 2]:
                return "ab"
            return ""
    state = _StreamState()
    assert _delta_text(_MergingTokenizer(), [1], state) == "a"
    assert _delta_text(_MergingTokenizer(), [1, 2], state) == "b"


def test_stream_state_starts_at_zero():
    s = _StreamState()
    assert s.printed_len == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
python3 -m pytest test_stream_helpers.py -v
```

Expected: ImportError — `_StreamState` and `_delta_text` not yet defined.

- [ ] **Step 3: Add the helpers to `llama3_inference.py`**

Insert this block in `programming_examples/llama3/llama3_inference.py` immediately after the existing imports (after line 51, before the `# Backend kwarg presets` comment around line 53):

```python
# ---------------------------------------------------------------------------
# Streaming-decode helpers (BPE-safe incremental output)
# ---------------------------------------------------------------------------


class _StreamState:
    """Tracks how many characters of the running decoded text have been emitted.

    BPE tokens may decode to '' in isolation but combine into characters when
    paired with later tokens. The safest streaming pattern is to decode the
    full id list each call and emit only the suffix we have not printed yet.
    """

    def __init__(self):
        self.printed_len = 0


def _delta_text(tokenizer, ids, state):
    """Return the new text fragment since the last call, advancing state."""
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
python3 -m pytest test_stream_helpers.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/llama3_inference.py \
        programming_examples/llama3/test_stream_helpers.py
git commit -m "Add BPE-safe streaming-decode helpers for llama3 inference"
```

---

## Task 2: Add `on_token` callback to `generate()`, preserving today's behavior when None

**Files:**
- Modify: `programming_examples/llama3/llama3_inference.py:546-669` (the `generate` function)

The change must be a no-op when `on_token=None` so today's `make run` produces bit-identical stdout. When set, the callback is invoked once per generated decode token with `(token_id, delta_text)`.

- [ ] **Step 1: Update the `generate()` signature**

Find this line in `llama3_inference.py`:

```python
def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    tokenizer,
    n_tokens=10,
    profile=False,
    verify=False,
    cpu_attn=True,
):
```

Replace it with:

```python
def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    tokenizer,
    n_tokens=10,
    profile=False,
    verify=False,
    cpu_attn=True,
    on_token=None,
):
```

- [ ] **Step 2: Initialize stream state and emit the prefill token**

Find this block (around line 590-595):

```python
    # --- Phase 2: NPU Decode ---
    generated_tokens = [prefill_token]  # Token 0 = from prefill
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    print(f"\nDecoding {n_tokens} tokens (token 1 to {n_tokens})...")
    t_decode_start = time.time()
```

Replace with:

```python
    # --- Phase 2: NPU Decode ---
    generated_tokens = [prefill_token]  # Token 0 = from prefill
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    # Streaming state — only used when on_token is provided.
    stream_state = _StreamState() if on_token is not None else None
    if on_token is not None:
        on_token(prefill_token, _delta_text(tokenizer, generated_tokens, stream_state))

    print(f"\nDecoding {n_tokens} tokens (token 1 to {n_tokens})...")
    t_decode_start = time.time()
```

- [ ] **Step 3: Emit each decode token via the callback**

Find this block (around line 649-660):

```python
        generated_tokens.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if profile:
            print(
                f"  Token {token_idx + 1}: id={next_token}, time={t_token*1000:.0f}ms"
            )

        # Stop on EOS or EOT (instruct model emits <|eot_id|> = 128009)
        if next_token in (tokenizer.eos_token_id, 128009):
            break
```

Replace with:

```python
        generated_tokens.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if on_token is not None:
            on_token(next_token, _delta_text(tokenizer, generated_tokens, stream_state))

        if profile:
            print(
                f"  Token {token_idx + 1}: id={next_token}, time={t_token*1000:.0f}ms"
            )

        # Stop on EOS or EOT (instruct model emits <|eot_id|> = 128009)
        if next_token in (tokenizer.eos_token_id, 128009):
            break
```

- [ ] **Step 4: Verify one-shot path is unchanged**

Run the existing one-shot path with no callback to confirm behavior is preserved:

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make run N_TOKENS=10 PROMPT="The capital of France is" 2>&1 | tail -15
```

Expected: same final output as before this task (token-by-token timing absent unless `--profile`, generated text ends with " Paris" or similar). If output is different, you broke the no-callback path — re-check Step 2 and 3.

- [ ] **Step 5: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/llama3_inference.py
git commit -m "Add on_token streaming callback to llama3 generate()"
```

---

## Task 3: Refactor `__main__` into `build_session()` + `run_once()`

**Files:**
- Modify: `programming_examples/llama3/llama3_inference.py:676-809`

This is a pure refactor — no behavior change. After this task, `make run` must produce bit-identical output to before.

- [ ] **Step 1: Add a `Session` dataclass and module-level imports**

At the top of `llama3_inference.py`, find the existing import block (lines 25-31):

```python
import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16
```

Replace with:

```python
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from ml_dtypes import bfloat16
```

Then immediately after the streaming helpers block from Task 1 (and before `_LM_GEMV_BACKEND`), add:

```python
# ---------------------------------------------------------------------------
# Session: long-lived state created once per process
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Everything `run_once` needs that should not be rebuilt per turn."""

    config: Any                  # LlamaConfig
    seq_len: int                 # padded prompt length (today: 2048)
    weights: Any                 # LlamaWeights, mutated by prepare_runtime()
    tokenizer: Any               # transformers AutoTokenizer
    prefill_cache: Any           # KernelCache
    decode_cache: Any            # KernelCache
    rope_lut_bf16: np.ndarray    # (max_seq, head_dim) bfloat16
    model_variant: str           # "base" | "instruct"
```

- [ ] **Step 2: Add `build_session()` above the `if __name__ == "__main__"` block**

Insert this just before the `if __name__ == "__main__":` line near the end of the file:

```python
# ---------------------------------------------------------------------------
# Session lifecycle and per-turn execution
# ---------------------------------------------------------------------------


def build_session(args) -> Session:
    """One-time setup: load kernel caches, weights, tokenizer, RoPE LUT,
    and run prepare_runtime(). Safe to call once per process; do not call
    twice (prepare_runtime mutates `weights` with idempotency guards but the
    intent is one-shot)."""
    config = LlamaConfig()
    seq_len = 2048

    prefill_cache = KernelCache("prefill_kernel_cache", verbose=args.verbose)
    decode_cache = KernelCache("decode_kernel_cache", verbose=args.verbose)

    if not args.run_only:
        print("Compiling prefill kernels...")
        compile_all_kernels(prefill_cache, config, seq_len, cpu_attn=args.cpu_attn)
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config)

    if args.compile_only:
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    model_id = (
        "meta-llama/Llama-3.2-1B-Instruct"
        if args.model == "instruct"
        else "meta-llama/Llama-3.2-1B"
    )
    print(f"\nLoading weights ({model_id})...")
    weights = load_weights(model_id)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    prepare_runtime(
        prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
    )

    return Session(
        config=config,
        seq_len=seq_len,
        weights=weights,
        tokenizer=tokenizer,
        prefill_cache=prefill_cache,
        decode_cache=decode_cache,
        rope_lut_bf16=rope_lut_bf16,
        model_variant=args.model,
    )


def _tokenize_prompt(session: Session, prompt_text: str) -> list:
    """Apply chat template if instruct model, then tokenize. Does NOT pad."""
    if session.model_variant == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return session.tokenizer.encode(chat_text)
    return session.tokenizer.encode(prompt_text)


def run_once(
    session: Session,
    prompt_text: str,
    *,
    n_tokens: int,
    profile: bool = False,
    verify: bool = False,
    cpu_attn: bool = True,
    on_token: Optional[Callable[[int, str], None]] = None,
) -> tuple:
    """Tokenize, pad to seq_len, and call generate(). Returns
    (generated_token_ids, prompt_len_actual)."""
    tokens = _tokenize_prompt(session, prompt_text)
    prompt_len_actual = len(tokens)
    if len(tokens) < session.seq_len:
        tokens = tokens + [session.tokenizer.eos_token_id] * (
            session.seq_len - len(tokens)
        )

    generated = generate(
        tokens,
        session.weights,
        session.config,
        session.prefill_cache,
        session.decode_cache,
        session.rope_lut_bf16,
        tokenizer=session.tokenizer,
        n_tokens=n_tokens,
        profile=profile,
        verify=verify,
        cpu_attn=cpu_attn,
        on_token=on_token,
    )
    return generated, prompt_len_actual


def _print_one_shot_output(session, args, generated, prompt_len_actual):
    """Format and print the final output for non-interactive mode."""
    print(f"\n{'='*60}")
    if session.model_variant == "instruct":
        response = session.tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Q: {args.prompt}")
        print(f"A: {response}")
    else:
        # Reconstruct the unpadded prompt + generated tokens.
        prompt_tokens = _tokenize_prompt(session, args.prompt)
        print(f"Generated text:")
        print(f"{'='*60}")
        all_tokens = prompt_tokens[:prompt_len_actual] + generated
        print(session.tokenizer.decode(all_tokens))
```

- [ ] **Step 3: Replace the `__main__` block**

Replace lines 676-809 (the entire `if __name__ == "__main__":` block) with:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-1B Inference (NPU)")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile both prefill and decode kernels, then exit",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Use cached kernels (skip compilation)",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number of decode tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-token timing instrumentation",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against CPU F32 reference",
    )
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use CPU attention for prefill (default: NPU flash attention)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="base",
        help="Model variant: base (completion) or instruct (Q&A)",
    )
    args = parser.parse_args()

    session = build_session(args)

    generated, prompt_len_actual = run_once(
        session,
        args.prompt,
        n_tokens=args.n_tokens,
        profile=args.profile,
        verify=args.verify,
        cpu_attn=args.cpu_attn,
    )

    _print_one_shot_output(session, args, generated, prompt_len_actual)
```

- [ ] **Step 4: Verify behavior is unchanged**

Run the regression check:

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make run N_TOKENS=10 PROMPT="The capital of France is" 2>&1 | tail -10
```

Expected: same final block as before (last line is the decoded text including " Paris"). If formatting drifted (e.g., extra/missing blank lines), check `_print_one_shot_output`.

Also check the instruct variant:

```bash
make run MODEL=instruct N_TOKENS=10 PROMPT="What is 2+2?" 2>&1 | tail -10
```

Expected: a `Q: What is 2+2?` / `A: ...` block.

- [ ] **Step 5: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/llama3_inference.py
git commit -m "Refactor llama3_inference __main__ into build_session/run_once"
```

---

## Task 4: Add `--interactive` flag, mutual-exclusion rules, and `repl_loop()`

**Files:**
- Modify: `programming_examples/llama3/llama3_inference.py` (add `--interactive` arg, add `repl_loop`, update `__main__` dispatch)

- [ ] **Step 1: Add `repl_loop()` helper**

Insert this immediately after `_print_one_shot_output` (added in Task 3):

```python
def repl_loop(session: Session, args) -> None:
    """Interactive REPL: prompt-> stream-> repeat. Each turn is independent."""
    print("\nInteractive mode — Ctrl-D or /quit to exit.")
    print("Each prompt is independent (no chat memory).\n")

    def _stream_cb(_token_id, delta):
        sys.stdout.write(delta)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return

        # Length guard.
        check_ids = _tokenize_prompt(session, prompt)
        if len(check_ids) > session.seq_len:
            print(
                f"Prompt too long ({len(check_ids)} > {session.seq_len} tokens). "
                "Skipped."
            )
            continue

        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        try:
            run_once(
                session,
                prompt,
                n_tokens=args.n_tokens,
                profile=False,
                verify=False,
                cpu_attn=args.cpu_attn,
                on_token=_stream_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue

        print("\n")
```

- [ ] **Step 2: Add the `--interactive` argparse flag**

In the `__main__` block, find:

```python
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="base",
        help="Model variant: base (completion) or instruct (Q&A)",
    )
    args = parser.parse_args()
```

Replace with:

```python
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="base",
        help="Model variant: base (completion) or instruct (Q&A)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into a REPL after runtime prep. Loops on prompts; each is independent.",
    )
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if args.profile:
            print(
                "WARNING: --profile is ignored in --interactive mode.",
                file=sys.stderr,
            )
            args.profile = False
        if args.verify:
            print(
                "WARNING: --verify is ignored in --interactive mode.",
                file=sys.stderr,
            )
            args.verify = False
```

- [ ] **Step 3: Dispatch to `repl_loop` when `--interactive` is set**

Replace the `run_once` + `_print_one_shot_output` lines at the end of `__main__`:

```python
    generated, prompt_len_actual = run_once(
        session,
        args.prompt,
        n_tokens=args.n_tokens,
        profile=args.profile,
        verify=args.verify,
        cpu_attn=args.cpu_attn,
    )

    _print_one_shot_output(session, args, generated, prompt_len_actual)
```

with:

```python
    if args.interactive:
        repl_loop(session, args)
    else:
        generated, prompt_len_actual = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            profile=args.profile,
            verify=args.verify,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args, generated, prompt_len_actual)
```

- [ ] **Step 4: Verify mutual-exclusion rules**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
mkdir -p build_peano && cd build_peano
python3 ../llama3_inference.py --interactive --compile-only 2>&1 | tail -3
```

Expected: argparse error containing `--interactive cannot be combined with --compile-only`. Process exits non-zero.

- [ ] **Step 5: Verify the REPL flow end-to-end**

This is the manual smoke test — feed the REPL via stdin:

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
printf 'The capital of France is\nOnce upon a time\n/quit\n' | \
  make chat N_TOKENS=10 2>&1 | tail -30
```

(`make chat` is added in Task 5; this step assumes it exists. If you are doing tasks strictly in order, run instead:

```bash
cd build_peano
printf 'The capital of France is\nOnce upon a time\n/quit\n' | \
  python3 ../llama3_inference.py --run-only --interactive --n-tokens 10 2>&1 | tail -30
```
)

Expected output ends with two `Response: ...` blocks of streamed text, then a clean exit (no traceback). If you see a traceback, something in `repl_loop` mishandled stdin or the callback.

- [ ] **Step 6: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/llama3_inference.py
git commit -m "Add --interactive REPL mode to llama3_inference"
```

---

## Task 5: Add `make chat` target and update help text

**Files:**
- Modify: `programming_examples/llama3/Makefile`

- [ ] **Step 1: Add `chat` to `.PHONY`**

In `programming_examples/llama3/Makefile`, find:

```makefile
.PHONY: help compile run profile verify clean \
        compile-prefill compile-decode run-prefill run-decode
```

Replace with:

```makefile
.PHONY: help compile run profile verify chat clean \
        compile-prefill compile-decode run-prefill run-decode
```

- [ ] **Step 2: Add the `chat` target**

In the same file, find the `## Run with CPU reference verification` block and the `verify:` target (around line 95-97):

```makefile
## Run with CPU reference verification
verify:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --n-tokens $(N_TOKENS) --verify --profile --prompt "$(PROMPT)" --model $(MODEL)
```

Add immediately after that block, before `## Compile and run in one step`:

```makefile
## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)
```

- [ ] **Step 3: Update help text**

Find this block in the `help:` target:

```makefile
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make profile          Run with profiling breakdown"
```

Replace with:

```makefile
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
```

- [ ] **Step 4: Verify help renders**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make help | head -15
```

Expected: the new `make chat` line appears under "Quick start".

- [ ] **Step 5: Verify the `make chat` target works end-to-end**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
printf 'The capital of France is\n/quit\n' | make chat N_TOKENS=10 2>&1 | tail -20
```

Expected: streaming response containing "Paris" or similar, then clean exit.

- [ ] **Step 6: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/Makefile
git commit -m "Add 'make chat' target for llama3 interactive mode"
```

---

## Task 6: Final verification & docs note

**Files:**
- Modify: `programming_examples/llama3/CLAUDE.md` (one-line addition)

- [ ] **Step 1: Run the full smoke matrix**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3

# A. Regression: one-shot base must still work bit-identically
make run N_TOKENS=10 PROMPT="The capital of France is" 2>&1 | tail -5

# B. Regression: one-shot instruct must still work
make run MODEL=instruct N_TOKENS=10 PROMPT="What is 2+2?" 2>&1 | tail -5

# C. New: interactive base, two prompts, /quit
printf 'The capital of France is\nOnce upon a time\n/quit\n' | \
  make chat N_TOKENS=10 2>&1 | tail -20

# D. New: interactive instruct
printf 'What is the capital of France?\n/quit\n' | \
  make chat MODEL=instruct N_TOKENS=20 2>&1 | tail -10

# E. New: interactive long-prompt rejection
python3 -c "print('x ' * 3000)" | \
  make chat N_TOKENS=10 2>&1 | grep -i 'too long' || echo "FAIL: no length warning"

# F. Unit tests
python3 -m pytest test_stream_helpers.py -v
```

Expected: A and B match pre-refactor output; C/D show streamed responses and clean exit; E prints the "Prompt too long" message; F shows 4 passed.

- [ ] **Step 2: Add one line to CLAUDE.md**

In `programming_examples/llama3/CLAUDE.md`, find:

```bash
make compile   # One-time kernel compilation (~4 min, cached to disk)
make run       # Run inference (prefill + 100 tokens decode)
make profile   # Run with per-kernel timing breakdown
make verify    # Run with CPU reference verification
```

Replace with:

```bash
make compile   # One-time kernel compilation (~4 min, cached to disk)
make run       # Run inference (prefill + 100 tokens decode)
make chat      # Interactive REPL: prep runtime once, loop on prompts (streaming)
make profile   # Run with per-kernel timing breakdown
make verify    # Run with CPU reference verification
```

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama3/CLAUDE.md
git commit -m "Document 'make chat' target in llama3 CLAUDE.md"
```

---

## Self-Review Notes (already incorporated)

- All spec sections (refactor, streaming hook, REPL loop, Makefile target, length guard, mutual-exclusion rules, profile-disabled-in-interactive) are covered by Tasks 1–5.
- No placeholder text; all code blocks are complete.
- Type and identifier names are consistent: `Session`, `_StreamState`, `_delta_text`, `build_session`, `run_once`, `repl_loop`, `_tokenize_prompt`, `_print_one_shot_output`, `on_token`.
- `_tokenize_prompt` is shared between `run_once`, `_print_one_shot_output`, and `repl_loop`'s length guard — no duplication.
- Smoke test E in Task 6 covers the length-guard edge case from the spec's testing matrix.
