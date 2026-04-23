# LLAMA-3 Interactive Chat Mode — Design

**Date:** 2026-04-22
**Scope:** `programming_examples/llama3/`
**Status:** Approved (design phase)

## Goal

Today, `make run PROMPT="..."` performs the full inference cycle every invocation: kernel-cache load, weights load, tokenizer load, `prepare_runtime()` (BO preloading, weight transposes, external-kernel build), and then `generate()`. The pre-inference work takes tens of seconds, then the model produces output for one prompt and exits.

We want an **interactive mode** where the heavy one-time setup runs once, then the process loops on user-supplied prompts. Each prompt is independent (no chat memory — the implementation does not currently support multi-turn KV-cache reuse). Generated tokens stream to stdout incrementally as they are produced.

## Non-Goals

- Multi-turn conversation / persistent KV cache across turns.
- Per-token timing / profile output during interactive mode (`--profile` is not honored when `--interactive` is set).
- Verification (`--verify`) during interactive mode.
- Sampling (top-k, temperature) — keep greedy argmax, matching today.
- Server/daemon mode, multiple clients, or async generation.
- Touching `llama3_prefill.py` / `llama3_decode.py` standalone scripts.

## User-Visible Behavior

### New Make target

```
make chat                      # default N_TOKENS=100, MODEL=base
make chat MODEL=instruct
make chat N_TOKENS=50
```

### Session

```
$ make chat MODEL=instruct
... (compile/load output)
Preparing runtime (one-time init)...
  Runtime prepared in 35.2s

Interactive mode — Ctrl-D or /quit to exit.
Each prompt is independent (no chat memory).

Prompt> What is the capital of France?

Response: The capital of France is Paris.

Prompt> Write a haiku about NPUs.

Response: Silicon whispers ...

Prompt> /quit
$
```

### Exit / interrupt

- `Ctrl-D` (EOF), `Ctrl-C` at the input prompt, or `/quit` / `/exit` cleanly exit the loop.
- `Ctrl-C` mid-generation aborts the current turn, prints `[interrupted]`, and returns to the input prompt. Safe because all per-turn state (KV cache, `current_pos`, `generated_tokens`) is local to `generate()`.

### Length guard

Today, prompts longer than `seq_len` (2048) silently truncate via padding logic. In interactive mode we tokenize first; if `n_in > seq_len`, print `Prompt too long (N > 2048 tokens). Skipped.` and re-prompt without invoking the NPU.

## Architecture

### File touched

Only `programming_examples/llama3/llama3_inference.py` and `programming_examples/llama3/Makefile`.

### Refactor of `llama3_inference.py`

Today the `if __name__ == "__main__"` block does five things in line:
1. Parse args.
2. Build / load kernel caches.
3. Load weights, tokenizer, build RoPE LUT.
4. Call `prepare_runtime()`.
5. Tokenize, pad to `seq_len`, call `generate()`, print output.

We extract steps 2–4 into `build_session(args)` and step 5 into `run_once(session, prompt_text, ...)`.

```python
@dataclass
class Session:
    config: LlamaConfig
    seq_len: int
    weights: ...                # LlamaWeights
    tokenizer: ...              # AutoTokenizer
    prefill_cache: KernelCache
    decode_cache: KernelCache
    rope_lut_bf16: np.ndarray
    model_variant: str          # "base" | "instruct"

def build_session(args) -> Session:
    """Steps 1-3 of today's __main__ — runs once."""
    ...

def run_once(
    session: Session,
    prompt_text: str,
    *,
    n_tokens: int,
    profile: bool = False,
    verify: bool = False,
    cpu_attn: bool = True,
    on_token: Callable[[int, str], None] | None = None,
) -> list[int]:
    """Tokenize + (optional chat template) + pad + generate. Returns generated ids."""
    ...
```

The new `__main__`:

```python
args = parser.parse_args()
session = build_session(args)
if args.interactive:
    repl_loop(session, args)
else:
    generated = run_once(session, args.prompt, n_tokens=args.n_tokens,
                         profile=args.profile, verify=args.verify,
                         cpu_attn=args.cpu_attn)
    _print_one_shot_output(session, args, generated)
```

`_print_one_shot_output` holds today's final formatting (Q/A for instruct, full decoded text for base).

### Streaming hook in `generate()`

`generate()` gains one optional kwarg `on_token`. Inside the existing decode loop, after `next_token` is appended:

```python
if on_token is not None:
    on_token(next_token, _delta_text(tokenizer, generated_tokens, state))
```

`_delta_text` is the standard incremental-decode pattern:

```python
class _StreamState:
    printed_len: int = 0

def _delta_text(tokenizer, ids, state):
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len:]
    state.printed_len = len(decoded)
    return delta
```

This handles BPE merges (a single token may decode to nothing until a later token completes a multi-byte sequence) — safer than decoding one id at a time.

When `on_token is None`, behavior is bit-identical to today.

When `on_token is not None`, `profile=False` is enforced (per the user decision: profile is not supported with interactive mode). `run_once` passes `profile=False` whenever it passes a callback.

### REPL loop

```python
def repl_loop(session, args):
    print("\nInteractive mode — Ctrl-D or /quit to exit.")
    print("Each prompt is independent (no chat memory).")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return

        # Length guard.
        check_ids = _tokenize_for_check(session, prompt)
        if len(check_ids) > session.seq_len:
            print(f"Prompt too long ({len(check_ids)} > {session.seq_len} tokens). Skipped.")
            continue

        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        state = _StreamState()
        try:
            run_once(
                session, prompt,
                n_tokens=args.n_tokens,
                profile=False, verify=False, cpu_attn=args.cpu_attn,
                on_token=lambda _id, delta: (sys.stdout.write(delta), sys.stdout.flush()),
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print()  # trailing newline after streamed text
```

`_tokenize_for_check` mirrors `run_once`'s tokenization (chat template for instruct) but does not pad — its only purpose is the length check.

### Argparse changes

Add:

```python
parser.add_argument(
    "--interactive",
    action="store_true",
    help="Drop into a REPL after runtime prep. Loops on prompts; each is independent.",
)
```

Mutual-exclusion / interaction rules enforced in `__main__`:

- `--interactive` requires `--run-only` (cannot pair with `--compile-only`). Hard error.
- `--interactive` ignores `--prompt` (warn once on stderr).
- `--interactive` forces `profile=False` and `verify=False` even if those flags are set (warn once on stderr each).

### Makefile changes

Add one target and one help line.

```makefile
## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)
```

Add `chat` to the `.PHONY:` line. Add a help line under "Quick start": `  make chat             Interactive chat mode (REPL)`.

## Data Flow

```
build_session()         (once)
  ├── KernelCache.load_manifest() x2
  ├── load_weights(model_id)
  ├── AutoTokenizer.from_pretrained(model_id)
  ├── generate_rope_lut(seq_len + n_tokens)   # uses CLI n_tokens as upper bound
  └── prepare_runtime(prefill_cache, decode_cache, weights, config, seq_len, rope_lut)

repl_loop()             (per turn)
  ├── input("Prompt> ")
  ├── tokenize (+ chat template if instruct)
  ├── length guard
  ├── run_once()
  │     ├── tokenize again (or reuse from guard)
  │     ├── pad to seq_len with EOS
  │     └── generate(on_token=stream_cb)
  │           ├── run_npu_prefill()
  │           └── decode loop
  │                 ├── per-token NPU calls
  │                 └── on_token(id, delta_text)  → stdout
  └── trailing newline
```

The `Session` object owns long-lived state (caches, BOs, weights). Per-turn state — KV cache, current_pos, generated_tokens — lives entirely in `generate()`'s frame and is freed at function exit.

## Error Handling

| Case | Behavior |
|------|----------|
| Empty prompt | Skip (re-prompt). |
| `/quit` / `/exit` / EOF / Ctrl-C at input | Clean exit. |
| Prompt > seq_len tokens | Print warning, skip turn. |
| Ctrl-C during generation | Catch in REPL, print `[interrupted]`, continue loop. |
| Tokenizer error | Let exception propagate (rare; signals a bug). |
| Kernel cache missing | `--run-only` already errors out today; unchanged. |
| `--interactive` with `--compile-only` | argparse-level error before any work. |

KeyboardInterrupt mid-generation does NOT corrupt session state because:
- `prefill_cache` / `decode_cache` BOs are reused across turns and are written by `prepare_runtime` (idempotent on repeat). Per-turn writes (input embeddings, KV cache buffers) are local arrays freed by GC.
- `generate()` allocates its own `k_cache` / `v_cache` arrays each call.
- The NPU does not retain state between XRT runs that affects a future run.

## Testing

Manual smoke (no automated coverage — interactive mode resists CI):

1. `make chat MODEL=base` — type `The capital of France is`, confirm `Paris` streams out.
2. Same session, second prompt `Once upon a time` — confirm independent generation, no memory of turn 1.
3. `make chat MODEL=instruct` — type `What is the capital of France?`, confirm response streams.
4. Type `/quit` — clean exit.
5. Ctrl-D at prompt — clean exit.
6. Ctrl-C mid-generation — prints `[interrupted]`, returns to prompt, next prompt works.
7. Paste a >2048-token prompt — confirm "Prompt too long" warning and re-prompt.
8. Regression: `make run PROMPT="The capital of France is"` — confirm bit-identical output to before refactor.

## Risks & Open Questions

- **BPE streaming edge case:** A single token may produce no visible delta (delta is `""`); this is correct, no flush needed but harmless if we flush anyway.
- **`tokenizer.decode` cost:** Called once per token, on a growing id list (~100 tokens). Worst-case ~100 decodes of ~100 ids = negligible vs. NPU work (~92ms/token).
- **Refactor regression risk:** `build_session` / `run_once` must reproduce today's exact tokenize→pad→generate path. Mitigated by smoke test #8.
- **`prepare_runtime` is non-idempotent in subtle ways:** It mutates `weights` (adds `_wq_t`, `_layer_idx`, `_decode_weights_preloaded_to_bos`, etc). Calling it twice would early-return on the guard flags. Since we only call it once per process from `build_session`, this is fine — but flagging it because anyone tempted to "reset state" between turns by re-running `prepare_runtime` would get unexpected behavior.
