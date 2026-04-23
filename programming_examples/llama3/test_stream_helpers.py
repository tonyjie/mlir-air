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


def test_delta_text_with_empty_ids():
    """Calling with an empty id list returns '' and leaves state unchanged.
    This path triggers when generate() is invoked with n_tokens=0."""
    tok = _FakeTokenizer({1: "Hello"})
    state = _StreamState()
    assert _delta_text(tok, [], state) == ""
    assert state.printed_len == 0
