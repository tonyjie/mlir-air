"""Pinned official DFlash weights and an explicit adapter for the NPU decoder."""
import os
from pathlib import Path
import json
import struct

import numpy as np
from ml_dtypes import bfloat16

DRAFT_ID = 'z-lab/Qwen3-4B-DFlash-b16'
DRAFT_REVISION = 'b74e3a329c4d963783143b1e970d95b002be72bd'
SNAPSHOT = Path(os.environ.get(
    'DFLASH_DRAFT_SNAPSHOT',
    Path.home()/'.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots'/DRAFT_REVISION,
)).expanduser()
PACKED_CACHE = Path(os.environ.get(
    'DFLASH_DRAFT_CACHE',
    Path.home()/'.cache/qwen3_4b_dflash_npu/draft_requant_w2ch.npz',
)).expanduser()


class DraftWeights:
    """Read BF16 checkpoint tensors; share the declared target embedding/head."""

    def __init__(self, target_model=None, snapshot=SNAPSHOT):
        self.snapshot = Path(snapshot)
        self.path = self.snapshot / 'model.safetensors'
        self.config = json.loads((self.snapshot / 'config.json').read_text())
        expected = {'hidden_size': 2560, 'num_hidden_layers': 5,
                    'num_attention_heads': 32, 'num_key_value_heads': 8,
                    'head_dim': 128, 'intermediate_size': 9728, 'vocab_size': 151936}
        for key, value in expected.items():
            if self.config[key] != value:
                raise ValueError((key, self.config[key], value))
        self.tap_slots = [i + 1 for i in self.config['dflash_config']['target_layer_ids']]
        self.mask_token_id = self.config['dflash_config']['mask_token_id']
        self.target_model = target_model
        with self.path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.base = 8 + n

    def tensor(self, name):
        entry = self.header[name]
        if entry['dtype'] != 'BF16':
            raise ValueError((name, entry['dtype']))
        lo, hi = entry['data_offsets']
        with self.path.open('rb') as f:
            f.seek(self.base + lo)
            return np.frombuffer(f.read(hi - lo), bfloat16).reshape(entry['shape'])

    def layer_qk_norm(self, layer):
        return tuple(self.tensor(f'layers.{layer}.self_attn.{kind}_norm.weight')
                     for kind in ('q', 'k'))

    def embed_norm_lmhead(self):
        if self.target_model is None:
            raise ValueError('Drafter embedding/head must come from the declared target')
        embedding, _, head = self.target_model.embed_norm_lmhead()
        return embedding, self.tensor('norm.weight'), head
