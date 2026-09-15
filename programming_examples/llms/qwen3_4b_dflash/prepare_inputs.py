"""Freeze the saved iGPU prompts and compute CPU prefill with target hidden taps."""
import argparse
from collections import Counter
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

from target_config import HERE
from model_paths import TARGET_MODEL, resolve_target_model
sys.path.insert(0, str(HERE.parent / 'qwen3_4b_q4nx'))
from qwen3_4b_q4nx_weights import Q4nxModel
from prefill_reference import forward_prompt_with_taps

SLOTS = [2, 10, 18, 26, 34]


def load_prefill(root, case):
    with np.load(Path(root) / case['cache_file']) as z:
        if not np.array_equal(z['input_ids'], case['input_ids']):
            raise ValueError('Prefill token IDs differ from the frozen case')
        return {'input_ids': z['input_ids'].tolist(), 'k': z['k'].view(bfloat16),
                'v': z['v'].view(bfloat16), 'taps': z['taps'].view(bfloat16),
                'first': int(z['first'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--per-dataset', type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'prefill').mkdir(exist_ok=True)
    source = json.loads(args.cases.read_text())
    counts = Counter(); cases = []
    for case in source['cases']:
        if counts[case['dataset']] >= args.per_dataset:
            continue
        counts[case['dataset']] += 1
        case = {key: value for key, value in case.items() if key != 'cache_sha256'}
        cases.append(dict(case, cache_file=f'prefill/{len(cases):02d}.npz',
                          dataset_ordinal=counts[case['dataset']]-1))
    manifest = {k: v for k, v in source.items() if k != 'cases'}
    manifest.update(cases=cases, source_cases=str(args.cases),
                    source_cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                    target_source=TARGET_MODEL, tap_slots=SLOTS)
    destination = args.output / 'cases.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError('Cannot change a frozen case manifest')
    destination.write_text(json.dumps(manifest, indent=2)+'\n')
    model = Q4nxModel(resolve_target_model(manifest['target_source']))
    model.layer_weights = functools.lru_cache(maxsize=36)(model.layer_weights)
    model.embed_norm_lmhead = functools.lru_cache(maxsize=1)(model.embed_norm_lmhead)
    for case in cases:
        path = args.output / case['cache_file']
        if path.exists():
            load_prefill(args.output, case)
            continue
        print(json.dumps({'prefill': case['id'], 'tokens': len(case['input_ids']), 'status': 'start'}), flush=True)
        start = time.perf_counter()
        k, v, logits, boundaries = forward_prompt_with_taps(model, case['input_ids'], SLOTS)
        taps = np.stack([boundaries[i] for i in SLOTS], axis=1).reshape(len(case['input_ids']), -1)
        values = {'input_ids': np.asarray(case['input_ids'], np.int64),
                  'k': k.astype(bfloat16).view(np.uint16), 'v': v.astype(bfloat16).view(np.uint16),
                  'taps': taps.astype(bfloat16).view(np.uint16), 'first': np.asarray(logits[-1].argmax())}
        seconds = time.perf_counter()-start
        np.savez(path, **values)
        record = {'id': case['id'], 'prefill_seconds': seconds,
                  'first': int(values['first']), 'input_tokens': len(case['input_ids']),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        del k, v, logits, boundaries, taps, values


if __name__ == '__main__':
    main()
