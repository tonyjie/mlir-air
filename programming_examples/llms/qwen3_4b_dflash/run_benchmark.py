"""Matched saved-prompt NPU benchmarks: production AR, DFlash, and GEMM control."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from context_prepass import ContextPrepass
from draft_weights import DraftWeights, PACKED_CACHE, DRAFT_ID, DRAFT_REVISION
from prepare_inputs import load_prefill
from speculative import DFlash, autoregressive
from target_runtime import TargetDecoder


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def load_pair(target_path, draft_path):
    target = TargetDecoder(target_path)
    weights = DraftWeights(target.model)
    drafter = TargetDecoder(draft_path, packed_cache=PACKED_CACHE, model_weights=weights)
    prepass = ContextPrepass(weights)
    return DFlash(target, drafter, prepass, weights.tap_slots, weights.mask_token_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=('spec', 'baseline', 'control'), required=True)
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--draft', type=Path)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--per-dataset', type=int, default=1)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    cases = [case for case in json.loads((args.inputs/'cases.json').read_text())['cases']
             if case['dataset_ordinal'] < args.per_dataset]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            assert (row['mode'], row['max_new_tokens'], row['repeat']) == (args.mode, args.tokens, args.repeat)
            done.add(row['id'])
    if done == {case['id'] for case in cases}:
        print('already complete', flush=True)
        return
    if args.mode == 'spec':
        if args.draft is None:
            raise ValueError('--draft is required for DFlash')
        loop = load_pair(args.target, args.draft)
        decoder = loop.target
    else:
        decoder = TargetDecoder(args.target)
        if decoder.batch != (1 if args.mode == 'baseline' else 8):
            raise ValueError('Wrong batch arithmetic for requested mode')
    metadata = {'mode': args.mode, 'tokens': args.tokens, 'case_ids': [c['id'] for c in cases],
                'repeat': args.repeat, 'target': str(args.target.resolve()),
                'target_manifest_sha256': sha(args.target/'manifest.json'),
                'capacity': decoder.capacity, 'block_size': decoder.batch,
                'draft_id': DRAFT_ID if args.mode == 'spec' else None,
                'draft_revision': DRAFT_REVISION if args.mode == 'spec' else None,
                'draft_manifest_sha256': sha(args.draft/'manifest.json') if args.draft else None,
                'prepass_epsilon': loop.prepass.eps if args.mode == 'spec' else None,
                'thinking': False, 'temperature': 0, 'threads': 8,
                'timing': 'wall-clock decode; excludes loading, warmup, CPU prefill, first token, initial target KV upload and zeroing; includes initial drafter context preparation/upload and every round',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.with_suffix('.config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    warm = load_prefill(args.inputs, cases[0])
    if args.mode == 'spec':
        loop.reset(warm)
        loop.generate(32, eos_ids=())
    else:
        autoregressive(decoder, warm, 32, eos_ids=())
    with args.output.open('a' if args.resume else 'x') as out:
        for case in cases:
            if case['id'] in done:
                continue
            prefill = load_prefill(args.inputs, case)
            print(json.dumps({'starting': case['id'], 'mode': args.mode, 'tokens': args.tokens}), flush=True)
            if args.mode == 'spec':
                loop.reset(prefill)
                generated = loop.generate(args.tokens)
            else:
                generated = autoregressive(decoder, prefill, args.tokens)
            n = max(0, len(generated.output_ids)-1)
            rounds = generated.rounds
            accepted = sum(r['accepted_drafts'] for r in rounds)
            proposed = sum(r['proposed'] for r in rounds)
            nr = len(rounds)
            attempts = [sum(r['accepted_drafts'] >= j for r in rounds) for j in range(decoder.batch-1)]
            successes = [sum(r['accepted_drafts'] > j for r in rounds) for j in range(decoder.batch-1)]
            row = {'id': case['id'], 'dataset': case['dataset'], 'repeat': args.repeat, 'mode': args.mode,
                   'prompt_sha256': case['prompt_sha256'], 'input_tokens': len(case['input_ids']),
                   'max_new_tokens': args.tokens, 'capacity': decoder.capacity, 'block_size': decoder.batch,
                   'output_ids': generated.output_ids, 'num_output_tokens': len(generated.output_ids),
                   'decode_count': n, 'decode_seconds': generated.wall_seconds,
                   'decode_tps': n/generated.wall_seconds if n and generated.wall_seconds else None,
                   'stop_type': 'eos' if generated.stopped_on_eos else 'limit',
                   'verification_steps': nr, 'draft_accepted': accepted, 'draft_proposed': proposed,
                   'mean_acceptance_length': 1+accepted/nr if nr else None,
                   'emitted_per_round': n/nr if nr else None,
                   'draft_acceptance_fraction': accepted/proposed if proposed else None,
                   'conditional_attempts': attempts, 'conditional_accepts': successes,
                   'stage_seconds': generated.stages, 'rounds': rounds}
            out.write(json.dumps(row)+'\n'); out.flush()
            print(json.dumps({k: row[k] for k in ('id','mode','decode_count','decode_seconds','decode_tps','mean_acceptance_length','stop_type')}), flush=True)


if __name__ == '__main__':
    main()
