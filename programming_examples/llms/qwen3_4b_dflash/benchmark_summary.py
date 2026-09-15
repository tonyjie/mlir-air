# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Summarize saved decode timings; throughput is total tokens / total seconds."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(rows):
    tokens = sum(row["decode_count"] for row in rows)
    seconds = sum(row["decode_seconds"] for row in rows)
    accepted = sum(row["draft_accepted"] for row in rows)
    proposed = sum(row["draft_proposed"] for row in rows)
    rounds = sum(row["verification_steps"] for row in rows)
    taus = [row["mean_acceptance_length"] for row in rows if row["mean_acceptance_length"] is not None]
    return {
        "requests": len(rows),
        "decode_tokens": tokens,
        "decode_seconds": seconds,
        "pooled_decode_tps": tokens / seconds if seconds else None,
        "mean_request_tau": statistics.mean(taus) if taus else None,
        "accepted_proposed_fraction": accepted / proposed if proposed else None,
        "emitted_tokens_per_round": tokens / rounds if rounds else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.results for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("No completed requests in the supplied files")
    if len({row["mode"] for row in rows}) != 1 or len({row["max_new_tokens"] for row in rows}) != 1:
        parser.error("Summarize one mode and one output cap at a time")
    keys = [(row["id"], row["repeat"]) for row in rows]
    if len(set(keys)) != len(keys):
        parser.error("Duplicate request/repeat entries")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    report = {
        "mode": rows[0]["mode"],
        "max_new_tokens": rows[0]["max_new_tokens"],
        "overall": summarize(rows),
        "datasets": {dataset: summarize(items) for dataset, items in sorted(grouped.items())},
        "timing": "Decode only; excludes initialization, prompt prefill, first token, initial KV reset, and warmup",
    }
    print("\nDataset       Requests   Decode tok/s   Mean tau   Accepted/proposed")
    for name, summary in [*report["datasets"].items(), ("ALL", report["overall"])]:
        speed = summary["pooled_decode_tps"]
        tau = summary["mean_request_tau"]
        acceptance = summary["accepted_proposed_fraction"]
        print(f"{name:<13} {summary['requests']:>8}   "
              f"{speed if speed is not None else 0:>12.3f}   "
              f"{f'{tau:.3f}' if tau is not None else '-':>8}   "
              f"{f'{acceptance:.2%}' if acceptance is not None else '-':>17}")
    print(report["timing"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
