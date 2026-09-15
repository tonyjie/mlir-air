# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate a paired benchmark and compare it with the historical 80 prompts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_rows(rows, cases, mode, tokens):
    expected = {case["id"]: case for case in cases}
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"{mode}: missing, extra, or duplicate requests")
    for row in rows:
        case = expected[row["id"]]
        if (row["mode"], row["max_new_tokens"], row["capacity"], row["block_size"], row["repeat"]) != (mode, tokens, 4096, 1 if mode == "baseline" else 8, 0):
            raise ValueError(f"Wrong benchmark configuration: {row['id']}")
        if row["prompt_sha256"] != case["prompt_sha256"] or row["input_tokens"] != len(case["input_ids"]):
            raise ValueError(f"Wrong prompt identity: {row['id']}")
        if row["dataset"] != case["dataset"] or row["decode_count"] != len(row["output_ids"]) - 1:
            raise ValueError(f"Wrong dataset or output count: {row['id']}")
        if row["decode_seconds"] <= 0 or row["decode_count"] <= 0:
            raise ValueError(f"Empty or invalid timed request: {row['id']}")
        if mode == "spec":
            rounds = row["rounds"]
            if (row["verification_steps"] != len(rounds)
                or row["draft_accepted"] != sum(r["accepted_drafts"] for r in rounds)
                or row["draft_proposed"] != sum(r["proposed"] for r in rounds)
                or row["decode_count"] != sum(r["emitted"] for r in rounds)
                or any(not 0 <= r["accepted_drafts"] <= r["proposed"] <= 7 for r in rounds)):
                raise ValueError(f"Acceptance accounting failed: {row['id']}")
    return {row["id"]: row for row in rows}


def mode_summary(rows):
    tokens = sum(row["decode_count"] for row in rows)
    seconds = sum(row["decode_seconds"] for row in rows)
    return {"requests": len(rows), "decode_tokens": tokens, "decode_seconds": seconds,
            "pooled_tps": tokens / seconds,
            "equal_request_tps": 1 / statistics.mean(r["decode_seconds"] / r["decode_count"] for r in rows)}


def paired_summary(baseline, spec):
    base, candidate = mode_summary(baseline), mode_summary(spec)
    rounds = sum(row["verification_steps"] for row in spec)
    accepted = sum(row["draft_accepted"] for row in spec)
    proposed = sum(row["draft_proposed"] for row in spec)
    stages = defaultdict(float)
    for row in spec:
        for name, seconds in row["stage_seconds"].items():
            stages[name] += seconds
    return {"baseline": base, "dflash": candidate,
            "pooled_speedup": candidate["pooled_tps"] / base["pooled_tps"],
            "equal_request_speedup": candidate["equal_request_tps"] / base["equal_request_tps"],
            "paper_style_mean_tau": statistics.mean(row["mean_acceptance_length"] for row in spec),
            "round_weighted_tau": 1 + accepted / rounds,
            "accepted_proposed_fraction": accepted / proposed,
            "emitted_tokens_per_round": candidate["decode_tokens"] / rounds,
            "verification_rounds": rounds,
            "mean_round_ms": candidate["decode_seconds"] * 1000 / rounds,
            "stage_ms_per_round": {k: v * 1000 / rounds for k, v in stages.items()}}


def report(baseline, spec, cases, tokens, reference=None):
    modes = {"baseline": validate_rows(baseline, cases, "baseline", tokens),
             "spec": validate_rows(spec, cases, "spec", tokens)}
    def group(selected):
        return paired_summary(*[[modes[mode][c["id"]] for c in selected] for mode in ("baseline", "spec")])
    datasets = {name: group([c for c in cases if c["dataset"] == name]) for name in DATASETS}
    result = {"schema": "dflash.reviewer.study.v1", "passed_accounting": True,
              "output_cap": tokens, "requests_per_dataset": len(cases) // 5,
              "overall": group(cases), "datasets": datasets,
              "macro": {"equal_request_speedup": statistics.mean(d["equal_request_speedup"] for d in datasets.values()),
                        "paper_style_mean_tau": statistics.mean(d["paper_style_mean_tau"] for d in datasets.values())},
              "timing": "Decode wall time; excludes loading, warmup, prefill, first token and initial target KV reset. Includes initial drafter context preparation and all DFlash rounds.",
              "correctness_scope": "B1 GEMV is a performance baseline. Exact correctness uses sequential row zero of B8, checked by make verify."}
    if reference:
        comparison = {}
        for mode in ("baseline", "spec"):
            old = {r["id"]: r for r in read_rows(Path(reference) / f"{mode}.jsonl")}
            changed, acceptance_changed = [], []
            for c in cases:
                row, ref = modes[mode][c["id"]], old[c["id"]]
                expected = ref["output_ids"][:tokens]
                if row["output_ids"] != expected:
                    changed.append(c["id"])
                if mode == "spec" and tokens == ref["max_new_tokens"] and any(row[k] != ref[k] for k in ("draft_accepted", "draft_proposed", "verification_steps")):
                    acceptance_changed.append(c["id"])
            item = {"matching_output_prefixes": len(cases) - len(changed),
                    "different_output_ids": changed, "different_acceptance_ids": acceptance_changed}
            if tokens == 2048:
                original = mode_summary([old[c["id"]] for c in cases])
                current = mode_summary(list(modes[mode].values()))
                item.update(reference_pooled_tps=original["pooled_tps"],
                            throughput_vs_reference=current["pooled_tps"] / original["pooled_tps"])
            comparison[mode] = item
        result["reference_comparison"] = comparison
    return result


def print_report(result):
    print("\nDataset       Requests   B1 tok/s  DFlash tok/s  Speedup  Mean tau  Acceptance")
    for name, d in [*result["datasets"].items(), ("ALL", result["overall"])]:
        print(f"{name:<13} {d['baseline']['requests']:>8} {d['baseline']['pooled_tps']:>10.3f} "
              f"{d['dflash']['pooled_tps']:>13.3f} {d['pooled_speedup']:>7.3f}x "
              f"{d['paper_style_mean_tau']:>9.3f} {d['accepted_proposed_fraction']:>10.2%}")
    print(f"Macro equal-request speedup: {result['macro']['equal_request_speedup']:.3f}x")
    print(result["timing"])
    if "reference_comparison" in result:
        print("Historical output comparison: " + json.dumps(result["reference_comparison"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", type=Path)
    ap.add_argument("--reference", type=Path, help="Reference directory when summarizing a study copied from another machine")
    args = ap.parse_args()
    config = json.loads((args.study / "study.json").read_text())
    cases = config["cases"]
    rows = {mode: [row for p in sorted((args.study / "runs").glob(f"shard_*_{mode}.jsonl")) for row in read_rows(p)] for mode in ("baseline", "spec")}
    reference = args.reference
    if reference is None:
        candidates = [config.get("reference"), HERE / "build/benchmark_bundle/reference"]
        reference = next((Path(p) for p in candidates if p and (Path(p) / "spec.jsonl").is_file()), None)
    result = report(rows["baseline"], rows["spec"], cases, config["identity"]["tokens"], reference)
    (args.study / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print_report(result)


if __name__ == "__main__":
    main()
