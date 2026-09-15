# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One paired B1/DFlash pass, alternating process order in two-prompt shards."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_data import sha, verify_inputs
from study_report import DATASETS, print_report, read_rows, report

HERE = Path(__file__).resolve().parent


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def artifact(path, batch):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "compiled" or manifest["config"]["batch"] != batch or manifest["config"]["capacity"] != 4096:
        raise ValueError(f"Wrong artifact configuration: {path}")
    from dataclasses import asdict
    from selected_config import baseline_config, require_selected
    from target_config import TargetConfig
    cfg = TargetConfig(**manifest["config"])
    if batch == 1:
        if asdict(cfg) != asdict(baseline_config()):
            raise ValueError("B1 configuration differs from the historical baseline")
    else:
        require_selected(cfg, cfg.role)
    for name, expected in manifest["artifacts"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {path / name}")
    return {"manifest_sha256": sha(path / "manifest.json"), "config": manifest["config"],
            "compiler": manifest.get("compiler"), "artifacts": manifest["artifacts"]}


def system_info():
    result = {"platform": list(os.uname()), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "threads": 8, "python": sys.version}
    for name, command in {
        "cpu": ["lscpu"], "memory": ["free", "-h"],
        "xrt_platform": ["/opt/xilinx/xrt/bin/xrt-smi", "examine", "-r", "platform"],
        "xrt_version": ["/opt/xilinx/xrt/bin/xrt-smi", "--version"],
        "runtime_packages": ["dpkg-query", "-W", "xrt-base", "xrt_plugin-amdxdna"],
    }.items():
        try:
            p = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            result[name] = {"returncode": p.returncode, "output": p.stdout}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--per-dataset", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--label", default="reviewer")
    args = ap.parse_args()
    if not 1 <= args.per_dataset <= 16 or not 2 <= args.tokens <= 2048:
        ap.error("Use 1..16 prompts per dataset and 2..2048 output tokens")
    inputs, dest = args.inputs.resolve(), args.output.resolve()
    source = verify_inputs(inputs)
    cases = [c for c in source["cases"] if c["dataset_ordinal"] < args.per_dataset]
    paths = {name: getattr(args, name).resolve() for name in ("baseline", "target", "draft")}
    from model_paths import default_target_cache, default_draft_cache
    identity = {"schema": "dflash.reviewer.study-identity.v1", "tokens": args.tokens,
                "per_dataset": args.per_dataset, "threads": 8,
                "inputs_sha256": sha(inputs / "cases.json"),
                "driver_sha256": sha(HERE / "run_benchmark.py"),
                "source_commit": subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip(),
                "artifacts": {name: artifact(path, 1 if name == "baseline" else 8) for name, path in paths.items()},
                "weight_caches": {"target": sha(default_target_cache()), "draft": sha(default_draft_cache())}}
    expected_weights = json.loads((HERE / "KNOWN_GOOD.json").read_text())["weights"]
    for role in ("target", "draft"):
        if identity["weight_caches"][role] != expected_weights[role + "_cache_sha256"]:
            raise ValueError(f"{role} packed cache differs from the measured study; use the pinned environment and a fresh make weights cache")
    if dest.exists() and not args.resume:
        raise FileExistsError(f"{dest}; use a new STUDY_DIR or RESUME=1")
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / "study.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Resume identity differs: source, inputs, weights, artifacts, or options changed")
    else:
        reference = inputs.parent / "reference"
        state = {"identity": identity, "label": args.label, "cases": cases,
                 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "reference": str(reference) if (reference / "spec.jsonl").exists() else None,
                 "stages": [], "status": "prepared"}
        save(state_path, state)
        save(dest / "system.json", system_info())
    runs = dest / "runs"
    runs.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "8"
    for shard in range((args.per_dataset + 1) // 2):
        selected = [c for name in DATASETS for c in cases
                    if c["dataset"] == name and 2 * shard <= c["dataset_ordinal"] < 2 * shard + 2]
        shard_dir = runs / f"shard_{shard:02d}_inputs"
        shard_dir.mkdir(exist_ok=True)
        manifest = dict(source, cases=[dict(c, cache_file=str(inputs / c["cache_file"])) for c in selected])
        save(shard_dir / "cases.json", manifest)
        order = ("baseline", "spec") if shard % 2 == 0 else ("spec", "baseline")
        for mode in order:
            out = runs / f"shard_{shard:02d}_{mode}.jsonl"
            existing = read_rows(out) if out.exists() else []
            if [r["id"] for r in existing] != [c["id"] for c in selected[:len(existing)]]:
                raise ValueError(f"Unexpected partial output: {out}")
            if len(existing) == len(selected):
                continue
            command = [sys.executable, str(HERE / "run_benchmark.py"),
                       "--inputs", str(shard_dir), "--output", str(out), "--mode", mode,
                       "--target", str(paths["baseline"] if mode == "baseline" else paths["target"]),
                       "--tokens", str(args.tokens), "--per-dataset", str(args.per_dataset)]
            if mode == "spec":
                command += ["--draft", str(paths["draft"])]
            if out.exists():
                command += ["--resume"]
            print(f"\nShard {shard + 1}/{(args.per_dataset + 1) // 2}: {mode}, {len(selected)} requests", flush=True)
            started = time.monotonic()
            state["status"] = "running"
            save(state_path, state)
            with out.with_suffix(".log").open("a") as log:
                with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
                    for line in child.stdout:
                        log.write(line); log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
            state["stages"].append({"shard": shard, "mode": mode, "returncode": code,
                                   "process_seconds": time.monotonic() - started})
            state["status"] = "running" if code == 0 else "failed"
            save(state_path, state)
            if code:
                raise RuntimeError(f"Benchmark failed; see {out.with_suffix('.log')}")
    rows = {mode: [r for p in sorted(runs.glob(f"shard_*_{mode}.jsonl")) for r in read_rows(p)] for mode in ("baseline", "spec")}
    result = report(rows["baseline"], rows["spec"], cases, args.tokens, state["reference"])
    save(dest / "summary.json", result)
    state["status"] = "passed"
    state["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(state_path, state)
    print_report(result)


if __name__ == "__main__":
    main()
