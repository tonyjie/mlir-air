# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Static identity/configuration checks for selected target and drafter artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from selected_config import require_selected
from target_config import ABI_VERSION, TargetConfig


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(role: str, path: Path, reference: Path | None) -> dict:
    path = path.resolve()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "compiled"
    assert manifest["abi"]["version"] == ABI_VERSION
    config = TargetConfig(**manifest["config"])
    require_selected(config, role)
    artifacts = {}
    for name, expected in manifest["artifacts"].items():
        artifact = path / name
        actual = sha(artifact)
        assert actual == expected, (role, name, actual, expected)
        artifacts[name] = {"sha256": actual, "bytes": artifact.stat().st_size}
    route = manifest["routing_diagnostic"]
    assert route["fix_shim_arbiter"] is True
    assert route["isolate_main_output"] is True
    assert route["isolate_normalization_replay"] is False
    assert len(route["core_elf_hashes"]) == 30
    comparison = None
    if reference:
        reference = reference.resolve()
        old = json.loads((reference / "manifest.json").read_text())
        assert old["config"] == manifest["config"]
        assert old["abi"] == manifest["abi"]
        transaction = f"decode_dyn_L{config.capacity}.txn.h"
        comparison = {
            "directory": str(reference),
            "config_exact": True,
            "abi_exact": True,
            "transaction_exact": sha(path / transaction) == sha(reference / transaction),
            "xclbin_same_size": (path / f"decode_dyn_L{config.capacity}.xclbin").stat().st_size
            == (reference / f"decode_dyn_L{config.capacity}.xclbin").stat().st_size,
            "note": "Xclbin container hashes may vary across fresh builds; runtime behavior is checked separately.",
        }
        assert comparison["transaction_exact"] and comparison["xclbin_same_size"]
    return {
        "role": role,
        "directory": str(path),
        "manifest_sha256": sha(manifest_path),
        "config": manifest["config"],
        "abi": manifest["abi"],
        "artifacts": artifacts,
        "routing": {
            "fix_shim_arbiter": True,
            "isolate_main_output": True,
            "core_elf_count": len(route["core_elf_hashes"]),
            "physical_ir_hashes": route["physical_ir_hashes"],
        },
        "reference": comparison,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=here / "build/target")
    parser.add_argument("--draft", type=Path, default=here / "build/draft")
    parser.add_argument("--reference-target", type=Path)
    parser.add_argument("--reference-draft", type=Path)
    parser.add_argument("--output", type=Path, default=here / "build/artifact_check.json")
    args = parser.parse_args()
    result = {
        "passed": True,
        "target": inspect("target", args.target, args.reference_target),
        "draft": inspect("drafter", args.draft, args.reference_draft),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "passed": True,
        "target_manifest": result["target"]["manifest_sha256"],
        "draft_manifest": result["draft"]["manifest_sha256"],
        "target_reference": result["target"]["reference"],
        "draft_reference": result["draft"]["reference"],
    }, indent=2))


if __name__ == "__main__":
    main()
