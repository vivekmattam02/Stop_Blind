#!/usr/bin/env python3
"""Create a compact read-only comparison of two verified native-C0 runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "spatialvlm"
    / "training"
    / "native_c0_action_balance_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("native_c0_action_balance_comparison", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load the pure native-C0 action-balance comparison module.")
COMPARISON_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON_MODULE)
build_comparison = COMPARISON_MODULE.build_comparison

SOURCE_PATHS = (
    "src/spatialvlm/training/native_c0_action_balance_comparison.py",
    "scripts/analyze_native_c0_action_balance_comparison.py",
    "scripts/check_native_c0_action_balance_comparison.py",
    "tests/test_native_c0_action_balance_comparison.py",
    "docs/NATIVE_C0_ACTION_BALANCE_COMPARISON.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        source_path = repo / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"Comparison source is missing: {source_path}")
        locks[relative] = file_sha256(source_path)
    return locks


def evidence_bundle(root: Path, checkpoint: Path, label: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    updates_path = root / "updates.jsonl"
    verification_path = root / "verification.json"
    result = load_json(result_path, f"{label} result")
    selection = result.get("selection", {})
    selected = selection.get("checkpoint", {}) if isinstance(selection, dict) else {}
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {label} selected checkpoint: {checkpoint}")
    return {
        "manifest": load_json(manifest_path, f"{label} manifest"),
        "result": result,
        "verification": load_json(verification_path, f"{label} verification"),
        "digests": {
            "manifest_sha256": file_sha256(manifest_path),
            "result_sha256": file_sha256(result_path),
            "updates_sha256": file_sha256(updates_path),
            "verification_sha256": file_sha256(verification_path),
        },
        "selected_checkpoint": {
            "path": str(checkpoint.resolve()),
            "relative_path": checkpoint.name,
            "source_record_relative_path": selected.get("relative_path"),
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
            "update_step": selection.get("selected_update"),
        },
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--unweighted-root", type=Path, required=True)
    parser.add_argument("--unweighted-checkpoint", type=Path, required=True)
    parser.add_argument("--action-balanced-root", type=Path, required=True)
    parser.add_argument("--action-balanced-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    comparison = build_comparison(
        unweighted=evidence_bundle(
            args.unweighted_root.expanduser().resolve(),
            args.unweighted_checkpoint.expanduser().resolve(),
            "unweighted",
        ),
        action_balanced=evidence_bundle(
            args.action_balanced_root.expanduser().resolve(),
            args.action_balanced_checkpoint.expanduser().resolve(),
            "action-balanced",
        ),
        source_locks=source_locks(repo),
        recorded_at_utc=utc_now(),
    )
    write_exclusive(args.output.expanduser().resolve(), comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
