#!/usr/bin/env python3
"""Create a read-only per-action analysis of a verified native-C0 development failure."""

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
    / "native_c0_failure_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("native_c0_failure_analysis_runtime", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load the pure native-C0 failure-analysis module.")
ANALYSIS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS_MODULE)
build_failure_analysis = ANALYSIS_MODULE.build_failure_analysis

SOURCE_PATHS = (
    "src/spatialvlm/training/native_c0_failure_analysis.py",
    "scripts/analyze_native_c0_failure.py",
    "scripts/check_native_c0_failure_analysis.py",
    "tests/test_native_c0_failure_analysis.py",
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
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Failure-analysis source is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def selected_checkpoint_record(result: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    selected = result.get("selection", {})
    record = selected.get("checkpoint", {}) if isinstance(selected, dict) else {}
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint mirror is missing: {checkpoint}")
    return {
        "path": str(checkpoint.resolve()),
        "relative_path": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "update_step": selected.get("selected_update"),
        "source_record_relative_path": record.get("relative_path"),
    }


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite failure analysis: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    updates_path = root / "updates.jsonl"
    verification_path = root / "verification.json"
    result = load_json(result_path, "development result")
    analysis = build_failure_analysis(
        manifest=load_json(manifest_path, "development manifest"),
        result=result,
        verification=load_json(verification_path, "development verification"),
        manifest_sha256=file_sha256(manifest_path),
        result_sha256=file_sha256(result_path),
        updates_sha256=file_sha256(updates_path),
        verification_sha256=file_sha256(verification_path),
        selected_checkpoint=selected_checkpoint_record(
            result, args.selected_checkpoint.expanduser().resolve()
        ),
        source_locks=source_locks(repo),
        recorded_at_utc=utc_now(),
    )
    write_exclusive_json(args.output.expanduser().resolve(), analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
