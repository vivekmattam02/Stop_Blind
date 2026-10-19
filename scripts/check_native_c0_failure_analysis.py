#!/usr/bin/env python3
"""Independently reconstruct a native-C0 per-action failure-analysis artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.analyze_native_c0_failure import (
    ANALYSIS_MODULE,
    file_sha256,
    load_json,
    selected_checkpoint_record,
    source_locks,
    utc_now,
)

NATIVE_C0_FAILURE_ANALYSIS_GATE = ANALYSIS_MODULE.NATIVE_C0_FAILURE_ANALYSIS_GATE
NATIVE_C0_FAILURE_ANALYSIS_SCHEMA_VERSION = (
    ANALYSIS_MODULE.NATIVE_C0_FAILURE_ANALYSIS_SCHEMA_VERSION
)
build_failure_analysis = ANALYSIS_MODULE.build_failure_analysis

VERIFIER_GATE = "independent_native_qwen_c0_failure_analysis_verification_v1"


def verify_analysis(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    try:
        repo = args.repo.expanduser().resolve()
        root = args.result_root.expanduser().resolve()
        manifest_path = root / "manifest.json"
        result_path = root / "result.json"
        updates_path = root / "updates.jsonl"
        verification_path = root / "verification.json"
        analysis_path = args.analysis.expanduser().resolve()
        observed = load_json(analysis_path, "failure analysis")
        expected = build_failure_analysis(
            manifest=load_json(manifest_path, "development manifest"),
            result=load_json(result_path, "development result"),
            verification=load_json(verification_path, "development verification"),
            manifest_sha256=file_sha256(manifest_path),
            result_sha256=file_sha256(result_path),
            updates_sha256=file_sha256(updates_path),
            verification_sha256=file_sha256(verification_path),
            selected_checkpoint=selected_checkpoint_record(
                load_json(result_path, "development result"),
                args.selected_checkpoint.expanduser().resolve(),
            ),
            source_locks=source_locks(repo),
            recorded_at_utc=str(observed.get("recorded_at_utc")),
        )
        if observed != expected:
            errors.append("failure analysis does not independently reconstruct")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite failure-analysis verification: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify_analysis(args)
    analysis_path = args.analysis.expanduser().resolve()
    payload = {
        "schema_version": NATIVE_C0_FAILURE_ANALYSIS_SCHEMA_VERSION,
        "gate": VERIFIER_GATE,
        "analysis_gate": NATIVE_C0_FAILURE_ANALYSIS_GATE,
        "recorded_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "analysis_path": str(analysis_path),
        "analysis_sha256": file_sha256(analysis_path) if analysis_path.is_file() else None,
        "analysis_reconstruction_performed": True,
        "selected_checkpoint_fully_hashed": True,
    }
    write_exclusive(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
