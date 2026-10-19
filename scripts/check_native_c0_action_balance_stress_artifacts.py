#!/usr/bin/env python3
"""Independently verify native-C0 action-balance numerical-stress artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_spark_native_c0_action_balance_stress import (
    HARD_MAX_SECONDS,
    STRESS_VERIFIER_GATE,
    _load_json,
    expected_manifest,
)
from spatialvlm.training.native_c0_action_balance import (
    EXPECTED_STRESS_HISTOGRAM,
    NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
    NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
    STRESS_EXPOSURES,
    STRESS_UPDATES,
    verify_stress_records,
)
from spatialvlm.training.native_c0_gate import file_sha256


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Stress JSONL row is not an object.")
            rows.append(payload)
    return rows


def verify(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    events_path = root / "updates.jsonl"
    for path in (manifest_path, result_path, events_path):
        if not path.is_file():
            errors.append(f"required action-balance stress artifact is missing: {path}")
    if errors:
        return errors
    manifest = _load_json(manifest_path, "action-balance stress manifest")
    result = _load_json(result_path, "action-balance stress result")
    expected, _ = expected_manifest(args)
    if manifest != expected:
        errors.append("action-balance stress manifest does not independently reconstruct")
    manifest_sha = file_sha256(manifest_path)
    events_sha = file_sha256(events_path)
    if (
        result.get("gate") != NATIVE_C0_ACTION_BALANCE_STRESS_GATE
        or result.get("schema_version") != NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION
        or result.get("run_id") != args.run_id
        or result.get("status") != "passed"
        or result.get("manifest_sha256") != manifest_sha
        or result.get("updates_sha256") != events_sha
        or result.get("completed_updates") != STRESS_UPDATES
        or result.get("completed_exposures") != STRESS_EXPOSURES
        or result.get("observed_action_histogram") != EXPECTED_STRESS_HISTOGRAM
        or result.get("weights") != manifest.get("contract", {}).get("weights")
        or result.get("scientific_result_emitted") is not False
        or result.get("full_action_balance_run_authorized") is not True
        or float(result.get("runtime", {}).get("elapsed_seconds", HARD_MAX_SECONDS + 1))
        > HARD_MAX_SECONDS
    ):
        errors.append("action-balance stress result identity/status/provenance mismatch")
    events = _read_jsonl(events_path)
    errors.extend(
        verify_stress_records(
            contract=manifest.get("contract", {}),
            events=events,
            samples=manifest.get("samples", []),
            initial_trainable_sha256=str(result.get("initial_trainable_state_sha256")),
            final_trainable_sha256=str(result.get("final_trainable_state_sha256")),
        )
    )
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--failure-verification", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify(args)
    root = args.result_root.expanduser().resolve()
    payload = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": STRESS_VERIFIER_GATE,
        "recorded_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(root / "manifest.json"),
        "manifest_sha256": file_sha256(root / "manifest.json")
        if (root / "manifest.json").is_file()
        else None,
        "result_path": str(root / "result.json"),
        "result_sha256": file_sha256(root / "result.json")
        if (root / "result.json").is_file()
        else None,
        "updates_path": str(root / "updates.jsonl"),
        "updates_sha256": file_sha256(root / "updates.jsonl")
        if (root / "updates.jsonl").is_file()
        else None,
        "contract_reconstruction_performed": True,
        "event_chain_reconstruction_performed": True,
        "scientific_result_emitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

