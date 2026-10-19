#!/usr/bin/env python3
"""Independently verify the full native-C0 action-balanced development comparison."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import check_native_c0_development_artifacts as base_verifier
from scripts.run_spark_native_c0_action_balance import expected_manifest, source_locks
from spatialvlm.training.native_c0_action_balance import (
    NATIVE_C0_ACTION_BALANCE_GATE,
    NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
    NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
)
from spatialvlm.training.native_c0_action_balance_full import (
    NATIVE_C0_ACTION_BALANCE_VERIFIER_GATE,
    NativeC0ActionBalanceProtocol,
    validate_action_balance_protocol_record,
    verify_action_balance_training_records,
)
from spatialvlm.training.native_c0_gate import file_sha256


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _verify_events(
    events: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    protocol: NativeC0ActionBalanceProtocol,
    initial_sha256: str,
    final_sha256: str,
) -> list[str]:
    contract = getattr(_verify_events, "contract", None)
    if not isinstance(contract, dict):
        return ["action-balance verifier did not bind the frozen treatment contract"]
    return verify_action_balance_training_records(
        contract=contract,
        events=events,
        samples=samples,
        protocol=protocol,
        initial_trainable_sha256=initial_sha256,
        final_trainable_sha256=final_sha256,
    )


def verify_action_balance(args: argparse.Namespace) -> list[str]:
    """Reuse the original outcome/checkpoint verifier with weighted contracts injected."""
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return ["required full action-balance manifest or result is missing"]
    manifest = base_verifier._load_json(manifest_path, "action-balance manifest")
    result = base_verifier._load_json(result_path, "action-balance result")
    _verify_events.contract = manifest.get("action_balance_contract")

    replacements = {
        "NATIVE_C0_DEVELOPMENT_GATE": NATIVE_C0_ACTION_BALANCE_GATE,
        "NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "NATIVE_C0_OBJECTIVE": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
        "NativeC0DevelopmentProtocol": NativeC0ActionBalanceProtocol,
        "validate_protocol_record": validate_action_balance_protocol_record,
        "_expected_manifest": expected_manifest,
        "_source_locks": source_locks,
        "_verify_events": _verify_events,
    }
    originals = {name: getattr(base_verifier, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base_verifier, name, value)
        errors = base_verifier.verify_development(args)
    finally:
        for name, value in originals.items():
            setattr(base_verifier, name, value)
        delattr(_verify_events, "contract")

    if result.get("action_balance_contract") != manifest.get("action_balance_contract"):
        errors.append("result action-balance contract differs from the frozen manifest")
    if result.get("stress_authorization") != manifest.get("stress_authorization"):
        errors.append("result stress authorization differs from the frozen manifest")
    if result.get("adaptive_development_comparison") is not True:
        errors.append("result does not retain its adaptive-development interpretation boundary")
    expected, _ = expected_manifest(args)
    if manifest != expected:
        errors.append("action-balance manifest does not independently reconstruct")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--failure-verification", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify_action_balance(args)
    root = args.result_root.expanduser().resolve()
    payload = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_VERIFIER_GATE,
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
        "dataset_reconstruction_performed": True,
        "weighted_event_chain_reconstruction_performed": True,
        "all_checkpoint_files_fully_loaded_and_hashed": True,
        "outcome_reconstruction_performed": True,
        "stress_authorization_reconstruction_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
