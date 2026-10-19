#!/usr/bin/env python3
"""Correctly reverify the preserved native-Qwen C0 v1 result without rerunning training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatialvlm.training.native_c0_artifacts import file_sha256
from spatialvlm.training.native_c0_artifacts_v2 import (
    NATIVE_C0_V1_CHECKPOINT_SHA256,
    NATIVE_C0_V1_RESULT_SHA256,
    NATIVE_C0_VERIFIER_V2_CORRECTION,
    NATIVE_C0_VERIFIER_V2_GATE_NAME,
    NATIVE_C0_VERIFIER_V2_SCHEMA_VERSION,
    NATIVE_C0_VERIFIER_V2_SOURCE_PATHS,
    verify_native_c0_result_v2,
    verify_v1_habitat_false_negative,
)


def _reconstruct_expected_probe_v1(args: argparse.Namespace) -> dict[str, Any]:
    """Load the preserved v1 reconstructor without modifying its source-locked file."""
    # Direct script execution places ``scripts/`` rather than the repository root on sys.path.
    repo_import_root = Path(__file__).resolve().parents[1]
    if str(repo_import_root) not in sys.path:
        sys.path.insert(0, str(repo_import_root))
    from scripts.check_native_c0_artifacts import _reconstruct_expected_probe

    return _reconstruct_expected_probe(args)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--processor-reverification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prior-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    prior_path = args.prior_verification.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    result_hash = file_sha256(result_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    errors: list[str] = []
    if result_hash != NATIVE_C0_V1_RESULT_SHA256:
        errors.append("preserved native-C0 v1 result SHA-256 mismatch")
    if checkpoint_hash != NATIVE_C0_V1_CHECKPOINT_SHA256:
        errors.append("preserved native-C0 v1 checkpoint SHA-256 mismatch")
    errors.extend(
        verify_v1_habitat_false_negative(
            prior,
            result_sha256=result_hash,
            checkpoint_sha256=checkpoint_hash,
        )
    )

    reconstruction_performed = False
    try:
        expected_probe = _reconstruct_expected_probe_v1(args)
        reconstruction_performed = True
    except Exception as exc:
        expected_probe = {}
        errors.append(
            f"independent native-C0 probe reconstruction failed: {type(exc).__name__}: {exc}"
        )
    errors.extend(
        verify_native_c0_result_v2(
            result,
            repo=repo,
            checkpoint=checkpoint_path,
            model_snapshot=args.model_snapshot,
            processor_result=args.processor_result,
            processor_verification=args.processor_verification,
            processor_reverification=args.processor_reverification,
            expected_probe=expected_probe,
        )
    )

    verifier_source_locks: dict[str, str] = {}
    for relative in NATIVE_C0_VERIFIER_V2_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            errors.append(f"native-C0 v2 verifier source is missing: {relative}")
        else:
            verifier_source_locks[relative] = file_sha256(path)

    payload = {
        "schema_version": NATIVE_C0_VERIFIER_V2_SCHEMA_VERSION,
        "gate": NATIVE_C0_VERIFIER_V2_GATE_NAME,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "error_count": len(errors),
        "errors": errors,
        "dataset_reconstruction_performed": reconstruction_performed,
        "result_path": str(result_path),
        "result_sha256": result_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "prior_verification_path": str(prior_path),
        "prior_verification_sha256": file_sha256(prior_path),
        "supersedes_verifier_gate": prior.get("gate"),
        "correction_scope": NATIVE_C0_VERIFIER_V2_CORRECTION,
        "verifier_source_locks": verifier_source_locks,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
