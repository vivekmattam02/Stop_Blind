#!/usr/bin/env python3
"""Independently verify one paired-seed native-C0 replication result."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import check_native_c0_action_balance_artifacts as weighted_verifier
from scripts import check_native_c0_development_artifacts as unweighted_verifier
from scripts.run_spark_native_c0_seed_replication import expected_manifest, source_locks
from spatialvlm.training.native_c0_action_balance_full import (
    NativeC0ActionBalanceProtocol,
)
from spatialvlm.training.native_c0_development import NativeC0DevelopmentProtocol
from spatialvlm.training.native_c0_gate import file_sha256
from spatialvlm.training.native_c0_seed_replication import (
    NATIVE_C0_SEED_REPLICATION_GATE,
    NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
    NATIVE_C0_SEED_REPLICATION_VERIFIER_GATE,
    validate_replication_contract,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _protocol(treatment: str, seed: int) -> NativeC0DevelopmentProtocol:
    cls = (
        NativeC0DevelopmentProtocol if treatment == "unweighted" else NativeC0ActionBalanceProtocol
    )
    return cls(sampler_seed=seed, model_seed=seed)


def _validate_protocol(expected: NativeC0DevelopmentProtocol):
    def validate(record: dict[str, Any]) -> list[str]:
        return [] if record == expected.to_record() else ["replication protocol differs"]

    return validate


def verify_replication(args: argparse.Namespace) -> list[str]:
    """Run the original treatment verifier plus replication-specific checks."""
    root = args.result_root.resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return ["replication manifest or result is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected, _ = expected_manifest(args)
    protocol = _protocol(args.treatment, args.replication_seed)
    errors: list[str] = []
    if manifest != expected:
        errors.append("replication manifest does not independently reconstruct")
    errors.extend(validate_replication_contract(manifest.get("replication_contract", {})))
    if (
        manifest.get("replication_gate") != NATIVE_C0_SEED_REPLICATION_GATE
        or manifest.get("schema_version") != NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION
        or result.get("replication_gate") != NATIVE_C0_SEED_REPLICATION_GATE
        or result.get("replication_schema_version") != NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION
        or result.get("replication_contract") != manifest.get("replication_contract")
        or result.get("original_evidence") != manifest.get("original_evidence")
        or result.get("replication_stress_authorization")
        != manifest.get("replication_stress_authorization")
    ):
        errors.append("replication-specific result provenance mismatch")

    if args.treatment == "unweighted":
        replacements = {
            "NativeC0DevelopmentProtocol": lambda: protocol,
            "validate_protocol_record": _validate_protocol(protocol),
            "_expected_manifest": expected_manifest,
            "_source_locks": lambda repo: source_locks(Path(repo)),
        }
        originals = {name: getattr(unweighted_verifier, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(unweighted_verifier, name, value)
            errors.extend(unweighted_verifier.verify_development(args))
        finally:
            for name, value in originals.items():
                setattr(unweighted_verifier, name, value)
    else:
        replacements = {
            "NativeC0ActionBalanceProtocol": lambda: protocol,
            "expected_manifest": expected_manifest,
            "source_locks": lambda repo: source_locks(Path(repo)),
        }
        originals = {name: getattr(weighted_verifier, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(weighted_verifier, name, value)
            errors.extend(weighted_verifier.verify_action_balance(args))
        finally:
            for name, value in originals.items():
                setattr(weighted_verifier, name, value)
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--treatment", choices=("unweighted", "inverse_frequency"), required=True)
    parser.add_argument("--replication-seed", type=int, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--unweighted-root", type=Path, required=True)
    parser.add_argument("--inverse-frequency-root", type=Path, required=True)
    parser.add_argument("--replication-stress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--failure-analysis", type=Path)
    parser.add_argument("--failure-verification", type=Path)
    parser.add_argument("--stress-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify_replication(args)
    root = args.result_root.resolve()
    payload = {
        "schema_version": NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
        "gate": NATIVE_C0_SEED_REPLICATION_VERIFIER_GATE,
        "recorded_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "treatment": args.treatment,
        "replication_seed": args.replication_seed,
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
        "all_checkpoint_files_fully_loaded_and_hashed": True,
        "event_chain_reconstruction_performed": True,
        "outcome_reconstruction_performed": True,
        "replication_contract_reconstruction_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
