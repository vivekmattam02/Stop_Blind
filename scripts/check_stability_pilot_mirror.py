#!/usr/bin/env python3
"""Verify a compact Mac mirror of Spark stability-pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIRROR_SCHEMA_VERSION = 1
MIRROR_GATE_NAME = "stability_pilot_compact_mac_mirror_verification_v1"
SUITE_GATE_NAME = "spark_r2r_ce_optimization_stability_pilot_v1"
VERIFIER_V1_GATE_NAME = "independent_stability_pilot_artifact_verification_v1"
VERIFIER_V2_GATE_NAME = "independent_stability_pilot_artifact_verification_v2"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{label} is missing")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"{label} is invalid JSON")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} is not a JSON object")
        return {}
    return value


def verify_mirror(
    *,
    repo: Path,
    evidence_root: Path,
    selected_checkpoint: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Return mirror-integrity errors and the derived evidence summary."""
    errors: list[str] = []
    manifest_path = evidence_root / "manifest.json"
    suite_path = evidence_root / "suite.json"
    verifier_v1_path = evidence_root / "verification.json"
    verifier_v2_path = evidence_root / "verification_v2.json"
    manifest = _load_json(manifest_path, errors, "manifest")
    suite = _load_json(suite_path, errors, "suite")
    verifier_v1 = _load_json(verifier_v1_path, errors, "verifier v1")
    verifier_v2 = _load_json(verifier_v2_path, errors, "verifier v2")

    top_level_hashes = {}
    for label, path in (
        ("manifest", manifest_path),
        ("suite", suite_path),
        ("verification_v1", verifier_v1_path),
        ("verification_v2", verifier_v2_path),
    ):
        if path.is_file():
            top_level_hashes[label] = file_sha256(path)

    if manifest.get("gate") != SUITE_GATE_NAME or manifest.get("status") != "passed":
        errors.append("manifest gate/status mismatch")
    if suite.get("gate") != SUITE_GATE_NAME or suite.get("status") != "passed":
        errors.append("suite gate/status mismatch")
    if verifier_v1.get("gate") != VERIFIER_V1_GATE_NAME or verifier_v1.get("status") != "failed":
        errors.append("preserved verifier-v1 gate/status mismatch")
    if (
        verifier_v2.get("gate") != VERIFIER_V2_GATE_NAME
        or verifier_v2.get("status") != "passed"
        or verifier_v2.get("error_count") != 0
        or verifier_v2.get("errors") != []
        or verifier_v2.get("dataset_reconstruction_performed") is not True
    ):
        errors.append("verifier-v2 pass record is invalid")
    if top_level_hashes.get("manifest") != verifier_v2.get("manifest_sha256"):
        errors.append("mirrored manifest SHA-256 mismatch")
    if top_level_hashes.get("suite") != verifier_v2.get("suite_sha256"):
        errors.append("mirrored suite SHA-256 mismatch")
    if top_level_hashes.get("verification_v1") != verifier_v2.get("prior_verification_sha256"):
        errors.append("mirrored verifier-v1 SHA-256 mismatch")

    preflight_hashes = {}
    preflights = manifest.get("preflights")
    if not isinstance(preflights, dict):
        errors.append("manifest preflight index is missing")
    else:
        for label, row in preflights.items():
            relative = row.get("relative_path") if isinstance(row, dict) else None
            if not isinstance(relative, str):
                errors.append(f"{label} preflight path is invalid")
                continue
            path = (evidence_root / relative).resolve()
            if not path.is_relative_to(evidence_root.resolve()) or not path.is_file():
                errors.append(f"{label} preflight file is missing or unsafe")
                continue
            observed = file_sha256(path)
            preflight_hashes[label] = observed
            if observed != row.get("sha256"):
                errors.append(f"{label} preflight SHA-256 mismatch")
            record = _load_json(path, errors, f"{label} preflight")
            if record.get("status") != "passed" or record.get("gate") != row.get("gate"):
                errors.append(f"{label} preflight gate/status mismatch")

    source_locks = manifest.get("source_locks")
    verified_source_count = 0
    if not isinstance(source_locks, dict) or not source_locks:
        errors.append("manifest source locks are missing")
    else:
        for relative, expected in source_locks.items():
            path = repo / relative
            if not path.is_file() or file_sha256(path) != expected:
                errors.append(f"source-lock mismatch: {relative}")
            else:
                verified_source_count += 1

    verifier_source_locks = verifier_v2.get("verifier_source_locks")
    verified_verifier_source_count = 0
    if not isinstance(verifier_source_locks, dict) or not verifier_source_locks:
        errors.append("verifier-v2 source locks are missing")
    else:
        for relative, expected in verifier_source_locks.items():
            path = repo / relative
            if not path.is_file() or file_sha256(path) != expected:
                errors.append(f"verifier-v2 source-lock mismatch: {relative}")
            else:
                verified_verifier_source_count += 1

    protocol = suite.get("protocol")
    expected_arm_ids = protocol.get("arm_ids") if isinstance(protocol, dict) else None
    arm_index = suite.get("arms")
    arm_result_hashes = {}
    if (
        not isinstance(expected_arm_ids, list)
        or not isinstance(arm_index, dict)
        or set(arm_index) != set(expected_arm_ids)
    ):
        errors.append("suite arm index mismatch")
    else:
        for arm_id in expected_arm_ids:
            row = arm_index[arm_id]
            path = evidence_root / "arms" / arm_id / "result.json"
            if not path.is_file():
                errors.append(f"mirrored arm result is missing: {arm_id}")
                continue
            observed = file_sha256(path)
            arm_result_hashes[arm_id] = observed
            if observed != row.get("result_sha256"):
                errors.append(f"mirrored arm result SHA-256 mismatch: {arm_id}")
            arm = _load_json(path, errors, f"{arm_id} result")
            if arm.get("status") != "passed" or arm.get("manifest_sha256") != top_level_hashes.get(
                "manifest"
            ):
                errors.append(f"{arm_id} result gate linkage mismatch")

    selected = suite.get("selected_checkpoint")
    checkpoint_record = selected.get("checkpoint") if isinstance(selected, dict) else None
    checkpoint_hash = None
    checkpoint_bytes = None
    if not isinstance(checkpoint_record, dict):
        errors.append("suite selected-checkpoint record is missing")
    elif not selected_checkpoint.is_file():
        errors.append("selected checkpoint was not mirrored")
    else:
        checkpoint_bytes = selected_checkpoint.stat().st_size
        checkpoint_hash = file_sha256(selected_checkpoint)
        if checkpoint_bytes != checkpoint_record.get("bytes"):
            errors.append("selected checkpoint byte-size mismatch")
        if checkpoint_hash != checkpoint_record.get("sha256"):
            errors.append("selected checkpoint SHA-256 mismatch")

    summary = {
        "top_level_hashes": top_level_hashes,
        "preflight_hashes": preflight_hashes,
        "arm_result_hashes": arm_result_hashes,
        "verified_manifest_source_lock_count": verified_source_count,
        "verified_verifier_v2_source_lock_count": verified_verifier_source_count,
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_hash,
            "arm_id": selected.get("arm_id") if isinstance(selected, dict) else None,
            "step": selected.get("step") if isinstance(selected, dict) else None,
        },
    }
    return errors, summary


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    selected_checkpoint = args.selected_checkpoint.expanduser().resolve()
    errors, summary = verify_mirror(
        repo=repo,
        evidence_root=evidence_root,
        selected_checkpoint=selected_checkpoint,
    )
    payload = {
        "schema_version": MIRROR_SCHEMA_VERSION,
        "gate": MIRROR_GATE_NAME,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "repo": str(repo),
        "evidence_root": str(evidence_root),
        **summary,
        "evidence_boundary": {
            "established": (
                "The compact evidence, current locked sources, and selected checkpoint copied "
                "from Spark match the identities independently verified on Spark."
            ),
            "not_established": (
                "Local byte verification of the five intentionally unmirrored checkpoints, "
                "navigation performance, rollout improvement, generalization, or ablation effects."
            ),
        },
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
