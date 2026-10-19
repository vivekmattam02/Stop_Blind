#!/usr/bin/env python3
"""Verify the compact Mac mirror of the admitted Spark native-Qwen C0 mechanics gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIRROR_SCHEMA_VERSION = 1
MIRROR_GATE_NAME = "native_qwen_c0_compact_mac_mirror_verification_v1"
RESULT_GATE_NAME = "spark_native_qwen_c0_one_real_h0_update_v1"
VERIFIER_V1_GATE_NAME = "independent_native_qwen_c0_real_batch_verification_v1"
VERIFIER_V2_GATE_NAME = "independent_native_qwen_c0_real_batch_verification_v2"
EXPECTED_RESULT_SHA256 = "f8ed157a4b9def07e31c33a3a9378272b30ef535adc5a829661a49c1c2da9005"
EXPECTED_CHECKPOINT_SHA256 = "d564ad6794486602625a72930cf185b715c9ac6199d28fabffbd18baa5d8093a"
EXPECTED_VERIFIER_V1_SHA256 = "03e2634abdbf431c6cb1ae47f750957ebd88b53985dd296e3c02ba3e8b71e122"
EXPECTED_VERIFIER_V1_ERRORS = ["native-C0 Habitat runtime reconstruction mismatch"]


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


def _verify_source_locks(
    repo: Path,
    locks: Any,
    *,
    label: str,
    errors: list[str],
) -> int:
    if not isinstance(locks, dict) or not locks:
        errors.append(f"{label} source locks are missing")
        return 0
    verified = 0
    for relative, expected in locks.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append(f"{label} source-lock entry is invalid")
            continue
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            errors.append(f"{label} source is missing or unsafe: {relative}")
        elif file_sha256(path) != expected:
            errors.append(f"{label} source-lock mismatch: {relative}")
        else:
            verified += 1
    return verified


def verify_mirror(
    *,
    repo: Path,
    evidence_root: Path,
    checkpoint: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Return transfer-integrity errors and a derived evidence summary."""
    repo = repo.resolve()
    evidence_root = evidence_root.resolve()
    checkpoint = checkpoint.resolve()
    errors: list[str] = []
    result_path = evidence_root / "result.json"
    verifier_v1_path = evidence_root / "verification.json"
    verifier_v2_path = evidence_root / "verification_v2.json"
    processor_reverification_path = evidence_root / "processor_reverification.json"

    result = _load_json(result_path, errors, "result")
    verifier_v1 = _load_json(verifier_v1_path, errors, "verifier v1")
    verifier_v2 = _load_json(verifier_v2_path, errors, "verifier v2")
    processor_reverification = _load_json(
        processor_reverification_path,
        errors,
        "processor reverification",
    )

    file_hashes: dict[str, str] = {}
    for label, path in (
        ("result", result_path),
        ("verification_v1", verifier_v1_path),
        ("verification_v2", verifier_v2_path),
        ("processor_reverification", processor_reverification_path),
    ):
        if path.is_file():
            file_hashes[label] = file_sha256(path)

    if result.get("gate") != RESULT_GATE_NAME or result.get("status") != "passed":
        errors.append("native-C0 result gate/status mismatch")
    if file_hashes.get("result") != EXPECTED_RESULT_SHA256:
        errors.append("native-C0 mirrored result SHA-256 mismatch")
    if (
        verifier_v1.get("gate") != VERIFIER_V1_GATE_NAME
        or verifier_v1.get("status") != "failed"
        or verifier_v1.get("error_count") != 1
        or verifier_v1.get("errors") != EXPECTED_VERIFIER_V1_ERRORS
    ):
        errors.append("preserved verifier-v1 record is not the exact known false-negative")
    if file_hashes.get("verification_v1") != EXPECTED_VERIFIER_V1_SHA256:
        errors.append("native-C0 mirrored verifier-v1 SHA-256 mismatch")
    if (
        verifier_v2.get("gate") != VERIFIER_V2_GATE_NAME
        or verifier_v2.get("schema_version") != 2
        or verifier_v2.get("status") != "passed"
        or verifier_v2.get("error_count") != 0
        or verifier_v2.get("errors") != []
        or verifier_v2.get("dataset_reconstruction_performed") is not True
    ):
        errors.append("verifier-v2 pass record is invalid")
    if verifier_v2.get("result_sha256") != file_hashes.get("result"):
        errors.append("verifier-v2/result linkage mismatch")
    if verifier_v2.get("prior_verification_sha256") != file_hashes.get("verification_v1"):
        errors.append("verifier-v2/verifier-v1 linkage mismatch")

    checkpoint_hash = file_sha256(checkpoint) if checkpoint.is_file() else None
    checkpoint_bytes = checkpoint.stat().st_size if checkpoint.is_file() else None
    checkpoint_record = result.get("checkpoint")
    if not checkpoint.is_file():
        errors.append("native-C0 checkpoint mirror is missing")
    elif not isinstance(checkpoint_record, dict):
        errors.append("native-C0 result checkpoint record is missing")
    else:
        if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            errors.append("native-C0 mirrored checkpoint SHA-256 mismatch")
        if checkpoint_hash != checkpoint_record.get("sha256"):
            errors.append("native-C0 result/checkpoint hash linkage mismatch")
        if checkpoint_hash != verifier_v2.get("checkpoint_sha256"):
            errors.append("native-C0 verifier-v2/checkpoint hash linkage mismatch")
        if checkpoint_bytes != checkpoint_record.get("bytes"):
            errors.append("native-C0 mirrored checkpoint byte-size mismatch")

    parent = result.get("parent_processor_gate")
    if not isinstance(parent, dict):
        errors.append("native-C0 parent processor record is missing")
    else:
        observed_parent_hash = file_hashes.get("processor_reverification")
        if observed_parent_hash != parent.get("reverification_sha256"):
            errors.append("processor reverification/result linkage mismatch")
        if observed_parent_hash != verifier_v1.get("processor_reverification_sha256"):
            errors.append("processor reverification/verifier-v1 linkage mismatch")
        if (
            processor_reverification.get("status") != "passed"
            or processor_reverification.get("error_count") != 0
        ):
            errors.append("processor reverification pass record is invalid")

    result_source_count = _verify_source_locks(
        repo,
        result.get("source_locks"),
        label="result",
        errors=errors,
    )
    verifier_source_count = _verify_source_locks(
        repo,
        verifier_v2.get("verifier_source_locks"),
        label="verifier v2",
        errors=errors,
    )

    summary = {
        "file_hashes": file_hashes,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_hash,
        },
        "verified_result_source_lock_count": result_source_count,
        "verified_verifier_v2_source_lock_count": verifier_source_count,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    errors, summary = verify_mirror(
        repo=repo,
        evidence_root=evidence_root,
        checkpoint=checkpoint,
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
                "The compact native-C0 evidence, current locked sources, and checkpoint copied "
                "from Spark match the identities independently verified on Spark."
            ),
            "not_established": (
                "Local reconstruction of licensed simulator data or Qwen weight shards, history "
                "selection, navigation competence, convergence, generalization, or C0-C4 effects."
            ),
        },
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
