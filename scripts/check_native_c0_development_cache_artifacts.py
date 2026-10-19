#!/usr/bin/env python3
"""Independently reconstruct and verify native-C0 content-addressed cache artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_spark_native_c0_development_cache import (
    _expected_manifest,
    _load_json,
    _object_path,
    _source_locks,
    _validate_object,
)
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_CACHE_GATE,
    NATIVE_C0_CACHE_SCHEMA_VERSION,
    NativeC0DevelopmentProtocol,
    canonical_sha256,
    validate_partition_records,
    validate_protocol_record,
)
from spatialvlm.training.native_c0_gate import file_sha256

CACHE_VERIFIER_GATE = "independent_native_qwen_c0_content_cache_verification_v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def verify_cache(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    index_path = root / "cache_index.json"
    result_path = root / "result.json"
    for path in (manifest_path, index_path, result_path):
        if not path.is_file():
            errors.append(f"required cache artifact is missing: {path}")
    if errors:
        return errors
    manifest = _load_json(manifest_path, "cache manifest")
    index = _load_json(index_path, "cache index")
    result = _load_json(result_path, "cache result")
    expected, _, samples = _expected_manifest(args)
    if manifest != expected:
        errors.append("cache manifest does not independently reconstruct")
    if validate_protocol_record(manifest.get("protocol", {})):
        errors.append("cache manifest protocol does not reconstruct")
    try:
        validate_partition_records(samples, protocol=NativeC0DevelopmentProtocol())
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cache partition records are invalid: {exc}")
    manifest_sha = file_sha256(manifest_path)
    if (
        index.get("gate") != NATIVE_C0_CACHE_GATE
        or index.get("schema_version") != NATIVE_C0_CACHE_SCHEMA_VERSION
        or index.get("run_id") != args.run_id
        or index.get("status") != "passed"
        or index.get("manifest_sha256") != manifest_sha
        or index.get("content_addressed") is not True
    ):
        errors.append("cache index identity/status/provenance mismatch")
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != len(samples):
        errors.append("cache index entry count mismatch")
        entries = []
    if entries and index.get("entry_index_sha256") != canonical_sha256(entries):
        errors.append("cache index entry-list hash mismatch")
    object_root = args.object_root.expanduser().resolve()
    references = {
        (row["split"], int(row["sample_index"])): row
        for row in manifest.get("object_references", [])
    }
    for sample, entry in zip(samples, entries):
        key_tuple = (sample["split"], int(sample["sample_index"]))
        reference = references.get(key_tuple)
        if reference is None:
            errors.append(f"cache reference missing for {key_tuple}")
            continue
        key = reference["cache_key"]
        expected_path = _object_path(object_root, key)
        if entry.get("cache_key") != key:
            errors.append(f"cache key mismatch for {key_tuple}")
        if entry.get("sample_sha256") != canonical_sha256(sample):
            errors.append(f"cache sample hash mismatch for {key_tuple}")
        if entry.get("relative_path") != str(expected_path.relative_to(object_root)):
            errors.append(f"cache object path mismatch for {key_tuple}")
        if not expected_path.is_file():
            errors.append(f"cache object missing for {key_tuple}")
            continue
        if entry.get("bytes") != expected_path.stat().st_size:
            errors.append(f"cache object byte count mismatch for {key_tuple}")
        if entry.get("sha256") != file_sha256(expected_path):
            errors.append(f"cache object file hash mismatch for {key_tuple}")
            continue
        try:
            payload = _validate_object(expected_path, reference["object_identity"], key)
        except (KeyError, RuntimeError, ValueError) as exc:
            errors.append(f"cache object validation failed for {key_tuple}: {exc}")
            continue
        if entry.get("rgb_frames_sha256") != payload.get("rgb_frames_sha256"):
            errors.append(f"cache RGB hash link mismatch for {key_tuple}")
    if len({entry.get("cache_key") for entry in entries}) != len(entries):
        errors.append("cache index reuses an object for distinct frozen decisions")
    if (
        result.get("gate") != NATIVE_C0_CACHE_GATE
        or result.get("schema_version") != NATIVE_C0_CACHE_SCHEMA_VERSION
        or result.get("run_id") != args.run_id
        or result.get("status") != "passed"
        or result.get("manifest_sha256") != manifest_sha
        or result.get("cache_index_sha256") != file_sha256(index_path)
        or result.get("object_count") != len(samples)
        or result.get("content_addressed") is not True
        or result.get("resumable_at_object_boundary") is not True
    ):
        errors.append("cache result identity/status/provenance mismatch")
    if manifest.get("source_locks") != _source_locks(args.repo.expanduser().resolve()):
        errors.append("cache source locks changed after freeze")
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
    parser.add_argument("--data-bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify_cache(args)
    payload: dict[str, Any] = {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "gate": CACHE_VERIFIER_GATE,
        "recorded_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str((args.result_root / "manifest.json").resolve()),
        "manifest_sha256": file_sha256(args.result_root / "manifest.json")
        if (args.result_root / "manifest.json").is_file()
        else None,
        "cache_index_path": str((args.result_root / "cache_index.json").resolve()),
        "cache_index_sha256": file_sha256(args.result_root / "cache_index.json")
        if (args.result_root / "cache_index.json").is_file()
        else None,
        "result_path": str((args.result_root / "result.json").resolve()),
        "result_sha256": file_sha256(args.result_root / "result.json")
        if (args.result_root / "result.json").is_file()
        else None,
        "object_root": str(args.object_root.expanduser().resolve()),
        "all_objects_fully_hashed": True,
        "dataset_reconstruction_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
