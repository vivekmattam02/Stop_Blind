"""Tests for native-Qwen C0 compact cross-machine mirror verification."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.check_native_c0_mirror as mirror


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(root: Path, monkeypatch) -> tuple[Path, Path, Path]:
    repo = root / "repo"
    evidence = root / "evidence"
    checkpoint = root / "checkpoint.pt"
    repo.mkdir()
    evidence.mkdir()
    checkpoint.write_bytes(b"native-c0-checkpoint")

    original_source = repo / "original.py"
    original_source.write_text("original\n", encoding="utf-8")
    verifier_source = repo / "verifier.py"
    verifier_source.write_text("verifier\n", encoding="utf-8")
    processor_path = evidence / "processor_reverification.json"
    _write_json(processor_path, {"status": "passed", "error_count": 0})

    result = {
        "gate": mirror.RESULT_GATE_NAME,
        "status": "passed",
        "source_locks": {"original.py": mirror.file_sha256(original_source)},
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "sha256": mirror.file_sha256(checkpoint),
        },
        "parent_processor_gate": {"reverification_sha256": mirror.file_sha256(processor_path)},
    }
    result_path = evidence / "result.json"
    _write_json(result_path, result)
    verifier_v1 = {
        "gate": mirror.VERIFIER_V1_GATE_NAME,
        "status": "failed",
        "error_count": 1,
        "errors": mirror.EXPECTED_VERIFIER_V1_ERRORS,
        "processor_reverification_sha256": mirror.file_sha256(processor_path),
    }
    verifier_v1_path = evidence / "verification.json"
    _write_json(verifier_v1_path, verifier_v1)
    verifier_v2 = {
        "gate": mirror.VERIFIER_V2_GATE_NAME,
        "schema_version": 2,
        "status": "passed",
        "error_count": 0,
        "errors": [],
        "dataset_reconstruction_performed": True,
        "result_sha256": mirror.file_sha256(result_path),
        "checkpoint_sha256": mirror.file_sha256(checkpoint),
        "prior_verification_sha256": mirror.file_sha256(verifier_v1_path),
        "verifier_source_locks": {"verifier.py": mirror.file_sha256(verifier_source)},
    }
    _write_json(evidence / "verification_v2.json", verifier_v2)

    monkeypatch.setattr(mirror, "EXPECTED_RESULT_SHA256", mirror.file_sha256(result_path))
    monkeypatch.setattr(mirror, "EXPECTED_CHECKPOINT_SHA256", mirror.file_sha256(checkpoint))
    monkeypatch.setattr(
        mirror,
        "EXPECTED_VERIFIER_V1_SHA256",
        mirror.file_sha256(verifier_v1_path),
    )
    return repo, evidence, checkpoint


def test_native_c0_mirror_passes_and_detects_checkpoint_tamper(tmp_path, monkeypatch) -> None:
    repo, evidence, checkpoint = _fixture(tmp_path, monkeypatch)
    errors, summary = mirror.verify_mirror(
        repo=repo,
        evidence_root=evidence,
        checkpoint=checkpoint,
    )
    assert errors == []
    assert summary["verified_result_source_lock_count"] == 1
    assert summary["verified_verifier_v2_source_lock_count"] == 1

    checkpoint.write_bytes(b"tampered")
    errors, _ = mirror.verify_mirror(
        repo=repo,
        evidence_root=evidence,
        checkpoint=checkpoint,
    )
    assert "native-C0 mirrored checkpoint SHA-256 mismatch" in errors
    assert "native-C0 mirrored checkpoint byte-size mismatch" in errors


def test_native_c0_mirror_detects_verifier_and_source_tamper(tmp_path, monkeypatch) -> None:
    repo, evidence, checkpoint = _fixture(tmp_path, monkeypatch)
    (repo / "verifier.py").write_text("tampered\n", encoding="utf-8")
    verifier_v2_path = evidence / "verification_v2.json"
    verifier_v2 = json.loads(verifier_v2_path.read_text(encoding="utf-8"))
    verifier_v2["error_count"] = 1
    _write_json(verifier_v2_path, verifier_v2)

    errors, _ = mirror.verify_mirror(
        repo=repo,
        evidence_root=evidence,
        checkpoint=checkpoint,
    )
    assert "verifier-v2 pass record is invalid" in errors
    assert "verifier v2 source-lock mismatch: verifier.py" in errors
