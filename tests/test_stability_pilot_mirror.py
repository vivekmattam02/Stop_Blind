"""Tests for compact cross-machine stability-pilot mirror verification."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_stability_pilot_mirror import (
    SUITE_GATE_NAME,
    VERIFIER_V1_GATE_NAME,
    VERIFIER_V2_GATE_NAME,
    file_sha256,
    verify_mirror,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    repo = root / "repo"
    evidence = root / "evidence"
    checkpoint = root / "checkpoints" / "lr_5em05" / "step_0008.pt"
    repo.mkdir()
    evidence.mkdir()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selected-checkpoint")

    source = repo / "source.py"
    source.write_text("source\n", encoding="utf-8")
    verifier_source = repo / "verifier_v2.py"
    verifier_source.write_text("verifier\n", encoding="utf-8")
    preflight = evidence / "action_preflight.json"
    _write_json(preflight, {"gate": "preflight_v1", "status": "passed"})
    manifest = {
        "gate": SUITE_GATE_NAME,
        "status": "passed",
        "preflights": {
            "action": {
                "relative_path": "action_preflight.json",
                "sha256": file_sha256(preflight),
                "gate": "preflight_v1",
            }
        },
        "source_locks": {"source.py": file_sha256(source)},
    }
    manifest_path = evidence / "manifest.json"
    _write_json(manifest_path, manifest)

    arm_id = "lr_5em05"
    arm_path = evidence / "arms" / arm_id / "result.json"
    _write_json(
        arm_path,
        {
            "status": "passed",
            "manifest_sha256": file_sha256(manifest_path),
        },
    )
    suite = {
        "gate": SUITE_GATE_NAME,
        "status": "passed",
        "protocol": {"arm_ids": [arm_id]},
        "arms": {arm_id: {"result_sha256": file_sha256(arm_path)}},
        "selected_checkpoint": {
            "arm_id": arm_id,
            "step": 8,
            "checkpoint": {
                "bytes": checkpoint.stat().st_size,
                "sha256": file_sha256(checkpoint),
            },
        },
    }
    suite_path = evidence / "suite.json"
    _write_json(suite_path, suite)
    verifier_v1_path = evidence / "verification.json"
    _write_json(verifier_v1_path, {"gate": VERIFIER_V1_GATE_NAME, "status": "failed"})
    _write_json(
        evidence / "verification_v2.json",
        {
            "gate": VERIFIER_V2_GATE_NAME,
            "status": "passed",
            "error_count": 0,
            "errors": [],
            "dataset_reconstruction_performed": True,
            "manifest_sha256": file_sha256(manifest_path),
            "suite_sha256": file_sha256(suite_path),
            "prior_verification_sha256": file_sha256(verifier_v1_path),
            "verifier_source_locks": {"verifier_v2.py": file_sha256(verifier_source)},
        },
    )
    return repo, evidence, checkpoint


def test_compact_mirror_passes_and_detects_selected_checkpoint_tampering(tmp_path: Path) -> None:
    repo, evidence, checkpoint = _fixture(tmp_path)
    errors, summary = verify_mirror(
        repo=repo,
        evidence_root=evidence,
        selected_checkpoint=checkpoint,
    )
    assert errors == []
    assert summary["verified_manifest_source_lock_count"] == 1
    assert summary["verified_verifier_v2_source_lock_count"] == 1

    checkpoint.write_bytes(b"tampered")
    errors, _ = verify_mirror(
        repo=repo,
        evidence_root=evidence,
        selected_checkpoint=checkpoint,
    )
    assert "selected checkpoint byte-size mismatch" in errors
    assert "selected checkpoint SHA-256 mismatch" in errors
