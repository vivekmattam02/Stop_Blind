"""Tests for independent real-batch evidence verification."""

from __future__ import annotations

from pathlib import Path

from spatialvlm.training.real_batch_artifacts import (
    REAL_BATCH_SOURCE_LOCK_PATHS,
    verify_real_batch_result,
)
from spatialvlm.training.real_batch_gate import file_sha256
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS


def _result(repo: Path, checkpoint: Path) -> dict:
    source = repo / "source.py"
    source.write_text("ok\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    gradients = {
        group: {
            "trainable_tensor_count": 1,
            "gradient_tensor_count": 1,
            "all_gradients_finite": True,
            "gradient_l2_norm": 1.0,
        }
        for group in FREEZE_GROUP_PATTERNS
    }
    updates = {
        group: {
            "all_updates_finite": True,
            "update_l2_norm": 0.1,
        }
        for group in FREEZE_GROUP_PATTERNS
    }
    scores = {
        name: {"total_nll": float(index + 1)}
        for index, name in enumerate(("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"))
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": "real_batch_v2",
        "source_locks": {"source.py": file_sha256(source)},
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
            "trainable_state_sha256": "after",
            "reloaded_state_sha256": "after",
            "perturbed_state_sha256": "perturbed",
            "reload_target_loss_absolute_delta": 0.0,
        },
        "optimization": {
            "trainable_state_sha256_before": "before",
            "trainable_state_sha256_after": "after",
            "gradients_by_group": gradients,
            "updates_by_group": updates,
        },
        "constrained_decode": {"scores": scores, "predicted_action": "STOP"},
    }


def test_independent_checker_accepts_complete_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    result = _result(tmp_path, checkpoint)
    assert verify_real_batch_result(result, repo=tmp_path, checkpoint=checkpoint) == []


def test_independent_checker_detects_source_and_prediction_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    result = _result(tmp_path, checkpoint)
    (tmp_path / "source.py").write_text("changed\n", encoding="utf-8")
    result["constrained_decode"]["predicted_action"] = "TURN_RIGHT"

    errors = verify_real_batch_result(result, repo=tmp_path, checkpoint=checkpoint)
    assert any("source hash mismatch" in error for error in errors)
    assert any("minimum-total-NLL" in error for error in errors)

