"""Regression tests for the native-C0 Habitat JSON-type verifier correction."""

from __future__ import annotations

from pathlib import Path

import spatialvlm.training.native_c0_artifacts_v2 as verifier_v2
from spatialvlm.training.native_c0_artifacts import NATIVE_C0_VERIFIER_GATE_NAME
from spatialvlm.training.native_c0_artifacts_v2 import (
    EXPECTED_V1_HABITAT_FALSE_NEGATIVE_ERRORS,
    json_normalize,
    verify_native_c0_result_v2,
    verify_v1_habitat_false_negative,
)


def test_json_normalization_changes_only_serialization_types() -> None:
    original = {
        "config": {
            "rgb_height_width": (224, 224),
            "depth_height_width": (256, 256),
            "seed": 100,
        }
    }
    assert json_normalize(original) == {
        "config": {
            "rgb_height_width": [224, 224],
            "depth_height_width": [256, 256],
            "seed": 100,
        }
    }
    assert isinstance(original["config"]["rgb_height_width"], tuple)


def test_v1_failure_must_be_exactly_the_known_habitat_mismatch() -> None:
    result_hash = "a" * 64
    checkpoint_hash = "b" * 64
    prior = {
        "schema_version": 1,
        "gate": NATIVE_C0_VERIFIER_GATE_NAME,
        "status": "failed",
        "result_sha256": result_hash,
        "checkpoint_sha256": checkpoint_hash,
        "dataset_reconstruction_performed": True,
        "error_count": 1,
        "errors": list(EXPECTED_V1_HABITAT_FALSE_NEGATIVE_ERRORS),
    }
    assert (
        verify_v1_habitat_false_negative(
            prior,
            result_sha256=result_hash,
            checkpoint_sha256=checkpoint_hash,
        )
        == []
    )
    prior["errors"] = ["unrelated failure"]
    assert verify_v1_habitat_false_negative(
        prior,
        result_sha256=result_hash,
        checkpoint_sha256=checkpoint_hash,
    )


def test_v2_normalizes_only_habitat_before_delegating(monkeypatch) -> None:
    captured = {}

    def fake_v1(result, **kwargs):
        captured.update(kwargs["expected_probe"])
        return []

    monkeypatch.setattr(verifier_v2, "verify_native_c0_result", fake_v1)
    result = {"runtime": {"docker_image": {"id": verifier_v2.NATIVE_C0_V1_DOCKER_IMAGE_ID}}}
    expected = {
        "habitat_runtime": {"config": {"rgb_height_width": (224, 224)}},
        "native_batch": {"shape": [1, 8]},
    }
    errors = verify_native_c0_result_v2(
        result,
        repo=".",
        checkpoint="checkpoint.pt",
        model_snapshot="snapshot",
        processor_result="processor_result.json",
        processor_verification="processor_verification.json",
        processor_reverification="processor_reverification.json",
        expected_probe=expected,
    )
    assert errors == []
    assert captured["habitat_runtime"]["config"]["rgb_height_width"] == [224, 224]
    assert captured["native_batch"] == expected["native_batch"]


def test_v2_requires_the_exact_v1_image_identity(monkeypatch) -> None:
    monkeypatch.setattr(verifier_v2, "verify_native_c0_result", lambda *args, **kwargs: [])
    errors = verify_native_c0_result_v2(
        {"runtime": {"docker_image": {"id": "sha256:" + "0" * 64}}},
        repo=".",
        checkpoint="checkpoint.pt",
        model_snapshot="snapshot",
        processor_result="processor_result.json",
        processor_verification="processor_verification.json",
        processor_reverification="processor_reverification.json",
        expected_probe={},
    )
    assert errors == ["native-C0 v1 Docker image identity mismatch"]

