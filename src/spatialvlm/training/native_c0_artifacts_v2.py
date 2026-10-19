"""JSON-type correction for native-Qwen C0 artifact verification.

Verifier v1 compared the JSON-decoded runtime record against an in-memory Habitat metadata
mapping. Two sensor-size fields are tuples in memory and lists after JSON serialization, so the
otherwise identical mappings compared unequal. This module normalizes only the independently
reconstructed Habitat record through JSON before delegating every v1 scientific and integrity
check.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from spatialvlm.training.native_c0_artifacts import (
    NATIVE_C0_VERIFIER_GATE_NAME,
    NATIVE_C0_VERIFIER_SCHEMA_VERSION,
    verify_native_c0_result,
)

NATIVE_C0_VERIFIER_V2_SCHEMA_VERSION = 2
NATIVE_C0_VERIFIER_V2_GATE_NAME = "independent_native_qwen_c0_real_batch_verification_v2"
NATIVE_C0_VERIFIER_V2_CORRECTION = (
    "Normalize the independently reconstructed Habitat runtime mapping through JSON so tuple "
    "sensor sizes compare with their serialized list representation; preserve every v1 data, "
    "processor, source, model-weight, checkpoint, optimization, and action-decision check."
)
NATIVE_C0_VERIFIER_V2_SOURCE_PATHS = (
    "scripts/__init__.py",
    "scripts/check_native_c0_artifacts_v2.py",
    "scripts/native_c0_v2_verifier_sync_files.txt",
    "scripts/reverify_spark_native_c0_v2.sh",
    "src/spatialvlm/training/native_c0_artifacts_v2.py",
    "tests/test_native_c0_artifacts_v2.py",
)
NATIVE_C0_V1_RESULT_SHA256 = "f8ed157a4b9def07e31c33a3a9378272b30ef535adc5a829661a49c1c2da9005"
NATIVE_C0_V1_CHECKPOINT_SHA256 = "d564ad6794486602625a72930cf185b715c9ac6199d28fabffbd18baa5d8093a"
NATIVE_C0_V1_DOCKER_IMAGE_ID = (
    "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
)
EXPECTED_V1_HABITAT_FALSE_NEGATIVE_ERRORS = ("native-C0 Habitat runtime reconstruction mismatch",)


def json_normalize(value: Any) -> Any:
    """Return the exact standard-JSON representation of a JSON-compatible value."""
    return json.loads(json.dumps(value, sort_keys=True))


def verify_v1_habitat_false_negative(
    prior: dict[str, Any],
    *,
    result_sha256: str,
    checkpoint_sha256: str,
) -> list[str]:
    """Require the preserved v1 decision to be exactly the known tuple/list false-negative."""
    errors: list[str] = []
    if prior.get("schema_version") != NATIVE_C0_VERIFIER_SCHEMA_VERSION:
        errors.append("prior native-C0 verification schema is not v1")
    if prior.get("gate") != NATIVE_C0_VERIFIER_GATE_NAME:
        errors.append("prior native-C0 verification gate is not v1")
    if prior.get("status") != "failed":
        errors.append("prior native-C0 verification status is not failed")
    if prior.get("result_sha256") != result_sha256:
        errors.append("prior native-C0 verification result linkage mismatch")
    if prior.get("checkpoint_sha256") != checkpoint_sha256:
        errors.append("prior native-C0 verification checkpoint linkage mismatch")
    if prior.get("dataset_reconstruction_performed") is not True:
        errors.append("prior native-C0 verification did not reconstruct the dataset")
    if prior.get("errors") != list(EXPECTED_V1_HABITAT_FALSE_NEGATIVE_ERRORS):
        errors.append("prior native-C0 errors are not exactly the known Habitat JSON-type issue")
    if prior.get("error_count") != len(EXPECTED_V1_HABITAT_FALSE_NEGATIVE_ERRORS):
        errors.append("prior native-C0 error count does not match the known v1 issue")
    return errors


def verify_native_c0_result_v2(
    result: dict[str, Any],
    *,
    repo: str | Path,
    checkpoint: str | Path,
    model_snapshot: str | Path,
    processor_result: str | Path,
    processor_verification: str | Path,
    processor_reverification: str | Path,
    expected_probe: dict[str, Any],
) -> list[str]:
    """Delegate all v1 checks after normalizing only reconstructed Habitat JSON types."""
    normalized_probe = deepcopy(expected_probe)
    if "habitat_runtime" in normalized_probe:
        normalized_probe["habitat_runtime"] = json_normalize(normalized_probe["habitat_runtime"])
    errors = verify_native_c0_result(
        result,
        repo=repo,
        checkpoint=checkpoint,
        model_snapshot=model_snapshot,
        processor_result=processor_result,
        processor_verification=processor_verification,
        processor_reverification=processor_reverification,
        expected_probe=normalized_probe,
    )
    if result.get("runtime", {}).get("docker_image", {}).get("id") != NATIVE_C0_V1_DOCKER_IMAGE_ID:
        errors.append("native-C0 v1 Docker image identity mismatch")
    return errors
