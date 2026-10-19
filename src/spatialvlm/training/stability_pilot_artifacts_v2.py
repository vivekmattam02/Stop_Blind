"""Order-independent correction for stability-pilot artifact verification.

The v1 verifier treated JSON object key order as evidence. The suite writer serializes with
``sort_keys=True``, so a valid arm mapping is alphabetized on disk and v1 rejects it before checking
the recorded arm hashes. This module changes only that invalid invariant: arm IDs are validated as
an exact set and then placed in protocol order before delegating every scientific and integrity
check to the source-locked v1 verifier.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.training.stability_pilot import StabilityPilotProtocol
from spatialvlm.training.stability_pilot_artifacts import (
    STABILITY_PILOT_VERIFIER_GATE_NAME,
    STABILITY_PILOT_VERIFIER_SCHEMA_VERSION,
    verify_stability_pilot,
)

STABILITY_PILOT_VERIFIER_V2_SCHEMA_VERSION = 2
STABILITY_PILOT_VERIFIER_V2_GATE_NAME = "independent_stability_pilot_artifact_verification_v2"
STABILITY_PILOT_VERIFIER_V2_CORRECTION = (
    "Treat the suite arms JSON object as an exact arm-ID mapping rather than requiring insertion "
    "order; preserve every v1 scientific, source, checkpoint, hash, and selection check."
)
STABILITY_PILOT_VERIFIER_V2_SOURCE_PATHS = (
    "scripts/check_stability_pilot_artifacts_v2.py",
    "scripts/reverify_spark_stability_pilot_v2.sh",
    "src/spatialvlm/training/stability_pilot_artifacts_v2.py",
)
EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS = (
    "suite arm index mismatch",
    "suite arm result hash mismatch: lr_5em06",
    "suite arm result hash mismatch: lr_1em05",
    "suite arm result hash mismatch: lr_5em05",
)


def verify_v1_order_false_negative(
    prior: dict[str, Any],
    *,
    suite_sha256: str,
    manifest_sha256: str,
) -> list[str]:
    """Validate that the preserved v1 failure is exactly the known ordering false-negative."""
    errors: list[str] = []
    if prior.get("schema_version") != STABILITY_PILOT_VERIFIER_SCHEMA_VERSION:
        errors.append("prior verification schema version is not v1")
    if prior.get("gate") != STABILITY_PILOT_VERIFIER_GATE_NAME:
        errors.append("prior verification gate is not the preserved v1 verifier")
    if prior.get("status") != "failed":
        errors.append("prior verification status is not failed")
    if prior.get("suite_sha256") != suite_sha256:
        errors.append("prior verification suite SHA-256 mismatch")
    if prior.get("manifest_sha256") != manifest_sha256:
        errors.append("prior verification manifest SHA-256 mismatch")
    if prior.get("dataset_reconstruction_performed") is not True:
        errors.append("prior verification did not reconstruct the released dataset")
    observed = prior.get("errors")
    if observed != list(EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS):
        errors.append("prior verification errors are not exactly the known v1 ordering cascade")
    if prior.get("error_count") != len(EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS):
        errors.append("prior verification error count does not match the known v1 cascade")
    return errors


def verify_stability_pilot_v2(
    suite: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    arms_root: Path,
    checkpoint_root: Path,
    repo: Path,
    dataset: R2RCEDataset | None = None,
) -> list[str]:
    """Run all v1 checks after canonicalizing a valid arm mapping without mutating input."""
    normalized_suite = deepcopy(suite)
    arm_index = normalized_suite.get("arms")
    protocol = StabilityPilotProtocol()
    expected_arm_ids = [protocol.arm_id(rate) for rate in protocol.learning_rates]
    if isinstance(arm_index, dict) and set(arm_index) == set(expected_arm_ids):
        normalized_suite["arms"] = {arm_id: arm_index[arm_id] for arm_id in expected_arm_ids}
    return verify_stability_pilot(
        normalized_suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=arms_root,
        checkpoint_root=checkpoint_root,
        repo=repo,
        dataset=dataset,
    )
