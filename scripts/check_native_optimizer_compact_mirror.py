#!/usr/bin/env python3
"""Verify the compact Mac mirror of the native-Qwen optimizer pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATE = "native_qwen_optimizer_compact_mac_mirror_verification_v1"
OPTIMIZER_GATE = "native_qwen_optimizer_admission_v1"
OPTIMIZER_VERIFIER_GATE = "independent_native_qwen_optimizer_admission_verification_v1"
SENSITIVITY_GATE = "native_qwen_c0_visual_sensitivity_diagnostic_v2"
SENSITIVITY_VERIFIER_GATE = "independent_native_qwen_c0_visual_sensitivity_verification_v2"
EXPECTED_RATES = (2e-7, 2e-6, 2e-5, 5e-5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} is not a JSON object")
        return {}
    return value


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return False


def verify_sensitivity(root: Path, repo: Path, errors: list[str]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    verification_path = root / "verification.json"
    manifest = load_json(manifest_path, errors, "sensitivity manifest")
    result = load_json(result_path, errors, "sensitivity result")
    verification = load_json(verification_path, errors, "sensitivity verification")
    manifest_hash = file_sha256(manifest_path) if manifest_path.is_file() else None
    result_hash = file_sha256(result_path) if result_path.is_file() else None
    verification_hash = file_sha256(verification_path) if verification_path.is_file() else None
    if manifest.get("gate") != SENSITIVITY_GATE or manifest.get("status") != "frozen":
        errors.append("sensitivity manifest gate/status mismatch")
    if result.get("gate") != SENSITIVITY_GATE or result.get("status") != "passed":
        errors.append("sensitivity result gate/status mismatch")
    if result.get("manifest_sha256") != manifest_hash:
        errors.append("sensitivity result/manifest hash mismatch")
    if (
        verification.get("gate") != SENSITIVITY_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("manifest_sha256") != manifest_hash
        or verification.get("result_sha256") != result_hash
    ):
        errors.append("sensitivity independent verification mismatch")
    if result.get("summary", {}).get("verdict") != "logit_sensitive_admit_optimizer_diagnostic":
        errors.append("sensitivity did not record the required admit verdict")
    locks = result.get("source_locks")
    verified_locks = 0
    if not isinstance(locks, dict) or not locks:
        errors.append("sensitivity source locks are missing")
    else:
        for relative, expected in locks.items():
            path = repo / relative
            if not path.is_file() or file_sha256(path) != expected:
                errors.append(f"sensitivity source-lock mismatch: {relative}")
            else:
                verified_locks += 1
    return {
        "manifest_sha256": manifest_hash,
        "result_sha256": result_hash,
        "verification_sha256": verification_hash,
        "verified_source_lock_count": verified_locks,
        "decision_count": len(result.get("decisions", [])),
        "verdict": result.get("summary", {}).get("verdict"),
    }


def verify_optimizer_run(root: Path, repo: Path, profile: str, errors: list[str]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    suite_path = root / "suite.json"
    verification_path = root / "verification.json"
    manifest = load_json(manifest_path, errors, f"{profile} manifest")
    suite = load_json(suite_path, errors, f"{profile} suite")
    verification = load_json(verification_path, errors, f"{profile} verification")
    manifest_hash = file_sha256(manifest_path) if manifest_path.is_file() else None
    suite_hash = file_sha256(suite_path) if suite_path.is_file() else None
    verification_hash = file_sha256(verification_path) if verification_path.is_file() else None
    if (
        manifest.get("gate") != OPTIMIZER_GATE
        or manifest.get("status") != "frozen"
        or manifest.get("profile") != profile
    ):
        errors.append(f"{profile} manifest gate/status/profile mismatch")
    if (
        suite.get("gate") != OPTIMIZER_GATE
        or suite.get("status") != "passed"
        or suite.get("profile") != profile
        or suite.get("manifest_sha256") != manifest_hash
    ):
        errors.append(f"{profile} suite linkage mismatch")
    if (
        verification.get("gate") != OPTIMIZER_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("profile") != profile
        or verification.get("manifest_sha256") != manifest_hash
        or verification.get("suite_sha256") != suite_hash
    ):
        errors.append(f"{profile} independent verification mismatch")
    if not _finite_numbers(suite):
        errors.append(f"{profile} suite contains a non-finite numeric value")
    locks = suite.get("source_locks")
    verified_locks = 0
    if not isinstance(locks, dict) or locks != manifest.get("source_locks"):
        errors.append(f"{profile} source-lock mapping mismatch")
        locks = {}
    for relative, expected in locks.items():
        path = repo / relative
        if not path.is_file() or file_sha256(path) != expected:
            errors.append(f"{profile} source-lock mismatch: {relative}")
        else:
            verified_locks += 1

    protocol = suite.get("protocol", {})
    planned_updates = protocol.get("planned_optimizer_updates")
    if tuple(protocol.get("learning_rates", ())) != EXPECTED_RATES:
        errors.append(f"{profile} learning-rate set mismatch")
    index = suite.get("arm_index")
    if not isinstance(index, list) or len(index) != len(EXPECTED_RATES):
        errors.append(f"{profile} arm index is incomplete")
        index = []
    initial_hashes: set[str] = set()
    arm_summaries: list[dict[str, Any]] = []
    for expected_rate, row in zip(EXPECTED_RATES, index):
        result_path = root / str(row.get("relative_path"))
        arm = load_json(result_path, errors, f"{profile} arm result")
        if not result_path.is_file() or file_sha256(result_path) != row.get("sha256"):
            errors.append(f"{profile} arm-result hash mismatch: {result_path}")
        if (
            arm.get("status") != "passed"
            or arm.get("manifest_sha256") != manifest_hash
            or arm.get("learning_rate") != expected_rate
            or arm.get("completed_updates") != planned_updates
            or arm.get("changed_update_count") != planned_updates
            or arm.get("all_updates_finite") is not True
        ):
            errors.append(f"{profile} arm contract mismatch: {result_path}")
        initial_hashes.add(str(arm.get("initial_trainable_state_sha256")))
        events_path = result_path.parent / "updates.jsonl"
        if not events_path.is_file() or file_sha256(events_path) != arm.get("events_sha256"):
            errors.append(f"{profile} event hash mismatch: {events_path}")
            events: list[dict[str, Any]] = []
        else:
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len(events) != planned_updates or not _finite_numbers(events):
            errors.append(f"{profile} event count/finiteness mismatch: {events_path}")
        previous_hash = str(arm.get("initial_trainable_state_sha256"))
        for expected_step, event in enumerate(events, start=1):
            if (
                event.get("update_step") != expected_step
                or event.get("before_trainable_state_sha256") != previous_hash
            ):
                errors.append(
                    f"{profile} event hash chain mismatch at {row.get('arm_id')}/{expected_step}"
                )
                break
            previous_hash = str(event.get("after_trainable_state_sha256"))
        if events and previous_hash != arm.get("final_trainable_state_sha256"):
            errors.append(f"{profile} final trainable-state hash mismatch: {row.get('arm_id')}")
        calibrations = arm.get("calibrations", [])
        arm_summaries.append(
            {
                "arm_id": row.get("arm_id"),
                "learning_rate": expected_rate,
                "completed_updates": arm.get("completed_updates"),
                "changed_update_count": arm.get("changed_update_count"),
                "result_sha256": file_sha256(result_path) if result_path.is_file() else None,
                "events_sha256": file_sha256(events_path) if events_path.is_file() else None,
                "calibration": [
                    {
                        "update_step": item.get("update_step"),
                        "mean_restricted_action_ce": item.get("summary", {}).get(
                            "mean_restricted_action_ce"
                        ),
                        "descriptive_accuracy": item.get("summary", {}).get("descriptive_accuracy"),
                        "distinct_predicted_action_count": item.get("summary", {}).get(
                            "distinct_predicted_action_count"
                        ),
                    }
                    for item in calibrations
                ],
            }
        )
    if len(initial_hashes) != 1 or next(iter(initial_hashes), None) != suite.get(
        "identical_initial_trainable_state_sha256"
    ):
        errors.append(f"{profile} arms do not share the declared initial state")
    selection = suite.get("selection", {})
    if (
        selection.get("status") != "rate_admitted"
        or selection.get("selected_learning_rate") != 2e-7
    ):
        errors.append(f"{profile} selection mismatch")
    return {
        "manifest_sha256": manifest_hash,
        "suite_sha256": suite_hash,
        "verification_sha256": verification_hash,
        "verified_source_lock_count": verified_locks,
        "selected_learning_rate": selection.get("selected_learning_rate"),
        "elapsed_suite_seconds": suite.get("runtime", {}).get("elapsed_suite_seconds"),
        "initial_trainable_state_sha256": suite.get("identical_initial_trainable_state_sha256"),
        "arms": arm_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    errors: list[str] = []
    sensitivity = verify_sensitivity(artifact_root / "native_sensitivity_v2", repo, errors)
    stress = verify_optimizer_run(
        artifact_root / "native_optimizer_stress_v1", repo, "stress", errors
    )
    main_run = verify_optimizer_run(
        artifact_root / "native_optimizer_admission_v1", repo, "main", errors
    )
    pipeline_log = artifact_root / "native_optimizer_pipeline_v4" / "run.log"
    pipeline_hash = file_sha256(pipeline_log) if pipeline_log.is_file() else None
    if not pipeline_log.is_file() or not pipeline_log.read_text(encoding="utf-8").rstrip().endswith(
        "Native optimizer-admission pipeline PASSED all four independently verified stages."
    ):
        errors.append("pipeline-v4 terminal pass marker is missing")
    failed_root = artifact_root / "native_sensitivity_v2_attempt1_import_failure"
    failed_log = failed_root / "run.log"
    failed_text = failed_log.read_text(encoding="utf-8") if failed_log.is_file() else ""
    if (
        "52 passed" not in failed_text
        or "ModuleNotFoundError: No module named 'scripts.run_spark_native_sensitivity'"
        not in failed_text
        or (failed_root / "result.json").exists()
        or (failed_root / "verification.json").exists()
    ):
        errors.append("preserved sensitivity-v2 import failure boundary mismatch")
    payload = {
        "schema_version": 1,
        "gate": GATE,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "repo": str(repo),
        "artifact_root": str(artifact_root),
        "sensitivity_v2": sensitivity,
        "optimizer_stress_v1": stress,
        "optimizer_admission_v1": main_run,
        "pipeline_v4_log_sha256": pipeline_hash,
        "evidence_boundary": {
            "established": (
                "The compact Spark artifacts, current source locks, arm/event hash chains, "
                "selection record, and independent-verifier linkages survived the Mac transfer."
            ),
            "not_established": (
                "Local bytes for unmirrored Spark checkpoints/caches/model weights, navigation "
                "competence, generalization, learning-rate optimality, or paper performance."
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
