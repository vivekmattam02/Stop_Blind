#!/usr/bin/env python3
"""Independently reconstruct the paired-seed replication stress gate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from scripts.run_spark_native_c0_seed_replication_stress import expected_stress_manifest
from spatialvlm.training.native_c0_gate import file_sha256
from spatialvlm.training.native_c0_seed_replication import (
    NATIVE_C0_SEED_REPLICATION_STRESS_GATE,
    NATIVE_C0_SEED_REPLICATION_STRESS_VERIFIER_GATE,
    NEW_REPLICATION_SEEDS,
    TREATMENTS,
    canonical_sha256,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def verify_stress(args: argparse.Namespace) -> list[str]:
    root = args.result_root.resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return ["replication stress manifest or result is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected, _ = expected_stress_manifest(args)
    errors: list[str] = []
    if manifest != expected:
        errors.append("replication stress manifest does not reconstruct")
    events = result.get("events")
    if not isinstance(events, list) or len(events) != 4:
        return [*errors, "replication stress must contain four configuration events"]
    expected_pairs = {
        (seed, treatment) for seed in NEW_REPLICATION_SEEDS for treatment in TREATMENTS
    }
    observed_pairs = {(row.get("replication_seed"), row.get("treatment")) for row in events}
    if observed_pairs != expected_pairs:
        errors.append("replication stress treatment/seed pairs differ")
    if (
        result.get("gate") != NATIVE_C0_SEED_REPLICATION_STRESS_GATE
        or result.get("status") != "passed"
        or result.get("full_replication_authorized") is not True
        or result.get("manifest_sha256") != file_sha256(manifest_path)
        or result.get("events_sha256") != canonical_sha256(events)
        or float(result.get("elapsed_seconds", math.inf))
        > float(result.get("hard_max_seconds", -math.inf))
    ):
        errors.append("replication stress result identity/status/provenance mismatch")
    try:
        completed = datetime.fromisoformat(str(result["completed_at_utc"]).replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(
            str(manifest["stress_deadline_utc"]).replace("Z", "+00:00")
        )
        if completed >= deadline:
            errors.append("replication stress completed after the two-hour deadline")
        if datetime.now(UTC) >= deadline:
            errors.append("replication stress was not independently verified before the deadline")
    except (KeyError, TypeError, ValueError):
        errors.append("replication stress completion/deadline timestamp is invalid")
    expected_indices = manifest.get("stress_sample_indices")
    for event in events:
        microbatches = event.get("microbatches")
        if (
            not isinstance(microbatches, list)
            or [row.get("sample_index") for row in microbatches] != expected_indices
        ):
            errors.append("replication stress microbatch identity/order mismatch")
            continue
        if any(row.get("supervised_token_count") != 1 for row in microbatches):
            errors.append("replication stress supervision count mismatch")
        if any(
            not math.isfinite(float(row.get("optimized_restricted_action_ce", math.nan)))
            or not math.isclose(
                float(row.get("optimized_restricted_action_ce", math.nan)),
                float(row.get("unweighted_restricted_action_ce", math.nan))
                * float(row.get("target_action_weight", math.nan)),
                rel_tol=0.0,
                abs_tol=5e-6,
            )
            for row in microbatches
        ):
            errors.append("replication stress optimized loss does not reconstruct")
        gradients = event.get("gradient_report", {})
        gradients_after = event.get("gradient_report_after_clip", {})
        updates = event.get("update_report", {})
        if (
            gradients.get("all_gradients_finite") is not True
            or gradients.get("nonzero_gradient_tensor_count", 0) <= 0
            or gradients_after.get("all_gradients_finite") is not True
            or float(gradients_after.get("gradient_l2_norm", math.inf)) > 1.0002
            or updates.get("all_updates_finite") is not True
            or updates.get("changed_tensor_count", 0) <= 0
            or event.get("initial_trainable_state_sha256")
            == event.get("final_trainable_state_sha256")
        ):
            errors.append("replication stress gradient/update report is invalid")
    for seed in NEW_REPLICATION_SEEDS:
        paired = [row for row in events if row.get("replication_seed") == seed]
        if len({row.get("initial_trainable_state_sha256") for row in paired}) != 1:
            errors.append(f"seed {seed} treatments do not share initial trainable state")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--unweighted-root", type=Path, required=True)
    parser.add_argument("--inverse-frequency-root", type=Path, required=True)
    parser.add_argument("--replication-stress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--failure-analysis", type=Path)
    parser.add_argument("--failure-verification", type=Path)
    parser.add_argument("--stress-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify_stress(args)
    root = args.result_root.resolve()
    payload = {
        "schema_version": 1,
        "gate": NATIVE_C0_SEED_REPLICATION_STRESS_VERIFIER_GATE,
        "recorded_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(root / "manifest.json"),
        "manifest_sha256": file_sha256(root / "manifest.json")
        if (root / "manifest.json").is_file()
        else None,
        "result_path": str(root / "result.json"),
        "result_sha256": file_sha256(root / "result.json")
        if (root / "result.json").is_file()
        else None,
        "contract_reconstruction_performed": True,
        "event_reconstruction_performed": True,
        "paired_initialization_reconstruction_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
