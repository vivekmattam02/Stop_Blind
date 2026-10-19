"""Scientific-design and integrity tests for the Spark policy-control gate."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from spatialvlm.eval.policy_ordering import (
    build_policy_ordering_manifest,
    canonical_sha256,
    evaluate_pre_registered_checks,
    policy_run_specs,
)

_RUNNER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_spark_policy_ordering.py")
)


def _parent_manifest() -> dict:
    entries = []
    definitions = (
        ("1", "trajectory-a", "scene-a", "scene_coverage_stress"),
        ("2", "trajectory-a", "scene-a", "full_split"),
        ("3", "trajectory-b", "scene-b", "scene_coverage_stress"),
        ("4", "trajectory-b", "scene-b", "full_split"),
        ("5", "trajectory-c", "scene-a", "full_split"),
        ("6", "trajectory-c", "scene-a", "full_split"),
    )
    for index, (episode_id, trajectory_id, scene_id, phase) in enumerate(definitions):
        entries.append(
            {
                "index": index,
                "episode_id": episode_id,
                "trajectory_id": trajectory_id,
                "scene_id": scene_id,
                "phase": phase,
                "artifact_stem": f"{index:04d}-{episode_id}",
                "features": {
                    "action_count": 3,
                    "forward_steps": 2,
                    "turn_left_count": 0,
                    "turn_right_count": 0,
                    "no_displacement_forward_count": 0,
                    "short_displacement_forward_count": 0,
                    "minimum_forward_displacement_m": 0.25,
                    "maximum_forward_displacement_m": 0.25,
                },
            }
        )
    payload = {
        "schema_version": 2,
        "suite_id": "parent-test-suite",
        "selection": {
            "target_episode_count": 6,
            "scene_count": 2,
        },
        "release": {
            "release_id": "R2R_VLNCE_v1-3",
            "split": "val_seen",
            "episodes_path": "/datasets/val_seen.json.gz",
            "episodes_sha256": "a" * 64,
            "source_episode_count": 6,
            "selected_episode_count": 6,
            "ground_truth_path": "/datasets/val_seen_gt.json.gz",
            "ground_truth_sha256": "b" * 64,
            "ground_truth_count": 6,
        },
        "software_sha256": {},
        "episodes": entries,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _parent_reports(manifest: dict) -> list[dict]:
    reports = []
    for entry in manifest["episodes"]:
        drift = entry["trajectory_id"] == "trajectory-c"
        reports.append(
            {
                "schema_version": 2,
                "status": "passed",
                "index": entry["index"],
                "episode_id": entry["episode_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "ground_truth_replay": {
                    "diagnostics": {
                        "release_metadata_geodesic_within_tolerance": not drift,
                    },
                    "observed": {
                        "runtime_reset_geodesic_m": 5.0 + entry["index"],
                        "release_metadata_geodesic_drift_m": 0.2 if drift else 0.0,
                    },
                },
            }
        )
    return reports


def _policy_reports(manifest: dict) -> list[dict]:
    reports = []
    for episode in manifest["episodes"]:
        for spec in manifest["policies"]:
            family = spec["family"]
            metric = {
                "episode_id": episode["episode_id"],
                "navigation_error": 0.5,
                "oracle_success": 1.0,
                "success": 1.0,
                "path_length": 5.0,
                "shortest_path_length": 5.0,
                "spl": 1.0,
                "ndtw": 1.0,
                "sdtw": 1.0,
                "step_count": 3,
                "collision_count": 0,
                "invalid_action_count": 0,
                "stopped": True,
            }
            actions = [1, 1, 0]
            if family == "immediate_stop":
                metric.update(
                    success=0.0,
                    oracle_success=0.0,
                    path_length=0.0,
                    spl=0.0,
                    ndtw=0.4,
                    sdtw=0.0,
                    step_count=1,
                )
                actions = [0]
            elif family == "always_forward":
                metric.update(
                    success=0.0,
                    spl=0.0,
                    ndtw=0.2,
                    sdtw=0.0,
                    step_count=500,
                    stopped=False,
                )
                actions = [1] * 500
            elif family == "invalid_action":
                metric.update(
                    success=0.0,
                    oracle_success=0.0,
                    path_length=0.0,
                    spl=0.0,
                    ndtw=0.3,
                    sdtw=0.0,
                    step_count=1,
                    invalid_action_count=1,
                    stopped=False,
                )
                actions = []
            elif family == "official_random":
                metric.update(success=0.1, spl=0.05, ndtw=0.3, sdtw=0.03)
            elif family == "official_handcrafted":
                metric.update(success=0.2, spl=0.15, ndtw=0.4, sdtw=0.08)
            reports.append(
                {
                    "schema_version": 1,
                    "status": "passed",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "index": episode["index"],
                    "episode_id": episode["episode_id"],
                    "policy_run_id": spec["policy_run_id"],
                    "family": family,
                    "seed": spec["seed"],
                    "evaluation": {
                        "metrics": metric,
                        "trace": {"actions": actions},
                    },
                }
            )
    return reports


def test_manifest_uses_one_instruction_per_trajectory_and_retains_drift_stress():
    parent = _parent_manifest()
    manifest = build_policy_ordering_manifest(parent, _parent_reports(parent))
    assert len(manifest["episodes"]) == 3
    assert {row["trajectory_id"] for row in manifest["episodes"]} == {
        "trajectory-a",
        "trajectory-b",
        "trajectory-c",
    }
    drift_row = next(row for row in manifest["episodes"] if row["trajectory_id"] == "trajectory-c")
    assert drift_row["episode_id"] == "5"
    assert drift_row["selection_reasons"] == ["runtime_geodesic_drift"]
    assert manifest["design"]["policy_outcomes_used_for_selection"] is False


def test_manifest_and_policy_seed_schedule_are_deterministic():
    parent = _parent_manifest()
    reports = _parent_reports(parent)
    first = build_policy_ordering_manifest(parent, reports)
    second = build_policy_ordering_manifest(parent, list(reversed(reports)))
    assert first == second
    assert len(policy_run_specs()) == 14
    assert first["design"]["stochastic_seeds"] == [17, 29, 43, 71, 101]


def test_parent_manifest_tampering_fails_closed():
    parent = _parent_manifest()
    parent["episodes"][0]["episode_id"] = "tampered"
    with pytest.raises(ValueError, match="self-hash"):
        build_policy_ordering_manifest(parent, _parent_reports(_parent_manifest()))


def test_pre_registered_checks_pass_without_random_handcrafted_total_order():
    parent = _parent_manifest()
    manifest = build_policy_ordering_manifest(parent, _parent_reports(parent))
    analysis = evaluate_pre_registered_checks(_policy_reports(manifest), manifest)
    assert analysis["status"] == "passed"
    assert analysis["all_required_checks_pass"] is True
    assert "No random-versus-handcrafted ordering" in analysis["non_required_comparison"]


def test_pre_registered_check_detects_missing_stop_regression():
    parent = _parent_manifest()
    manifest = build_policy_ordering_manifest(parent, _parent_reports(parent))
    reports = _policy_reports(manifest)
    forward = next(row for row in reports if row["family"] == "always_forward")
    forward["evaluation"]["metrics"]["stopped"] = True
    analysis = evaluate_pre_registered_checks(reports, manifest)
    assert analysis["status"] == "failed_scientific_gate"
    assert analysis["checks"]["always_forward_never_stopped"] is False


def test_resume_checkpoint_rejects_tampered_step_trace(tmp_path):
    steps_path = tmp_path / "steps.jsonl"
    steps_path.write_text('{"step": 1}\n', encoding="utf-8")
    digest = hashlib.sha256(steps_path.read_bytes()).hexdigest()
    report_path = tmp_path / "episode.json"
    spec = {
        "policy_run_id": "immediate_stop",
        "family": "immediate_stop",
        "seed": None,
    }
    entry = {"index": 0, "episode_id": "1"}
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "manifest_sha256": "a" * 64,
                "policy_run_id": "immediate_stop",
                "family": "immediate_stop",
                "seed": None,
                "index": 0,
                "episode_id": "1",
                "steps_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    validator = _RUNNER["_validate_existing_report"]
    validator(report_path, steps_path, spec, entry, "a" * 64)
    steps_path.write_text('{"step": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="inconsistent"):
        validator(report_path, steps_path, spec, entry, "a" * 64)
