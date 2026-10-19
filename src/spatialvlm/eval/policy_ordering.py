"""Frozen control-policy design and analysis for the Spark R2R-CE evaluator gate.

This module does not evaluate a learned navigation claim.  It constructs a trajectory-level
manifest and pre-registers contrasts that can falsify evaluator, termination, action-parser, or
runtime behavior before training begins.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.eval.habitat_suite import safe_episode_filename
from spatialvlm.eval.r2r_ce import (
    R2R_CE_SUCCESS_DISTANCE_M,
    AlwaysForwardPolicy,
    GroundTruthActionPolicy,
    ImmediateStopPolicy,
    InvalidActionPolicy,
    R2RCEPolicy,
    SeededHandcraftedPolicy,
    SeededRandomPolicy,
)

POLICY_ORDERING_SCHEMA_VERSION = 1
POLICY_ORDERING_SEEDS = (17, 29, 43, 71, 101)
OFFICIAL_VLNCE_NONLEARNING_SOURCE = (
    "https://github.com/jacobkrantz/VLN-CE/blob/master/vlnce_baselines/nonlearning_agents.py"
)
OFFICIAL_VLNCE_BASELINE_README = (
    "https://github.com/jacobkrantz/VLN-CE/blob/master/"
    "vlnce_baselines/config/r2r_baselines/README.md"
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one mapping using the JSON representation used by frozen manifests."""
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _episode_sort_key(episode_id: str) -> tuple[int, int, str]:
    try:
        return (0, int(episode_id), "")
    except ValueError:
        return (1, 0, episode_id)


@dataclass(frozen=True)
class PolicyRunSpec:
    """One deterministic policy/seed combination in the frozen suite."""

    policy_run_id: str
    family: str
    role: str
    seed: int | None
    provenance: str
    implementation_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def policy_run_specs(
    seeds: Sequence[int] = POLICY_ORDERING_SEEDS,
) -> tuple[PolicyRunSpec, ...]:
    """Return the pre-registered deterministic and stochastic control runs."""
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Policy-ordering seeds must be non-empty and unique.")
    deterministic = (
        PolicyRunSpec(
            policy_run_id="released_ground_truth_replay",
            family="released_ground_truth_replay",
            role="positive evaluator control; not a deployable policy",
            seed=None,
            provenance="R2R_VLNCE_v1-3 released action trace",
            implementation_note="exact released actions for the selected episode",
        ),
        PolicyRunSpec(
            policy_run_id="immediate_stop",
            family="immediate_stop",
            role="termination-only negative control",
            seed=None,
            provenance="GridNav pre-registered control",
            implementation_note="one STOP action at reset",
        ),
        PolicyRunSpec(
            policy_run_id="always_forward",
            family="always_forward",
            role="missing-STOP negative control",
            seed=None,
            provenance="GridNav pre-registered control",
            implementation_note="MOVE_FORWARD until the official 500-step cap",
        ),
        PolicyRunSpec(
            policy_run_id="invalid_action",
            family="invalid_action",
            role="fail-closed action-parser control; not a navigation baseline",
            seed=None,
            provenance="GridNav pre-registered control",
            implementation_note="integer 999 rejected without simulator execution",
        ),
    )
    stochastic: list[PolicyRunSpec] = []
    for seed in normalized_seeds:
        stochastic.extend(
            (
                PolicyRunSpec(
                    policy_run_id=f"official_random_seed_{seed}",
                    family="official_random",
                    role="recognized stochastic non-learning baseline",
                    seed=seed,
                    provenance=OFFICIAL_VLNCE_NONLEARNING_SOURCE,
                    implementation_note=(
                        "official action prior with SHA256(seed, episode_id) and Python PRNG "
                        "for auditable episode-local determinism"
                    ),
                ),
                PolicyRunSpec(
                    policy_run_id=f"official_handcrafted_seed_{seed}",
                    family="official_handcrafted",
                    role="recognized stochastic language-blind baseline",
                    seed=seed,
                    provenance=OFFICIAL_VLNCE_NONLEARNING_SOURCE,
                    implementation_note=(
                        "official heading/37-forward/STOP schedule with SHA256(seed, episode_id) "
                        "and Python PRNG for auditable episode-local determinism"
                    ),
                ),
            )
        )
    return deterministic + tuple(stochastic)


def make_policy(spec: PolicyRunSpec) -> R2RCEPolicy:
    """Instantiate one policy from its immutable run specification."""
    factories: dict[str, Callable[[], R2RCEPolicy]] = {
        "released_ground_truth_replay": GroundTruthActionPolicy,
        "immediate_stop": ImmediateStopPolicy,
        "always_forward": AlwaysForwardPolicy,
        "invalid_action": InvalidActionPolicy,
    }
    if spec.family in factories:
        if spec.seed is not None:
            raise ValueError(f"Deterministic policy {spec.family} must not have a seed.")
        return factories[spec.family]()
    if spec.seed is None:
        raise ValueError(f"Stochastic policy {spec.family} requires a seed.")
    if spec.family == "official_random":
        return SeededRandomPolicy(seed=spec.seed)
    if spec.family == "official_handcrafted":
        return SeededHandcraftedPolicy(seed=spec.seed)
    raise ValueError(f"Unknown policy family: {spec.family!r}")


def _validate_parent_evidence(
    parent_manifest: Mapping[str, Any],
    conformance_reports: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    episodes = parent_manifest.get("episodes")
    manifest_sha256 = parent_manifest.get("manifest_sha256")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Policy ordering requires a non-empty parent conformance manifest.")
    unhashed = dict(parent_manifest)
    unhashed.pop("manifest_sha256", None)
    if manifest_sha256 != canonical_sha256(unhashed):
        raise ValueError("Parent conformance manifest self-hash is invalid.")

    entries_by_id: dict[str, Mapping[str, Any]] = {}
    for entry in episodes:
        episode_id = str(entry["episode_id"])
        if episode_id in entries_by_id:
            raise ValueError(f"Duplicate parent episode ID: {episode_id}")
        entries_by_id[episode_id] = entry

    reports_by_id: dict[str, Mapping[str, Any]] = {}
    for report in conformance_reports:
        episode_id = str(report.get("episode_id"))
        if episode_id in reports_by_id:
            raise ValueError(f"Duplicate conformance report for episode {episode_id}.")
        if episode_id not in entries_by_id:
            raise ValueError(f"Conformance report episode {episode_id} is absent from manifest.")
        if report.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"Conformance report {episode_id} has the wrong manifest hash.")
        if report.get("status") != "passed":
            raise ValueError(f"Conformance report {episode_id} did not pass.")
        reports_by_id[episode_id] = report
    if set(reports_by_id) != set(entries_by_id):
        missing = sorted(set(entries_by_id) - set(reports_by_id), key=_episode_sort_key)
        raise ValueError(f"Parent conformance evidence is incomplete; missing={missing[:10]}.")
    return entries_by_id, reports_by_id


def build_policy_ordering_manifest(
    parent_manifest: Mapping[str, Any],
    conformance_reports: Sequence[Mapping[str, Any]],
    *,
    software_sha256: Mapping[str, str] | None = None,
    seeds: Sequence[int] = POLICY_ORDERING_SEEDS,
) -> dict[str, Any]:
    """Freeze one unique-trajectory, all-scene control suite from passed conformance evidence."""
    entries_by_id, reports_by_id = _validate_parent_evidence(parent_manifest, conformance_reports)
    entries = list(entries_by_id.values())
    coverage_entries = [entry for entry in entries if entry.get("phase") == "scene_coverage_stress"]
    scene_ids = {str(entry["scene_id"]) for entry in entries}
    if {str(entry["scene_id"]) for entry in coverage_entries} != scene_ids:
        raise ValueError("Parent coverage prefix must contain a representative from every scene.")
    if len(coverage_entries) != len(scene_ids):
        raise ValueError("Parent coverage prefix must contain exactly one episode per scene.")

    entries_by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        entries_by_trajectory[str(entry["trajectory_id"])].append(entry)

    drift_trajectory_ids = {
        str(entries_by_id[episode_id]["trajectory_id"])
        for episode_id, report in reports_by_id.items()
        if report["ground_truth_replay"]["diagnostics"][
            "release_metadata_geodesic_within_tolerance"
        ]
        is False
    }
    selected_by_trajectory: dict[str, Mapping[str, Any]] = {
        str(entry["trajectory_id"]): entry for entry in coverage_entries
    }
    for trajectory_id in sorted(drift_trajectory_ids):
        if trajectory_id not in selected_by_trajectory:
            selected_by_trajectory[trajectory_id] = min(
                entries_by_trajectory[trajectory_id],
                key=lambda entry: _episode_sort_key(str(entry["episode_id"])),
            )

    selected_rows: list[dict[str, Any]] = []
    for trajectory_id, entry in selected_by_trajectory.items():
        reasons = ["scene_coverage_stress"] if entry in coverage_entries else []
        if trajectory_id in drift_trajectory_ids:
            reasons.append("runtime_geodesic_drift")
        report = reports_by_id[str(entry["episode_id"])]
        selected_rows.append(
            {
                "episode_id": str(entry["episode_id"]),
                "trajectory_id": trajectory_id,
                "scene_id": str(entry["scene_id"]),
                "parent_index": int(entry["index"]),
                "features": dict(entry["features"]),
                "selection_reasons": reasons,
                "runtime_reset_geodesic_m": report["ground_truth_replay"]["observed"][
                    "runtime_reset_geodesic_m"
                ],
                "release_metadata_geodesic_drift_m": report["ground_truth_replay"]["observed"][
                    "release_metadata_geodesic_drift_m"
                ],
            }
        )

    selected_rows.sort(
        key=lambda row: (
            row["scene_id"],
            row["parent_index"],
            _episode_sort_key(row["episode_id"]),
        )
    )
    for index, row in enumerate(selected_rows):
        row["index"] = index
        row["artifact_stem"] = f"{index:04d}-{safe_episode_filename(row['episode_id'])}"
    minimum_reset_geodesic = min(float(row["runtime_reset_geodesic_m"]) for row in selected_rows)
    if minimum_reset_geodesic < R2R_CE_SUCCESS_DISTANCE_M:
        raise ValueError(
            "Immediate-STOP control is confounded because a selected reset pose is already "
            "inside the strict 3 m success radius."
        )

    run_specs = policy_run_specs(seeds)
    payload: dict[str, Any] = {
        "schema_version": POLICY_ORDERING_SCHEMA_VERSION,
        "suite_id": "r2r_ce_val_seen_trajectory_level_policy_controls_v1",
        "parent_evidence": {
            "suite_id": parent_manifest.get("suite_id"),
            "manifest_sha256": parent_manifest["manifest_sha256"],
            "parent_episode_count": len(entries),
            "parent_status_required": "passed",
        },
        "release": dict(parent_manifest["release"]),
        "design": {
            "experimental_unit": "unique physical trajectory",
            "instruction_sampling": "one deterministic instruction episode per trajectory",
            "selection_strategy": (
                "union of the parent suite's one stress representative per scene and one "
                "representative from every trajectory with release/runtime geodesic drift"
            ),
            "policy_outcomes_used_for_selection": False,
            "selected_episode_count": len(selected_rows),
            "selected_trajectory_count": len(selected_rows),
            "selected_scene_count": len({row["scene_id"] for row in selected_rows}),
            "source_scene_count": len(scene_ids),
            "drift_trajectory_count": len(drift_trajectory_ids),
            "minimum_runtime_reset_geodesic_m": minimum_reset_geodesic,
            "success_distance_m": R2R_CE_SUCCESS_DISTANCE_M,
            "stochastic_seeds": [int(seed) for seed in seeds],
            "stochastic_summary": "seed-wise metrics plus descriptive mean and range",
            "inferential_claim": "none; this is an evaluator/control gate",
        },
        "policies": [spec.to_dict() for spec in run_specs],
        "pre_registered_checks": {
            "released_ground_truth_replay": (
                "all episodes succeed and stop, no outputs are invalid, and minimum nDTW is 0.9999"
            ),
            "immediate_stop": (
                "all episodes stop with zero path length and fail because every locked reset "
                "geodesic exceeds the 3 m success radius"
            ),
            "always_forward": (
                "all episodes reach the 500-step cap without STOP and therefore have zero SR"
            ),
            "invalid_action": (
                "exactly one invalid output and zero executed actions per episode, with zero SR"
            ),
            "recognized_nonlearning_controls": (
                "every official-random and official-handcrafted seed is below released replay "
                "on SR, SPL, nDTW, and SDTW"
            ),
            "not_required": (
                "no ordering is required between random and handcrafted controls; neither is a "
                "learned-model result"
            ),
        },
        "learned_baseline_boundary": {
            "status": "not_part_of_this_gate",
            "official_cma_checkpoint_source": OFFICIAL_VLNCE_BASELINE_README,
            "reason": (
                "legacy Habitat 0.1.7 reproduction and modern-runtime port results must be "
                "reported separately"
            ),
        },
        "software_sha256": dict(sorted((software_sha256 or {}).items())),
        "episodes": selected_rows,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def summarize_policy_reports(
    reports: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize completed reports without treating seeds as independent trajectories."""
    expected_episode_ids = {str(row["episode_id"]) for row in manifest["episodes"]}
    expected_specs = {str(row["policy_run_id"]): row for row in manifest["policies"]}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for report in reports:
        policy_run_id = str(report["policy_run_id"])
        episode_id = str(report["episode_id"])
        identity = (policy_run_id, episode_id)
        if identity in seen:
            raise ValueError(f"Duplicate policy report: {identity}")
        if policy_run_id not in expected_specs or episode_id not in expected_episode_ids:
            raise ValueError(f"Policy report is outside the frozen manifest: {identity}")
        seen.add(identity)
        grouped[policy_run_id].append(report)

    metric_names = (
        "navigation_error",
        "oracle_success",
        "success",
        "spl",
        "ndtw",
        "sdtw",
        "path_length",
        "step_count",
    )
    run_summaries: dict[str, dict[str, Any]] = {}
    for policy_run_id, spec in expected_specs.items():
        rows = grouped.get(policy_run_id, [])
        metrics = [row["evaluation"]["metrics"] for row in rows if row.get("status") == "passed"]
        summary: dict[str, Any] = {
            "family": spec["family"],
            "seed": spec["seed"],
            "completed_episode_count": len(rows),
            "passed_episode_count": len(metrics),
            "failed_episode_count": sum(row.get("status") != "passed" for row in rows),
        }
        if metrics:
            for metric_name in metric_names:
                summary[metric_name] = sum(float(row[metric_name]) for row in metrics) / len(
                    metrics
                )
            summary["stopped_rate"] = sum(bool(row["stopped"]) for row in metrics) / len(metrics)
            summary["collision_count"] = sum(int(row["collision_count"]) for row in metrics)
            summary["invalid_action_count"] = sum(
                int(row["invalid_action_count"]) for row in metrics
            )
            summary["minimum_ndtw"] = min(float(row["ndtw"]) for row in metrics)
        run_summaries[policy_run_id] = summary

    family_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in run_summaries.values():
        if summary["passed_episode_count"]:
            family_runs[str(summary["family"])].append(summary)
    family_summaries: dict[str, dict[str, Any]] = {}
    family_metric_names = ("success", "spl", "ndtw", "sdtw", "navigation_error")
    for family, rows in sorted(family_runs.items()):
        family_summary: dict[str, Any] = {
            "run_count": len(rows),
            "seeds": [row["seed"] for row in rows if row["seed"] is not None],
        }
        for metric_name in family_metric_names:
            values = [float(row[metric_name]) for row in rows]
            family_summary[metric_name] = {
                "mean_across_seed_runs": sum(values) / len(values),
                "minimum_seed_run": min(values),
                "maximum_seed_run": max(values),
            }
        family_summaries[family] = family_summary

    target = len(expected_episode_ids) * len(expected_specs)
    return {
        "target_report_count": target,
        "completed_report_count": len(reports),
        "remaining_report_count": target - len(reports),
        "policy_runs": run_summaries,
        "families": family_summaries,
    }


def evaluate_pre_registered_checks(
    reports: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the contrasts frozen in the manifest; never invent a total ranking."""
    summary = summarize_policy_reports(reports, manifest)
    if summary["remaining_report_count"]:
        return {
            "status": "incomplete",
            "all_required_checks_pass": False,
            "checks": {},
            "summary": summary,
        }
    if any(row.get("status") != "passed" for row in reports):
        return {
            "status": "infrastructure_failure",
            "all_required_checks_pass": False,
            "checks": {},
            "summary": summary,
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report["policy_run_id"])].append(report)

    def metrics(policy_run_id: str) -> list[Mapping[str, Any]]:
        return [row["evaluation"]["metrics"] for row in grouped[policy_run_id]]

    gt = metrics("released_ground_truth_replay")
    immediate = metrics("immediate_stop")
    forward = metrics("always_forward")
    invalid_reports = grouped["invalid_action"]
    invalid = [row["evaluation"]["metrics"] for row in invalid_reports]
    checks: dict[str, bool] = {
        "released_replay_all_success": all(row["success"] == 1.0 for row in gt),
        "released_replay_all_stopped": all(row["stopped"] for row in gt),
        "released_replay_no_invalid": all(row["invalid_action_count"] == 0 for row in gt),
        "released_replay_minimum_ndtw": min(row["ndtw"] for row in gt) >= 0.9999,
        "immediate_stop_all_stopped": all(row["stopped"] for row in immediate),
        "immediate_stop_zero_motion": all(
            math.isclose(row["path_length"], 0.0, abs_tol=1e-12) for row in immediate
        ),
        "immediate_stop_zero_success": all(row["success"] == 0.0 for row in immediate),
        "always_forward_never_stopped": all(not row["stopped"] for row in forward),
        "always_forward_reaches_step_cap": all(row["step_count"] == 500 for row in forward),
        "always_forward_zero_success": all(row["success"] == 0.0 for row in forward),
        "invalid_action_exactly_one_per_episode": all(
            row["invalid_action_count"] == 1 for row in invalid
        ),
        "invalid_action_zero_executed_actions": all(
            not report["evaluation"]["trace"]["actions"] for report in invalid_reports
        ),
        "invalid_action_zero_success": all(row["success"] == 0.0 for row in invalid),
    }

    gt_summary = summary["policy_runs"]["released_ground_truth_replay"]
    for family in ("official_random", "official_handcrafted"):
        family_rows = [row for row in summary["policy_runs"].values() if row["family"] == family]
        for metric_name in ("success", "spl", "ndtw", "sdtw"):
            checks[f"{family}_every_seed_below_replay_{metric_name}"] = all(
                row[metric_name] < gt_summary[metric_name] for row in family_rows
            )
        checks[f"{family}_no_invalid_actions"] = all(
            row["invalid_action_count"] == 0 for row in family_rows
        )

    return {
        "status": "passed" if all(checks.values()) else "failed_scientific_gate",
        "all_required_checks_pass": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "non_required_comparison": (
            "No random-versus-handcrafted ordering was tested or required."
        ),
    }
