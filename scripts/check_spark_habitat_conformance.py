#!/usr/bin/env python3
"""Run the first real Habitat-Sim/R2R-CE conformance gate on DGX Spark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import (
    HABITAT_SIM_COMMIT,
    HABITAT_SIM_VERSION,
    HabitatSimR2RConfig,
    HabitatSimR2REnvironment,
)
from spatialvlm.eval.habitat_conformance import (
    HabitatConformanceThresholds,
    compare_ground_truth_replay,
    describe_observation_contract,
    quaternion_angular_error_radians,
)
from spatialvlm.eval.metrics import aggregate_r2r_ce_metrics
from spatialvlm.eval.r2r_ce import (
    GroundTruthActionPolicy,
    R2RCEEvaluationRun,
    build_r2r_ce_artifact,
    run_r2r_ce_episode,
    write_r2r_ce_step_jsonl,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps-output", required=True, type=Path)
    parser.add_argument(
        "--habitat-build-lock",
        type=Path,
        default=Path("/opt/gridnav/habitat-sim-commit"),
    )
    parser.add_argument("--require-architecture", default="aarch64")
    return parser.parse_args()


def _failure_report(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "failed",
        "recorded_at_utc": _timestamp(),
        "gate": "spark_habitat_sim_r2r_ce_ground_truth_replay_runtime_geodesic_v2",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc().splitlines()[-40:],
        },
        "inputs": {
            "scene_root": str(args.scene_root),
            "episodes": str(args.episodes),
            "ground_truth": str(args.ground_truth),
            "split": args.split,
            "episode_count": args.episode_count,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_count <= 0:
        raise ValueError("Conformance episode count must be positive.")
    build_commit = args.habitat_build_lock.read_text(encoding="utf-8").strip()
    architecture = platform.machine().lower()
    dataset = R2RCEDataset.from_release(
        args.episodes,
        args.ground_truth,
        split=args.split,
        limit=args.episode_count,
    )
    if dataset.release_manifest is None:
        raise RuntimeError("Strict R2R-CE release loading did not produce a manifest.")
    if len(dataset.official_episodes) != args.episode_count:
        raise RuntimeError(
            f"Requested {args.episode_count} episodes, loaded {len(dataset.official_episodes)}."
        )

    thresholds = HabitatConformanceThresholds()
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(
            scene_root=args.scene_root,
            gpu_device_id=args.gpu_device_id,
        )
    )
    results = []
    episodes_report: list[dict[str, Any]] = []
    try:
        for episode in dataset.official_episodes:
            ground_truth = dataset.ground_truth_record(episode.episode_id)
            reset_state = environment.reset(episode)
            observation_report = describe_observation_contract(dict(reset_state.observation))
            rotation_error = quaternion_angular_error_radians(
                environment.agent_rotation_xyzw(), episode.start_rotation
            )
            result = run_r2r_ce_episode(
                episode,
                ground_truth,
                environment,
                GroundTruthActionPolicy(),
            )
            replay_report = compare_ground_truth_replay(
                episode,
                ground_truth,
                result,
                reset_rotation_error_radians=rotation_error,
                thresholds=thresholds,
            )
            results.append(result)
            episodes_report.append(
                {
                    "episode_id": episode.episode_id,
                    "scene_id": episode.scene_id,
                    "observation_contract": observation_report,
                    "ground_truth_replay": replay_report,
                    "status": (
                        "passed"
                        if observation_report["status"] == "passed"
                        and replay_report["status"] == "passed"
                        else "failed"
                    ),
                }
            )
        runtime = environment.runtime_metadata()
    finally:
        environment.close()

    evaluation_run = R2RCEEvaluationRun(
        results=tuple(results),
        aggregate=aggregate_r2r_ce_metrics([result.metrics for result in results]),
    )
    evaluation = build_r2r_ce_artifact(
        evaluation_run,
        dataset.release_manifest,
        policy_id="released_ground_truth_actions",
    )
    write_r2r_ce_step_jsonl(args.steps_output, results)

    checks = {
        "architecture_matches": architecture == args.require_architecture.lower(),
        "habitat_build_commit_matches": build_commit == HABITAT_SIM_COMMIT,
        "habitat_runtime_version_matches": runtime["habitat_sim_version"] == HABITAT_SIM_VERSION,
        "navmesh_loaded": runtime["navmesh_loaded"],
        "requested_episode_count_ran": len(results) == args.episode_count,
        "all_observation_contracts_passed": all(
            row["observation_contract"]["status"] == "passed" for row in episodes_report
        ),
        "all_ground_truth_replays_passed": all(
            row["ground_truth_replay"]["status"] == "passed" for row in episodes_report
        ),
    }
    return {
        "schema_version": 2,
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": _timestamp(),
        "gate": "spark_habitat_sim_r2r_ce_ground_truth_replay_runtime_geodesic_v2",
        "evidence_boundary": {
            "established_if_passed": (
                "Pinned modern Habitat-Sim loads licensed MP3D, renders the official native "
                "sensor contract, executes released actions, and reproduces the selected "
                "released position trace within locked tolerances. Metrics use the simulator "
                "geodesic measured at reset; released geodesic metadata drift is recorded."
            ),
            "not_established": (
                "Full-split equivalence, released-checkpoint performance, baseline ordering, "
                "training correctness, or paper-level model claims."
            ),
        },
        "checks": checks,
        "platform": {
            "platform": platform.platform(),
            "architecture": architecture,
            "python": platform.python_version(),
        },
        "runtime": {
            **runtime,
            "habitat_sim_build_lock_path": str(args.habitat_build_lock),
            "habitat_sim_build_commit_observed": build_commit,
        },
        "thresholds": thresholds.to_dict(),
        "episodes": episodes_report,
        "evaluation": evaluation,
        "steps_output": str(args.steps_output),
    }


def main() -> int:
    args = _arguments()
    try:
        report = run(args)
    except Exception as error:
        report = _failure_report(args, error)
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 1

    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
