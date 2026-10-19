"""Integrity tests for resumable expanded-runner episode checkpoints."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.eval.r2r_ce import R2RCERolloutState

_RUNNER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_spark_habitat_expanded.py")
)
_validate_existing_report = _RUNNER["_validate_existing_report"]
_summary = _RUNNER["_summary"]


def _runner_dataset() -> R2RCEDataset:
    episodes = tuple(
        R2RCEEpisode(
            episode_id=str(index),
            trajectory_id=f"trajectory-{index}",
            scene_id=f"mp3d/scene-{index}/scene-{index}.glb",
            instruction_text="Move forward and stop.",
            instruction_tokens=(),
            start_position=(0.0, 0.0, 0.0),
            start_rotation=(0.0, 0.0, 0.0, 1.0),
            goals=(R2RCEGoal(position=(0.25, 0.0, 0.0), radius=3.0),),
            reference_path=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
            geodesic_distance=0.25,
            raw={"episode_id": index},
        )
        for index in (1, 2)
    )
    ground_truth = {
        episode.episode_id: R2RCEGroundTruth(
            episode_id=episode.episode_id,
            actions=(R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP),
            forward_steps=1,
            locations=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
        )
        for episode in episodes
    }
    manifest = R2RCEReleaseManifest(
        release_id="R2R_VLNCE_v1-3",
        split="val_seen",
        episodes_path="/datasets/val_seen.json.gz",
        episodes_sha256="a" * 64,
        source_episode_count=2,
        selected_episode_count=2,
        ground_truth_path="/datasets/val_seen_gt.json.gz",
        ground_truth_sha256="b" * 64,
        ground_truth_count=2,
    )
    return R2RCEDataset(
        records=[episode.raw for episode in episodes],
        official_episodes=episodes,
        ground_truth=ground_truth,
        release_manifest=manifest,
    )


class _FakeHabitatEnvironment:
    construction_count = 0

    def __init__(self, config):
        del config
        type(self).construction_count += 1
        self.position = (0.0, 0.0, 0.0)
        self.done = False

    @staticmethod
    def _observation():
        return {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "depth": np.zeros((256, 256, 1), dtype=np.float32),
            "depth_meters": np.zeros((256, 256), dtype=np.float32),
        }

    def _state(self):
        return R2RCERolloutState(
            observation=self._observation(),
            position=self.position,
            distance_to_goal=abs(0.25 - self.position[0]),
            done=self.done,
        )

    def reset(self, episode):
        self.position = episode.start_position
        self.done = False
        return self._state()

    def step(self, action):
        if action == R2RCEAction.MOVE_FORWARD:
            self.position = (0.25, 0.0, 0.0)
        elif action == R2RCEAction.STOP:
            self.done = True
        return self._state()

    @staticmethod
    def agent_rotation_xyzw():
        return (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def runtime_metadata():
        return {
            "habitat_sim_version": "0.3.3",
            "navmesh_loaded": True,
            "scene_path": "/datasets/fake.glb",
        }

    def close(self):
        return None


def test_resume_checkpoint_rejects_a_tampered_step_trace(tmp_path):
    steps_path = tmp_path / "steps.jsonl"
    steps_path.write_text('{"step": 1}\n', encoding="utf-8")
    digest = hashlib.sha256(steps_path.read_bytes()).hexdigest()
    report_path = tmp_path / "episode.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "passed",
                "manifest_sha256": "a" * 64,
                "index": 0,
                "episode_id": "17",
                "steps_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    entry = {"index": 0, "episode_id": "17"}
    report = _validate_existing_report(report_path, steps_path, entry, "a" * 64)
    assert report["status"] == "passed"

    steps_path.write_text('{"step": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="inconsistent"):
        _validate_existing_report(report_path, steps_path, entry, "a" * 64)


def test_passing_resume_checkpoint_requires_a_step_hash(tmp_path):
    report_path = tmp_path / "episode.json"
    steps_path = tmp_path / "steps.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "passed",
                "manifest_sha256": "a" * 64,
                "index": 0,
                "episode_id": "17",
                "steps_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        _validate_existing_report(
            report_path,
            steps_path,
            {"index": 0, "episode_id": "17"},
            "a" * 64,
        )


def test_v1_checkpoint_cannot_be_mixed_into_v2_evidence(tmp_path):
    report_path = tmp_path / "episode.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "manifest_sha256": "a" * 64,
                "index": 0,
                "episode_id": "17",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        _validate_existing_report(
            report_path,
            tmp_path / "steps.jsonl",
            {"index": 0, "episode_id": "17"},
            "a" * 64,
        )


def test_partial_summary_cannot_look_complete():
    manifest = {
        "selection": {"target_episode_count": 2, "scene_count": 2},
    }
    reports = [
        {
            "status": "passed",
            "episode_id": "1",
            "scene_id": "scene-a",
            "ground_truth_replay": {
                "diagnostics": {"release_metadata_geodesic_within_tolerance": False},
                "observed": {
                    "simulator_collision_count_on_forward": 0,
                    "maximum_location_error_m": 0.0,
                    "release_metadata_geodesic_drift_m": 0.5,
                    "metrics": {"ndtw": 1.0, "success": 1.0, "spl": 1.0},
                },
            },
        }
    ]
    summary = _summary(reports, manifest)
    assert summary["completed_episode_count"] == 1
    assert summary["remaining_episode_count"] == 1
    assert summary["completed_scene_count"] == 1
    assert summary["release_metadata_geodesic_drift_episode_count"] == 1
    assert summary["maximum_release_metadata_geodesic_drift_m"] == pytest.approx(0.5)


def test_runner_completes_and_resumes_without_reexecuting(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    for relative in _RUNNER["_SOFTWARE_PATHS"]:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    build_lock = tmp_path / "habitat-sim-commit"
    build_lock.write_text(_RUNNER["HABITAT_SIM_COMMIT"], encoding="utf-8")
    dataset = _runner_dataset()
    loader = SimpleNamespace(from_release=lambda *args, **kwargs: dataset)
    runner_globals = _RUNNER["run"].__globals__
    monkeypatch.setitem(runner_globals, "R2RCEDataset", loader)
    monkeypatch.setitem(runner_globals, "HabitatSimR2REnvironment", _FakeHabitatEnvironment)
    monkeypatch.setattr(_RUNNER["platform"], "machine", lambda: "aarch64")
    _FakeHabitatEnvironment.construction_count = 0
    output_root = tmp_path / "artifacts"
    args = SimpleNamespace(
        repo_root=repo_root,
        scene_root=tmp_path / "scenes",
        episodes=tmp_path / "val_seen.json.gz",
        ground_truth=tmp_path / "val_seen_gt.json.gz",
        output_root=output_root,
        split="val_seen",
        max_hours=8.0,
        gpu_device_id=0,
        require_architecture="aarch64",
        habitat_build_lock=build_lock,
    )

    assert _RUNNER["run"](args) == 0
    suite = json.loads((output_root / "suite.json").read_text(encoding="utf-8"))
    assert suite["schema_version"] == 2
    assert suite["gate"].endswith("runtime_geodesic_v2")
    assert suite["status"] == "passed"
    assert suite["summary"]["completed_episode_count"] == 2
    assert len(list((output_root / "episodes").glob("*.json"))) == 2
    assert len(list((output_root / "steps").glob("*.jsonl"))) == 2
    assert _FakeHabitatEnvironment.construction_count == 1

    assert _RUNNER["run"](args) == 0
    assert _FakeHabitatEnvironment.construction_count == 1
