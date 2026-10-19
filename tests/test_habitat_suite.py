"""Tests for deterministic expanded Habitat-Sim suite manifests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.eval.habitat_suite import (
    build_full_split_manifest,
    released_motion_features,
    safe_episode_filename,
)


def _episode(episode_id: str, scene: str) -> R2RCEEpisode:
    return R2RCEEpisode(
        episode_id=episode_id,
        trajectory_id=f"trajectory-{episode_id}",
        scene_id=f"mp3d/{scene}/{scene}.glb",
        instruction_text="Navigate.",
        instruction_tokens=(),
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        goals=(R2RCEGoal(position=(1.0, 0.0, 0.0), radius=3.0),),
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        geodesic_distance=1.0,
        raw={"episode_id": episode_id},
    )


def _ground_truth(
    episode_id: str,
    locations: tuple[tuple[float, float, float], ...],
) -> R2RCEGroundTruth:
    actions = tuple(R2RCEAction.MOVE_FORWARD for _ in locations[1:]) + (R2RCEAction.STOP,)
    return R2RCEGroundTruth(
        episode_id=episode_id,
        actions=actions,
        forward_steps=len(locations) - 1,
        locations=locations,
    )


def _dataset(order: tuple[str, ...] = ("1", "2", "3", "4")) -> R2RCEDataset:
    episodes_by_id = {
        "1": _episode("1", "scene-a"),
        "2": _episode("2", "scene-a"),
        "3": _episode("3", "scene-b"),
        "4": _episode("4", "scene-b"),
    }
    ground_truth = {
        "1": _ground_truth("1", ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0))),
        "2": _ground_truth("2", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        "3": _ground_truth("3", ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0))),
        "4": _ground_truth(
            "4",
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.35, 0.0, 0.0)),
        ),
    }
    episodes = tuple(episodes_by_id[episode_id] for episode_id in order)
    manifest = R2RCEReleaseManifest(
        release_id="R2R_VLNCE_v1-3",
        split="val_seen",
        episodes_path="/datasets/val_seen.json.gz",
        episodes_sha256="a" * 64,
        source_episode_count=4,
        selected_episode_count=4,
        ground_truth_path="/datasets/val_seen_gt.json.gz",
        ground_truth_sha256="b" * 64,
        ground_truth_count=4,
    )
    return R2RCEDataset(
        records=[episode.raw for episode in episodes],
        official_episodes=episodes,
        ground_truth=ground_truth,
        release_manifest=manifest,
    )


def test_released_motion_features_distinguish_no_and_short_displacement():
    features = released_motion_features(
        _ground_truth(
            "x",
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.35, 0.0, 0.0)),
        )
    )
    assert features["no_displacement_forward_count"] == 1
    assert features["short_displacement_forward_count"] == 2


def test_manifest_covers_each_scene_first_then_every_episode():
    manifest = build_full_split_manifest(_dataset(), software_sha256={"code.py": "c" * 64})
    entries = manifest["episodes"]
    assert manifest["schema_version"] == 2
    assert manifest["suite_id"] == "r2r_ce_val_seen_full_runtime_geodesic_v2"
    assert [entry["episode_id"] for entry in entries] == ["2", "4", "1", "3"]
    assert [entry["phase"] for entry in entries[:2]] == [
        "scene_coverage_stress",
        "scene_coverage_stress",
    ]
    assert manifest["selection"]["target_episode_count"] == 4
    assert manifest["selection"]["scene_count"] == 2
    assert manifest["selection"]["no_displacement_candidate_episode_count"] == 1


def test_manifest_is_independent_of_release_row_order():
    first = build_full_split_manifest(_dataset())
    second = build_full_split_manifest(_dataset(("4", "2", "1", "3")))
    assert first == second


def test_manifest_rejects_a_limited_release_selection():
    dataset = _dataset()
    assert dataset.release_manifest is not None
    dataset.release_manifest = replace(dataset.release_manifest, selected_episode_count=3)
    with pytest.raises(ValueError, match="complete split"):
        build_full_split_manifest(dataset)


def test_safe_episode_filename_never_preserves_path_traversal():
    assert safe_episode_filename("episode-17") == "episode-17"
    assert safe_episode_filename("../../escape").startswith("sha256-")
