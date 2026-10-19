"""Source-lock reporting for the official R2R-CE v1-3 release."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from .datasets import R2R_CE_RELEASE_ID, R2RCEAction, R2RCEDataset

R2R_CE_V1_3_SPLIT_COUNTS = {
    "train": 10_819,
    "val_seen": 778,
    "val_unseen": 1_839,
    "test": 3_408,
}


def build_r2r_ce_release_report(dataset: R2RCEDataset) -> dict[str, Any]:
    """Build a deterministic conformance report for one official release split."""
    if dataset.release_manifest is None:
        raise ValueError("Release checker requires R2RCEDataset.from_release().")
    episodes = dataset.official_episodes
    ground_truth = dataset.ground_truth
    split = dataset.release_manifest.split
    expected_count = R2R_CE_V1_3_SPLIT_COUNTS.get(split)
    scene_ids = {episode.scene_id for episode in episodes}
    trajectory_ids = {episode.trajectory_id for episode in episodes}
    goal_radii = [goal.radius for episode in episodes for goal in episode.goals]
    ground_truth_expected = split != "test"
    action_histogram: Counter[str] = Counter()
    action_lengths: list[int] = []
    location_lengths: list[int] = []
    for record in ground_truth.values():
        action_histogram.update(action.name for action in record.actions)
        action_lengths.append(len(record.actions))
        location_lengths.append(len(record.locations))

    checks = {
        "release_id_matches": dataset.release_manifest.release_id == R2R_CE_RELEASE_ID,
        "split_count_matches": expected_count is None or len(episodes) == expected_count,
        "episode_ids_unique": len({episode.episode_id for episode in episodes}) == len(episodes),
        "trajectory_ids_present": len(trajectory_ids) > 0,
        "scene_ids_present": len(scene_ids) > 0,
        "goal_radii_are_three_metres": bool(goal_radii)
        and all(abs(radius - 3.0) <= 1e-6 for radius in goal_radii),
        "ground_truth_presence_matches_split": bool(ground_truth) == ground_truth_expected,
        "ground_truth_coverage_matches": (not ground_truth_expected and not ground_truth)
        or len(ground_truth) == len(episodes),
        "ground_truth_terminates_with_stop": (not ground_truth_expected and not ground_truth)
        or (
            bool(ground_truth)
            and all(record.actions[-1] == R2RCEAction.STOP for record in ground_truth.values())
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "release": dataset.release_manifest.to_dict(),
        "expected_source_episode_count": expected_count,
        "observed": {
            "episode_count": len(episodes),
            "scene_count": len(scene_ids),
            "trajectory_count": len(trajectory_ids),
            "goal_radii_m": sorted(set(goal_radii)),
            "action_histogram": dict(sorted(action_histogram.items())),
            "action_length_min": min(action_lengths) if action_lengths else None,
            "action_length_max": max(action_lengths) if action_lengths else None,
            "action_length_mean": statistics.fmean(action_lengths) if action_lengths else None,
            "location_length_min": min(location_lengths) if location_lengths else None,
            "location_length_max": max(location_lengths) if location_lengths else None,
            "location_length_mean": statistics.fmean(location_lengths)
            if location_lengths
            else None,
        },
        "checks": checks,
    }
