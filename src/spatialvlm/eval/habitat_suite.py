"""Deterministic manifest design for expanded Habitat-Sim conformance suites."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEGroundTruth

HABITAT_SUITE_SCHEMA_VERSION = 2
NO_DISPLACEMENT_TOLERANCE_M = 1e-4
SHORT_DISPLACEMENT_THRESHOLD_M = 0.249
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _episode_sort_key(episode_id: str) -> tuple[int, int, str]:
    try:
        return (0, int(episode_id), "")
    except ValueError:
        return (1, 0, episode_id)


def safe_episode_filename(episode_id: str) -> str:
    """Return a stable path component without trusting release IDs as paths."""
    if _SAFE_EPISODE_ID.fullmatch(episode_id) and episode_id not in {".", ".."}:
        return episode_id
    return f"sha256-{hashlib.sha256(episode_id.encode('utf-8')).hexdigest()[:20]}"


def released_motion_features(ground_truth: R2RCEGroundTruth) -> dict[str, Any]:
    """Describe released actions and forward-step displacements without inventing collisions."""
    if len(ground_truth.locations) != ground_truth.forward_steps + 1:
        raise ValueError(
            f"Ground truth {ground_truth.episode_id} must have reset plus one location per "
            "MOVE_FORWARD action."
        )
    displacements = [
        math.dist(start, end)
        for start, end in zip(ground_truth.locations[:-1], ground_truth.locations[1:])
    ]
    no_displacement = sum(value <= NO_DISPLACEMENT_TOLERANCE_M for value in displacements)
    short_displacement = sum(value < SHORT_DISPLACEMENT_THRESHOLD_M for value in displacements)
    return {
        "action_count": len(ground_truth.actions),
        "forward_steps": ground_truth.forward_steps,
        "turn_left_count": ground_truth.actions.count(R2RCEAction.TURN_LEFT),
        "turn_right_count": ground_truth.actions.count(R2RCEAction.TURN_RIGHT),
        "no_displacement_forward_count": no_displacement,
        "short_displacement_forward_count": short_displacement,
        "minimum_forward_displacement_m": min(displacements) if displacements else None,
        "maximum_forward_displacement_m": max(displacements) if displacements else None,
    }


def build_full_split_manifest(
    dataset: R2RCEDataset,
    *,
    expected_split: str = "val_seen",
    software_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a full-split, coverage-first manifest with deterministic stress representatives.

    One stress representative from every scene appears first. Remaining episodes are grouped by
    scene to reduce repeated scene loads. The manifest still contains every released episode.
    """
    release = dataset.release_manifest
    if release is None:
        raise ValueError("Expanded Habitat suite requires R2RCEDataset.from_release().")
    if release.split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, found {release.split!r}.")
    if release.selected_episode_count != release.source_episode_count:
        raise ValueError(
            "Expanded Habitat suite must use the complete split, not a limited selection."
        )
    episodes = dataset.official_episodes
    if not episodes:
        raise ValueError("Expanded Habitat suite cannot use an empty release split.")
    if len(dataset.ground_truth) != len(episodes):
        raise ValueError("Expanded Habitat suite requires ground truth for every episode.")

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        ground_truth = dataset.ground_truth_record(episode.episode_id)
        features = released_motion_features(ground_truth)
        rows_by_scene[episode.scene_id].append(
            {
                "episode_id": episode.episode_id,
                "trajectory_id": episode.trajectory_id,
                "scene_id": episode.scene_id,
                "features": features,
            }
        )

    def stress_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        features = row["features"]
        return (
            features["no_displacement_forward_count"],
            features["short_displacement_forward_count"],
            features["action_count"],
            features["turn_left_count"] + features["turn_right_count"],
            _episode_sort_key(str(row["episode_id"])),
        )

    representatives: list[dict[str, Any]] = []
    representative_ids: set[str] = set()
    for scene_id in sorted(rows_by_scene):
        representative = max(rows_by_scene[scene_id], key=stress_key)
        representatives.append(representative)
        representative_ids.add(representative["episode_id"])

    remaining = sorted(
        (
            row
            for scene_rows in rows_by_scene.values()
            for row in scene_rows
            if row["episode_id"] not in representative_ids
        ),
        key=lambda row: (row["scene_id"], _episode_sort_key(row["episode_id"])),
    )
    ordered = representatives + remaining
    entries = [
        {
            **row,
            "index": index,
            "phase": "scene_coverage_stress" if index < len(representatives) else "full_split",
            "artifact_stem": f"{index:04d}-{safe_episode_filename(row['episode_id'])}",
        }
        for index, row in enumerate(ordered)
    ]
    no_displacement_episode_count = sum(
        row["features"]["no_displacement_forward_count"] > 0 for row in ordered
    )
    short_displacement_episode_count = sum(
        row["features"]["short_displacement_forward_count"] > 0 for row in ordered
    )
    payload: dict[str, Any] = {
        "schema_version": HABITAT_SUITE_SCHEMA_VERSION,
        "suite_id": "r2r_ce_val_seen_full_runtime_geodesic_v2",
        "selection": {
            "strategy": "one_stress_representative_per_scene_then_remaining_grouped_by_scene",
            "target": "complete_release_split",
            "target_episode_count": len(entries),
            "scene_count": len(rows_by_scene),
            "coverage_prefix_count": len(representatives),
            "no_displacement_tolerance_m": NO_DISPLACEMENT_TOLERANCE_M,
            "short_displacement_threshold_m": SHORT_DISPLACEMENT_THRESHOLD_M,
            "no_displacement_candidate_episode_count": no_displacement_episode_count,
            "short_displacement_candidate_episode_count": short_displacement_episode_count,
        },
        "release": release.to_dict(),
        "software_sha256": dict(sorted((software_sha256 or {}).items())),
        "episodes": entries,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload
