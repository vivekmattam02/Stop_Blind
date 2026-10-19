"""Tests for the R2R-CE release source-lock report."""

from __future__ import annotations

import gzip
import json

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.r2r_ce_release import (
    R2R_CE_V1_3_SPLIT_COUNTS,
    build_r2r_ce_release_report,
)


def _write_json_gz(path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_release_report_passes_locked_checks_for_conformant_fixture(tmp_path, monkeypatch):
    episodes_path = tmp_path / "val_seen.json.gz"
    gt_path = tmp_path / "val_seen_gt.json.gz"
    episode = {
        "episode_id": 1,
        "trajectory_id": 11,
        "scene_id": "mp3d/example/example.glb",
        "instruction": {"instruction_text": "Walk forward.", "instruction_tokens": [2, 0]},
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [1.0, 0.0, 0.0], "radius": 3.0}],
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "info": {"geodesic_distance": 1.0},
    }
    ground_truth = {
        "1": {
            "actions": [1, 0],
            "forward_steps": 1,
            "locations": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
    }
    _write_json_gz(
        episodes_path,
        {"episodes": [episode], "instruction_vocab": {"word_list": ["<pad>"]}},
    )
    _write_json_gz(gt_path, ground_truth)
    monkeypatch.setitem(R2R_CE_V1_3_SPLIT_COUNTS, "val_seen", 1)

    dataset = R2RCEDataset.from_release(
        episodes_path,
        gt_path,
        split="val_seen",
    )
    report = build_r2r_ce_release_report(dataset)

    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["observed"]["action_histogram"] == {"MOVE_FORWARD": 1, "STOP": 1}
