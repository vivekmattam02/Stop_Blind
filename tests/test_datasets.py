"""Tests for dataset loaders in the data pipeline."""

from __future__ import annotations

import gzip
import json

import pytest

from spatialvlm.data.datasets import (
    R2R_CE_RELEASE_ID,
    R2RCEAction,
    R2RCEDataset,
    RxRCEDataset,
    SQA3DDataset,
    _read_json_or_jsonl,
    build_dataset,
    iter_instructions,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_json_gz(path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _official_episode(episode_id: int, start_x: float = 0.0) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id + 100,
        "scene_id": "mp3d/example/example.glb",
        "instruction": {
            "instruction_text": f"Walk to goal {episode_id}.",
            "instruction_tokens": [2, 3, 0],
        },
        "start_position": [start_x, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [start_x + 2.0, 0.0, 0.0], "radius": 3.0}],
        "reference_path": [
            [start_x, 0.0, 0.0],
            [start_x + 1.0, 0.0, 0.0],
            [start_x + 2.0, 0.0, 0.0],
        ],
        "info": {"geodesic_distance": 2.0},
    }


def _official_gt(start_x: float = 0.0) -> dict:
    return {
        "actions": [1, 1, 0],
        "forward_steps": 2,
        "locations": [
            [start_x, 0.0, 0.0],
            [start_x + 1.0, 0.0, 0.0],
            [start_x + 2.0, 0.0, 0.0],
        ],
    }


def test_read_jsonl_records(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "go left", "episode_id": "1"}),
                json.dumps({"instruction": "go right", "episode_id": "2"}),
            ]
        ),
        encoding="utf-8",
    )
    rows = _read_json_or_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["instruction"] == "go left"


def test_read_json_episodes_structure(tmp_path):
    path = tmp_path / "episodes.json"
    _write_json(
        path,
        {
            "episodes": [
                {"instruction": "walk to chair", "episode_id": "e1"},
                {"instruction": "turn around", "episode_id": "e2"},
            ]
        },
    )
    rows = _read_json_or_jsonl(path)
    assert len(rows) == 2
    assert rows[1]["episode_id"] == "e2"


def test_r2r_dataset_split_and_limit(tmp_path):
    path = tmp_path / "r2r.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "a", "episode_id": "1", "split": "train"}),
                json.dumps({"instruction": "b", "episode_id": "2", "split": "val"}),
                json.dumps({"instruction": "c", "episode_id": "3", "split": "train"}),
            ]
        ),
        encoding="utf-8",
    )
    ds = R2RCEDataset.from_file(path, split="train", limit=1)
    assert len(ds) == 1
    sample = ds[0]
    assert sample.source == "r2r-ce"
    assert sample.instruction == "a"


def test_r2r_v1_3_release_adapter_cross_checks_schema_ids_and_hashes(tmp_path):
    episodes_path = tmp_path / "train.json.gz"
    gt_path = tmp_path / "train_gt.json.gz"
    _write_json_gz(
        episodes_path,
        {
            "episodes": [_official_episode(1), _official_episode(2, start_x=10.0)],
            "instruction_vocab": {"word_list": ["<pad>", "walk"]},
        },
    )
    _write_json_gz(gt_path, {"1": _official_gt(), "2": _official_gt(start_x=10.0)})

    dataset = R2RCEDataset.from_release(
        episodes_path,
        gt_path,
        split="train",
        limit=1,
    )

    assert len(dataset) == 1
    assert dataset[0].instruction == "Walk to goal 1."
    assert dataset.episode_record("1").trajectory_id == "101"
    assert dataset.ground_truth_record(1).actions[-1] == R2RCEAction.STOP
    assert dataset.ground_truth_record(1).forward_steps == 2
    assert dataset.release_manifest is not None
    assert dataset.release_manifest.release_id == R2R_CE_RELEASE_ID
    assert dataset.release_manifest.source_episode_count == 2
    assert dataset.release_manifest.selected_episode_count == 1
    assert len(dataset.release_manifest.episodes_sha256) == 64
    assert len(dataset.release_manifest.ground_truth_sha256 or "") == 64


def test_r2r_v1_3_release_adapter_rejects_episode_gt_id_mismatch(tmp_path):
    episodes_path = tmp_path / "val_seen.json.gz"
    gt_path = tmp_path / "val_seen_gt.json.gz"
    _write_json_gz(
        episodes_path,
        {
            "episodes": [_official_episode(1)],
            "instruction_vocab": {"word_list": ["<pad>"]},
        },
    )
    _write_json_gz(gt_path, {"2": _official_gt()})

    with pytest.raises(ValueError, match="ID mismatch"):
        R2RCEDataset.from_release(episodes_path, gt_path, split="val_seen")


def test_r2r_v1_3_release_adapter_rejects_invalid_action_id(tmp_path):
    episodes_path = tmp_path / "train.json.gz"
    gt_path = tmp_path / "train_gt.json.gz"
    _write_json_gz(
        episodes_path,
        {
            "episodes": [_official_episode(1)],
            "instruction_vocab": {"word_list": ["<pad>"]},
        },
    )
    invalid_gt = _official_gt()
    invalid_gt["actions"] = [1, 7, 0]
    invalid_gt["forward_steps"] = 1
    _write_json_gz(gt_path, {"1": invalid_gt})

    with pytest.raises(ValueError, match="unsupported ID 7"):
        R2RCEDataset.from_release(episodes_path, gt_path, split="train")


def test_r2r_v1_3_release_adapter_rejects_start_position_mismatch(tmp_path):
    episodes_path = tmp_path / "train.json.gz"
    gt_path = tmp_path / "train_gt.json.gz"
    _write_json_gz(
        episodes_path,
        {
            "episodes": [_official_episode(1)],
            "instruction_vocab": {"word_list": ["<pad>"]},
        },
    )
    _write_json_gz(gt_path, {"1": _official_gt(start_x=5.0)})

    with pytest.raises(ValueError, match="start does not match"):
        R2RCEDataset.from_release(episodes_path, gt_path, split="train")


def test_dataset_accepts_instruction_lists(tmp_path):
    path = tmp_path / "rxr.json"
    _write_json(
        path,
        [{"instructions": ["first instruction", "fallback"], "episode_id": "42", "split": "train"}],
    )
    ds = RxRCEDataset.from_file(path, split="train")
    assert ds[0].instruction == "first instruction"


def test_missing_instruction_raises(tmp_path):
    path = tmp_path / "broken.json"
    _write_json(path, [{"episode_id": "1"}])
    ds = SQA3DDataset.from_file(path)
    with pytest.raises(ValueError, match="Missing instruction"):
        _ = ds[0]


def test_build_dataset_aliases_and_iter_instructions(tmp_path):
    path = tmp_path / "sqa3d.json"
    _write_json(path, [{"instruction": "where is the sofa?", "id": "x"}])

    ds = build_dataset("vln-r2r", path)
    assert isinstance(ds, R2RCEDataset)

    ds2 = build_dataset("sqa3d", path)
    instructions = list(iter_instructions(ds2))
    assert instructions == ["where is the sofa?"]


def test_build_dataset_unsupported_name(tmp_path):
    path = tmp_path / "data.json"
    _write_json(path, [{"instruction": "x"}])
    with pytest.raises(ValueError, match="Unsupported dataset name"):
        build_dataset("unknown-dataset", path)
