"""CPU tests for the direct Habitat-Sim R2R-CE adapter boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from spatialvlm.data.habitat_sim_r2r import (
    HabitatSimR2RConfig,
    postprocess_habitat_observations,
    resolve_mp3d_scene_path,
)


def test_resolve_mp3d_scene_path_accepts_release_prefix_and_mp3d_root(tmp_path):
    scene = tmp_path / "mp3d" / "17DRP5sb8fy" / "17DRP5sb8fy.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"glb")

    from_dataset_root = resolve_mp3d_scene_path(
        tmp_path,
        "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
    )
    from_mp3d_root = resolve_mp3d_scene_path(tmp_path / "mp3d", "17DRP5sb8fy")
    assert from_dataset_root == scene.resolve()
    assert from_mp3d_root == scene.resolve()


def test_resolve_mp3d_scene_path_fails_closed_without_recursive_guessing(tmp_path):
    (tmp_path / "unrelated").mkdir()
    with pytest.raises(FileNotFoundError, match="Canonical MP3D scene"):
        resolve_mp3d_scene_path(tmp_path, "mp3d/missing/missing.glb")
    with pytest.raises(ValueError, match="dataset-relative"):
        resolve_mp3d_scene_path(tmp_path, "/tmp/escape.glb")


def test_postprocess_preserves_native_shapes_and_separates_metric_depth(tmp_path):
    config = HabitatSimR2RConfig(scene_root=tmp_path)
    rgb = np.zeros((224, 224, 4), dtype=np.uint8)
    depth = np.linspace(0.0, 12.0, 256 * 256, dtype=np.float32).reshape(256, 256)

    result = postprocess_habitat_observations({"rgb": rgb, "depth": depth}, config)
    assert result["rgb"].shape == (224, 224, 3)
    assert result["rgb"].dtype == np.uint8
    assert result["depth"].shape == (256, 256, 1)
    assert result["depth"].dtype == np.float32
    assert result["depth"].min() >= 0.0
    assert result["depth"].max() <= 1.0
    assert result["depth_meters"].shape == (256, 256)
    assert result["depth_meters"].max() == pytest.approx(10.0)


def test_postprocess_rejects_legacy_518_square_observations(tmp_path):
    config = HabitatSimR2RConfig(scene_root=tmp_path)
    with pytest.raises(ValueError, match="Official R2R-CE RGB"):
        postprocess_habitat_observations(
            {
                "rgb": np.zeros((518, 518, 3), dtype=np.uint8),
                "depth": np.zeros((256, 256), dtype=np.float32),
            },
            config,
        )
