"""Offline tests for the authorized Spark dataset bootstrap."""

from __future__ import annotations

import io
import json
from zipfile import ZipFile

import pytest

from spatialvlm.data.spark_bootstrap import (
    MP3D_HABITAT_ARCHIVE,
    MP3D_SCAN_IDS,
    R2R_CE_V1_3_ARCHIVE,
    ArchiveSpec,
    BootstrapError,
    download_archive,
    extract_zip_resumable,
    locate_mp3d_root,
    locate_r2r_ce_root,
    prepare_spark_data,
    require_mp3d_terms_acceptance,
    validate_mp3d_root,
    validate_r2r_ce_root,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int):
        super().__init__(payload)
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_locked_archive_metadata_and_scan_count():
    assert MP3D_HABITAT_ARCHIVE.expected_bytes == 16_085_306_031
    assert R2R_CE_V1_3_ARCHIVE.expected_bytes == 261_610_900
    assert len(MP3D_SCAN_IDS) == 90
    assert MP3D_HABITAT_ARCHIVE.url.startswith("https://")
    assert R2R_CE_V1_3_ARCHIVE.url.startswith("https://")


def test_terms_must_be_explicitly_accepted():
    with pytest.raises(BootstrapError, match="accept-mp3d-terms"):
        require_mp3d_terms_acceptance(False)
    require_mp3d_terms_acceptance(True)


def test_download_resumes_partial_file_and_promotes_only_at_locked_size(tmp_path, monkeypatch):
    destination = tmp_path / "source.zip"
    partial = tmp_path / "source.zip.part"
    partial.write_bytes(b"abc")
    spec = ArchiveSpec("source.zip", "https://example.test/source.zip", 6)

    def fake_urlopen(request, timeout):
        assert request.headers["Range"] == "bytes=3-"
        assert timeout == 60
        return _FakeResponse(b"def", status=206)

    monkeypatch.setattr("spatialvlm.data.spark_bootstrap.urlopen", fake_urlopen)

    assert download_archive(spec, destination) == destination
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_download_refuses_server_that_ignores_range_request(tmp_path, monkeypatch):
    destination = tmp_path / "source.zip"
    partial = tmp_path / "source.zip.part"
    partial.write_bytes(b"abc")
    spec = ArchiveSpec("source.zip", "https://example.test/source.zip", 6)
    monkeypatch.setattr(
        "spatialvlm.data.spark_bootstrap.urlopen",
        lambda request, timeout: _FakeResponse(b"abcdef", status=200),
    )

    with pytest.raises(BootstrapError, match="did not honor byte-range"):
        download_archive(spec, destination)
    assert partial.read_bytes() == b"abc"
    assert not destination.exists()


def test_resumable_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")

    with pytest.raises(BootstrapError, match="Unsafe ZIP member"):
        extract_zip_resumable(archive, tmp_path / "staging")
    assert not (tmp_path / "escape.txt").exists()


def test_locates_and_validates_canonical_mp3d_layout(tmp_path):
    root = tmp_path / "archive" / "mp3d"
    for scan_id in MP3D_SCAN_IDS:
        scene_dir = root / scan_id
        scene_dir.mkdir(parents=True)
        (scene_dir / f"{scan_id}.glb").write_bytes(b"glb")

    located = locate_mp3d_root(tmp_path)
    report = validate_mp3d_root(located)

    assert located == root.resolve()
    assert report["scene_count"] == 90


def test_mp3d_validation_fails_closed_on_missing_scene(tmp_path):
    root = tmp_path / "mp3d"
    root.mkdir()
    with pytest.raises(BootstrapError, match="missing 90"):
        validate_mp3d_root(root)


def test_locates_and_validates_r2r_split_layout(tmp_path):
    root = tmp_path / "R2R_VLNCE_v1-3_preprocessed"
    for split in ("train", "val_seen", "val_unseen"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / f"{split}.json.gz").write_bytes(b"episodes")
        (split_dir / f"{split}_gt.json.gz").write_bytes(b"ground truth")

    located = locate_r2r_ce_root(tmp_path)
    report = validate_r2r_ce_root(located)

    assert located == root.resolve()
    assert report["validated_splits"] == ["train", "val_seen", "val_unseen"]


def test_prepare_pipeline_promotes_validated_layouts_and_writes_manifest(tmp_path, monkeypatch):
    mp3d_archive = tmp_path / "fixture_mp3d.zip"
    with ZipFile(mp3d_archive, "w") as handle:
        for scan_id in MP3D_SCAN_IDS:
            handle.writestr(f"mp3d/{scan_id}/{scan_id}.glb", b"glb")

    r2r_archive = tmp_path / "fixture_r2r.zip"
    with ZipFile(r2r_archive, "w") as handle:
        for split in ("train", "val_seen", "val_unseen"):
            prefix = f"R2R_VLNCE_v1-3_preprocessed/{split}"
            handle.writestr(f"{prefix}/{split}.json.gz", b"episodes")
            handle.writestr(f"{prefix}/{split}_gt.json.gz", b"ground truth")

    def fake_download(spec, destination):
        return mp3d_archive if spec.name == MP3D_HABITAT_ARCHIVE.name else r2r_archive

    def fake_release_checks(**kwargs):
        artifact_root = kwargs["artifact_root"]
        artifact_root.mkdir(parents=True, exist_ok=True)
        outputs = []
        for split in ("train", "val_seen", "val_unseen"):
            output = artifact_root / f"{split}_source_lock.json"
            output.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            outputs.append(output)
        return outputs

    monkeypatch.setattr("spatialvlm.data.spark_bootstrap.download_archive", fake_download)
    monkeypatch.setattr("spatialvlm.data.spark_bootstrap.run_release_checks", fake_release_checks)
    monkeypatch.setattr(
        "spatialvlm.data.spark_bootstrap.ensure_free_space", lambda path: 100 * 1024**3
    )

    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    manifest_path = prepare_spark_data(
        repo_root=tmp_path,
        data_root=data_root,
        artifact_root=artifact_root,
        image="test-image",
        inside_container=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "passed"
    assert manifest["datasets"]["mp3d"]["scene_count"] == 90
    assert (data_root / "scene_datasets" / "mp3d" / "17DRP5sb8fy").is_dir()
    assert (
        data_root / "episodes" / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen" / "val_unseen.json.gz"
    ).is_file()
