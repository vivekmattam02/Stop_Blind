"""No-network tests for the frozen Qwen metadata hydration boundary."""

from pathlib import Path

from scripts.hydrate_spark_qwen_metadata import _inventory, _snapshot_path
from spatialvlm.data.native_qwen_artifacts import NATIVE_QWEN_PROCESSOR_METADATA_FILES


def test_metadata_allow_list_cannot_select_model_weights() -> None:
    assert NATIVE_QWEN_PROCESSOR_METADATA_FILES
    assert all(
        not relative.endswith((".safetensors", ".bin"))
        for relative in NATIVE_QWEN_PROCESSOR_METADATA_FILES
    )


def test_inventory_records_presence_size_and_hash_without_following_missing_files(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    present = snapshot / NATIVE_QWEN_PROCESSOR_METADATA_FILES[0]
    present.write_text("metadata\n", encoding="utf-8")

    inventory = _inventory(snapshot)

    assert inventory[present.name]["exists"] is True
    assert inventory[present.name]["bytes"] == len(b"metadata\n")
    assert len(inventory[present.name]["sha256"]) == 64
    assert inventory[NATIVE_QWEN_PROCESSOR_METADATA_FILES[1]]["exists"] is False


def test_snapshot_path_uses_exact_full_revision(tmp_path: Path) -> None:
    revision = "a" * 40
    assert _snapshot_path(tmp_path, revision) == (
        tmp_path / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / revision
    )
