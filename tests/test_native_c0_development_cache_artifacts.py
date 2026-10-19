from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.run_spark_native_c0_development_cache import _object_path, _validate_object
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_CACHE_GATE,
    NATIVE_C0_CACHE_SCHEMA_VERSION,
    canonical_sha256,
)
from spatialvlm.training.native_c0_gate import tensor_sha256


def _payload(identity: dict[str, object]) -> dict[str, object]:
    frames = torch.zeros((4, 224, 224, 3), dtype=torch.uint8)
    return {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "gate": NATIVE_C0_CACHE_GATE,
        "cache_key": canonical_sha256(identity),
        "object_identity": identity,
        "rgb_frames": frames,
        "rgb_frames_sha256": tensor_sha256(frames),
        "history_action_ids": (1, 2, 3),
    }


def test_content_object_path_is_sharded_by_key_prefix(tmp_path: Path) -> None:
    key = "ab" + "0" * 62
    assert _object_path(tmp_path, key) == tmp_path / "objects" / "ab" / f"{key}.pt"


def test_content_object_validation_reconstructs_key_and_tensor_hash(tmp_path: Path) -> None:
    identity = {"decision": "physical", "render": {"rgb": [224, 224]}}
    payload = _payload(identity)
    path = tmp_path / "object.pt"
    torch.save(payload, path)
    observed = _validate_object(path, identity, str(payload["cache_key"]))
    assert observed["rgb_frames_sha256"] == payload["rgb_frames_sha256"]


def test_content_object_validation_rejects_identity_or_rgb_drift(tmp_path: Path) -> None:
    identity = {"decision": "physical"}
    payload = _payload(identity)
    path = tmp_path / "object.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_object(path, {"decision": "different"}, str(payload["cache_key"]))
    payload["rgb_frames"][0, 0, 0, 0] = 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="RGB content hash mismatch"):
        _validate_object(path, identity, str(payload["cache_key"]))
