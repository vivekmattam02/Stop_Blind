#!/usr/bin/env python3
"""Freeze and render the separate resumable content-addressed native-C0 H4 cache."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.run_spark_native_history_select import (
    EXPECTED_DOCKER_IMAGE_ID,
    _atomic_json,
    _load_dataset,
    _render_cache_entry,
)
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_CACHE_GATE,
    NATIVE_C0_CACHE_SCHEMA_VERSION,
    MP3D_HABITAT_ARCHIVE_SHA256,
    NativeC0DevelopmentProtocol,
    cache_object_identity,
    cache_object_key,
    canonical_sha256,
    native_c0_render_contract,
    select_native_c0_partitions,
    validate_partition_records,
)
from spatialvlm.training.native_c0_gate import file_sha256, tensor_sha256

MEASUREMENTS_GATE = "native_c0_development_measurements_v2"
MEASUREMENTS_VERIFIER_GATE = "independent_native_c0_development_measurements_verification_v2"
_STOP_REQUESTED = False

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_DEVELOPMENT_PROTOCOL.md",
    "docs/NATIVE_C0_DEVELOPMENT_PROTOCOL_DRAFT.md",
    "scripts/run_spark_native_c0_development_cache.py",
    "scripts/run_spark_native_history_select.py",
    "scripts/check_native_c0_development_cache_artifacts.py",
    "scripts/run_spark_native_c0_development_cache.sh",
    "scripts/start_spark_native_c0_development_cache.sh",
    "scripts/native_c0_development_sync_files.txt",
    "scripts/spark_launch_helpers.sh",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/training/native_c0_development.py",
    "src/spatialvlm/training/native_history_select.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_c0_development.py",
    "tests/test_native_c0_development_cache_artifacts.py",
)


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Termination requested; stopping after the current complete cache object.", flush=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required {label} must be a JSON object: {resolved}")
    return payload


def _measurement_evidence(args: argparse.Namespace) -> dict[str, Any]:
    result = _load_json(args.measurements_result, "measurement result")
    verification = _load_json(args.measurements_verification, "measurement verification")
    if result.get("gate") != MEASUREMENTS_GATE or result.get("status") != "passed":
        raise RuntimeError("The measured native-C0 runtime artifact has not passed.")
    if (
        verification.get("gate") != MEASUREMENTS_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("result_sha256") != file_sha256(args.measurements_result)
    ):
        raise RuntimeError("The measured native-C0 runtime artifact lacks linked verification.")
    measured_cache_decisions = int(result["protocol"]["unique_train_decisions"]) + int(
        result["protocol"]["unique_calibration_decisions"]
    )
    measured = {
        "cache_prepare_seconds_per_decision": float(
            result["timing"]["cache_prepare_seconds"]
        )
        / measured_cache_decisions,
        "cache_prepare_decision_count": measured_cache_decisions,
        "model_load_seconds": float(result["timing"]["model_load_seconds"]),
        "timed_update_p95_nearest_rank_seconds": float(
            result["timing"]["timed_update_p95_nearest_rank_seconds"]
        ),
        "calibration_seconds_per_decision": float(
            result["timing"]["calibration_seconds_per_decision"]
        ),
        "checkpoint_seconds": float(result["timing"]["checkpoint_seconds"]),
        "peak_allocated_bytes": int(result["runtime"]["peak_allocated_bytes"]),
    }
    return {
        "result_path": str(args.measurements_result.expanduser().resolve()),
        "result_sha256": file_sha256(args.measurements_result),
        "verification_path": str(args.measurements_verification.expanduser().resolve()),
        "verification_sha256": file_sha256(args.measurements_verification),
        "measured_constants": measured,
    }


def _scene_release_evidence(args: argparse.Namespace) -> dict[str, Any]:
    path = args.data_bootstrap_manifest.expanduser().resolve()
    manifest = _load_json(path, "R2R-CE/MP3D data-bootstrap manifest")
    archive = manifest.get("archives", {}).get("mp3d_habitat.zip", {})
    dataset = manifest.get("datasets", {}).get("mp3d", {})
    if (
        manifest.get("status") != "passed"
        or archive.get("sha256") != MP3D_HABITAT_ARCHIVE_SHA256
        or int(dataset.get("scene_count", -1)) != 90
    ):
        raise RuntimeError("The Matterport Habitat scene release lacks passing locked evidence.")
    return {
        "data_bootstrap_manifest_path": str(path),
        "data_bootstrap_manifest_sha256": file_sha256(path),
        "archive_name": "mp3d_habitat.zip",
        "archive_sha256": MP3D_HABITAT_ARCHIVE_SHA256,
        "scene_count": 90,
    }
def _expected_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted Habitat/Qwen image.")
    protocol = NativeC0DevelopmentProtocol()
    dataset = _load_dataset(args)
    samples = [row.to_record() for row in select_native_c0_partitions(dataset, protocol=protocol)]
    partition_summary = validate_partition_records(samples, protocol=protocol)
    episodes_sha = file_sha256(args.episodes)
    ground_truth_sha = file_sha256(args.ground_truth)
    scene_release = _scene_release_evidence(args)
    objects = [
        {
            "split": sample["split"],
            "sample_index": sample["sample_index"],
            "sample_sha256": canonical_sha256(sample),
            "cache_key": cache_object_key(
                sample,
                episodes_sha256=episodes_sha,
                ground_truth_sha256=ground_truth_sha,
                scene_archive_sha256=scene_release["archive_sha256"],
            ),
            "object_identity": cache_object_identity(
                sample,
                episodes_sha256=episodes_sha,
                ground_truth_sha256=ground_truth_sha,
                scene_archive_sha256=scene_release["archive_sha256"],
            ),
        }
        for sample in samples
    ]
    manifest = {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "gate": NATIVE_C0_CACHE_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": {
            "release_manifest": asdict(dataset.release_manifest),
            "episodes_path": str(args.episodes.expanduser().resolve()),
            "episodes_sha256": episodes_sha,
            "ground_truth_path": str(args.ground_truth.expanduser().resolve()),
            "ground_truth_sha256": ground_truth_sha,
            "split_used": "train",
            "sealed_splits": ["val_seen", "val_unseen"],
            "scene_release": scene_release,
        },
        "samples": samples,
        "sample_records_sha256": canonical_sha256(samples),
        "partition_summary": partition_summary,
        "render_contract": native_c0_render_contract(),
        "object_references": objects,
        "object_reference_sha256": canonical_sha256(objects),
        "cache_contract": {
            "content_addressed": True,
            "key_excludes": ["run_id", "split", "sample_index", "host_paths"],
            "key_includes": [
                "release_hashes",
                "scene_id",
                "physical_trajectory_id",
                "action_step_index",
                "replay_prefix",
                "simulator_sensor_history_contract",
            ],
            "resumable_at_object_boundary": True,
            "immutable_objects": True,
            "licensed_derived_rgb_cache": True,
            "contains_depth": False,
            "contains_simulator_pose": False,
        },
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "fully_offline": True,
        },
        "measurements": _measurement_evidence(args),
        "source_locks": _source_locks(args.repo.expanduser().resolve()),
        "evidence_boundary": {
            "established_if_passed": (
                "All 4,224 frozen train-release decisions have validated immutable H4 RGB "
                "objects keyed by decision plus release and simulator/render identity."
            ),
            "not_established": (
                "Qwen loading, optimization, checkpoint selection, C0 learning, closed-loop "
                "navigation, or val_unseen performance."
            ),
        },
    }
    return manifest, dataset, samples


def _freeze(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    expected, _, _ = _expected_manifest(args)
    if path.exists():
        existing = _load_json(path, "cache manifest")
        if existing != expected:
            raise RuntimeError("Existing C0 cache manifest differs from reconstruction.")
        print(f"Reusing identical frozen C0 cache manifest: {path}")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _object_path(object_root: Path, key: str) -> Path:
    return object_root / "objects" / key[:2] / f"{key}.pt"


def _validate_object(path: Path, expected_identity: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != NATIVE_C0_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Cache object schema mismatch: {path}")
    if payload.get("gate") != NATIVE_C0_CACHE_GATE:
        raise RuntimeError(f"Cache object gate mismatch: {path}")
    if payload.get("cache_key") != key or payload.get("object_identity") != dict(expected_identity):
        raise RuntimeError(f"Cache object identity mismatch: {path}")
    if canonical_sha256(dict(expected_identity)) != key:
        raise RuntimeError(f"Cache object key does not reconstruct: {path}")
    frames = payload.get("rgb_frames")
    if (
        not isinstance(frames, torch.Tensor)
        or frames.dtype != torch.uint8
        or frames.ndim != 4
        or tuple(frames.shape) != (4, 224, 224, 3)
    ):
        raise RuntimeError(f"Cache object RGB tensor is invalid: {path}")
    if payload.get("rgb_frames_sha256") != tensor_sha256(frames):
        raise RuntimeError(f"Cache object RGB content hash mismatch: {path}")
    actions = payload.get("history_action_ids")
    if not isinstance(actions, (list, tuple)) or len(actions) != 3 or 0 in actions:
        raise RuntimeError(f"Cache object history-action window is invalid: {path}")
    return payload


def load_content_cache_object(
    entry: Mapping[str, Any], object_root: Path, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Load one object from a verified cache-index entry."""
    path = object_root / str(entry["relative_path"])
    if not path.is_file() or file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Content cache object file identity mismatch: {path}")
    return _validate_object(path, expected_identity, str(entry["cache_key"]))


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_root = args.output_root.expanduser().resolve()
    object_root = args.object_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Freeze the C0 cache manifest before rendering.")
    manifest = _load_json(manifest_path, "cache manifest")
    expected, dataset, samples = _expected_manifest(args)
    if manifest != expected:
        raise RuntimeError("C0 cache manifest changed or no longer reconstructs.")
    manifest_sha = file_sha256(manifest_path)
    episodes = {episode.episode_id: episode for episode in dataset.official_episodes}
    references = {
        (row["split"], int(row["sample_index"])): row
        for row in manifest["object_references"]
    }
    ordered = sorted(
        samples,
        key=lambda row: (
            row["scene_id"],
            references[(row["split"], int(row["sample_index"]))]["cache_key"],
        ),
    )
    object_root.mkdir(parents=True, exist_ok=True)
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root.expanduser().resolve())
    )
    entries_by_sample: dict[tuple[str, int], dict[str, Any]] = {}
    rendered_count = 0
    reused_count = 0
    hard_cap_seconds = NativeC0DevelopmentProtocol().hard_total_hours * 3600
    try:
        for ordinal, sample in enumerate(ordered, start=1):
            if _STOP_REQUESTED:
                raise RuntimeError("C0 cache preparation was interrupted at an object boundary.")
            if time.perf_counter() - started >= hard_cap_seconds:
                raise TimeoutError(
                    "C0 cache reached the frozen 4.5-hour end-to-end recovery boundary. "
                    "Completed objects remain resumable."
                )
            reference = references[(sample["split"], int(sample["sample_index"]))]
            key = reference["cache_key"]
            identity = reference["object_identity"]
            path = _object_path(object_root, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                payload = _validate_object(path, identity, key)
                reused_count += 1
            else:
                legacy = _render_cache_entry(
                    environment,
                    episodes[sample["episode_id"]],
                    sample,
                    gate_name=NATIVE_C0_CACHE_GATE,
                )
                payload = {
                    "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
                    "gate": NATIVE_C0_CACHE_GATE,
                    "cache_key": key,
                    "object_identity": identity,
                    "rgb_frames": legacy["rgb_frames"],
                    "rgb_frames_sha256": legacy["rgb_frames_sha256"],
                    "history_action_ids": legacy["history_action_ids"],
                    "policy_visible_fields": legacy["policy_visible_fields"],
                    "explicitly_excluded_policy_fields": legacy[
                        "explicitly_excluded_policy_fields"
                    ],
                }
                temporary = path.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
                _validate_object(path, identity, key)
                rendered_count += 1
            relative = str(path.relative_to(object_root))
            entries_by_sample[(sample["split"], int(sample["sample_index"]))] = {
                "split": sample["split"],
                "sample_index": int(sample["sample_index"]),
                "sample_sha256": canonical_sha256(sample),
                "cache_key": key,
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rgb_frames_sha256": payload["rgb_frames_sha256"],
            }
            if ordinal % 16 == 0 or ordinal == len(ordered):
                _atomic_json(
                    output_root / "progress.json",
                    {
                        "status": "running",
                        "completed_objects": ordinal,
                        "target_objects": len(ordered),
                        "rendered_objects": rendered_count,
                        "reused_objects": reused_count,
                        "elapsed_seconds": time.perf_counter() - started,
                        "recorded_at_utc": _utc_now(),
                    },
                )
                print(f"[cache {ordinal}/{len(ordered)}] {sample['scene_id']}", flush=True)
    finally:
        environment.close()

    entries = [
        entries_by_sample[(sample["split"], int(sample["sample_index"]))]
        for sample in samples
    ]
    index = {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "gate": NATIVE_C0_CACHE_GATE,
        "run_id": args.run_id,
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "object_root": str(object_root),
        "entry_count": len(entries),
        "entries": entries,
        "entry_index_sha256": canonical_sha256(entries),
        "rendered_object_count_this_invocation": rendered_count,
        "reused_object_count_this_invocation": reused_count,
        "content_addressed": True,
    }
    _atomic_json(output_root / "cache_index.json", index)
    result = {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "gate": NATIVE_C0_CACHE_GATE,
        "run_id": args.run_id,
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "manifest_sha256": manifest_sha,
        "cache_index_path": str(output_root / "cache_index.json"),
        "cache_index_sha256": file_sha256(output_root / "cache_index.json"),
        "object_count": len(entries),
        "rendered_object_count_this_invocation": rendered_count,
        "reused_object_count_this_invocation": reused_count,
        "elapsed_seconds": time.perf_counter() - started,
        "content_addressed": True,
        "resumable_at_object_boundary": True,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    _atomic_json(output_root / "result.json", result)
    _atomic_json(
        output_root / "progress.json",
        {
            "status": "passed",
            "completed_objects": len(entries),
            "target_objects": len(entries),
            "rendered_objects": rendered_count,
            "reused_objects": reused_count,
            "elapsed_seconds": result["elapsed_seconds"],
            "recorded_at_utc": _utc_now(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--data-bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    _common(freeze)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--scene-root", type=Path, required=True)
    run.add_argument("--object-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        return _freeze(args) if args.command == "freeze" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
            "gate": NATIVE_C0_CACHE_GATE,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_root = getattr(args, "output_root", None)
        if isinstance(output_root, Path):
            try:
                _atomic_json(output_root.expanduser().resolve() / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
