#!/usr/bin/env python3
"""Hydrate only frozen Qwen processor/tokenizer metadata into the Spark HF cache."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_MODEL_ID,
    NATIVE_QWEN_HISTORY_MODEL_REVISION,
    NATIVE_QWEN_METADATA_HYDRATION_GATE_NAME,
    NATIVE_QWEN_METADATA_HYDRATION_SCHEMA_VERSION,
    NATIVE_QWEN_PROCESSOR_METADATA_FILES,
    file_sha256,
    processor_metadata_locks,
    source_locks,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _snapshot_path(hf_home: Path, revision: str) -> Path:
    return hf_home / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / revision


def _inventory(snapshot: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for relative in NATIVE_QWEN_PROCESSOR_METADATA_FILES:
        path = snapshot / relative
        record: dict[str, Any] = {
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_symlink": path.is_symlink(),
        }
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
        inventory[relative] = record
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default=NATIVE_QWEN_HISTORY_MODEL_ID)
    parser.add_argument("--revision", default=NATIVE_QWEN_HISTORY_MODEL_REVISION)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    hf_home = args.hf_home.expanduser().resolve()
    output = args.output.expanduser().resolve()
    snapshot = _snapshot_path(hf_home, args.revision)
    payload: dict[str, Any] = {
        "schema_version": NATIVE_QWEN_METADATA_HYDRATION_SCHEMA_VERSION,
        "gate": NATIVE_QWEN_METADATA_HYDRATION_GATE_NAME,
        "run_id": args.run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "allow_patterns": list(NATIVE_QWEN_PROCESSOR_METADATA_FILES),
        "explicitly_excluded_patterns": ["*.safetensors", "*.bin"],
        "docker_image": {"reference": args.docker_image_ref, "id": args.docker_image_id},
        "snapshot_path": str(snapshot),
        "before": _inventory(snapshot),
        "source_locks": source_locks(repo),
        "evidence_boundary": {
            "established_if_passed": (
                "Only the nine named processor/tokenizer metadata files were requested at the "
                "frozen full Qwen commit, and every resulting file was hashed."
            ),
            "not_established": (
                "Processor construction, history rendering, model execution, training, or any "
                "navigation result."
            ),
        },
    }
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing hydration result: {output}")
        if args.model_id != NATIVE_QWEN_HISTORY_MODEL_ID:
            raise ValueError("Model ID differs from the frozen hydration contract.")
        if args.revision != NATIVE_QWEN_HISTORY_MODEL_REVISION:
            raise ValueError("Model revision differs from the frozen hydration contract.")

        downloaded = Path(
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                cache_dir=hf_home / "hub",
                allow_patterns=list(NATIVE_QWEN_PROCESSOR_METADATA_FILES),
            )
        ).resolve()
        if downloaded != snapshot.resolve():
            raise RuntimeError(
                "Hugging Face resolved an unexpected snapshot: "
                f"{downloaded} != {snapshot.resolve()}"
            )

        payload.update(
            {
                "status": "passed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "resolved_snapshot_path": str(downloaded),
                "after": _inventory(snapshot),
                "processor_metadata_locks": processor_metadata_locks(snapshot),
            }
        )
        _atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "after": _inventory(snapshot),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
