#!/usr/bin/env python3
"""Independently verify one completed Spark development SFT run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.training.development_sft_artifacts import (
    DEVELOPMENT_SFT_VERIFIER_GATE_NAME,
    DEVELOPMENT_SFT_VERIFIER_SCHEMA_VERSION,
    verify_development_sft,
)
from spatialvlm.training.real_batch_gate import file_sha256


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-cache-file-hashes",
        action="store_true",
        help="Development-only shortcut; never use for final evidence admission.",
    )
    parser.add_argument(
        "--skip-all-checkpoint-file-hashes",
        action="store_true",
        help="Development-only shortcut; selected checkpoint is still payload-checked.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = _load(args.result)
    manifest = _load(args.manifest)
    cache_manifest = _load(args.cache_manifest)
    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    errors = verify_development_sft(
        result,
        manifest=manifest,
        manifest_path=args.manifest,
        cache_manifest=cache_manifest,
        cache_manifest_path=args.cache_manifest,
        checkpoint_root=args.checkpoint_root,
        warmstart_checkpoint=args.warmstart_checkpoint,
        repo=args.repo,
        dataset=dataset,
        verify_cache_files=not args.skip_cache_file_hashes,
        verify_all_checkpoint_files=not args.skip_all_checkpoint_file_hashes,
    )
    payload = {
        "schema_version": DEVELOPMENT_SFT_VERIFIER_SCHEMA_VERSION,
        "gate": DEVELOPMENT_SFT_VERIFIER_GATE_NAME,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "error_count": len(errors),
        "errors": errors,
        "result_path": str(args.result.resolve()),
        "result_sha256": file_sha256(args.result),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "cache_manifest_path": str(args.cache_manifest.resolve()),
        "cache_manifest_sha256": file_sha256(args.cache_manifest),
        "warmstart_checkpoint_sha256": file_sha256(args.warmstart_checkpoint),
        "cache_files_fully_hashed": not args.skip_cache_file_hashes,
        "all_checkpoint_files_fully_hashed": not args.skip_all_checkpoint_file_hashes,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
