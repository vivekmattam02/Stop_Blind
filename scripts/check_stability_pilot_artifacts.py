#!/usr/bin/env python3
"""Independently verify a complete Spark optimization/stability pilot."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.training.real_batch_gate import file_sha256
from spatialvlm.training.stability_pilot_artifacts import (
    STABILITY_PILOT_VERIFIER_GATE_NAME,
    STABILITY_PILOT_VERIFIER_SCHEMA_VERSION,
    verify_stability_pilot,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.episodes is None) != (args.ground_truth is None):
        parser.error("--episodes and --ground-truth must be supplied together")

    suite_path = args.suite.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = None
    if args.episodes is not None and args.ground_truth is not None:
        dataset = R2RCEDataset.from_release(
            args.episodes,
            args.ground_truth,
            split="train",
        )
    errors = verify_stability_pilot(
        suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=args.arms_root.expanduser().resolve(),
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        repo=args.repo.expanduser().resolve(),
        dataset=dataset,
    )
    payload = {
        "schema_version": STABILITY_PILOT_VERIFIER_SCHEMA_VERSION,
        "gate": STABILITY_PILOT_VERIFIER_GATE_NAME,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "suite_path": str(suite_path),
        "suite_sha256": file_sha256(suite_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_reconstruction_performed": dataset is not None,
        "checkpoint_root": str(args.checkpoint_root.expanduser().resolve()),
        "error_count": len(errors),
        "errors": errors,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
