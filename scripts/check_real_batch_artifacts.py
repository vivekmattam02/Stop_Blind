#!/usr/bin/env python3
"""Independently verify a completed one-real-batch result and its compact checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.training.real_batch_artifacts import verify_real_batch_result
from spatialvlm.training.real_batch_gate import file_sha256


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.result.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors = verify_real_batch_result(
        result,
        repo=args.repo.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
    )
    payload = {
        "schema_version": 1,
        "gate": "independent_spark_real_batch_artifact_verification_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "checkpoint_path": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint.is_file() else None,
        "error_count": len(errors),
        "errors": errors,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
