#!/usr/bin/env python3
"""Independently verify a native-Qwen C0 history processor preflight artifact."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME,
    NATIVE_QWEN_HISTORY_VERIFIER_SCHEMA_VERSION,
    file_sha256,
    verify_native_qwen_history_preflight,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("Preflight result must be a JSON object.")
    errors = verify_native_qwen_history_preflight(
        result, repo=args.repo, model_snapshot=args.model_snapshot
    )
    payload = {
        "schema_version": NATIVE_QWEN_HISTORY_VERIFIER_SCHEMA_VERSION,
        "gate": NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "error_count": len(errors),
        "errors": errors,
        "result_path": str(args.result.resolve()),
        "result_sha256": file_sha256(args.result),
        "repo": str(args.repo.resolve()),
        "model_snapshot": str(args.model_snapshot.resolve()),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
