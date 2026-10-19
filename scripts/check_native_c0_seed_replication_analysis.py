#!/usr/bin/env python3
"""Independently reconstruct the frozen three-seed native-C0 analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.analyze_native_c0_seed_replication import file_sha256, load_json, reconstruct

VERIFIER_GATE = "independent_native_qwen_c0_seed_replication_analysis_verification_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis_path = args.analysis.resolve()
    analysis = load_json(analysis_path, "seed-replication analysis")
    errors: list[str] = []
    try:
        rebuilt = reconstruct(
            args.repo.resolve(), recorded_at_utc=str(analysis.get("recorded_at_utc"))
        )
        if analysis != rebuilt:
            errors.append("seed-replication analysis does not independently reconstruct")
    except Exception as exc:
        errors.append(f"seed-replication reconstruction failed: {type(exc).__name__}: {exc}")
    payload = {
        "schema_version": 1,
        "gate": VERIFIER_GATE,
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_gate": analysis.get("gate"),
        "analysis_path": str(analysis_path),
        "analysis_sha256": file_sha256(analysis_path),
        "analysis_reconstruction_performed": True,
        "all_six_upstream_results_hash_linked": not errors,
        "all_preregistered_seeds_reported": not errors,
        "paired_initialization_reconstructed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
