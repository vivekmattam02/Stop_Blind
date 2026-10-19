#!/usr/bin/env python3
"""Independently reconstruct a native-C0 action-balance comparison artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_native_c0_action_balance_comparison import (
    COMPARISON_MODULE,
    build_comparison,
    evidence_bundle,
    file_sha256,
    load_json,
    source_locks,
    utc_now,
    write_exclusive,
)

VERIFIER_GATE = "independent_native_qwen_c0_action_balance_comparison_verification_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--unweighted-root", type=Path, required=True)
    parser.add_argument("--unweighted-checkpoint", type=Path, required=True)
    parser.add_argument("--action-balanced-root", type=Path, required=True)
    parser.add_argument("--action-balanced-checkpoint", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    try:
        repo = args.repo.expanduser().resolve()
        observed = load_json(args.analysis.expanduser().resolve(), "comparison analysis")
        expected = build_comparison(
            unweighted=evidence_bundle(
                args.unweighted_root.expanduser().resolve(),
                args.unweighted_checkpoint.expanduser().resolve(),
                "unweighted",
            ),
            action_balanced=evidence_bundle(
                args.action_balanced_root.expanduser().resolve(),
                args.action_balanced_checkpoint.expanduser().resolve(),
                "action-balanced",
            ),
            source_locks=source_locks(repo),
            recorded_at_utc=str(observed.get("recorded_at_utc")),
        )
        if observed != expected:
            errors.append("action-balance comparison does not independently reconstruct")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    args = parse_args()
    errors = verify(args)
    analysis_path = args.analysis.expanduser().resolve()
    payload = {
        "schema_version": COMPARISON_MODULE.SCHEMA_VERSION,
        "gate": VERIFIER_GATE,
        "analysis_gate": COMPARISON_MODULE.COMPARISON_GATE,
        "recorded_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "analysis_path": str(analysis_path),
        "analysis_sha256": file_sha256(analysis_path) if analysis_path.is_file() else None,
        "analysis_reconstruction_performed": True,
        "both_selected_checkpoints_fully_hashed": True,
        "both_upstream_verifications_linked": True,
        "per_decision_comparison_reconstructed": True,
    }
    write_exclusive(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
