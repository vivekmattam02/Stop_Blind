#!/usr/bin/env python3
"""Independently reconstruct a saved native-C0 closed-loop analysis artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def _analysis_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/spatialvlm/eval/native_closed_loop_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("gridnav_native_closed_loop_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = _analysis_module()
    analysis_path = args.analysis.resolve()
    observed = analysis.load_json(analysis_path)
    expected = analysis.analyze_closed_loop_artifacts(
        args.result_root.resolve(), args.control_root.resolve()
    )
    errors = [] if observed == expected else ["analysis does not exactly reconstruct"]
    payload = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "gate": "independent_native_qwen_c0_closed_loop_analysis_verification_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_count": len(errors),
        "errors": errors,
        "analysis_path": str(analysis_path),
        "analysis_sha256": analysis.sha256(analysis_path),
        "analysis_reconstruction_performed": True,
        "source_result_sha256": expected["source"]["result_sha256"],
        "source_verification_sha256": expected["source"]["verification_sha256"],
        "all_366_episode_reports_fully_hashed": (
            sum(
                len(records)
                for records in expected["source"]["episode_report_sha256"].values()
            )
            == 366
        ),
        "all_61_always_forward_control_reports_fully_hashed": (
            len(expected["source"]["always_forward_control_report_sha256"]) == 61
        ),
    }
    analysis.write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
