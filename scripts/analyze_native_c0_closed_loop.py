#!/usr/bin/env python3
"""Generate deterministic episode-level analysis from verified native-C0 artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = _analysis_module()
    payload = analysis.analyze_closed_loop_artifacts(
        args.result_root.resolve(), args.control_root.resolve()
    )
    analysis.write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
