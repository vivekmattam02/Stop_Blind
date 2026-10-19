#!/usr/bin/env python3
"""Validate an official R2R_VLNCE_v1-3 split and write a source-lock artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.r2r_ce_release import (
    R2R_CE_V1_3_SPLIT_COUNTS,
    build_r2r_ce_release_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--split", required=True, choices=sorted(R2R_CE_V1_3_SPLIT_COUNTS))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = R2RCEDataset.from_release(
        args.episodes,
        args.ground_truth,
        split=args.split,
    )
    report = build_r2r_ce_release_report(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
