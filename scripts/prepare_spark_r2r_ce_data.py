#!/usr/bin/env python3
"""Prepare authorized MP3D Habitat scenes and official R2R-CE v1-3 data on Spark."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from spatialvlm.data.spark_bootstrap import (
    MP3D_HABITAT_ARCHIVE,
    R2R_CE_V1_3_ARCHIVE,
    BootstrapError,
    prepare_spark_data,
    require_mp3d_terms_acceptance,
)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-mp3d-terms",
        action="store_true",
        help="Confirm authorization and acceptance of the Matterport3D Terms of Use.",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets/GridNav")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "artifacts" / "spark" / "r2r_ce_data",
    )
    parser.add_argument("--image", default="gridnav-spark:cuda13-py25.10")
    parser.add_argument(
        "--inside-container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the locked plan without downloading or accepting terms.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.plan:
        print(f"MP3D: {MP3D_HABITAT_ARCHIVE.url}")
        print(f"MP3D bytes: {MP3D_HABITAT_ARCHIVE.expected_bytes}")
        print(f"R2R-CE: {R2R_CE_V1_3_ARCHIVE.url}")
        print(f"R2R-CE bytes: {R2R_CE_V1_3_ARCHIVE.expected_bytes}")
        print(f"Data root: {args.data_root}")
        print(f"Artifact root: {args.artifact_root}")
        print("No files were created or downloaded.")
        return 0
    try:
        require_mp3d_terms_acceptance(args.accept_mp3d_terms)
        manifest = prepare_spark_data(
            repo_root=args.repo_root.resolve(),
            data_root=args.data_root.resolve(),
            artifact_root=args.artifact_root.resolve(),
            image=args.image,
            inside_container=args.inside_container,
        )
    except (BootstrapError, OSError, subprocess.SubprocessError) as exc:
        print(f"DATA BOOTSTRAP FAILED: {exc}")
        return 1
    print(f"DATA BOOTSTRAP PASSED: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
