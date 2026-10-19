#!/usr/bin/env python3
"""Record a read-only DGX Spark host and repository preflight.

This script intentionally uses only the Python standard library so it can run
before the SpatialVLM environment exists. It does not install packages, modify
the Git checkout, download models, or start GPU work.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(name: str, command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    """Run one diagnostic command and return its complete result."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "name": name,
            "command": command,
            "status": "missing",
            "returncode": None,
            "stdout": "",
            "stderr": f"Executable not found: {command[0]}",
        }

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "name": name,
            "command": command,
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "name": name,
        "command": command,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def command_stdout(results: list[dict[str, Any]], name: str) -> str:
    """Return stdout for a named diagnostic command."""
    return next(result["stdout"] for result in results if result["name"] == name)


def command_ok(results: list[dict[str, Any]], name: str) -> bool:
    """Return whether a named diagnostic command succeeded."""
    return next(result["status"] == "ok" for result in results if result["name"] == name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Spark checkout to inspect (default: current directory).",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Git commit that the Spark checkout must match.",
    )
    parser.add_argument(
        "--expected-remote",
        default="https://github.com/Ashish-Reddy-T/elv.git",
        help="Expected origin URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spark/host_preflight.json"),
        help="JSON artifact path, relative to the repository unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.output if args.output.is_absolute() else repo / args.output

    results = [
        run_command("hostname", ["hostname"]),
        run_command("identity", ["id"]),
        run_command("kernel", ["uname", "-srvo"]),
        run_command("architecture", ["uname", "-m"]),
        run_command("cpu", ["lscpu"]),
        run_command("memory", ["free", "-h"]),
        run_command("disk", ["df", "-h", str(repo)]),
        run_command("python", ["python3", "--version"]),
        run_command("git_version", ["git", "--version"]),
        run_command("rsync_version", ["rsync", "--version"]),
        run_command("gpu_list", ["nvidia-smi", "-L"]),
        run_command(
            "gpu_driver",
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
        ),
        run_command("nvidia_smi", ["nvidia-smi"]),
        run_command("cuda_compiler", ["nvcc", "--version"]),
        run_command("docker_version", ["docker", "version"]),
        run_command("docker_runtime", ["docker", "info"]),
        run_command("git_commit", ["git", "rev-parse", "HEAD"], cwd=repo),
        run_command(
            "git_remote",
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo,
        ),
        run_command(
            "git_tracked_status",
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
        ),
        run_command(
            "git_untracked_files",
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo,
        ),
        run_command(
            "git_status_summary",
            ["git", "status", "--short", "--branch"],
            cwd=repo,
        ),
    ]

    architecture = command_stdout(results, "architecture")
    actual_commit = command_stdout(results, "git_commit")
    actual_remote = command_stdout(results, "git_remote")
    git_tracked_status = command_stdout(results, "git_tracked_status")
    checks = {
        "repo_exists": repo.is_dir(),
        "architecture_is_arm64": architecture in {"aarch64", "arm64"},
        "gpu_visible": command_ok(results, "gpu_list"),
        "docker_accessible": command_ok(results, "docker_runtime"),
        "git_commit_matches": actual_commit == args.expected_commit,
        "git_remote_matches": actual_remote.rstrip("/") == args.expected_remote.rstrip("/"),
        "git_tracked_worktree_clean": (
            command_ok(results, "git_tracked_status") and not git_tracked_status
        ),
    }

    os_release_path = Path("/etc/os-release")
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "expected_commit": args.expected_commit,
        "expected_remote": args.expected_remote,
        "platform": platform.platform(),
        "os_release": (
            os_release_path.read_text(encoding="utf-8") if os_release_path.exists() else None
        ),
        "checks": checks,
        "ready_for_environment_setup": all(checks.values()),
        "commands": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"DGX Spark preflight written to: {output}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not payload["ready_for_environment_setup"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
