#!/usr/bin/env python3
"""Recurring autonomous checkpoint: verify artifacts against the filesystem, report live compute.

Written because this checkpoint is run repeatedly and re-typing it invites drift. Two rules it
enforces that were learned from failures in this project:

- A manifest is never evidence. Every entry that names a file is checked against the host
  filesystem; a build once reported ``status=completed`` with ``entry_count`` in the thousands and
  zero objects actually written, because the object root was not mounted into the container.
- Rate is measured between progress rows, not from ``elapsed_seconds / update``. The latter folds
  in the one-off startup pass that opens and validates every cache object, which inflated an ETA
  far enough to suggest a run would be truncated by its wall-clock cap when it would not.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
from pathlib import Path
from typing import Any

OBJECT_ROOT = Path.home() / "datasets/GridNav/checkpoints/geometry_depth_cache_v1"
FROZEN = {"ce": 0.6843, "update": 450, "constant": "189/240", "stop_recall": "0/4"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_artifacts(spark: Path) -> list[str]:
    """Check every manifest reference against the filesystem. Returns the disagreeing runs."""
    bad: list[str] = []
    for manifest_path in sorted(spark.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries = manifest.get("entries")
        if not (isinstance(entries, list) and entries and "relative_path" in entries[0]):
            continue
        missing = sum(1 for e in entries if not (OBJECT_ROOT / e["relative_path"]).exists())
        run = manifest_path.parent.name
        flag = "  <== DISAGREES" if missing else "  ok"
        print(f"  {run}: {len(entries):,} refs, {missing} missing{flag}")
        if missing:
            bad.append(run)

    for run_dir in sorted(spark.glob("*/episodes")):
        run = run_dir.parent.name
        on_disk = len(list(run_dir.glob("*.json")))
        result = run_dir.parent / "result.json"
        if not result.is_file():
            continue
        claimed = json.loads(result.read_text())["aggregate"]["episodes_evaluated"]
        flag = "  ok" if on_disk == claimed else "  <== DISAGREES"
        print(f"  {run}: claims {claimed} eps, {on_disk} on disk{flag}")
        if on_disk != claimed:
            bad.append(run)

    for result_path in sorted(spark.glob("*/result.json")):
        run = result_path.parent.name
        result = json.loads(result_path.read_text())
        if "completed_updates" not in result:
            continue
        adapters = len(glob.glob(str(result_path.parent / "adapter_u*.pt")))
        print(f"  {run}: {result['status']} {result['completed_updates']}u | {adapters} adapters")

    print(f"  objects on host: {len(glob.glob(str(OBJECT_ROOT / '*/*.pt'))):,}")
    return bad


def report_training(run_dir: Path, max_updates: int, cap_hours: float) -> list[str]:
    """Report progress, marginal rate, ETA and the calibration series. Returns flags raised."""
    progress = run_dir / "progress.jsonl"
    if not (progress.is_file() and progress.stat().st_size):
        print(f"  {run_dir.name}: launched, no progress row yet")
        return []
    rows = _rows(progress)
    first, last = rows[0], rows[-1]
    span = max(last["update"] - first["update"], 1)
    marginal = (last["elapsed_seconds"] - first["elapsed_seconds"]) / span
    projected = (first["elapsed_seconds"] + max_updates * marginal) / 3600.0
    print(
        f"  {run_dir.name}: update {last['update']}/{max_updates} | {marginal:.2f} s/update "
        f"(marginal) | {last['elapsed_seconds']/3600:.2f} h elapsed | "
        f"ETA {(max_updates - last['update']) * marginal/3600:.2f} h | "
        f"projected total {projected:.2f} h vs {cap_hours:g} h cap"
    )

    flags: list[str] = []
    if projected > cap_hours:
        flags.append(f"{run_dir.name} projects {projected:.2f} h, past its {cap_hours:g} h cap")
    nonfinite = [
        r["update"]
        for r in rows
        if not isinstance(r.get("train_chunk_ce_mean"), (int, float))
        or not math.isfinite(r["train_chunk_ce_mean"])
    ]
    if nonfinite:
        flags.append(f"{run_dir.name} non-finite train loss at updates {nonfinite[:5]}")
    print(f"  non-finite train-loss rows: {nonfinite or 'none'}")

    print(f"  {'':<8}{'CE':>8}{'acc':>8}{'const':>9}{'STOPpred':>10}{'STOPrec':>9}")
    print(
        f"  {'frozen':<8}{FROZEN['ce']:>8.4f}{'--':>8}{FROZEN['constant']:>9}{'--':>10}"
        f"{FROZEN['stop_recall']:>9}   (3,072 decisions, u{FROZEN['update']})"
    )
    cal = [r for r in rows if "calibration" in r]
    for r in cal:
        c = r["calibration"]
        stop = c["per_action"]["STOP"]
        print(
            f"  u{r['update']:<7}{c['selection_chunk_ce_mean']:>8.4f}"
            f"{c['position_accuracy']:>8.4f}{str(c['constant_predicted_chunks']) + '/240':>9}"
            f"{stop['predicted_count']:>10}"
            f"{str(stop['recall_numerator']) + '/' + str(stop['recall_denominator']):>9}"
        )
    if cal:
        best = min(cal, key=lambda r: r["calibration"]["selection_chunk_ce_mean"])
        total_stop = sum(r["calibration"]["per_action"]["STOP"]["predicted_count"] for r in cal)
        print(
            f"  min CE {best['calibration']['selection_chunk_ce_mean']:.4f} @u{best['update']} | "
            f"STOP predictions summed over {len(cal)} points: {total_stop}"
        )
    else:
        print("  (no calibration point yet)")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--run", type=str, default="native_chunk_scale_134k_v1")
    parser.add_argument("--container", type=str, default="scale_train_134k")
    parser.add_argument("--max-updates", type=int, default=3000)
    parser.add_argument("--cap-hours", type=float, default=10.0)
    args = parser.parse_args()
    spark = args.repo / "artifacts" / "spark"

    print("=== 1. MANIFEST vs FILESYSTEM ===")
    bad = verify_artifacts(spark)

    print("=== 2. CONTAINERS / GPU ===")
    ps = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}} {{.Status}}"], capture_output=True, text=True
    ).stdout
    live = [line for line in ps.splitlines() if line and "openshell" not in line]
    print("\n".join(f"  {line}" for line in live) or "  none running")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    mem = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", args.container],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  gpu: {gpu}" + (f" | container mem: {mem}" if mem else ""))

    print("=== 3-4. TRAINING vs FROZEN CURVE ===")
    flags = report_training(spark / args.run, args.max_updates, args.cap_hours)

    print("=== 5. FLAGS ===")
    if bad:
        flags.append(f"manifests disagreeing with the filesystem: {bad}")
    if not any(args.container in line for line in live):
        flags.append(f"container {args.container} is not running")
    print("  " + ("\n  ".join(flags) if flags else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
