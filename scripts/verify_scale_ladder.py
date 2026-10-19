#!/usr/bin/env python3
"""Independently re-derive every claim made about the nested scale ladder.

This exists because the code that produced these runs cannot be the code that validates them. It
reads only raw artifacts --- `progress.jsonl`, the sample manifests, and the per-episode closed-loop
files --- and **refuses to open any `result.json`**, so agreement with a run's own summary is
evidence rather than a tautology.

Five checks, each of which can fail and one of which is expected to constrain what may be written:

1. minimum selection cross-entropy and its update, per rung, from the calibration rows alone;
2. the frozen 3,072-decision list is a genuine prefix of each expansion, decision by decision ---
   the property that makes the rungs nested rather than three unrelated samples;
3. STOP prediction counts and recall numerators summed over every calibration point of every rung;
4. closed-loop aggregates rebuilt from the per-episode artifacts;
5. whether each rung's minimum is interior or falls on the final update. A boundary minimum means
   the curve had not turned over and no claim of convergence is supportable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORBIDDEN = "result.json"


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.name == FORBIDDEN:
        raise AssertionError("This verifier must not read a run's own summary.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rung(name: str, progress: Path) -> dict[str, Any]:
    rows = _rows(progress)
    cal = [r for r in rows if "calibration" in r]
    best = min(cal, key=lambda r: r["calibration"]["selection_chunk_ce_mean"])
    stop_pred = sum(r["calibration"]["per_action"]["STOP"]["predicted_count"] for r in cal)
    stop_hit = sum(r["calibration"]["per_action"]["STOP"]["recall_numerator"] for r in cal)
    acc = [r["calibration"]["position_accuracy"] for r in cal]
    const = [r["calibration"]["constant_predicted_chunks"] for r in cal]
    return {
        "run": name,
        "calibration_points": len(cal),
        "min_selection_ce": best["calibration"]["selection_chunk_ce_mean"],
        "min_at_update": best["update"],
        "final_update": cal[-1]["update"],
        "minimum_is_on_final_update": best["update"] == cal[-1]["update"],
        "stop_predicted_total": stop_pred,
        "stop_recall_numerator_total": stop_hit,
        "position_accuracy_range": [min(acc), max(acc)],
        "constant_chunks_range": [min(const), max(const)],
        "non_finite_train_rows": [
            r["update"]
            for r in rows
            if not isinstance(r.get("train_chunk_ce_mean"), (int, float))
            or r["train_chunk_ce_mean"] != r["train_chunk_ce_mean"]
        ],
    }


def prefix_nesting(frozen: Path, expanded: Path) -> dict[str, Any]:
    """The frozen training decisions must be the opening segment of the expansion, in order."""
    f = [s for s in json.loads(frozen.read_text())["samples"] if s["split"] == "train"]
    e = [s for s in json.loads(expanded.read_text())["samples"] if s["split"] == "train"]
    f.sort(key=lambda s: int(s["sample_index"]))
    e.sort(key=lambda s: int(s["sample_index"]))
    key = lambda s: (str(s["trajectory_id"]), int(s["action_step_index"]))  # noqa: E731
    head_matches = sum(1 for i in range(min(len(f), len(e))) if key(f[i]) == key(e[i]))
    traj_f = {str(s["trajectory_id"]) for s in f}
    traj_e = {str(s["trajectory_id"]) for s in e}
    return {
        "frozen_train_decisions": len(f),
        "expanded_train_decisions": len(e),
        "prefix_positions_matching": head_matches,
        "is_true_prefix": head_matches == len(f),
        "expansion_left_frozen_trajectories": sorted(traj_e - traj_f)[:5],
        "trajectory_sets_identical": traj_e == traj_f,
        "duplicate_decisions_in_expansion": len(e) - len({key(s) for s in e}),
    }


def closed_loop(root: Path) -> dict[str, Any]:
    files = sorted((root / "episodes").glob("*.json"))
    acts: Counter[int] = Counter()
    succ = orac = stopped = const = capped = 0
    ne: list[float] = []
    for path in files:
        payload = json.loads(path.read_text())
        m = payload["metrics"]
        counts = Counter(int(a) for a in payload["trace"]["actions"])
        acts.update(counts)
        succ += int(float(m["success"]) > 0)
        orac += int(float(m["oracle_success"]) > 0)
        stopped += int(bool(m["stopped"]))
        const += int(len(counts) == 1)
        capped += int(int(m["step_count"]) >= 500)
        ne.append(float(m["navigation_error"]))
    names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    return {
        "episodes": len(files),
        "success": succ,
        "oracle_success": orac,
        "episodes_ending_in_stop": stopped,
        "constant_action_episodes": const,
        "episodes_at_step_cap": capped,
        "mean_navigation_error_m": sum(ne) / len(ne) if ne else None,
        "action_histogram": {names[k]: v for k, v in sorted(acts.items())},
        "total_actions": sum(acts.values()),
        "stop_actions_emitted": acts.get(0, 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/context/scale_ladder_verification.json")
    )
    args = parser.parse_args()
    spark = args.repo / "artifacts" / "spark"

    rungs = [
        rung("native_chunk_scale_40k_v1", spark / "native_chunk_scale_40k_v1" / "progress.jsonl"),
        rung("native_chunk_scale_134k_v1", spark / "native_chunk_scale_134k_v1" / "progress.jsonl"),
    ]
    frozen_manifest = spark / "native_c0_development_cache_v1" / "manifest.json"
    frozen_sha = hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()

    nesting = {}
    for name, path in (
        ("40k", spark / "scale_manifest_40k" / "manifest.json"),
        ("134k", spark / "scale_manifest_134k" / "manifest.json"),
    ):
        if not path.is_file():
            continue
        result = prefix_nesting(frozen_manifest, path)
        # The expansion records which manifest it was built from. Recompute that hash rather than
        # trusting the field, so a swapped source manifest cannot pass as the frozen one.
        recorded = json.loads(path.read_text())["source_manifest_sha256"]
        result["source_manifest_sha256_recorded"] = recorded
        result["source_manifest_sha256_recomputed"] = frozen_sha
        result["source_manifest_is_the_frozen_one"] = recorded == frozen_sha
        nesting[name] = result

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gate": "scale_ladder_verification_v1",
        "method": (
            "Every value re-derived from progress.jsonl, the sample manifests and the per-episode "
            "closed-loop artifacts. No run's result.json was opened."
        ),
        "frozen_reference": {"min_selection_ce": 0.6843, "at_update": 450, "stop_recall": "0/4"},
        "rungs": rungs,
        "prefix_nesting": nesting,
        "closed_loop_134k": closed_loop(spark / "native_chunk_closed_loop_134k_v1"),
        "closed_loop_pilot_u450": closed_loop(spark / "native_chunk_closed_loop_v1"),
    }
    total_stop = sum(r["stop_predicted_total"] for r in rungs)
    payload["findings"] = {
        "stop_predicted_across_all_calibration_points": total_stop,
        "boundary_selected_rungs": [
            r["run"] for r in rungs if r["minimum_is_on_final_update"]
        ],
        "any_non_finite_train_loss": any(r["non_finite_train_rows"] for r in rungs),
        "all_nesting_checks_pass": all(
            v.get("is_true_prefix") and v.get("source_manifest_is_the_frozen_one")
            for v in nesting.values()
        )
        if nesting
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["findings"], indent=2))
    for r in rungs:
        print(
            f"  {r['run']}: min CE {r['min_selection_ce']:.6f} @u{r['min_at_update']} "
            f"({'BOUNDARY' if r['minimum_is_on_final_update'] else 'interior'}), "
            f"{r['calibration_points']} points, STOP predicted {r['stop_predicted_total']}"
        )
    for name, v in nesting.items():
        print(
            f"  nesting {name}: {v['prefix_positions_matching']}/{v['frozen_train_decisions']} "
            f"prefix positions match -> true prefix: {v['is_true_prefix']}; "
            f"trajectory sets identical: {v['trajectory_sets_identical']}; "
            f"duplicates: {v['duplicate_decisions_in_expansion']}; "
            f"source is the frozen manifest: {v['source_manifest_is_the_frozen_one']}"
        )
    cl = payload["closed_loop_134k"]
    print(
        f"  closed loop 134k: {cl['episodes']} eps, success {cl['success']}, "
        f"ending in STOP {cl['episodes_ending_in_stop']}, "
        f"STOP actions emitted {cl['stop_actions_emitted']}/{cl['total_actions']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
