#!/usr/bin/env python3
"""Generate every value the chunk-pilot write-up cites, from verified artifacts only.

Project rule: prose never carries a typed number. Anything quoted in a document is produced here,
from the run artifacts, and the document cites this file. Nothing is recomputed from memory and
nothing is copied from a terminal summary.

Sources, all read directly:

- `native_chunk_pilot_v1/progress.jsonl` — the 26-point calibration series;
- `native_chunk_pilot_u450_v2/progress.jsonl` — the independent repeat of updates 150/300/450;
- `native_chunk_closed_loop_v1/episodes/*.json` — per-episode closed-loop evidence, reconstructed
  rather than read from that run's own `result.json`;
- `native_c0_closed_loop_v1/result.json` — the frozen manuscript comparison, read as published.

The released R2R-CE ground truth supplies the degenerate-strategy baselines that motivate the
objective.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction

CHUNK_SIZE = 4


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_series(root: Path) -> dict[str, Any]:
    rows = _rows(root / "progress.jsonl")
    cal = [r for r in rows if "calibration" in r]
    points = []
    for r in cal:
        c = r["calibration"]
        stop = c["per_action"]["STOP"]
        points.append(
            {
                "update": r["update"],
                "selection_chunk_ce_mean": c["selection_chunk_ce_mean"],
                "position_accuracy": c["position_accuracy"],
                "exact_chunk_match": c["exact_chunk_match"],
                "exact_chunk_match_total": c["exact_chunk_match_total"],
                "constant_predicted_chunks": c["constant_predicted_chunks"],
                "stop_predicted_count": stop["predicted_count"],
                "stop_recall_numerator": stop["recall_numerator"],
                "stop_recall_denominator": stop["recall_denominator"],
            }
        )
    best = min(points, key=lambda p: p["selection_chunk_ce_mean"]) if points else None
    best_full = min(cal, key=lambda r: r["calibration"]["selection_chunk_ce_mean"])["calibration"]
    per_action = {
        name: {
            "targets": v["recall_denominator"],
            "predicted": v["predicted_count"],
            "correct": v["recall_numerator"],
            "recall": v["recall_numerator"] / v["recall_denominator"]
            if v["recall_denominator"]
            else None,
            "precision": v["recall_numerator"] / v["predicted_count"]
            if v["predicted_count"]
            else None,
        }
        for name, v in best_full["per_action"].items()
    }
    supervised = sum(v["targets"] for v in per_action.values())
    forward = per_action["MOVE_FORWARD"]["targets"]
    result = json.loads((root / "result.json").read_text())
    return {
        "run_id": result["run_id"],
        "status": result["status"],
        "completed_updates": result["completed_updates"],
        "elapsed_hours": result["elapsed_seconds"] / 3600.0,
        "final_epochs": rows[-1]["epoch_fraction"],
        "first_train_chunk_ce": rows[0]["train_chunk_ce_mean"],
        "final_train_chunk_ce": rows[-1]["train_chunk_ce_mean"],
        "calibration_point_count": len(points),
        "calibration_points": points,
        "minimum": best,
        "final_selection_chunk_ce": points[-1]["selection_chunk_ce_mean"] if points else None,
        "uniform_reference_ln4": 2.0 * 0.6931471805599453,
        "best_checkpoint_per_action": per_action,
        "best_checkpoint_supervised_positions": supervised,
        "always_forward_position_accuracy": forward / supervised if supervised else None,
        "calibrations_with_nonzero_stop_recall": [
            p["update"] for p in points if p["stop_recall_numerator"] > 0
        ],
    }


def repeat_series(root: Path, original: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(root / "progress.jsonl")
    cal = [r for r in rows if "calibration" in r]
    by_update = {p["update"]: p for p in original["calibration_points"]}
    points = []
    for r in cal:
        c = r["calibration"]
        orig = by_update.get(r["update"])
        points.append(
            {
                "update": r["update"],
                "selection_chunk_ce_mean": c["selection_chunk_ce_mean"],
                "original_selection_chunk_ce_mean": orig["selection_chunk_ce_mean"]
                if orig
                else None,
                "position_accuracy": c["position_accuracy"],
                "constant_predicted_chunks": c["constant_predicted_chunks"],
                "original_constant_predicted_chunks": orig["constant_predicted_chunks"]
                if orig
                else None,
            }
        )
    return {
        "run_id": "native_chunk_pilot_u450_v2",
        "first_update_train_ce_matches_original": rows[0]["train_chunk_ce_mean"]
        == original["first_train_chunk_ce"],
        "calibration_points": points,
        "note": (
            "Seeding fixes initialization and sample order; the 8B backward pass uses "
            "non-deterministic GPU kernels, so trajectories diverge after the first optimizer "
            "step. "
            "Agreement here is reproducibility of outcome, not of arithmetic."
        ),
    }


def closed_loop(root: Path) -> dict[str, Any]:
    files = sorted((root / "episodes").glob("*.json"))
    actions: Counter[str] = Counter()
    successes = oracle = stopped = constant = capped = 0
    nav: list[float] = []
    spl: list[float] = []
    ndtw: list[float] = []
    sdtw: list[float] = []
    oracle_episodes: list[str] = []
    per_episode: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text())
        m = payload["metrics"]
        raw = payload["trace"]["actions"]
        counts = Counter(R2RCEAction(int(a)).name for a in raw)
        actions.update(counts)
        successes += int(float(m["success"]) > 0)
        if float(m["oracle_success"]) > 0:
            oracle += 1
            oracle_episodes.append(str(payload["episode_id"]))
        stopped += int(bool(m["stopped"]))
        constant += int(len(counts) == 1)
        capped += int(int(m["step_count"]) >= 500)
        nav.append(float(m["navigation_error"]))
        spl.append(float(m["spl"]))
        ndtw.append(float(m["ndtw"]))
        sdtw.append(float(m["sdtw"]))
        per_episode.append(
            {
                "episode_id": str(payload["episode_id"]),
                "success": float(m["success"]),
                "oracle_success": float(m["oracle_success"]),
                "navigation_error": float(m["navigation_error"]),
                "step_count": int(m["step_count"]),
                "stopped": bool(m["stopped"]),
                "distinct_actions": len(counts),
                "action_histogram": dict(counts),
            }
        )
    n = len(files)
    manifest = json.loads((root / "manifest.json").read_text())
    return {
        "run_id": manifest["run_id"],
        "checkpoint_update": manifest.get("checkpoint_update"),
        "checkpoint_selection_ce": manifest.get("checkpoint_selection_ce"),
        "chunk_size": manifest.get("chunk_size"),
        "episode_count": n,
        "success_rate": successes / n if n else None,
        "success_numerator": successes,
        "oracle_success_rate": oracle / n if n else None,
        "oracle_success_numerator": oracle,
        "oracle_success_episodes": oracle_episodes,
        "episodes_that_stopped": stopped,
        "constant_action_episodes": constant,
        "episodes_at_step_cap": capped,
        "mean_navigation_error": statistics.mean(nav) if nav else None,
        "mean_spl": statistics.mean(spl) if spl else None,
        "mean_ndtw": statistics.mean(ndtw) if ndtw else None,
        "mean_sdtw": statistics.mean(sdtw) if sdtw else None,
        "total_actions": sum(actions.values()),
        "action_histogram": dict(sorted(actions.items())),
        "distinct_actions_used": len(actions),
        "per_episode": per_episode,
    }


def frozen_comparison(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    unweighted = result["checkpoint_summaries"]["unweighted_s20260825"]
    control = result["control_comparison"]["always_forward"]["aggregate"]
    identical = {
        key: unweighted[key] == control[key]
        for key in ("navigation_error", "ndtw", "mean_path_length", "collision_count")
        if key in control
    }
    return {
        "run_id": "native_c0_closed_loop_v1",
        "completed_episode_runs": result["completed_episode_runs"],
        "unweighted_s20260825": {
            k: unweighted[k]
            for k in (
                "success_rate",
                "oracle_success_rate",
                "spl",
                "ndtw",
                "sdtw",
                "navigation_error",
                "stop_rate",
                "step_cap_rate",
                "mean_steps",
                "collision_count",
                "mean_path_length",
                "action_histogram",
            )
        },
        "always_forward_control": control,
        "unweighted_matches_control_on": identical,
    }


def dataset_baselines(gt_path: Path) -> dict[str, Any]:
    with gzip.open(gt_path, "rt") as handle:
        data = json.load(handle)
    single: Counter[int] = Counter()
    chunks: Counter[tuple[int, ...]] = Counter()
    stop_positions_non_final = 0
    for value in data.values():
        a = value["actions"]
        single.update(a)
        for i in range(0, len(a) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunks[tuple(a[i : i + CHUNK_SIZE])] += 1
        stop_positions_non_final += sum(
            1 for i, x in enumerate(a) if x == int(R2RCEAction.STOP) and i != len(a) - 1
        )
    ts = sum(single.values())
    tc = sum(chunks.values())
    bs = single.most_common(1)[0]
    bc = chunks.most_common(1)[0]
    turn_runs = []
    for value in data.values():
        for key, group in itertools.groupby(value["actions"]):
            if key in (int(R2RCEAction.TURN_LEFT), int(R2RCEAction.TURN_RIGHT)):
                turn_runs.append(len(list(group)))
    return {
        "episodes": len(data),
        "single_decisions": ts,
        "best_constant_single_action": R2RCEAction(bs[0]).name,
        "best_constant_single_action_matches": bs[1],
        "best_constant_single_action_rate": bs[1] / ts,
        "chunk_count": tc,
        "best_constant_chunk": list(bc[0]),
        "best_constant_chunk_matches": bc[1],
        "best_constant_chunk_rate": bc[1] / tc,
        "distinct_chunks_observed": len(chunks),
        "possible_chunks": 4**CHUNK_SIZE,
        "stop_occurrences_not_final": stop_positions_non_final,
        "turn_run_median": statistics.median(turn_runs),
        "turn_run_mean": statistics.mean(turn_runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path.home() / (
            "datasets/GridNav/episodes/"
            "R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/context/chunk_pilot_values.json")
    )
    args = parser.parse_args()
    spark = args.repo / "artifacts" / "spark"

    training = training_series(spark / "native_chunk_pilot_v1")
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gate": "chunk_pilot_values_v1",
        "evidence_boundary": (
            "Exploratory capability pilot. Values here are generated from run artifacts and are "
            "not admitted manuscript evidence: the pilot is single-seed, was not preregistered, "
            "and its closed-loop runner does not re-verify suite and checkpoint hashes."
        ),
        "dataset_baselines": dataset_baselines(args.ground_truth),
        "training": training,
        "repeat": repeat_series(spark / "native_chunk_pilot_u450_v2", training),
        "closed_loop": closed_loop(spark / "native_chunk_closed_loop_v1"),
        "frozen_comparison": frozen_comparison(spark / "native_c0_closed_loop_v1" / "result.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
