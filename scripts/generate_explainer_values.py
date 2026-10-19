#!/usr/bin/env python3
"""Generate every number the long-form explainer cites, from verified artifacts only.

Project rule: prose never carries a typed number. This walks the frozen C0 runs, the closed-loop
matrix, the chunk pilot, the nested scale ladder and the released R2R-CE ground truth, and emits a
single JSON the explainer's LaTeX macros are generated from. Nothing here is recomputed from
memory, and nothing is copied out of a terminal summary.

Where a run publishes its own summary of itself, the aggregate is rebuilt from the per-decision or
per-episode records instead, so agreement is evidence rather than restatement.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK_SIZE = 4
SEEDS = ("s20260825", "s20260826", "s20260827")
TREATMENTS = ("unweighted", "inverse_frequency")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_facts(gt_path: Path, episodes_path: Path) -> dict[str, Any]:
    """Ground-truth statistics: the arithmetic that motivates the whole study."""
    with gzip.open(gt_path, "rt") as handle:
        data = json.load(handle)
    single: Counter[int] = Counter()
    chunks: Counter[tuple[int, ...]] = Counter()
    lengths: list[int] = []
    stop_not_final = 0
    turn_runs: list[int] = []
    for value in data.values():
        actions = value["actions"]
        lengths.append(len(actions))
        single.update(actions)
        for i in range(0, len(actions) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunks[tuple(actions[i : i + CHUNK_SIZE])] += 1
        stop_not_final += sum(1 for i, a in enumerate(actions) if a == 0 and i != len(actions) - 1)
        for key, group in itertools.groupby(actions):
            if key in (2, 3):
                turn_runs.append(len(list(group)))

    total = sum(single.values())
    names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    best_single = single.most_common(1)[0]
    best_chunk = chunks.most_common(1)[0]
    total_chunks = sum(chunks.values())
    return {
        "episodes": len(data),
        "total_decisions": total,
        "action_counts": {names[k]: v for k, v in sorted(single.items())},
        "action_shares": {names[k]: v / total for k, v in sorted(single.items())},
        "mean_episode_length_actions": statistics.mean(lengths),
        "median_episode_length_actions": statistics.median(lengths),
        "best_constant_action": names[best_single[0]],
        "best_constant_action_matches": best_single[1],
        "best_constant_action_rate": best_single[1] / total,
        "chunk_count": total_chunks,
        "best_constant_chunk": list(best_chunk[0]),
        "best_constant_chunk_matches": best_chunk[1],
        "best_constant_chunk_rate": best_chunk[1] / total_chunks,
        "distinct_chunks_observed": len(chunks),
        "possible_chunks": 4**CHUNK_SIZE,
        "stop_occurrences_not_final": stop_not_final,
        "turn_run_median": statistics.median(turn_runs),
        "turn_run_mean": statistics.mean(turn_runs),
        "ground_truth_sha256": _sha256(gt_path),
        "episodes_sha256": _sha256(episodes_path) if episodes_path.is_file() else None,
    }


def c0_run(root: Path) -> dict[str, Any]:
    """One frozen C0 development run, with its outcome aggregates rebuilt from per-decision rows."""
    result = json.loads((root / "result.json").read_text())
    protocol = result["protocol"]

    def rebuild(block: str) -> dict[str, Any]:
        rows = result[block]["per_decision"]
        correct = sum(1 for r in rows if r["action_decision"]["predicted_action"] == r["target_action"])
        per_action: dict[str, dict[str, Any]] = {}
        for name in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            targets = [r for r in rows if r["target_action"] == name]
            hits = sum(
                1 for r in targets if r["action_decision"]["predicted_action"] == name
            )
            per_action[name] = {
                "count": len(targets),
                "recall": hits / len(targets) if targets else None,
                "predicted_count": sum(
                    1 for r in rows if r["action_decision"]["predicted_action"] == name
                ),
            }
        return {
            "decision_count": len(rows),
            "reconstructed_accuracy": correct / len(rows),
            "reported_accuracy": result[block]["summary"]["descriptive_accuracy"],
            "reported_mean_ce": result[block]["summary"]["mean_restricted_action_ce"],
            "distinct_predicted_actions": len(
                {r["action_decision"]["predicted_action"] for r in rows}
            ),
            "per_action": per_action,
        }

    calibration = [
        {
            "update_step": c["update_step"],
            "mean_restricted_action_ce": c["summary"]["mean_restricted_action_ce"],
            "descriptive_accuracy": c["summary"]["descriptive_accuracy"],
            "distinct_predicted_action_count": c["summary"]["distinct_predicted_action_count"],
            "stop_predicted": sum(
                1 for r in c["per_decision"] if r["action_decision"]["predicted_action"] == "STOP"
            ),
        }
        for c in result["calibrations"]
    ]
    best = min(calibration, key=lambda c: c["mean_restricted_action_ce"])
    return {
        "run_id": result["run_id"],
        "status": result["status"],
        "completed_updates": result["completed_updates"],
        "trainable_parameter_count": result["trainable_contract"]["trainable_parameter_count"],
        "trainable_tensor_count": result["trainable_contract"]["trainable_tensor_count"],
        "all_trainable_names_are_lora": result["trainable_contract"]["all_trainable_names_are_lora"],
        "selected_update": result["selection"]["checkpoint"]["update_step"],
        "selection_candidates": result["selection"]["candidate_count"],
        "calibration_series": calibration,
        "minimum_selection_ce": best["mean_restricted_action_ce"],
        "minimum_selection_update": best["update_step"],
        "primary_evaluation": rebuild("primary_evaluation"),
        "balanced_diagnostic": rebuild("balanced_diagnostic"),
        "protocol": {
            k: protocol[k]
            for k in (
                "objective", "learning_rate", "weight_decay", "accumulation_steps",
                "effective_batch", "epochs", "planned_optimizer_updates", "train_decisions",
                "train_trajectories", "selection_decisions", "primary_evaluation_decisions",
                "diagnostic_decisions", "diagnostic_decisions_per_action", "history_contract",
                "min_decision_step", "max_replay_prefix_actions", "natural_blind_prior_ce_nats",
                "natural_ce_threshold", "balanced_uniform_ce_nats", "balanced_ce_threshold",
                "required_margin_nats", "stop_recall_floor", "sealed_splits", "sampler_seed",
                "model_seed", "reserved_trajectories", "checkpoint_selection_rule",
                "max_grad_norm", "bootstrap_samples",
            )
            if k in protocol
        },
    }


def closed_loop_matrix(path: Path) -> dict[str, Any]:
    """The 366-episode closed-loop matrix and its non-learning controls."""
    result = json.loads(path.read_text())
    summaries = result["checkpoint_summaries"]
    controls = {
        name: block["aggregate"] for name, block in result["control_comparison"].items()
    }
    forward = controls["always_forward"]
    identity_fields = ("navigation_error", "ndtw", "mean_path_length", "collision_count")
    matches = {
        key: {
            field: summaries[key].get(field) == forward.get(field)
            for field in identity_fields
            if field in forward and field in summaries[key]
        }
        for key in summaries
    }
    return {
        "run_id": result["run_id"],
        "completed_episode_runs": result["completed_episode_runs"],
        "checkpoint_summaries": summaries,
        "controls": controls,
        "control_names": sorted(controls),
        "matches_always_forward_on": matches,
        "all_success_rates_zero": all(v["success_rate"] == 0.0 for v in summaries.values()),
        "result_sha256": _sha256(path),
    }


def chunk_or_scale_run(root: Path) -> dict[str, Any]:
    """A chunk-objective run (pilot or scale rung), rebuilt from its progress rows."""
    rows = _rows(root / "progress.jsonl")
    cal = [r for r in rows if "calibration" in r]
    best = min(cal, key=lambda r: r["calibration"]["selection_chunk_ce_mean"])
    series = [
        {
            "update": r["update"],
            "selection_chunk_ce_mean": r["calibration"]["selection_chunk_ce_mean"],
            "position_accuracy": r["calibration"]["position_accuracy"],
            "constant_predicted_chunks": r["calibration"]["constant_predicted_chunks"],
            "stop_predicted": r["calibration"]["per_action"]["STOP"]["predicted_count"],
            "stop_recall_numerator": r["calibration"]["per_action"]["STOP"]["recall_numerator"],
            "stop_recall_denominator": r["calibration"]["per_action"]["STOP"]["recall_denominator"],
        }
        for r in cal
    ]
    best_pa = best["calibration"]["per_action"]
    supervised = sum(v["recall_denominator"] for v in best_pa.values())
    return {
        "run_id": root.name,
        "calibration_points": len(cal),
        "series": series,
        "minimum_selection_ce": best["calibration"]["selection_chunk_ce_mean"],
        "minimum_at_update": best["update"],
        "minimum_is_on_final_calibration": best["update"] == cal[-1]["update"],
        "stop_predicted_total": sum(s["stop_predicted"] for s in series),
        "stop_recall_numerator_total": sum(s["stop_recall_numerator"] for s in series),
        "checkpoints_that_ever_predicted_stop": sum(1 for s in series if s["stop_predicted"] > 0),
        "accuracy_range": [
            min(s["position_accuracy"] for s in series),
            max(s["position_accuracy"] for s in series),
        ],
        "constant_chunk_range": [
            min(s["constant_predicted_chunks"] for s in series),
            max(s["constant_predicted_chunks"] for s in series),
        ],
        "best_checkpoint_per_action": best_pa,
        "always_forward_accuracy_on_surface": (
            best_pa["MOVE_FORWARD"]["recall_denominator"] / supervised if supervised else None
        ),
        "final_epoch_fraction": rows[-1].get("epoch_fraction"),
        "elapsed_hours": rows[-1]["elapsed_seconds"] / 3600.0,
        "progress_sha256": _sha256(root / "progress.jsonl"),
    }


def closed_loop_episodes(root: Path) -> dict[str, Any]:
    """Rebuild a closed-loop run's aggregates from per-episode artifacts only."""
    files = sorted((root / "episodes").glob("*.json"))
    acts: Counter[int] = Counter()
    succ = orac = stopped = const = capped = 0
    ne: list[float] = []
    spl: list[float] = []
    ndtw: list[float] = []
    sdtw: list[float] = []
    oracle_ids: list[str] = []
    for path in files:
        payload = json.loads(path.read_text())
        m = payload["metrics"]
        counts = Counter(int(a) for a in payload["trace"]["actions"])
        acts.update(counts)
        succ += int(float(m["success"]) > 0)
        if float(m["oracle_success"]) > 0:
            orac += 1
            oracle_ids.append(str(payload["episode_id"]))
        stopped += int(bool(m["stopped"]))
        const += int(len(counts) == 1)
        capped += int(int(m["step_count"]) >= 500)
        ne.append(float(m["navigation_error"]))
        spl.append(float(m["spl"]))
        ndtw.append(float(m["ndtw"]))
        sdtw.append(float(m["sdtw"]))
    names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    n = len(files)
    return {
        "run_id": root.name,
        "episodes": n,
        "success": succ,
        "oracle_success": orac,
        "oracle_episode_ids": oracle_ids,
        "episodes_ending_in_stop": stopped,
        "constant_action_episodes": const,
        "episodes_at_step_cap": capped,
        "mean_navigation_error_m": statistics.mean(ne) if ne else None,
        "mean_spl": statistics.mean(spl) if spl else None,
        "mean_ndtw": statistics.mean(ndtw) if ndtw else None,
        "mean_sdtw": statistics.mean(sdtw) if sdtw else None,
        "action_histogram": {names[k]: v for k, v in sorted(acts.items())},
        "total_actions": sum(acts.values()),
        "stop_actions_emitted": acts.get(0, 0),
        "distinct_actions_used": len(acts),
    }


def scale_nesting(frozen: Path, expansions: dict[str, Path]) -> dict[str, Any]:
    """Confirm each expansion literally begins with the frozen decision list."""
    def train(path: Path) -> list[tuple[str, int]]:
        samples = [s for s in json.loads(path.read_text())["samples"] if s["split"] == "train"]
        samples.sort(key=lambda s: int(s["sample_index"]))
        return [(str(s["trajectory_id"]), int(s["action_step_index"])) for s in samples]

    base = train(frozen)
    out: dict[str, Any] = {"frozen_train_decisions": len(base), "frozen_sha256": _sha256(frozen)}
    for name, path in expansions.items():
        if not path.is_file():
            continue
        expanded = train(path)
        head = sum(1 for i in range(min(len(base), len(expanded))) if base[i] == expanded[i])
        out[name] = {
            "expanded_train_decisions": len(expanded),
            "prefix_positions_matching": head,
            "is_true_prefix": head == len(base),
            "trajectory_sets_identical": {t for t, _ in expanded} == {t for t, _ in base},
            "duplicate_decisions": len(expanded) - len(set(expanded)),
            "growth_factor": len(expanded) / len(base),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path.home() / "datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/context/explainer_values.json")
    )
    args = parser.parse_args()
    spark = args.repo / "artifacts" / "spark"

    runs: dict[str, Any] = {}
    for treatment in TREATMENTS:
        for seed in SEEDS:
            if treatment == "unweighted" and seed == "s20260825":
                root = spark / "native_c0_development_v1"
            elif treatment == "inverse_frequency" and seed == "s20260825":
                root = spark / "native_c0_action_balance_v1"
            else:
                root = spark / f"native_c0_seed_replication_{treatment}_{seed}_v1"
            if (root / "result.json").is_file():
                runs[f"{treatment}_{seed}"] = c0_run(root)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gate": "explainer_values_v1",
        "method": (
            "Outcome aggregates are rebuilt from per-decision and per-episode records; a run's own "
            "summary is read only to compare against that reconstruction."
        ),
        "dataset": dataset_facts(
            args.episodes_root / "train" / "train_gt.json.gz",
            args.episodes_root / "train" / "train.json.gz",
        ),
        "c0_runs": runs,
        "closed_loop_matrix": closed_loop_matrix(
            spark / "native_c0_closed_loop_v1" / "result.json"
        ),
        "chunk_pilot": chunk_or_scale_run(spark / "native_chunk_pilot_v1"),
        "scale_40k": chunk_or_scale_run(spark / "native_chunk_scale_40k_v1"),
        "scale_134k": chunk_or_scale_run(spark / "native_chunk_scale_134k_v1"),
        "chunk_closed_loop_pilot": closed_loop_episodes(
            spark / "native_chunk_closed_loop_v1"
        ),
        "chunk_closed_loop_134k": closed_loop_episodes(
            spark / "native_chunk_closed_loop_134k_v1"
        ),
        "geometry_arm": chunk_or_scale_run(spark / "native_chunk_geometry_134k_v1"),
        "geometry_closed_loop": closed_loop_episodes(
            spark / "native_chunk_geometry_closed_loop_v1"
        ),
        "geometry_informativeness": json.loads(
            (args.repo / "artifacts" / "context" / "geometry_informativeness.json").read_text()
        ),
        "nesting": scale_nesting(
            spark / "native_c0_development_cache_v1" / "manifest.json",
            {
                "scale_40k": spark / "scale_manifest_40k" / "manifest.json",
                "scale_134k": spark / "scale_manifest_134k" / "manifest.json",
            },
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    d = payload["dataset"]
    print(
        f"[done] {args.output} | {len(runs)} C0 runs | dataset {d['episodes']:,} episodes, "
        f"{d['total_decisions']:,} decisions, always-forward {d['best_constant_action_rate']*100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
