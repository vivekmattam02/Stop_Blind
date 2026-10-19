#!/usr/bin/env python3
"""Emit LaTeX macros for the long-form explainer from the generated values file.

The explainer must not contain a typed number. Every quantity it cites is a macro defined here,
resolved from ``artifacts/context/explainer_values.json``, which is itself rebuilt from run
artifacts by ``generate_explainer_values.py``. If an artifact changes, both regenerate and the
document changes with them; there is no path by which prose and evidence can silently disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def pct(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values", type=Path, default=Path("artifacts/context/explainer_values.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v = json.loads(args.values.read_text())

    lines: list[str] = [
        "% GENERATED FILE - do not edit by hand.",
        "% Produced by scripts/generate_explainer_macros.py from",
        "% artifacts/context/explainer_values.json, which is rebuilt from run artifacts by",
        "% scripts/generate_explainer_values.py. Every number in the explainer is one of these.",
        f"\\newcommand{{\\ValuesGeneratedAt}}{{{v['generated_at_utc'][:19].replace('T',' ')} UTC}}",
        "",
    ]

    def add(name: str, value: Any) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---- dataset -----------------------------------------------------------------
    d = v["dataset"]
    add("DsEpisodes", fmt_int(d["episodes"]))
    add("DsDecisions", fmt_int(d["total_decisions"]))
    add("DsMeanLen", fmt(d["mean_episode_length_actions"], 1))
    add("DsMedianLen", fmt_int(d["median_episode_length_actions"]))
    for name, count in d["action_counts"].items():
        add(f"DsCount{name.title().replace('_','')}", fmt_int(count))
        add(f"DsShare{name.title().replace('_','')}", pct(d["action_shares"][name], 2))
    add("DsForwardMatches", fmt_int(d["best_constant_action_matches"]))
    add("DsForwardRate", pct(d["best_constant_action_rate"]))
    add("DsChunkCount", fmt_int(d["chunk_count"]))
    add("DsBestChunkMatches", fmt_int(d["best_constant_chunk_matches"]))
    add("DsBestChunkRate", pct(d["best_constant_chunk_rate"]))
    add("DsChunkObserved", fmt_int(d["distinct_chunks_observed"]))
    add("DsChunkPossible", fmt_int(d["possible_chunks"]))
    add("DsStopNotFinal", fmt_int(d["stop_occurrences_not_final"]))
    add("DsTurnRunMedian", fmt_int(d["turn_run_median"]))
    add("DsTurnRunMean", fmt(d["turn_run_mean"], 2))

    # ---- frozen protocol (from the reference run) --------------------------------
    ref = v["c0_runs"]["unweighted_s20260825"]
    p = ref["protocol"]
    add("PTrainDecisions", fmt_int(p["train_decisions"]))
    add("PTrainTrajectories", fmt_int(p["train_trajectories"]))
    add("PSelectionDecisions", fmt_int(p["selection_decisions"]))
    add("PPrimaryDecisions", fmt_int(p["primary_evaluation_decisions"]))
    add("PDiagnosticDecisions", fmt_int(p["diagnostic_decisions"]))
    add("PDiagnosticPerAction", fmt_int(p["diagnostic_decisions_per_action"]))
    add("PUpdates", fmt_int(p["planned_optimizer_updates"]))
    add("PAccumulation", fmt_int(p["accumulation_steps"]))
    add("PEffectiveBatch", fmt_int(p["effective_batch"]))
    add("PLearningRate", f"{p['learning_rate']:.0e}".replace("e-05", "\\times 10^{-5}"))
    add("PWeightDecay", str(p["weight_decay"]))
    add("PEpochs", fmt_int(p["epochs"]))
    add("PBlindPrior", fmt(p["natural_blind_prior_ce_nats"]))
    add("PNaturalThreshold", fmt(p["natural_ce_threshold"]))
    add("PUniformCE", fmt(p["balanced_uniform_ce_nats"]))
    add("PBalancedThreshold", fmt(p["balanced_ce_threshold"]))
    add("PMargin", fmt(p["required_margin_nats"], 2))
    add("PStopFloor", fmt(p["stop_recall_floor"], 2))
    add("PMinDecisionStep", fmt_int(p["min_decision_step"]))
    add("PMaxPrefix", fmt_int(p["max_replay_prefix_actions"]))
    add("PReservedTrajectories", fmt_int(p["reserved_trajectories"]))
    add("PSamplerSeed", str(p["sampler_seed"]))
    add("PBootstrap", fmt_int(p["bootstrap_samples"]))
    add("PMaxGradNorm", str(p["max_grad_norm"]))
    add("PSelectionRule", p["checkpoint_selection_rule"].replace("_", " "))
    add("PLoraParams", fmt_int(ref["trainable_parameter_count"]))
    add("PLoraTensors", fmt_int(ref["trainable_tensor_count"]))

    # ---- per-run outcomes --------------------------------------------------------
    label = {
        "unweighted_s20260825": "UwA", "unweighted_s20260826": "UwB",
        "unweighted_s20260827": "UwC", "inverse_frequency_s20260825": "IfA",
        "inverse_frequency_s20260826": "IfB", "inverse_frequency_s20260827": "IfC",
    }
    for key, tag in label.items():
        if key not in v["c0_runs"]:
            continue
        r = v["c0_runs"][key]
        add(f"Run{tag}Selected", fmt_int(r["selected_update"]))
        add(f"Run{tag}MinCE", fmt(r["minimum_selection_ce"]))
        add(f"Run{tag}PrimaryAcc", fmt(r["primary_evaluation"]["reconstructed_accuracy"]))
        add(f"Run{tag}PrimaryCE", fmt(r["primary_evaluation"]["reported_mean_ce"]))
        add(f"Run{tag}BalancedAcc", fmt(r["balanced_diagnostic"]["reconstructed_accuracy"]))
        add(f"Run{tag}Status", r["status"].replace("_", "\\_"))
        stop = r["balanced_diagnostic"]["per_action"]["STOP"]
        add(f"Run{tag}BalStopRecall", fmt(stop["recall"], 3) if stop["recall"] is not None else "--")
        add(f"Run{tag}PrimaryDistinct", fmt_int(r["primary_evaluation"]["distinct_predicted_actions"]))
        pstop = r["primary_evaluation"]["per_action"]["STOP"]
        add(f"Run{tag}PrimaryStopCount", fmt_int(pstop["count"]))
        add(f"Run{tag}PrimaryStopRecall",
            fmt(pstop["recall"], 3) if pstop["recall"] is not None else "--")

    # ---- closed-loop matrix ------------------------------------------------------
    m = v["closed_loop_matrix"]
    add("CLEpisodeRuns", fmt_int(m["completed_episode_runs"]))
    add("CLControlCount", fmt_int(len(m["control_names"])))
    s = m["checkpoint_summaries"]["unweighted_s20260825"]
    add("CLUwNavErr", fmt(s["navigation_error"], 3))
    add("CLUwNdtw", fmt(s["ndtw"]))
    add("CLUwSteps", fmt(s["mean_steps"], 1))
    add("CLUwPath", fmt(s["mean_path_length"], 3))
    add("CLUwCollisions", fmt_int(s["collision_count"]))
    add("CLUwForwardActions", fmt_int(s["action_histogram"].get("MOVE_FORWARD", 0)))
    add("CLUwStopActions", fmt_int(s["action_histogram"].get("STOP", 0)))
    add("CLUwStopRate", fmt(s["stop_rate"], 4))
    add("CLUwCapRate", fmt(s["step_cap_rate"], 4))
    add("CLUwOracle", fmt(s["oracle_success_rate"], 4))
    c = m["controls"]["always_forward"]
    add("CLAfNavErr", fmt(c["navigation_error"], 3))
    add("CLAfNdtw", fmt(c["ndtw"]))
    add("CLAfPath", fmt(c["mean_path_length"], 3))
    add("CLAfCollisions", fmt_int(c["collision_count"]))
    add("CLAfSteps", fmt(c["mean_steps"], 1))
    gt = m["controls"].get("released_ground_truth_replay", {})
    if gt:
        add("CLGtSuccess", fmt(gt.get("success_rate", 0), 4))
        add("CLGtNavErr", fmt(gt.get("navigation_error", 0), 3))
    rnd = m["controls"].get("official_random_seed_17", {})
    if rnd:
        add("CLRndSuccess", fmt(rnd.get("success_rate", 0), 4))
        add("CLRndNavErr", fmt(rnd.get("navigation_error", 0), 3))
        add("CLRndNdtw", fmt(rnd.get("ndtw", 0)))
    # per-checkpoint closed-loop rows, all six
    cl_label = {
        "unweighted_s20260825": "UwA", "unweighted_s20260826": "UwB",
        "unweighted_s20260827": "UwC", "inverse_frequency_s20260825": "IfA",
        "inverse_frequency_s20260826": "IfB", "inverse_frequency_s20260827": "IfC",
    }
    for key, tag in cl_label.items():
        row = m["checkpoint_summaries"].get(key)
        if not row:
            continue
        add(f"CL{tag}Sr", fmt(row["success_rate"], 3))
        add(f"CL{tag}Osr", fmt(row["oracle_success_rate"], 4))
        add(f"CL{tag}StopRate", fmt(row["stop_rate"], 4))
        add(f"CL{tag}CapRate", fmt(row["step_cap_rate"], 4))
        add(f"CL{tag}Ne", fmt(row["navigation_error"], 3))
        add(f"CL{tag}Dtw", fmt(row["ndtw"], 4))
        add(f"CL{tag}Fwd", fmt_int(row["action_histogram"].get("MOVE_FORWARD", 0)))
        add(f"CL{tag}Stops", fmt_int(row["action_histogram"].get("STOP", 0)))
        add(f"CL{tag}Distinct", fmt_int(len([k for k, x in row["action_histogram"].items() if x > 0])))
        matches = m["matches_always_forward_on"].get(key, {})
        add(f"CL{tag}MatchesAf", "yes" if matches and all(matches.values()) else "no")
    isr = m["controls"].get("immediate_stop", {})
    if isr:
        add("CLIsNavErr", fmt(isr.get("navigation_error", 0), 3))
        add("CLIsSuccess", fmt(isr.get("success_rate", 0), 4))

    # ---- chunk pilot and scale ladder -------------------------------------------
    for key, tag in (("chunk_pilot", "Pilot"), ("scale_40k", "Forty"), ("scale_134k", "OneThreeFour")):
        r = v[key]
        add(f"Sc{tag}Points", fmt_int(r["calibration_points"]))
        add(f"Sc{tag}MinCE", fmt(r["minimum_selection_ce"]))
        add(f"Sc{tag}MinUpdate", fmt_int(r["minimum_at_update"]))
        add(f"Sc{tag}StopTotal", fmt_int(r["stop_predicted_total"]))
        add(f"Sc{tag}StopCorrect", fmt_int(r["stop_recall_numerator_total"]))
        add(f"Sc{tag}EverStop", fmt_int(r["checkpoints_that_ever_predicted_stop"]))
        add(f"Sc{tag}AccLo", fmt(r["accuracy_range"][0]))
        add(f"Sc{tag}AccHi", fmt(r["accuracy_range"][1]))
        add(f"Sc{tag}ConstLo", fmt_int(r["constant_chunk_range"][0]))
        add(f"Sc{tag}ConstHi", fmt_int(r["constant_chunk_range"][1]))
        add(f"Sc{tag}Hours", fmt(r["elapsed_hours"], 2))
        add(f"Sc{tag}Boundary", "yes" if r["minimum_is_on_final_calibration"] else "no")
        if r["always_forward_accuracy_on_surface"]:
            add(f"Sc{tag}AfAcc", fmt(r["always_forward_accuracy_on_surface"]))

    # ---- chunk closed loops ------------------------------------------------------
    for key, tag in (("chunk_closed_loop_pilot", "ClPilot"), ("chunk_closed_loop_134k", "ClBig")):
        r = v[key]
        add(f"{tag}Episodes", fmt_int(r["episodes"]))
        add(f"{tag}Success", fmt_int(r["success"]))
        add(f"{tag}Oracle", fmt_int(r["oracle_success"]))
        add(f"{tag}Stopped", fmt_int(r["episodes_ending_in_stop"]))
        add(f"{tag}Constant", fmt_int(r["constant_action_episodes"]))
        add(f"{tag}Capped", fmt_int(r["episodes_at_step_cap"]))
        add(f"{tag}NavErr", fmt(r["mean_navigation_error_m"], 3))
        add(f"{tag}Ndtw", fmt(r["mean_ndtw"]))
        add(f"{tag}TotalActions", fmt_int(r["total_actions"]))
        add(f"{tag}StopActions", fmt_int(r["stop_actions_emitted"]))
        add(f"{tag}Distinct", fmt_int(r["distinct_actions_used"]))
        for a, n in r["action_histogram"].items():
            add(f"{tag}Act{a.title().replace('_','')}", fmt_int(n))

    # ---- plot coordinates --------------------------------------------------------
    # Charts must not carry typed coordinates: a label generated from an artifact beside a point
    # typed by hand is exactly how a figure drifts away from its own caption.
    ladder = [
        (v["nesting"]["frozen_train_decisions"], v["chunk_pilot"]["minimum_selection_ce"]),
        (v["nesting"]["scale_40k"]["expanded_train_decisions"], v["scale_40k"]["minimum_selection_ce"]),
        (v["nesting"]["scale_134k"]["expanded_train_decisions"], v["scale_134k"]["minimum_selection_ce"]),
    ]
    add("LadderCoords", " ".join(f"({int(x)},{y:.4f})" for x, y in ladder))
    ces = [y for _, y in ladder]
    add("LadderYMin", fmt(min(ces) - 0.04, 2))
    add("LadderYMax", fmt(max(ces) + 0.04, 2))
    add("LadderXMin", fmt_int(ladder[0][0] * 0.6).replace(",", ""))
    add("LadderXMax", fmt_int(ladder[-1][0] * 1.5).replace(",", ""))
    for i, (x, y) in enumerate(ladder):
        add(f"LadderX{'ABC'[i]}", str(int(x)))
        add(f"LadderY{'ABC'[i]}", fmt(y))
        add(f"LadderLbl{'ABC'[i]}", fmt(y + 0.005, 4))
    # closed-loop success rates as displayed strings
    add("SrZero", fmt(0.0, 3))
    for key, tag in (("chunk_pilot", "Pilot"), ("scale_40k", "Forty"), ("scale_134k", "OneThreeFour")):
        ef = v[key].get("final_epoch_fraction")
        if ef is not None:
            add(f"Sc{tag}Epochs", fmt(ef, 2))
    add("SuiteEpisodes", fmt_int(v["chunk_closed_loop_134k"]["episodes"]))

    # ---- geometry arm (second paper): smoke gate and encoder diagnostics ----------
    import json as _json
    from pathlib import Path as _P
    smoke = _P("artifacts/spark/geo_smoke_v1/progress.jsonl")
    if smoke.is_file():
        rows = [_json.loads(x) for x in smoke.read_text().splitlines() if x.strip()]
        cals = [r for r in rows if "calibration" in r]
        if cals:
            c = cals[-1]["calibration"]
            add("GeoSmokeOn", fmt(c["selection_chunk_ce_mean"]))
            add("GeoSmokeMasked", fmt(c["geometry_masked"]["selection_chunk_ce_mean"]))
    gman = _P("artifacts/spark/native_chunk_geometry_134k_v1/manifest.json")
    if gman.is_file():
        g = _json.loads(gman.read_text())["geometry"]
        kd = g["rope_v2_kernel_diagnostics"]
        add("GeoRopeShort", fmt(kd["shortest_wavelength_m"], 3))
        add("GeoRopeLong", fmt(kd["longest_wavelength_m"], 1))
        add("GeoRopeInversions", fmt_int(kd["near_range_rank_inversions"]))
        add("GeoTokens", fmt_int(g["n_geometry_tokens"]))
        add("GeoParams", fmt(g["geometry_param_count"] / 1e6, 1))

    # ---- geometry arm ------------------------------------------------------------
    if "geometry_arm" in v:
        g = v["geometry_arm"]
        add("GeoMinCE", fmt(g["minimum_selection_ce"]))
        add("GeoMinU", fmt_int(g["minimum_at_update"]))
        add("GeoStopTotal", fmt_int(g["stop_predicted_total"]))
        add("GeoPts", fmt_int(g["calibration_points"]))
        add("GeoBoundary", "yes" if g["minimum_is_on_final_calibration"] else "no")
        c = v["geometry_closed_loop"]
        add("GeoClEpisodes", fmt_int(c["episodes"]))
        add("GeoClOracle", fmt_int(c["oracle_success"]))
        add("GeoClStopped", fmt_int(c["episodes_ending_in_stop"]))
        add("GeoClConstant", fmt_int(c["constant_action_episodes"]))
        add("GeoClActs", fmt_int(c["total_actions"]))
        add("GeoClStopActs", fmt_int(c["stop_actions_emitted"]))
        gi = v["geometry_informativeness"]
        add("GeoWallShift", fmt(gi["flat_wall_shift_over_std"], 2))
        add("GeoBWRatio", fmt(gi["between_within_ratio"], 2))

    # ---- nesting -----------------------------------------------------------------
    n = v["nesting"]
    add("NestFrozen", fmt_int(n["frozen_train_decisions"]))
    for key, tag in (("scale_40k", "Forty"), ("scale_134k", "OneThreeFour")):
        if key in n:
            e = n[key]
            add(f"Nest{tag}Decisions", fmt_int(e["expanded_train_decisions"]))
            add(f"Nest{tag}Prefix", fmt_int(e["prefix_positions_matching"]))
            add(f"Nest{tag}Growth", fmt(e["growth_factor"], 1))
            add(f"Nest{tag}Dupes", fmt_int(e["duplicate_decisions"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {args.output}: {sum(1 for x in lines if x.startswith(chr(92)+'newcommand'))} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
