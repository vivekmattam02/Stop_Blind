#!/usr/bin/env python3
"""Generate native-C0 workshop-paper tables, figures, and provenance."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "src/spatialvlm/eval/native_c0_paper_assets.py"
SPEC = importlib.util.spec_from_file_location("native_c0_paper_assets_generator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load native-C0 paper asset module")
PAPER_ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAPER_ASSETS)

ACTION_ORDER = PAPER_ASSETS.ACTION_ORDER
DISPLAY_ACTIONS = PAPER_ASSETS.DISPLAY_ACTIONS
build_paper_values = PAPER_ASSETS.build_paper_values
build_provenance = PAPER_ASSETS.build_provenance
load_json = PAPER_ASSETS.load_json
render_action_recall_table = PAPER_ASSETS.render_action_recall_table
render_c0_gate_table = PAPER_ASSETS.render_c0_gate_table
render_evaluator_control_table = PAPER_ASSETS.render_evaluator_control_table
render_seed_replication_table = PAPER_ASSETS.render_seed_replication_table
sha256_file = PAPER_ASSETS.sha256_file
validate_linked_analysis = PAPER_ASSETS.validate_linked_analysis
validate_policy_controls = PAPER_ASSETS.validate_policy_controls


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _draw_action_recall(path: Path, values: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen.canvas import Canvas

    width, height = 540.0, 330.0
    canvas = Canvas(str(path), pagesize=(width, height), pageCompression=1, invariant=1)
    canvas.setTitle("Native-C0 balanced action recall")
    navy = colors.HexColor("#17324D")
    teal = colors.HexColor("#1B998B")
    coral = colors.HexColor("#E76F51")
    grid = colors.HexColor("#D7DEE5")
    text = colors.HexColor("#15212B")

    canvas.setFillColor(text)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(50, 302, "Action balancing restores support, but the frozen gate still fails")
    canvas.setFont("Helvetica", 8.8)
    canvas.setFillColor(colors.HexColor("#53616E"))
    canvas.drawString(50, 287, "Balanced diagnostic: 64 targets per action; labels show correct/64")

    left, bottom, chart_w, chart_h = 58.0, 58.0, 448.0, 205.0
    canvas.setStrokeColor(grid)
    canvas.setFillColor(colors.HexColor("#53616E"))
    canvas.setFont("Helvetica", 8)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom + tick * chart_h
        canvas.line(left, y, left + chart_w, y)
        label = f"{tick:.2f}"
        canvas.drawRightString(left - 7, y - 3, label)

    unweighted = values["c0"]["unweighted"]["balanced"]
    weighted = values["c0"]["action_balanced"]["balanced"]
    train = values["training_action_histogram"]
    group_w = chart_w / len(ACTION_ORDER)
    bar_w = 27.0
    for index, action in enumerate(ACTION_ORDER):
        center = left + group_w * (index + 0.5)
        records = (
            (unweighted, navy, center - bar_w - 2),
            (weighted, coral, center + 2),
        )
        for record, color, x in records:
            recall = record["per_action_recall"][action]
            correct = record["per_action_correct"][action]
            y_top = bottom + recall * chart_h
            canvas.setFillColor(color)
            canvas.rect(x, bottom, bar_w, recall * chart_h, fill=1, stroke=0)
            label = f"{correct}/64"
            canvas.setFillColor(text)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawCentredString(x + bar_w / 2, y_top + 5, label)
        canvas.setFillColor(text)
        canvas.setFont("Helvetica-Bold", 8.5)
        action_label = DISPLAY_ACTIONS[action]
        canvas.drawCentredString(center, 41, action_label)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#687580"))
        canvas.drawCentredString(center, 29, f"train n={train[action]}")

    canvas.setFillColor(navy)
    canvas.rect(60, 9, 10, 7, fill=1, stroke=0)
    canvas.setFillColor(text)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(75, 9, "Unweighted")
    canvas.setFillColor(coral)
    canvas.rect(145, 9, 10, 7, fill=1, stroke=0)
    canvas.setFillColor(text)
    canvas.drawString(160, 9, "Inverse-frequency weighted")
    note = "Frequency-aligned association; not a causal-mechanism claim"
    note_w = stringWidth(note, "Helvetica-Oblique", 7.5)
    canvas.setFont("Helvetica-Oblique", 7.5)
    canvas.setFillColor(teal)
    canvas.drawString(width - 35 - note_w, 9, note)
    canvas.showPage()
    canvas.save()


def _draw_stop_trace(path: Path, values: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas

    width, height = 540.0, 300.0
    canvas = Canvas(str(path), pagesize=(width, height), pageCompression=1, invariant=1)
    canvas.setTitle("Native-C0 STOP prediction suppression")
    navy = colors.HexColor("#17324D")
    coral = colors.HexColor("#E76F51")
    grid = colors.HexColor("#D7DEE5")
    text = colors.HexColor("#15212B")

    canvas.setFillColor(text)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(50, 272, "STOP disappears from the 256-decision selection surface")
    canvas.setFont("Helvetica", 8.8)
    canvas.setFillColor(colors.HexColor("#53616E"))
    canvas.drawString(
        50,
        257,
        "Prediction counts are safer than the target-specific STOP recall (only 3 STOP targets)",
    )

    trace = values["selection_stop_trace"]
    left, bottom, chart_w, chart_h = 62.0, 58.0, 430.0, 170.0
    canvas.setStrokeColor(grid)
    canvas.setFillColor(colors.HexColor("#53616E"))
    canvas.setFont("Helvetica", 8)
    for tick in (0, 7, 14, 21, 28):
        y = bottom + tick / 28 * chart_h
        canvas.line(left, y, left + chart_w, y)
        canvas.drawRightString(left - 8, y - 3, str(tick))
    canvas.saveState()
    canvas.translate(16, bottom + chart_h / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "STOP predictions / 256")
    canvas.restoreState()

    points = []
    for index, row in enumerate(trace):
        x = left + index * chart_w / (len(trace) - 1)
        y = bottom + row["stop_predictions"] / 28 * chart_h
        points.append((x, y, row))
    canvas.setStrokeColor(coral)
    canvas.setLineWidth(2.5)
    path_obj = canvas.beginPath()
    path_obj.moveTo(points[0][0], points[0][1])
    for x, y, _ in points[1:]:
        path_obj.lineTo(x, y)
    canvas.drawPath(path_obj, stroke=1, fill=0)
    for x, y, row in points:
        canvas.setFillColor(coral)
        canvas.circle(x, y, 4, fill=1, stroke=0)
        canvas.setFillColor(text)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawCentredString(x, y + 10, f"{row['stop_predictions']}/256")
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(x, 42, str(row["update"]))
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawCentredString(left + chart_w / 2, 27, "Optimizer update")
    canvas.setFillColor(navy)
    canvas.roundRect(311, 236, 181, 15, 4, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawCentredString(401.5, 240, "28/256 at update 0  ->  0/256 at update 32")
    canvas.showPage()
    canvas.save()


def _draw_seed_replication(path: Path, values: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas

    width, height = 540.0, 290.0
    canvas = Canvas(str(path), pagesize=(width, height), pageCompression=1, invariant=1)
    canvas.setTitle("Native-C0 three-seed paired replication")
    navy = colors.HexColor("#17324D")
    coral = colors.HexColor("#E76F51")
    grid = colors.HexColor("#D7DEE5")
    text = colors.HexColor("#15212B")
    canvas.setFillColor(text)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(40, 264, "The checkpoint-selection failure and weighting trade-off reproduce")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#53616E"))
    canvas.drawString(40, 249, "Three preregistered paired seeds; every point is shown")

    replication = values["seed_replication"]
    seeds = replication["seeds"]
    panels = (
        ("natural_mean_ce", "Natural CE", 0.70, 1.02, False),
        ("balanced_mean_ce", "Balanced CE", 1.20, 1.95, False),
        ("balanced_stop_recall", "Balanced STOP recall", 0.0, 0.25, True),
    )
    panel_width = 146.0
    panel_bottom = 55.0
    panel_height = 160.0
    for panel_index, (metric, title, low, high, higher_better) in enumerate(panels):
        left = 45.0 + panel_index * 165.0
        canvas.setFillColor(text)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawCentredString(left + panel_width / 2, 226, title)
        canvas.setStrokeColor(grid)
        canvas.setLineWidth(0.6)
        for tick_index in range(5):
            value = low + tick_index * (high - low) / 4
            y = panel_bottom + tick_index * panel_height / 4
            canvas.line(left, y, left + panel_width, y)
            canvas.setFillColor(colors.HexColor("#65727D"))
            canvas.setFont("Helvetica", 6.5)
            canvas.drawRightString(left - 4, y - 2, f"{value:.2f}")
        for seed_index, seed in enumerate(seeds):
            x = left + 22 + seed_index * 51
            unweighted = float(replication["runs"][str(seed)]["unweighted"][metric])
            weighted = float(replication["runs"][str(seed)]["inverse_frequency"][metric])
            y_unweighted = panel_bottom + (unweighted - low) / (high - low) * panel_height
            y_weighted = panel_bottom + (weighted - low) / (high - low) * panel_height
            canvas.setStrokeColor(colors.HexColor("#8C99A3"))
            canvas.setLineWidth(1.1)
            canvas.line(x - 5, y_unweighted, x + 5, y_weighted)
            canvas.setFillColor(navy)
            canvas.circle(x - 5, y_unweighted, 3.2, fill=1, stroke=0)
            canvas.setFillColor(coral)
            canvas.circle(x + 5, y_weighted, 3.2, fill=1, stroke=0)
            canvas.setFillColor(text)
            canvas.setFont("Helvetica", 6.5)
            canvas.drawCentredString(x, 42, str(seed)[-2:])
        direction = "higher is better" if higher_better else "lower is better"
        canvas.setFillColor(colors.HexColor("#65727D"))
        canvas.setFont("Helvetica-Oblique", 6.5)
        canvas.drawCentredString(left + panel_width / 2, 29, direction)

    canvas.setFillColor(navy)
    canvas.circle(205, 12, 3.2, fill=1, stroke=0)
    canvas.setFillColor(text)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(213, 9, "Unweighted")
    canvas.setFillColor(coral)
    canvas.circle(290, 12, 3.2, fill=1, stroke=0)
    canvas.setFillColor(text)
    canvas.drawString(298, 9, "Inverse-frequency")
    canvas.showPage()
    canvas.save()


def generate(repo: Path) -> dict[str, Any]:
    comparison_path = (
        repo / "artifacts/context/native_c0_action_balance_comparison_v1/analysis.json"
    )
    comparison_verification_path = (
        repo / "artifacts/context/native_c0_action_balance_comparison_v1/verification.json"
    )
    failure_path = repo / "artifacts/context/native_c0_failure_analysis_v1/analysis.json"
    failure_verification_path = (
        repo / "artifacts/context/native_c0_failure_analysis_v1/verification.json"
    )
    policy_path = repo / "artifacts/context/policy_ordering_v1_verification.json"
    manifest_path = repo / "artifacts/spark/native_c0_development_v1/manifest.json"
    seed_replication_path = (
        repo / "artifacts/context/native_c0_seed_replication_analysis_v1/analysis.json"
    )
    seed_replication_verification_path = (
        repo / "artifacts/context/native_c0_seed_replication_analysis_v1/verification.json"
    )

    comparison, _ = validate_linked_analysis(
        comparison_path,
        comparison_verification_path,
        expected_analysis_gate="native_qwen_c0_action_balance_comparison_v1",
        expected_verifier_gate="independent_native_qwen_c0_action_balance_comparison_verification_v1",
    )
    failure, _ = validate_linked_analysis(
        failure_path,
        failure_verification_path,
        expected_analysis_gate="native_qwen_c0_failure_analysis_v1",
        expected_verifier_gate="independent_native_qwen_c0_failure_analysis_verification_v1",
    )
    seed_replication, _ = validate_linked_analysis(
        seed_replication_path,
        seed_replication_verification_path,
        expected_analysis_gate="native_qwen_c0_seed_replication_analysis_v1",
        expected_verifier_gate=(
            "independent_native_qwen_c0_seed_replication_analysis_verification_v1"
        ),
    )
    policy = validate_policy_controls(policy_path, repo)
    manifest = load_json(manifest_path)
    expected_manifest_hash = comparison["upstream_evidence"]["unweighted"]["manifest_sha256"]
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError("C0 manifest is not the one linked by the verified comparison")

    values = build_paper_values(
        comparison=comparison,
        failure_analysis=failure,
        policy_controls=policy,
        c0_manifest=manifest,
        seed_replication=seed_replication,
    )
    generated = repo / "paper/generated"
    tables = repo / "paper/tables"
    figures = repo / "paper/figures"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    values_path = generated / "native_c0_paper_values.json"
    gate_table = tables / "native_c0_gate_results.tex"
    recall_table = tables / "native_c0_action_recall.tex"
    control_table = tables / "native_c0_evaluator_controls.tex"
    replication_table = tables / "native_c0_seed_replication.tex"
    recall_figure = figures / "native_c0_action_recall.pdf"
    stop_figure = figures / "native_c0_stop_suppression.pdf"
    replication_figure = figures / "native_c0_seed_replication.pdf"
    _write_json(values_path, values)
    _write_text(gate_table, render_c0_gate_table(values))
    _write_text(recall_table, render_action_recall_table(values))
    _write_text(control_table, render_evaluator_control_table(values))
    _write_text(replication_table, render_seed_replication_table(values))
    _draw_action_recall(recall_figure, values)
    _draw_stop_trace(stop_figure, values)
    _draw_seed_replication(replication_figure, values)

    source_paths = {
        "comparison_analysis": comparison_path,
        "comparison_verification": comparison_verification_path,
        "failure_analysis": failure_path,
        "failure_analysis_verification": failure_verification_path,
        "seed_replication_analysis": seed_replication_path,
        "seed_replication_verification": seed_replication_verification_path,
        "policy_control_verification": policy_path,
        "c0_manifest": manifest_path,
        "frozen_submission_scope": repo / "paper/SUBMISSION_SCOPE.md",
        "paper_asset_module": repo / "src/spatialvlm/eval/native_c0_paper_assets.py",
        "paper_asset_generator": repo / "scripts/generate_native_c0_paper_assets.py",
        "paper_asset_verifier": repo / "scripts/check_native_c0_paper_assets.py",
    }
    output_paths = {
        "c0_gate_table": gate_table,
        "action_recall_table": recall_table,
        "evaluator_control_table": control_table,
        "seed_replication_table": replication_table,
        "action_recall_figure": recall_figure,
        "stop_suppression_figure": stop_figure,
        "seed_replication_figure": replication_figure,
    }
    provenance = build_provenance(
        repo=repo,
        values_path=values_path,
        source_paths=source_paths,
        output_paths=output_paths,
    )
    provenance_path = generated / "native_c0_paper_provenance.json"
    _write_json(provenance_path, provenance)
    return {
        "status": "passed",
        "values": str(values_path),
        "values_sha256": sha256_file(values_path),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "outputs": {label: str(path) for label, path in output_paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    print(json.dumps(generate(repo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
