from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "src/spatialvlm/eval/native_c0_paper_assets.py"
SPEC = importlib.util.spec_from_file_location("native_c0_paper_assets_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load native-C0 paper asset module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VERIFIER_PATH = ROOT / "scripts/check_native_c0_paper_assets.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "native_c0_paper_assets_verifier_test", VERIFIER_PATH
)
if VERIFIER_SPEC is None or VERIFIER_SPEC.loader is None:
    raise RuntimeError("Unable to load native-C0 paper asset verifier")
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER)


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    comparison_path = (
        ROOT / "artifacts/context/native_c0_action_balance_comparison_v1/analysis.json"
    )
    failure_path = ROOT / "artifacts/context/native_c0_failure_analysis_v1/analysis.json"
    policy_path = ROOT / "artifacts/context/policy_ordering_v1_verification.json"
    manifest_path = ROOT / "artifacts/spark/native_c0_development_v1/manifest.json"
    seed_replication_path = (
        ROOT / "artifacts/context/native_c0_seed_replication_analysis_v1/analysis.json"
    )
    if not all(
        path.is_file()
        for path in (
            comparison_path,
            failure_path,
            policy_path,
            manifest_path,
            seed_replication_path,
        )
    ):
        pytest.skip("mirrored native-C0 paper evidence is unavailable")
    return tuple(
        json.loads(path.read_text())
        for path in (
            comparison_path,
            failure_path,
            policy_path,
            manifest_path,
            seed_replication_path,
        )
    )


def test_frozen_values_preserve_denominators_and_separate_references() -> None:
    values = MODULE.build_paper_values(
        comparison=_inputs()[0],
        failure_analysis=_inputs()[1],
        policy_controls=_inputs()[2],
        c0_manifest=_inputs()[3],
        seed_replication=_inputs()[4],
    )
    assert values["references"]["balanced_uniform_ce_nats"] == pytest.approx(1.3862943611198906)
    assert values["references"]["balanced_uniform_accuracy"] == 0.25
    unweighted = values["c0"]["unweighted"]["balanced"]
    weighted = values["c0"]["action_balanced"]["balanced"]
    assert unweighted["accuracy_correct"] == 84
    assert weighted["accuracy_correct"] == 113
    assert unweighted["per_action_correct"]["STOP"] == 0
    assert weighted["per_action_correct"]["STOP"] == 12
    assert unweighted["per_action_count"]["STOP"] == 64
    assert values["selection_stop_trace"][0]["stop_predictions"] == 28
    assert values["selection_stop_trace"][1]["stop_predictions"] == 0
    assert all(row["stop_target_count"] == 3 for row in values["selection_stop_trace"])
    assert values["trajectory_allocation"]["unused_trajectory_count"] == 0


def test_tables_label_recall_denominators_and_ce_accuracy_references() -> None:
    values = MODULE.build_paper_values(
        comparison=_inputs()[0],
        failure_analysis=_inputs()[1],
        policy_controls=_inputs()[2],
        c0_manifest=_inputs()[3],
        seed_replication=_inputs()[4],
    )
    gate_table = MODULE.render_c0_gate_table(values)
    recall_table = MODULE.render_action_recall_table(values)
    assert "\\ln 4=1.3863" in gate_table
    assert "uniform $=0.25$" in gate_table
    assert "0/64 (0.000)" in gate_table
    assert "12/64 (0.188)" in gate_table
    assert "58/64 (0.906)" in recall_table
    assert "association, not a causal mechanism" in recall_table

    replication_table = MODULE.render_seed_replication_table(values)
    assert "Three-seed paired replication" in replication_table
    assert "0/64 in 3/3 seeds" in replication_table
    assert "12/64, 1/64, 6/64" in replication_table
    assert "0/3 passed" in replication_table


def test_linked_analysis_refuses_tampered_hash(tmp_path: Path) -> None:
    analysis = ROOT / "artifacts/context/native_c0_failure_analysis_v1/analysis.json"
    verification = ROOT / "artifacts/context/native_c0_failure_analysis_v1/verification.json"
    if not analysis.is_file() or not verification.is_file():
        pytest.skip("mirrored failure analysis is unavailable")
    tampered = tmp_path / "analysis.json"
    payload = json.loads(analysis.read_text())
    payload["status"] = "tampered"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected passed gate"):
        MODULE.validate_linked_analysis(
            tampered,
            verification,
            expected_analysis_gate="native_qwen_c0_failure_analysis_v1",
            expected_verifier_gate="independent_native_qwen_c0_failure_analysis_verification_v1",
        )


def test_frozen_invariant_refuses_reframed_balanced_accuracy() -> None:
    comparison, failure, policy, manifest, seed_replication = _inputs()
    broken = copy.deepcopy(comparison)
    broken["balanced_diagnostic"]["unweighted"]["descriptive_accuracy"] = 0.24
    with pytest.raises(ValueError, match="balanced accuracy"):
        MODULE.build_paper_values(
            comparison=broken,
            failure_analysis=failure,
            policy_controls=policy,
            c0_manifest=manifest,
            seed_replication=seed_replication,
        )


def test_frozen_invariant_refuses_incomplete_seed_reporting() -> None:
    comparison, failure, policy, manifest, seed_replication = _inputs()
    broken = copy.deepcopy(seed_replication)
    broken["seeds"] = [20260825, 20260826]
    with pytest.raises(ValueError, match="Three-seed replication evidence"):
        MODULE.build_paper_values(
            comparison=comparison,
            failure_analysis=failure,
            policy_controls=policy,
            c0_manifest=manifest,
            seed_replication=broken,
        )


def test_generated_assets_pass_independent_provenance_verification(tmp_path: Path) -> None:
    provenance = ROOT / "paper/generated/native_c0_paper_provenance.json"
    if not provenance.is_file():
        pytest.skip("generated native-C0 paper provenance is unavailable")
    passed = VERIFIER.verify_paper_assets(ROOT, provenance)
    assert passed["status"] == "passed"
    assert passed["generated_output_count"] == 7
    assert passed["all_three_preregistered_seeds_reported"] is True

    broken = json.loads(provenance.read_text())
    broken["generated_outputs"]["seed_replication_figure"]["sha256"] = "0" * 64
    broken_path = tmp_path / "broken_provenance.json"
    broken_path.write_text(json.dumps(broken))
    failed = VERIFIER.verify_paper_assets(ROOT, broken_path)
    assert failed["status"] == "failed"
    assert "hash mismatch" in failed["errors"][0]
