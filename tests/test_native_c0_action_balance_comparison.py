from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "src/spatialvlm/training/native_c0_action_balance_comparison.py"
SPEC = importlib.util.spec_from_file_location(
    "native_c0_action_balance_comparison_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load comparison module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ANALYZER_PATH = ROOT / "scripts/analyze_native_c0_action_balance_comparison.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "analyze_native_c0_action_balance_comparison_test", ANALYZER_PATH
)
if ANALYZER_SPEC is None or ANALYZER_SPEC.loader is None:
    raise RuntimeError("Unable to load comparison analyzer")
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(ANALYZER)


def _bundle(run_name: str, checkpoint_name: str) -> dict:
    return ANALYZER.evidence_bundle(
        ROOT / "artifacts/spark" / run_name,
        ROOT / "checkpoints" / run_name / checkpoint_name,
        run_name,
    )


def _real_bundles() -> tuple[dict, dict]:
    required = (
        ROOT / "artifacts/spark/native_c0_development_v1/result.json",
        ROOT / "artifacts/spark/native_c0_action_balance_v1/result.json",
        ROOT / "checkpoints/native_c0_development_v1/step_0160.pt",
        ROOT / "checkpoints/native_c0_action_balance_v1/step_0192.pt",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("complete mirrored comparison evidence is unavailable")
    return (
        _bundle("native_c0_development_v1", "step_0160.pt"),
        _bundle("native_c0_action_balance_v1", "step_0192.pt"),
    )


def test_real_comparison_reconstructs_the_tradeoff() -> None:
    unweighted, treatment = _real_bundles()
    result = MODULE.build_comparison(
        unweighted=unweighted,
        action_balanced=treatment,
        source_locks={},
        recorded_at_utc="frozen",
    )
    assert result["comparison_classification"] == (
        "balanced_support_improved_natural_fit_degraded_no_gate_rescue"
    )
    observed = result["confirmed_descriptive_observations"]
    assert all(observed.values())
    assert result["balanced_diagnostic"]["unweighted"]["per_action_recall"]["STOP"] == 0.0
    assert result["balanced_diagnostic"]["action_balanced"]["per_action_recall"]["STOP"] == 0.1875
    assert result["action_balanced_threshold_gaps"]["stop_recall_gap_to_required_minimum"] == (
        -0.0625
    )


def test_comparison_refuses_sample_surface_drift() -> None:
    unweighted, treatment = _real_bundles()
    broken = copy.deepcopy(treatment)
    broken["manifest"]["samples"][0]["episode_id"] = "tampered"
    with pytest.raises(ValueError, match="samples"):
        MODULE.build_comparison(
            unweighted=unweighted,
            action_balanced=broken,
            source_locks={},
            recorded_at_utc="frozen",
        )


def test_comparison_refuses_unlinked_verification() -> None:
    unweighted, treatment = _real_bundles()
    broken = copy.deepcopy(treatment)
    broken["verification"]["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="verification"):
        MODULE.build_comparison(
            unweighted=unweighted,
            action_balanced=broken,
            source_locks={},
            recorded_at_utc="frozen",
        )


def test_expected_upstream_hashes_are_present() -> None:
    path = ROOT / "artifacts/spark/native_c0_action_balance_v1/verification.json"
    if not path.is_file():
        pytest.skip("balanced verifier is unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["error_count"] == 0
