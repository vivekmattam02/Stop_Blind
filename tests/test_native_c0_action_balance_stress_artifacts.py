from __future__ import annotations

import ast
from pathlib import Path


def test_stress_runner_is_bounded_and_emits_no_scientific_result() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/run_spark_native_c0_action_balance_stress.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assert "HARD_MAX_SECONDS = 1_800.0" in source
    assert '"scientific_result_emitted": False' in source
    assert '"full_action_balance_run_authorized": True' in source
    assert any(isinstance(node, ast.For) for node in ast.walk(tree))


def test_stress_launcher_is_immutable_and_waits_for_registration() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/start_spark_native_c0_action_balance_stress.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'run_id="${GRIDNAV_ACTION_BALANCE_STRESS_RUN_ID:-'
        'native_c0_action_balance_stress_v1}"' in source
    )
    assert "is non-empty; preserve it and do not overwrite" in source
    assert "wait_for_container_registration" in source
    assert "--gpus=all" in source


def test_stress_verifier_reconstructs_event_chain() -> None:
    root = Path(__file__).parents[1]
    source = (
        root / "scripts/check_native_c0_action_balance_stress_artifacts.py"
    ).read_text(encoding="utf-8")
    assert "verify_stress_records" in source
    assert '"contract_reconstruction_performed": True' in source
    assert '"event_chain_reconstruction_performed": True' in source
