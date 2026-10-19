from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_full_launcher_is_versioned_fail_closed_and_gpu_enabled() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/start_spark_native_c0_action_balance.sh").read_text(encoding="utf-8")
    assert 'run_id="${GRIDNAV_ACTION_BALANCE_RUN_ID:-native_c0_action_balance_v1}"' in source
    assert "full_action_balance_run_authorized" in source
    assert "scientific_result_emitted" in source
    assert "is non-empty; preserve it and do not overwrite" in source
    assert "wait_for_container_registration" in source
    assert "--gpus=all" in source
    assert "native_c0_action_balance_stress_v1" in source
    assert "327ffad9d2331f982b5b7a96d7ece0d9046fe137a936bdb1935fb672cbdaf9c9" in source


def test_full_runner_records_both_loss_scales_and_keeps_outcome_evaluation_unweighted() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/run_spark_native_c0_action_balance.py").read_text(encoding="utf-8")
    assert '"unweighted_restricted_action_ce"' in source
    assert '"weighted_restricted_action_ce"' in source
    assert "weighted_contribution_summary" in source
    assert "base._evaluate" in source
    assert "evaluate_existence_gate" in source
    assert '"adaptive_development_comparison": True' in source
    assert "EXPECTED_STRESS_ARTIFACT_SHA256" in source
    assert "Executed stress source lock changed" in source


def test_full_verifier_reconstructs_stress_weighted_events_checkpoints_and_outcome() -> None:
    root = Path(__file__).parents[1]
    source = (root / "scripts/check_native_c0_action_balance_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "verify_action_balance_training_records" in source
    assert '"weighted_event_chain_reconstruction_performed": True' in source
    assert '"stress_authorization_reconstruction_performed": True' in source
    assert '"all_checkpoint_files_fully_loaded_and_hashed": True' in source
    assert '"outcome_reconstruction_performed": True' in source


def test_executed_stress_source_locks_remain_reconstructable() -> None:
    root = Path(__file__).parents[1]
    path = root / "artifacts/spark/native_c0_action_balance_stress_v1/manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["source_locks"].items():
        source = root / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative
