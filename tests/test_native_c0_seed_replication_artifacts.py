from __future__ import annotations

from pathlib import Path

from spatialvlm.training.native_c0_action_balance_full import NativeC0ActionBalanceProtocol
from spatialvlm.training.native_c0_development import NativeC0DevelopmentProtocol
from spatialvlm.training.native_c0_seed_replication import NEW_REPLICATION_SEEDS


def test_replication_protocol_changes_training_rng_but_not_other_frozen_values() -> None:
    seed = NEW_REPLICATION_SEEDS[0]
    unweighted = NativeC0DevelopmentProtocol(sampler_seed=seed, model_seed=seed)
    weighted = NativeC0ActionBalanceProtocol(sampler_seed=seed, model_seed=seed)
    assert unweighted.sampler_seed == weighted.sampler_seed == seed
    assert unweighted.model_seed == weighted.model_seed == seed
    assert unweighted.bootstrap_seed == weighted.bootstrap_seed == 20260825
    assert unweighted.train_decisions == weighted.train_decisions == 3072
    assert unweighted.calibration_updates == weighted.calibration_updates


def test_runner_wraps_existing_verified_cores_without_editing_them() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_spark_native_c0_seed_replication.py"
    ).read_text(encoding="utf-8")
    assert "run_spark_native_c0_development as unweighted_runner" in text
    assert "run_spark_native_c0_action_balance as weighted_runner" in text
    assert 'replacements["_expected_manifest"] = expected_manifest' in text
    assert 'replacements["expected_manifest"] = expected_manifest' in text
