from __future__ import annotations

import pytest

from spatialvlm.training.native_c0_development import NativeC0DevelopmentProtocol
from spatialvlm.training.native_c0_seed_replication import (
    ALL_REPLICATION_SEEDS,
    NEW_REPLICATION_SEEDS,
    build_replication_contract,
    replication_run_id,
    replication_training_order,
    stress_sample_indices,
    validate_replication_contract,
)


def _records() -> list[dict[str, object]]:
    protocol = NativeC0DevelopmentProtocol()
    actions = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    return [
        {
            "split": "train",
            "sample_index": index,
            "target_action": actions[index % len(actions)],
        }
        for index in range(protocol.train_decisions)
    ]


def _contract(treatment: str = "unweighted", seed: int = NEW_REPLICATION_SEEDS[0]):
    weights = (
        None
        if treatment == "unweighted"
        else {"STOP": 10.0, "MOVE_FORWARD": 0.4, "TURN_LEFT": 1.4, "TURN_RIGHT": 1.7}
    )
    return build_replication_contract(
        treatment=treatment,
        replication_seed=seed,
        frozen_sample_records_sha256="a" * 64,
        frozen_cache_manifest_sha256="b" * 64,
        frozen_cache_index_sha256="c" * 64,
        original_unweighted_result_sha256="d" * 64,
        original_inverse_frequency_result_sha256="e" * 64,
        treatment_weights=weights,
    )


def test_seed_replication_changes_order_without_changing_membership() -> None:
    records = _records()
    orders = [replication_training_order(records, seed) for seed in ALL_REPLICATION_SEEDS]
    assert len(set(orders)) == len(orders)
    assert all(set(order) == set(range(len(records))) for order in orders)


def test_stress_surface_is_balanced_and_deterministic() -> None:
    records = _records()
    first = stress_sample_indices(records)
    assert first == stress_sample_indices(records)
    assert len(first) == len(set(first)) == 16
    by_index = {int(row["sample_index"]): row for row in records}
    assert {
        action: sum(by_index[index]["target_action"] == action for index in first)
        for action in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    } == {"STOP": 4, "MOVE_FORWARD": 4, "TURN_LEFT": 4, "TURN_RIGHT": 4}


@pytest.mark.parametrize("treatment", ["unweighted", "inverse_frequency"])
@pytest.mark.parametrize("seed", NEW_REPLICATION_SEEDS)
def test_contract_accepts_only_the_four_new_paired_runs(treatment: str, seed: int) -> None:
    contract = _contract(treatment, seed)
    assert validate_replication_contract(contract) == []
    assert replication_run_id(treatment, seed).endswith(f"_s{seed}_v1")


def test_contract_rejects_seed_search_and_tampering() -> None:
    with pytest.raises(ValueError, match="preregistered"):
        _contract(seed=99)
    contract = _contract()
    contract["partition_seed"] = 99
    assert validate_replication_contract(contract)
