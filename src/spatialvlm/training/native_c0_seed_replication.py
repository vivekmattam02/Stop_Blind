"""Frozen contracts for paired native-C0 training-seed replication.

The original train/selection/outcome partitions remain immutable.  A replication seed controls
only fresh LoRA initialization and the single-pass order of the already frozen training records.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_SEED,
    NativeC0DevelopmentProtocol,
    canonical_sha256,
)

NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION = 1
NATIVE_C0_SEED_REPLICATION_GATE = "native_qwen_c0_paired_seed_replication_v1"
NATIVE_C0_SEED_REPLICATION_VERIFIER_GATE = (
    "independent_native_qwen_c0_paired_seed_replication_verification_v1"
)
NATIVE_C0_SEED_REPLICATION_STRESS_GATE = "native_qwen_c0_paired_seed_replication_stress_v1"
NATIVE_C0_SEED_REPLICATION_STRESS_VERIFIER_GATE = (
    "independent_native_qwen_c0_paired_seed_replication_stress_verification_v1"
)

PARTITION_SEED = NATIVE_C0_SEED
BOOTSTRAP_SEEDS = (NATIVE_C0_SEED, NATIVE_C0_SEED + 1)
COMPLETED_SEED = NATIVE_C0_SEED
NEW_REPLICATION_SEEDS = (NATIVE_C0_SEED + 1, NATIVE_C0_SEED + 2)
ALL_REPLICATION_SEEDS = (COMPLETED_SEED, *NEW_REPLICATION_SEEDS)
TREATMENTS = ("unweighted", "inverse_frequency")


def replication_run_id(treatment: str, seed: int) -> str:
    """Return the only admitted run ID for a new treatment/seed pair."""
    validate_treatment_seed(treatment, seed)
    return f"native_c0_seed_replication_{treatment}_s{seed}_v1"


def validate_treatment_seed(treatment: str, seed: int) -> None:
    """Reject treatment or seed search beyond the preregistered paired design."""
    if treatment not in TREATMENTS:
        raise ValueError(f"Unknown replication treatment: {treatment}")
    if seed not in NEW_REPLICATION_SEEDS:
        raise ValueError(f"Seed {seed} is not one of the two preregistered new seeds.")


def replication_training_order(records: Sequence[Mapping[str, Any]], seed: int) -> tuple[int, ...]:
    """Shuffle only the already frozen training indices using the replication seed."""
    if seed not in ALL_REPLICATION_SEEDS:
        raise ValueError(f"Seed {seed} is outside the frozen three-seed set.")
    protocol = NativeC0DevelopmentProtocol()
    indices = [int(row["sample_index"]) for row in records if row.get("split") == "train"]
    if len(indices) != protocol.train_decisions or len(indices) != len(set(indices)):
        raise ValueError("Replication training records differ from the frozen 3,072 decisions.")
    random.Random(seed).shuffle(indices)
    return tuple(indices)


def stress_sample_indices(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """Choose a fixed 16-decision, four-per-action numerical stress microbatch."""
    grouped: dict[str, list[int]] = {action.name: [] for action in R2RCEAction}
    for row in records:
        if row.get("split") != "train":
            continue
        action = str(row["target_action"])
        grouped[action].append(int(row["sample_index"]))
    for values in grouped.values():
        values.sort()
    if any(len(values) < 4 for values in grouped.values()):
        raise ValueError("Replication stress requires at least four training decisions per action.")
    selected = tuple(index for action in R2RCEAction for index in grouped[action.name][:4])
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("Replication stress selection is not 16 unique decisions.")
    return selected


def build_replication_contract(
    *,
    treatment: str,
    replication_seed: int,
    frozen_sample_records_sha256: str,
    frozen_cache_manifest_sha256: str,
    frozen_cache_index_sha256: str,
    original_unweighted_result_sha256: str,
    original_inverse_frequency_result_sha256: str,
    treatment_weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Build the immutable record that separates partitioning from training randomness."""
    validate_treatment_seed(treatment, replication_seed)
    expected_actions = {action.name for action in R2RCEAction}
    if treatment == "unweighted":
        if treatment_weights is not None:
            raise ValueError("Unweighted replication cannot carry class weights.")
        weights = None
    else:
        if treatment_weights is None or set(treatment_weights) != expected_actions:
            raise ValueError("Inverse-frequency replication requires all four frozen weights.")
        weights = {name: float(treatment_weights[name]) for name in sorted(treatment_weights)}
        if any(value <= 0.0 for value in weights.values()):
            raise ValueError("Replication weights must be positive.")
    record = {
        "schema_version": NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
        "gate": NATIVE_C0_SEED_REPLICATION_GATE,
        "treatment": treatment,
        "replication_seed": replication_seed,
        "all_reported_seeds": list(ALL_REPLICATION_SEEDS),
        "new_seeds": list(NEW_REPLICATION_SEEDS),
        "partition_seed": PARTITION_SEED,
        "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
        "seed_controls": {
            "changes": ["lora_initialization", "single_pass_training_order"],
            "fixed": [
                "sample_membership",
                "partition_membership",
                "evaluation_order",
                "bootstrap_resamples",
                "model_revision",
                "optimizer",
                "learning_rate",
                "update_budget",
                "checkpoint_selection_rule",
                "outcome_thresholds",
            ],
        },
        "frozen_sample_records_sha256": frozen_sample_records_sha256,
        "frozen_cache_manifest_sha256": frozen_cache_manifest_sha256,
        "frozen_cache_index_sha256": frozen_cache_index_sha256,
        "original_unweighted_result_sha256": original_unweighted_result_sha256,
        "original_inverse_frequency_result_sha256": (original_inverse_frequency_result_sha256),
        "treatment_weights": weights,
        "interpretation": (
            "paired_seed_robustness_of_an_adaptive_train_release_development_comparison"
        ),
        "reporting_rule": "report_every_preregistered_seed_and_every_paired_delta_never_best_seed",
    }
    record["contract_sha256"] = canonical_sha256(record)
    return record


def validate_replication_contract(record: Mapping[str, Any]) -> list[str]:
    """Check internally reconstructable invariants without trusting prose."""
    errors: list[str] = []
    try:
        treatment = str(record["treatment"])
        seed = int(record["replication_seed"])
        validate_treatment_seed(treatment, seed)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"replication contract identity is invalid: {exc}"]
    if list(record.get("all_reported_seeds", [])) != list(ALL_REPLICATION_SEEDS):
        errors.append("replication contract changes the reported seed set")
    if list(record.get("new_seeds", [])) != list(NEW_REPLICATION_SEEDS):
        errors.append("replication contract changes the new seed set")
    if record.get("partition_seed") != PARTITION_SEED:
        errors.append("replication contract changes the partition seed")
    if list(record.get("bootstrap_seeds", [])) != list(BOOTSTRAP_SEEDS):
        errors.append("replication contract changes bootstrap seeds")
    without_hash = dict(record)
    observed_hash = without_hash.pop("contract_sha256", None)
    if observed_hash != canonical_sha256(without_hash):
        errors.append("replication contract hash does not reconstruct")
    weights = record.get("treatment_weights")
    if treatment == "unweighted" and weights is not None:
        errors.append("unweighted replication unexpectedly carries weights")
    if treatment == "inverse_frequency":
        if not isinstance(weights, Mapping):
            errors.append("inverse-frequency replication weights are missing")
        elif set(weights) != {action.name for action in R2RCEAction}:
            errors.append("inverse-frequency replication weight actions differ")
    return errors
