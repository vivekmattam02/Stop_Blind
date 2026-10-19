"""Frozen protocol and pure summaries for native-Qwen optimizer admission.

This gate answers one narrow question: which learning-rate region can move the FP32 LoRA adapters
without non-finite state, missing gradients, or immediate balanced-calibration collapse?  It is a
numerical range finder, not a policy-training result.  The explicit :class:`UnitLedger` prevents
unique decisions, repeated sample exposures, microbatches, and optimizer updates from being
silently conflated again.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.unit_ledger import (
    REPEATED_EXPOSURE_RESHUFFLED,
    SINGLE_PASS,
    UnitLedger,
    verify_unit_ledger_record,
)

NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION = 1
NATIVE_OPTIMIZER_ADMISSION_GATE = "native_qwen_optimizer_admission_v1"
NATIVE_OPTIMIZER_OBJECTIVE = "restricted_four_action_cross_entropy_v1"
NATIVE_OPTIMIZER_PHASE_A_RATES = (2e-7, 2e-6, 2e-5, 5e-5)
NATIVE_OPTIMIZER_SEED = 20260824


def _stable_hash(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NativeOptimizerAdmissionProtocol:
    """One immutable stress or main numerical-range-finder contract."""

    profile: str = "main"
    learning_rates: tuple[float, ...] = NATIVE_OPTIMIZER_PHASE_A_RATES
    train_samples_per_action: int = 16
    calibration_samples_per_action: int = 16
    epochs: int = 16
    microbatch_size: int = 1
    accumulation_steps: int = 16
    calibration_updates: tuple[int, ...] = (0, 8, 16, 32, 64)
    checkpoint_updates: tuple[int, ...] = (8, 16, 32, 64)
    sampler_seed: int = NATIVE_OPTIMIZER_SEED
    model_seed: int = NATIVE_OPTIMIZER_SEED
    weight_decay: float = 0.01
    warmup_fraction: float = 0.03
    minimum_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_restricted_ce: float = 100.0
    minimum_improvement_nats: float = 1e-4
    min_distinct_predicted_actions: int = 2
    max_hours: float = 5.0
    min_decision_step: int = 3
    max_replay_prefix_actions: int = 96

    def __post_init__(self) -> None:
        if self.profile not in {"stress", "main"}:
            raise ValueError("profile must be 'stress' or 'main'.")
        if not self.learning_rates or any(
            not math.isfinite(rate) or rate <= 0.0 for rate in self.learning_rates
        ):
            raise ValueError("learning_rates must be finite and positive.")
        if tuple(sorted(set(self.learning_rates))) != self.learning_rates:
            raise ValueError("learning_rates must be strictly increasing and unique.")
        if self.train_samples_per_action <= 0 or self.calibration_samples_per_action <= 0:
            raise ValueError("Per-action sample counts must be positive.")
        if self.microbatch_size != 1:
            raise ValueError("The measured GB10 contract currently requires microbatch_size=1.")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0,1).")
        if not 0.0 < self.minimum_lr_ratio <= 1.0:
            raise ValueError("minimum_lr_ratio must lie in (0,1].")
        if self.max_grad_norm <= 0.0 or self.hard_max_gradient_norm <= self.max_grad_norm:
            raise ValueError("Gradient thresholds are invalid.")
        if self.minimum_improvement_nats < 0.0:
            raise ValueError("minimum_improvement_nats must be non-negative.")
        if not 1 <= self.min_distinct_predicted_actions <= len(R2RCEAction):
            raise ValueError("min_distinct_predicted_actions is invalid.")
        if self.max_hours <= 0.0:
            raise ValueError("max_hours must be positive.")
        # Construction is the CR-022 consistency gate.
        self.unit_ledger

    @classmethod
    def stress(cls) -> NativeOptimizerAdmissionProtocol:
        """One update per arm, enough to exercise every expensive code and verifier path."""
        return cls(
            profile="stress",
            train_samples_per_action=4,
            calibration_samples_per_action=4,
            epochs=1,
            calibration_updates=(0, 1),
            checkpoint_updates=(1,),
            max_hours=1.0,
        )

    @property
    def unique_train_decisions(self) -> int:
        return self.train_samples_per_action * len(R2RCEAction)

    @property
    def unique_calibration_decisions(self) -> int:
        return self.calibration_samples_per_action * len(R2RCEAction)

    @property
    def planned_optimizer_updates(self) -> int:
        return self.unique_train_decisions * self.epochs // self.effective_batch

    @property
    def effective_batch(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    @property
    def warmup_updates(self) -> int:
        if self.planned_optimizer_updates <= 1 or self.warmup_fraction == 0.0:
            return 0
        return max(1, math.ceil(self.planned_optimizer_updates * self.warmup_fraction))

    @property
    def unit_ledger(self) -> UnitLedger:
        return UnitLedger(
            unique_train_decisions=self.unique_train_decisions,
            unique_calibration_decisions=self.unique_calibration_decisions,
            microbatch_size=self.microbatch_size,
            accumulation_steps=self.accumulation_steps,
            epochs=self.epochs,
            planned_optimizer_updates=self.planned_optimizer_updates,
            reuse_policy=(
                SINGLE_PASS if self.epochs == 1 else REPEATED_EXPOSURE_RESHUFFLED
            ),
            scheduler_unit="optimizer_update",
            calibration_updates=self.calibration_updates,
            checkpoint_updates=self.checkpoint_updates,
            sampler_seed=self.sampler_seed,
            notes=(
                "Stress-path execution check only."
                if self.profile == "stress"
                else "Repeated-exposure optimizer range finder; not a learning-quality study."
            ),
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
                "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
                "objective": NATIVE_OPTIMIZER_OBJECTIVE,
                "unique_train_decisions": self.unique_train_decisions,
                "unique_calibration_decisions": self.unique_calibration_decisions,
                "planned_optimizer_updates": self.planned_optimizer_updates,
                "effective_batch": self.effective_batch,
                "warmup_updates": self.warmup_updates,
                "unit_ledger": self.unit_ledger.to_record(),
                "development_splits_used": ["train"],
                "sealed_splits": ["val_seen", "val_unseen"],
                "automatic_high_rate_expansion": False,
                "paper_claim_boundary": "numerical_range_finder_only",
            }
        )
        record["learning_rates"] = list(self.learning_rates)
        record["calibration_updates"] = list(self.calibration_updates)
        record["checkpoint_updates"] = list(self.checkpoint_updates)
        return record


def validate_protocol_record(record: Mapping[str, Any]) -> list[str]:
    """Reconstruct a serialized protocol and independently check its unit ledger."""
    errors: list[str] = []
    if record.get("schema_version") != NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION:
        errors.append("optimizer-admission schema version mismatch")
        return errors
    if record.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE:
        errors.append("optimizer-admission gate mismatch")
    if record.get("objective") != NATIVE_OPTIMIZER_OBJECTIVE:
        errors.append("optimizer-admission objective mismatch")
    try:
        protocol = NativeOptimizerAdmissionProtocol(
            profile=str(record["profile"]),
            learning_rates=tuple(float(value) for value in record["learning_rates"]),
            train_samples_per_action=int(record["train_samples_per_action"]),
            calibration_samples_per_action=int(record["calibration_samples_per_action"]),
            epochs=int(record["epochs"]),
            microbatch_size=int(record["microbatch_size"]),
            accumulation_steps=int(record["accumulation_steps"]),
            calibration_updates=tuple(int(value) for value in record["calibration_updates"]),
            checkpoint_updates=tuple(int(value) for value in record["checkpoint_updates"]),
            sampler_seed=int(record["sampler_seed"]),
            model_seed=int(record["model_seed"]),
            weight_decay=float(record["weight_decay"]),
            warmup_fraction=float(record["warmup_fraction"]),
            minimum_lr_ratio=float(record["minimum_lr_ratio"]),
            max_grad_norm=float(record["max_grad_norm"]),
            hard_max_gradient_norm=float(record["hard_max_gradient_norm"]),
            hard_max_restricted_ce=float(record["hard_max_restricted_ce"]),
            minimum_improvement_nats=float(record["minimum_improvement_nats"]),
            min_distinct_predicted_actions=int(record["min_distinct_predicted_actions"]),
            max_hours=float(record["max_hours"]),
            min_decision_step=int(record["min_decision_step"]),
            max_replay_prefix_actions=int(record["max_replay_prefix_actions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [*errors, f"protocol record cannot reconstruct: {exc}"]
    expected = protocol.to_record()
    if dict(record) != expected:
        errors.append("protocol record differs from deterministic reconstruction")
    ledger = record.get("unit_ledger")
    if not isinstance(ledger, dict):
        errors.append("unit ledger is missing")
    else:
        errors.extend(verify_unit_ledger_record(ledger))
    return errors


def learning_rate_multiplier(update_step: int, protocol: NativeOptimizerAdmissionProtocol) -> float:
    """Return the preregistered warmup-plus-cosine multiplier for one accepted update."""
    if update_step < 1 or update_step > protocol.planned_optimizer_updates:
        raise ValueError("update_step is outside the protocol range.")
    warmup = protocol.warmup_updates
    if warmup and update_step <= warmup:
        return update_step / warmup
    remaining = protocol.planned_optimizer_updates - warmup
    if remaining <= 1:
        return 1.0
    progress = (update_step - warmup - 1) / (remaining - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return protocol.minimum_lr_ratio + (1.0 - protocol.minimum_lr_ratio) * cosine


def training_sample_order(
    samples: Sequence[Mapping[str, Any]], protocol: NativeOptimizerAdmissionProtocol
) -> tuple[int, ...]:
    """Return the deterministic reshuffled sample-index order for every declared epoch."""
    if len(samples) != protocol.unique_train_decisions:
        raise ValueError("Training sample count differs from the unit ledger.")
    base = [int(sample["sample_index"]) for sample in samples]
    if len(base) != len(set(base)):
        raise ValueError("Training sample indices must be unique within the frozen partition.")
    order: list[int] = []
    for epoch in range(protocol.epochs):
        epoch_values = list(base)
        random.Random(
            int(_stable_hash("native-optimizer-order-v1", protocol.sampler_seed, epoch), 16)
        ).shuffle(epoch_values)
        order.extend(epoch_values)
    if len(order) != protocol.unit_ledger.total_sample_exposures:
        raise RuntimeError("Training order does not match declared sample exposures.")
    return tuple(order)


def summarize_calibration(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one balanced held-out calibration surface."""
    if not decisions:
        raise ValueError("Calibration decisions must be non-empty.")
    expected_actions = {action.name for action in R2RCEAction}
    target_actions = {str(row["target_action"]) for row in decisions}
    if target_actions != expected_actions:
        raise ValueError("Calibration must contain every legal target action.")
    full_nll = [float(row["target_token_nll"]) for row in decisions]
    restricted_ce = [float(row["restricted_action_ce"]) for row in decisions]
    target_probability = [float(row["target_restricted_probability"]) for row in decisions]
    if not all(math.isfinite(value) for value in (*full_nll, *restricted_ce, *target_probability)):
        raise ValueError("Calibration contains a non-finite metric.")
    histogram = {
        action.name: sum(row["predicted_action"] == action.name for row in decisions)
        for action in R2RCEAction
    }
    per_action = {}
    for action in R2RCEAction:
        rows = [row for row in decisions if row["target_action"] == action.name]
        per_action[action.name] = {
            "count": len(rows),
            "mean_restricted_action_ce": sum(float(row["restricted_action_ce"]) for row in rows)
            / len(rows),
            "mean_target_restricted_probability": sum(
                float(row["target_restricted_probability"]) for row in rows
            )
            / len(rows),
            "recall": sum(row["predicted_action"] == action.name for row in rows) / len(rows),
        }
    return {
        "decision_count": len(decisions),
        "mean_target_token_nll": sum(full_nll) / len(full_nll),
        "mean_restricted_action_ce": sum(restricted_ce) / len(restricted_ce),
        "mean_target_restricted_probability": sum(target_probability) / len(target_probability),
        "descriptive_accuracy": sum(bool(row["correct"]) for row in decisions) / len(decisions),
        "predicted_action_histogram": histogram,
        "distinct_predicted_action_count": sum(value > 0 for value in histogram.values()),
        "per_target_action": per_action,
    }


def select_admitted_rate(
    arm_results: Sequence[Mapping[str, Any]], protocol: NativeOptimizerAdmissionProtocol
) -> dict[str, Any]:
    """Select the lowest numerically admitted rate without upgrading the gate to policy evidence."""
    if len(arm_results) != len(protocol.learning_rates):
        raise ValueError("Arm-result count differs from the frozen learning-rate set.")
    rows: list[dict[str, Any]] = []
    for expected_rate, result in zip(protocol.learning_rates, arm_results, strict=True):
        rate = float(result["learning_rate"])
        if rate != expected_rate:
            raise ValueError("Arm results are not in frozen learning-rate order.")
        calibrations = list(result["calibrations"])
        if [int(row["update_step"]) for row in calibrations] != list(
            protocol.calibration_updates
        ):
            raise ValueError("Arm calibration cadence differs from the unit ledger.")
        baseline = calibrations[0]["summary"]
        candidates = calibrations[1:]
        selected = min(
            candidates,
            key=lambda row: (
                float(row["summary"]["mean_restricted_action_ce"]),
                int(row["update_step"]),
            ),
        )
        improvement = float(baseline["mean_restricted_action_ce"]) - float(
            selected["summary"]["mean_restricted_action_ce"]
        )
        finite_updates = bool(result.get("all_updates_finite"))
        changed_updates = (
            int(result.get("changed_update_count", -1))
            == protocol.planned_optimizer_updates
        )
        diverse = (
            int(selected["summary"]["distinct_predicted_action_count"])
            >= protocol.min_distinct_predicted_actions
        )
        admitted = (
            result.get("status") == "passed"
            and finite_updates
            and changed_updates
            and diverse
            and improvement >= protocol.minimum_improvement_nats
        )
        rows.append(
            {
                "learning_rate": rate,
                "admitted": admitted,
                "selected_update": int(selected["update_step"]),
                "baseline_mean_restricted_action_ce": float(
                    baseline["mean_restricted_action_ce"]
                ),
                "selected_mean_restricted_action_ce": float(
                    selected["summary"]["mean_restricted_action_ce"]
                ),
                "improvement_nats": improvement,
                "selected_distinct_predicted_action_count": int(
                    selected["summary"]["distinct_predicted_action_count"]
                ),
                "finite_updates": finite_updates,
                "all_updates_changed_weights": changed_updates,
            }
        )
    admitted_rows = [row for row in rows if row["admitted"]]
    selected_rate = min((row["learning_rate"] for row in admitted_rows), default=None)
    return {
        "status": "rate_admitted" if selected_rate is not None else "no_rate_admitted",
        "selected_learning_rate": selected_rate,
        "selection_rule": (
            "lowest frozen rate with finite changed updates, balanced-calibration CE improvement "
            "at or above the frozen minimum, and at least the frozen number of predicted actions"
        ),
        "arm_decisions": rows,
        "paper_claim_boundary": "numerical_range_finder_only",
    }
