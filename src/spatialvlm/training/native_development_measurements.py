"""Pure contracts for the native-C0 data inventory and H4 runtime benchmark.

These measurements precede ``NATIVE-C0-DEVELOPMENT-v1``.  They do not choose the final sample
count, claim learning, or inspect ``val_unseen``.  The inventory describes the released train
surface at the physical-trajectory unit; the benchmark measures the exact causal-H4 training unit
on the admitted native-Qwen path.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.training.unit_ledger import SINGLE_PASS, UnitLedger, verify_unit_ledger_record

NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION = 2
NATIVE_DEVELOPMENT_MEASUREMENTS_GATE = "native_c0_development_measurements_v2"
NATIVE_DATA_INVENTORY_SCHEMA_VERSION = 1
NATIVE_H4_BENCHMARK_SCHEMA_VERSION = 1
NATIVE_H4_OBJECTIVE = "restricted_four_action_cross_entropy_v1"
NATIVE_H4_HISTORY = "H4_CAUSAL"
NATIVE_DEVELOPMENT_MEASUREMENTS_SEED = 20260825


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value.")
    if not 0.0 < probability <= 1.0:
        raise ValueError("Nearest-rank probability must lie in (0,1].")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


@dataclass(frozen=True)
class NativeDataInventoryProtocol:
    """Eligibility definition shared by inventory and benchmark selection."""

    min_decision_step: int = 3
    max_replay_prefix_actions: int = 96

    def __post_init__(self) -> None:
        if self.min_decision_step < 3:
            raise ValueError("Causal H4 requires at least three preceding actions.")
        if self.max_replay_prefix_actions < self.min_decision_step:
            raise ValueError("Replay ceiling must not precede the H4 decision floor.")

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "history_contract": NATIVE_H4_HISTORY,
            "stop_definition": "official terminal STOP decision in released ground truth",
            "prefix_policy": "exclude a decision when its visible replay prefix contains STOP",
            "instruction_policy": (
                "inventory all released instructions but group and partition by physical "
                "trajectory_id"
            ),
            "sealed_splits": ["val_seen", "val_unseen"],
        }


@dataclass(frozen=True)
class NativeH4BenchmarkProtocol:
    """A bounded real benchmark, not a development-training experiment."""

    learning_rate: float = 2e-5
    train_samples_per_action: int = 32
    calibration_samples_per_action: int = 4
    microbatch_size: int = 1
    accumulation_steps: int = 16
    warmup_updates_excluded_from_timing: int = 2
    measured_updates: int = 6
    sampler_seed: int = NATIVE_DEVELOPMENT_MEASUREMENTS_SEED
    model_seed: int = NATIVE_DEVELOPMENT_MEASUREMENTS_SEED
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_restricted_ce: float = 100.0
    max_hours: float = 1.0
    min_decision_step: int = 3
    max_replay_prefix_actions: int = 96

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Benchmark learning rate must be finite and positive.")
        if self.train_samples_per_action <= 0 or self.calibration_samples_per_action <= 0:
            raise ValueError("Benchmark sample counts must be positive.")
        if self.microbatch_size != 1 or self.accumulation_steps <= 0:
            raise ValueError("Benchmark requires microbatch one and positive accumulation.")
        if self.warmup_updates_excluded_from_timing < 1 or self.measured_updates < 4:
            raise ValueError(
                "Benchmark needs a cold/warm boundary and at least four timed updates."
            )
        if self.unique_train_decisions % self.effective_batch:
            raise ValueError("Training decisions must form complete effective batches.")
        if self.planned_optimizer_updates != (
            self.warmup_updates_excluded_from_timing + self.measured_updates
        ):
            raise ValueError("Train sample count must equal warmup plus measured updates.")
        if self.max_grad_norm <= 0.0 or self.hard_max_gradient_norm <= self.max_grad_norm:
            raise ValueError("Benchmark gradient thresholds are invalid.")
        if self.max_hours <= 0.0:
            raise ValueError("Benchmark time cap must be positive.")
        self.inventory_protocol
        self.unit_ledger

    @property
    def inventory_protocol(self) -> NativeDataInventoryProtocol:
        return NativeDataInventoryProtocol(
            min_decision_step=self.min_decision_step,
            max_replay_prefix_actions=self.max_replay_prefix_actions,
        )

    @property
    def unique_train_decisions(self) -> int:
        return self.train_samples_per_action * len(R2RCEAction)

    @property
    def unique_calibration_decisions(self) -> int:
        return self.calibration_samples_per_action * len(R2RCEAction)

    @property
    def effective_batch(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    @property
    def planned_optimizer_updates(self) -> int:
        return self.unique_train_decisions // self.effective_batch

    @property
    def timed_update_steps(self) -> tuple[int, ...]:
        return tuple(
            range(self.warmup_updates_excluded_from_timing + 1, self.planned_optimizer_updates + 1)
        )

    @property
    def unit_ledger(self) -> UnitLedger:
        return UnitLedger(
            unique_train_decisions=self.unique_train_decisions,
            unique_calibration_decisions=self.unique_calibration_decisions,
            microbatch_size=self.microbatch_size,
            accumulation_steps=self.accumulation_steps,
            epochs=1,
            planned_optimizer_updates=self.planned_optimizer_updates,
            reuse_policy=SINGLE_PASS,
            scheduler_unit="optimizer_update",
            calibration_updates=(self.planned_optimizer_updates,),
            checkpoint_updates=(self.planned_optimizer_updates,),
            sampler_seed=self.sampler_seed,
            notes=(
                "Causal-H4 throughput/memory measurement only; no checkpoint or rate selection."
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema_version": NATIVE_H4_BENCHMARK_SCHEMA_VERSION,
            "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
            "history_contract": NATIVE_H4_HISTORY,
            "objective": NATIVE_H4_OBJECTIVE,
            "unique_train_decisions": self.unique_train_decisions,
            "unique_calibration_decisions": self.unique_calibration_decisions,
            "effective_batch": self.effective_batch,
            "planned_optimizer_updates": self.planned_optimizer_updates,
            "timed_update_steps": list(self.timed_update_steps),
            "calibration_updates": [self.planned_optimizer_updates],
            "checkpoint_updates": [self.planned_optimizer_updates],
            "unit_ledger": self.unit_ledger.to_record(),
            "development_splits_used": ["train"],
            "sealed_splits": ["val_seen", "val_unseen"],
            "claim_boundary": "runtime_and_memory_measurement_only",
        }


def validate_h4_protocol_record(record: Mapping[str, Any]) -> list[str]:
    """Reconstruct a benchmark protocol and its unit ledger."""
    errors: list[str] = []
    try:
        protocol = NativeH4BenchmarkProtocol(
            learning_rate=float(record["learning_rate"]),
            train_samples_per_action=int(record["train_samples_per_action"]),
            calibration_samples_per_action=int(record["calibration_samples_per_action"]),
            microbatch_size=int(record["microbatch_size"]),
            accumulation_steps=int(record["accumulation_steps"]),
            warmup_updates_excluded_from_timing=int(record["warmup_updates_excluded_from_timing"]),
            measured_updates=int(record["measured_updates"]),
            sampler_seed=int(record["sampler_seed"]),
            model_seed=int(record["model_seed"]),
            weight_decay=float(record["weight_decay"]),
            max_grad_norm=float(record["max_grad_norm"]),
            hard_max_gradient_norm=float(record["hard_max_gradient_norm"]),
            hard_max_restricted_ce=float(record["hard_max_restricted_ce"]),
            max_hours=float(record["max_hours"]),
            min_decision_step=int(record["min_decision_step"]),
            max_replay_prefix_actions=int(record["max_replay_prefix_actions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"H4 benchmark protocol cannot reconstruct: {exc}"]
    if dict(record) != protocol.to_record():
        errors.append("H4 benchmark protocol differs from deterministic reconstruction")
    ledger = record.get("unit_ledger")
    if not isinstance(ledger, dict):
        errors.append("H4 benchmark unit ledger is missing")
    else:
        errors.extend(verify_unit_ledger_record(ledger))
    return errors


def h4_training_sample_order(
    samples: Sequence[Mapping[str, Any]], protocol: NativeH4BenchmarkProtocol
) -> tuple[int, ...]:
    """Return the frozen single-pass benchmark order."""
    if len(samples) != protocol.unique_train_decisions:
        raise ValueError("H4 benchmark training sample count differs from the unit ledger.")
    indices = [int(sample["sample_index"]) for sample in samples]
    if len(indices) != len(set(indices)):
        raise ValueError("H4 benchmark sample indices must be unique.")
    random.Random(protocol.sampler_seed).shuffle(indices)
    return tuple(indices)


def _eligible_steps(
    actions: Sequence[R2RCEAction], protocol: NativeDataInventoryProtocol
) -> list[tuple[int, R2RCEAction]]:
    rows: list[tuple[int, R2RCEAction]] = []
    for step, action in enumerate(actions):
        if step < protocol.min_decision_step or step > protocol.max_replay_prefix_actions:
            continue
        if R2RCEAction.STOP in actions[:step]:
            continue
        rows.append((step, action))
    return rows


def _max_flow_balanced_capacity(
    availability: Mapping[str, frozenset[str]], actions: Sequence[str]
) -> int:
    """Exact maximum equal per-action allocation with each trajectory used at most once."""

    trajectory_ids = sorted(availability)

    def feasible(per_action: int) -> bool:
        source = "__source__"
        sink = "__sink__"
        capacity: dict[str, dict[str, int]] = defaultdict(dict)

        def add_edge(left: str, right: str, value: int) -> None:
            capacity[left][right] = value
            capacity[right].setdefault(left, 0)

        for trajectory_id in trajectory_ids:
            node = f"trajectory:{trajectory_id}"
            add_edge(source, node, 1)
            for action in availability[trajectory_id]:
                add_edge(node, f"action:{action}", 1)
        for action in actions:
            add_edge(f"action:{action}", sink, per_action)

        flow = 0
        target = per_action * len(actions)
        while flow < target:
            parent: dict[str, str | None] = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                left = queue.popleft()
                for right, residual in capacity[left].items():
                    if residual > 0 and right not in parent:
                        parent[right] = left
                        queue.append(right)
            if sink not in parent:
                break
            node = sink
            while parent[node] is not None:
                previous = parent[node]
                capacity[previous][node] -= 1
                capacity[node][previous] = capacity[node].get(previous, 0) + 1
                node = previous
            flow += 1
        return flow == target

    upper = min(
        sum(action in available for available in availability.values()) for action in actions
    )
    upper = min(upper, len(trajectory_ids) // len(actions))
    low = 0
    while low < upper:
        middle = (low + upper + 1) // 2
        if feasible(middle):
            low = middle
        else:
            upper = middle - 1
    return low


def build_data_policy_inventory(
    dataset: R2RCEDataset,
    *,
    protocol: NativeDataInventoryProtocol | None = None,
) -> dict[str, Any]:
    """Summarize train data without choosing development split sizes or viewing outcomes."""

    protocol = protocol or NativeDataInventoryProtocol()
    if dataset.release_manifest is None or dataset.release_manifest.split != "train":
        raise ValueError("The development inventory requires the official train release.")

    grouped: dict[str, list[Any]] = defaultdict(list)
    for episode in dataset.official_episodes:
        grouped[episode.trajectory_id].append(episode)

    physical_counts = Counter({action.name: 0 for action in R2RCEAction})
    instruction_counts = Counter({action.name: 0 for action in R2RCEAction})
    full_physical_counts = Counter({action.name: 0 for action in R2RCEAction})
    availability: dict[str, frozenset[str]] = {}
    scene_rows: dict[str, dict[str, Any]] = {}
    availability_masks: Counter[str] = Counter()
    instruction_multiplicity: Counter[str] = Counter()
    eligible_lengths: list[int] = []
    full_lengths: list[int] = []

    for trajectory_id, episodes in sorted(grouped.items()):
        scene_ids = {episode.scene_id for episode in episodes}
        if len(scene_ids) != 1:
            raise ValueError(f"Physical trajectory {trajectory_id} spans multiple scenes.")
        action_sequences = {
            tuple(int(action) for action in dataset.ground_truth_record(episode.episode_id).actions)
            for episode in episodes
        }
        if len(action_sequences) != 1:
            raise ValueError(
                f"Instruction episodes for physical trajectory {trajectory_id} disagree on actions."
            )
        actions = tuple(R2RCEAction(value) for value in next(iter(action_sequences)))
        eligible = _eligible_steps(actions, protocol)
        per_trajectory = Counter(action.name for _, action in eligible)
        action_set = frozenset(per_trajectory)
        availability[trajectory_id] = action_set
        availability_masks["|".join(sorted(action_set)) or "NONE"] += 1
        instruction_multiplicity[str(len(episodes))] += 1
        full_lengths.append(len(actions))
        eligible_lengths.append(len(eligible))
        full_physical_counts.update(action.name for action in actions)
        physical_counts.update(per_trajectory)
        for action, count in per_trajectory.items():
            instruction_counts[action] += count * len(episodes)

        scene_id = next(iter(scene_ids))
        scene = scene_rows.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "physical_trajectory_count": 0,
                "instruction_episode_count": 0,
                "eligible_physical_decision_count": 0,
                "eligible_instruction_decision_count": 0,
                "eligible_physical_action_counts": {action.name: 0 for action in R2RCEAction},
                "eligible_unique_trajectory_counts": {action.name: 0 for action in R2RCEAction},
            },
        )
        scene["physical_trajectory_count"] += 1
        scene["instruction_episode_count"] += len(episodes)
        scene["eligible_physical_decision_count"] += len(eligible)
        scene["eligible_instruction_decision_count"] += len(eligible) * len(episodes)
        for action in R2RCEAction:
            scene["eligible_physical_action_counts"][action.name] += per_trajectory[action.name]
            scene["eligible_unique_trajectory_counts"][action.name] += int(
                action.name in action_set
            )

    action_names = [action.name for action in R2RCEAction]
    physical_total = sum(physical_counts.values())
    instruction_total = sum(instruction_counts.values())
    physical_prior = {action: physical_counts[action] / physical_total for action in action_names}
    instruction_prior = {
        action: instruction_counts[action] / instruction_total for action in action_names
    }
    unique_by_action = {
        action: sum(action in available for available in availability.values())
        for action in action_names
    }
    cooccurrence = {
        left: {
            right: sum(
                left in available and right in available for available in availability.values()
            )
            for right in action_names
        }
        for left in action_names
    }
    max_balanced = _max_flow_balanced_capacity(availability, action_names)

    def length_summary(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "minimum": min(values),
            "median": statistics.median(values),
            "p95_nearest_rank": _nearest_rank(values, 0.95),
            "maximum": max(values),
        }

    inventory = {
        "schema_version": NATIVE_DATA_INVENTORY_SCHEMA_VERSION,
        "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
        "status": "passed",
        "protocol": protocol.to_record(),
        "release_manifest": asdict(dataset.release_manifest),
        "summary": {
            "scene_count": len(scene_rows),
            "physical_trajectory_count": len(grouped),
            "instruction_episode_count": len(dataset.official_episodes),
            "instruction_multiplicity_histogram": dict(sorted(instruction_multiplicity.items())),
            "full_physical_action_counts": dict(full_physical_counts),
            "eligible_physical_decision_count": physical_total,
            "eligible_instruction_decision_count": instruction_total,
            "eligible_physical_action_counts": dict(physical_counts),
            "eligible_instruction_action_counts": dict(instruction_counts),
            "eligible_physical_action_prior": physical_prior,
            "eligible_instruction_action_prior": instruction_prior,
            "eligible_unique_trajectory_counts": unique_by_action,
            "eligible_action_cooccurrence_by_trajectory": cooccurrence,
            "trajectory_action_availability_histogram": dict(sorted(availability_masks.items())),
            "maximum_balanced_globally_disjoint_trajectories_per_action": max_balanced,
            "maximum_balanced_globally_disjoint_total": max_balanced * len(action_names),
            "full_action_sequence_length": length_summary(full_lengths),
            "eligible_decisions_per_trajectory": length_summary(eligible_lengths),
        },
        "scene_inventory": sorted(scene_rows.values(), key=lambda row: row["scene_id"]),
        "baselines": {
            "uniform_balanced_ce_nats": math.log(len(action_names)),
            "physical_prior_natural_ce_nats": -sum(
                probability * math.log(probability) for probability in physical_prior.values()
            ),
            "physical_prior_balanced_ce_nats": -sum(
                math.log(physical_prior[action]) for action in action_names
            )
            / len(action_names),
            "instruction_prior_natural_ce_nats": -sum(
                probability * math.log(probability) for probability in instruction_prior.values()
            ),
            "instruction_prior_balanced_ce_nats": -sum(
                math.log(instruction_prior[action]) for action in action_names
            )
            / len(action_names),
        },
        "inventory_sha256": "",
        "evidence_boundary": {
            "established_if_passed": (
                "Exact released-train scene, physical-trajectory, instruction, H4-eligible "
                "decision, action-frequency, overlap, and balanced-allocation capacity counts."
            ),
            "not_established": (
                "Final development sample sizes, cluster-aware power, model learning, runtime, "
                "closed-loop competence, or any val_seen/val_unseen result."
            ),
        },
    }
    hash_payload = {**inventory, "inventory_sha256": ""}
    inventory["inventory_sha256"] = _canonical_sha256(hash_payload)
    return inventory


def validate_inventory_record(
    record: Mapping[str, Any],
    dataset: R2RCEDataset,
    *,
    protocol: NativeDataInventoryProtocol | None = None,
) -> list[str]:
    """Recompute the entire inventory from the locked release."""
    try:
        expected = build_data_policy_inventory(dataset, protocol=protocol)
    except Exception as exc:
        return [f"inventory reconstruction failed: {exc}"]
    return [] if dict(record) == expected else ["inventory differs from full reconstruction"]


def summarize_h4_timings(
    update_events: Sequence[Mapping[str, Any]],
    *,
    protocol: NativeH4BenchmarkProtocol,
    model_load_seconds: float,
    cache_prepare_seconds: float,
    calibration_seconds: float,
    checkpoint_seconds: float,
) -> dict[str, Any]:
    """Produce deterministic throughput coefficients from complete event records."""
    if len(update_events) != protocol.planned_optimizer_updates:
        raise ValueError("Benchmark event count differs from the unit ledger.")
    steps = [int(row["update_step"]) for row in update_events]
    if steps != list(range(1, protocol.planned_optimizer_updates + 1)):
        raise ValueError("Benchmark update steps are not complete and ordered.")
    durations = [float(row["elapsed_seconds"]) for row in update_events]
    if not all(math.isfinite(value) and value > 0.0 for value in durations):
        raise ValueError("Benchmark update durations must be finite and positive.")
    timed = [
        float(row["elapsed_seconds"])
        for row in update_events
        if int(row["update_step"]) in protocol.timed_update_steps
    ]
    if len(timed) != protocol.measured_updates:
        raise ValueError("Benchmark timed-update set differs from the protocol.")
    fixed = {
        "model_load_seconds": float(model_load_seconds),
        "cache_prepare_seconds": float(cache_prepare_seconds),
        "calibration_seconds": float(calibration_seconds),
        "checkpoint_seconds": float(checkpoint_seconds),
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in fixed.values()):
        raise ValueError("Benchmark boundary timings must be finite and nonnegative.")
    return {
        "all_update_seconds": durations,
        "timed_update_steps": list(protocol.timed_update_steps),
        "timed_update_seconds": timed,
        "timed_update_mean_seconds": statistics.fmean(timed),
        "timed_update_p50_seconds": statistics.median(timed),
        "timed_update_p95_nearest_rank_seconds": _nearest_rank(timed, 0.95),
        "timed_update_minimum_seconds": min(timed),
        "timed_update_maximum_seconds": max(timed),
        "seconds_per_training_decision_from_timed_mean": statistics.fmean(timed)
        / protocol.effective_batch,
        "calibration_decision_count": protocol.unique_calibration_decisions,
        "calibration_seconds_per_decision": calibration_seconds
        / protocol.unique_calibration_decisions,
        **fixed,
        "projection_contract": {
            "training_seconds_for_u_updates_at_p50": "U * timed_update_p50_seconds",
            "training_seconds_for_u_updates_at_p95": ("U * timed_update_p95_nearest_rank_seconds"),
            "calibration_seconds_for_c_decisions": ("C * calibration_seconds_per_decision"),
            "fixed_checkpoint_seconds_per_boundary": checkpoint_seconds,
            "model_load_seconds_per_invocation": model_load_seconds,
            "cache_prepare_seconds_is_dataset_size_specific": True,
        },
    }
