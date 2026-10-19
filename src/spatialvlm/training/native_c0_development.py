"""Frozen contracts for the native-Qwen C0 development existence test.

This module contains only deterministic protocol, partitioning, cache-identity, selection, and
evaluation arithmetic.  Habitat rendering and Qwen execution live in the Spark runners.  Keeping
the scientific contract pure makes it independently testable without a simulator or GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode
from spatialvlm.data.habitat_sim_r2r import HABITAT_SIM_COMMIT, HABITAT_SIM_VERSION
from spatialvlm.training.native_history_select import NativeHistorySample
from spatialvlm.training.unit_ledger import SINGLE_PASS, UnitLedger, verify_unit_ledger_record

NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION = 1
NATIVE_C0_DEVELOPMENT_GATE = "native_qwen_c0_development_v1"
NATIVE_C0_CACHE_SCHEMA_VERSION = 1
NATIVE_C0_CACHE_GATE = "native_qwen_c0_content_addressed_cache_v1"
NATIVE_C0_OBJECTIVE = "restricted_four_action_cross_entropy_v1"
NATIVE_C0_HISTORY = "H4_CAUSAL"
NATIVE_C0_SEED = 20260825
MP3D_HABITAT_ARCHIVE_SHA256 = (
    "470948ee78ff4d6dc4c4870395d1e790a0f15c72065d1f62cdd229b2bcbe0d36"
)


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with a stable serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_hash(*parts: object) -> str:
    return canonical_sha256(list(parts))


@dataclass(frozen=True)
class NativeC0DevelopmentProtocol:
    """Exact measured C0 development protocol, frozen before its outcome is observed."""

    train_decisions: int = 3_072
    train_trajectories: int = 2_451
    selection_decisions: int = 256
    primary_evaluation_decisions: int = 640
    diagnostic_decisions_per_action: int = 64
    microbatch_size: int = 1
    accumulation_steps: int = 16
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_restricted_ce: float = 100.0
    calibration_updates: tuple[int, ...] = (0, 32, 64, 96, 128, 160, 192)
    checkpoint_updates: tuple[int, ...] = (0, 32, 64, 96, 128, 160, 192)
    natural_blind_prior_ce_nats: float = 0.92470
    balanced_uniform_ce_nats: float = math.log(4.0)
    required_margin_nats: float = 0.05
    stop_recall_floor: float = 0.25
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = NATIVE_C0_SEED
    sampler_seed: int = NATIVE_C0_SEED
    model_seed: int = NATIVE_C0_SEED
    min_decision_step: int = 3
    max_replay_prefix_actions: int = 96
    hard_total_hours: float = 4.5

    def __post_init__(self) -> None:
        if self.train_decisions != 3_072 or self.train_trajectories != 2_451:
            raise ValueError("C0 train size differs from the measured frozen protocol.")
        if self.selection_decisions != 256 or self.primary_evaluation_decisions != 640:
            raise ValueError("C0 held-out sizes differ from the measured frozen protocol.")
        if self.diagnostic_decisions_per_action != 64:
            raise ValueError("C0 diagnostic must contain 64 decisions per action.")
        if self.microbatch_size != 1 or self.accumulation_steps != 16 or self.epochs != 1:
            raise ValueError("C0 requires the measured single-pass effective-batch-16 contract.")
        if not math.isclose(self.learning_rate, 2e-5, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("C0 learning rate is the frozen 2e-5 numerical anchor.")
        if self.calibration_updates != self.checkpoint_updates:
            raise ValueError("Every C0 calibration point must have a checkpoint.")
        if self.calibration_updates != tuple(range(0, 193, 32)):
            raise ValueError("C0 calibration cadence must be 0/32/.../192.")
        if not 0.0 <= self.stop_recall_floor <= 1.0:
            raise ValueError("STOP recall floor must lie in [0,1].")
        if self.required_margin_nats <= 0.0 or self.bootstrap_samples != 10_000:
            raise ValueError("C0 margin/bootstrap contract changed.")
        if self.min_decision_step < 3 or self.max_replay_prefix_actions < self.min_decision_step:
            raise ValueError("C0 causal-H4 replay limits are invalid.")
        if self.hard_total_hours != 4.5:
            raise ValueError("C0 end-to-end hard cap must remain 4.5 hours.")
        self.unit_ledger

    @property
    def diagnostic_decisions(self) -> int:
        return self.diagnostic_decisions_per_action * len(R2RCEAction)

    @property
    def cached_decisions(self) -> int:
        return (
            self.train_decisions
            + self.selection_decisions
            + self.primary_evaluation_decisions
            + self.diagnostic_decisions
        )

    @property
    def reserved_trajectories(self) -> int:
        return (
            self.selection_decisions
            + self.primary_evaluation_decisions
            + self.diagnostic_decisions
        )

    @property
    def effective_batch(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    @property
    def planned_optimizer_updates(self) -> int:
        return self.train_decisions // self.effective_batch

    @property
    def natural_ce_threshold(self) -> float:
        return self.natural_blind_prior_ce_nats - self.required_margin_nats

    @property
    def balanced_ce_threshold(self) -> float:
        return self.balanced_uniform_ce_nats - self.required_margin_nats

    @property
    def unit_ledger(self) -> UnitLedger:
        return UnitLedger(
            unique_train_decisions=self.train_decisions,
            unique_calibration_decisions=self.selection_decisions,
            microbatch_size=self.microbatch_size,
            accumulation_steps=self.accumulation_steps,
            epochs=self.epochs,
            planned_optimizer_updates=self.planned_optimizer_updates,
            reuse_policy=SINGLE_PASS,
            scheduler_unit="optimizer_update",
            calibration_updates=self.calibration_updates,
            checkpoint_updates=self.checkpoint_updates,
            sampler_seed=self.sampler_seed,
            notes=(
                "Native-Qwen C0 existence test. Selection is train-only and trajectory-disjoint; "
                "primary and balanced outcomes are opened once after checkpoint selection."
            ),
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
                "gate": NATIVE_C0_DEVELOPMENT_GATE,
                "history_contract": NATIVE_C0_HISTORY,
                "objective": NATIVE_C0_OBJECTIVE,
                "diagnostic_decisions": self.diagnostic_decisions,
                "cached_decisions": self.cached_decisions,
                "reserved_trajectories": self.reserved_trajectories,
                "effective_batch": self.effective_batch,
                "planned_optimizer_updates": self.planned_optimizer_updates,
                "natural_ce_threshold": self.natural_ce_threshold,
                "balanced_ce_threshold": self.balanced_ce_threshold,
                "unit_ledger": self.unit_ledger.to_record(),
                "partition_unit": "physical_trajectory_id",
                "training_only_partition_permitting_repeated_trajectory_decisions": True,
                "checkpoint_selection_rule": (
                    "minimum finite selection-partition restricted CE; exact ties choose the "
                    "earlier optimizer update"
                ),
                "interval_method": (
                    "one-stage scene-cluster bootstrap; resample scenes with replacement and "
                    "retain all decisions from each drawn scene"
                ),
                "sealed_splits": ["val_seen", "val_unseen"],
                "claim_boundary": "train_release_c0_existence_test_only",
            }
        )
        for name in ("calibration_updates", "checkpoint_updates"):
            record[name] = list(record[name])
        return record


def validate_protocol_record(record: Mapping[str, Any]) -> list[str]:
    """Rebuild the frozen protocol and its exact optimizer unit ledger."""
    errors: list[str] = []
    try:
        protocol = NativeC0DevelopmentProtocol(
            train_decisions=int(record["train_decisions"]),
            train_trajectories=int(record["train_trajectories"]),
            selection_decisions=int(record["selection_decisions"]),
            primary_evaluation_decisions=int(record["primary_evaluation_decisions"]),
            diagnostic_decisions_per_action=int(record["diagnostic_decisions_per_action"]),
            microbatch_size=int(record["microbatch_size"]),
            accumulation_steps=int(record["accumulation_steps"]),
            epochs=int(record["epochs"]),
            learning_rate=float(record["learning_rate"]),
            weight_decay=float(record["weight_decay"]),
            max_grad_norm=float(record["max_grad_norm"]),
            hard_max_gradient_norm=float(record["hard_max_gradient_norm"]),
            hard_max_restricted_ce=float(record["hard_max_restricted_ce"]),
            calibration_updates=tuple(int(value) for value in record["calibration_updates"]),
            checkpoint_updates=tuple(int(value) for value in record["checkpoint_updates"]),
            natural_blind_prior_ce_nats=float(record["natural_blind_prior_ce_nats"]),
            balanced_uniform_ce_nats=float(record["balanced_uniform_ce_nats"]),
            required_margin_nats=float(record["required_margin_nats"]),
            stop_recall_floor=float(record["stop_recall_floor"]),
            bootstrap_samples=int(record["bootstrap_samples"]),
            bootstrap_seed=int(record["bootstrap_seed"]),
            sampler_seed=int(record["sampler_seed"]),
            model_seed=int(record["model_seed"]),
            min_decision_step=int(record["min_decision_step"]),
            max_replay_prefix_actions=int(record["max_replay_prefix_actions"]),
            hard_total_hours=float(record["hard_total_hours"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"C0 protocol cannot reconstruct: {exc}"]
    if dict(record) != protocol.to_record():
        errors.append("C0 protocol differs from deterministic reconstruction")
    ledger = record.get("unit_ledger")
    if not isinstance(ledger, dict):
        errors.append("C0 unit ledger is missing")
    else:
        errors.extend(verify_unit_ledger_record(ledger))
    return errors


@dataclass(frozen=True)
class _Candidate:
    episode: R2RCEEpisode
    action: R2RCEAction
    step_index: int
    replay_prefix: tuple[R2RCEAction, ...]

    @property
    def decision_identity(self) -> tuple[str, int]:
        return self.episode.trajectory_id, self.step_index


def _canonical_episodes(dataset: R2RCEDataset, seed: int) -> dict[str, R2RCEEpisode]:
    grouped: dict[str, list[R2RCEEpisode]] = defaultdict(list)
    for episode in dataset.official_episodes:
        grouped[episode.trajectory_id].append(episode)
    return {
        trajectory_id: min(
            episodes,
            key=lambda episode: _stable_hash(
                "native-c0-canonical-instruction-v1",
                seed,
                trajectory_id,
                episode.episode_id,
                episode.instruction_text,
            ),
        )
        for trajectory_id, episodes in grouped.items()
    }


def _eligible_candidates(
    dataset: R2RCEDataset,
    protocol: NativeC0DevelopmentProtocol,
) -> dict[str, list[_Candidate]]:
    candidates: dict[str, list[_Candidate]] = {}
    for trajectory_id, episode in _canonical_episodes(dataset, protocol.sampler_seed).items():
        actions = dataset.ground_truth_record(episode.episode_id).actions
        rows: list[_Candidate] = []
        for step_index, action in enumerate(actions):
            if step_index < protocol.min_decision_step:
                continue
            if step_index > protocol.max_replay_prefix_actions:
                continue
            prefix = actions[:step_index]
            if R2RCEAction.STOP in prefix:
                continue
            rows.append(
                _Candidate(
                    episode=episode,
                    action=action,
                    step_index=step_index,
                    replay_prefix=prefix,
                )
            )
        if not rows:
            raise ValueError(f"Physical trajectory {trajectory_id} has no eligible H4 decision.")
        candidates[trajectory_id] = rows
    return candidates


def _balanced_diagnostic(
    candidates: Mapping[str, Sequence[_Candidate]],
    protocol: NativeC0DevelopmentProtocol,
) -> list[_Candidate]:
    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for trajectory_rows in candidates.values():
        for action in R2RCEAction:
            rows = [row for row in trajectory_rows if row.action == action]
            if rows:
                buckets[action].append(
                    min(
                        rows,
                        key=lambda row: _stable_hash(
                            "native-c0-diagnostic-decision-v1",
                            protocol.sampler_seed,
                            row.episode.trajectory_id,
                            row.step_index,
                            int(row.action),
                        ),
                    )
                )
    action_order = sorted(R2RCEAction, key=lambda action: (len(buckets[action]), int(action)))
    for action in action_order:
        buckets[action].sort(
            key=lambda row: _stable_hash(
                "native-c0-diagnostic-trajectory-v1",
                protocol.sampler_seed,
                row.episode.trajectory_id,
                row.step_index,
                int(row.action),
            )
        )
    used: set[str] = set()
    selected: list[_Candidate] = []
    for _ in range(protocol.diagnostic_decisions_per_action):
        for action in action_order:
            chosen = next(
                (row for row in buckets[action] if row.episode.trajectory_id not in used),
                None,
            )
            if chosen is None:
                raise ValueError("Insufficient globally disjoint trajectories for diagnostic.")
            used.add(chosen.episode.trajectory_id)
            selected.append(chosen)
    return selected


def _natural_one_per_trajectory(
    candidates: Mapping[str, Sequence[_Candidate]],
    *,
    count: int,
    used_trajectories: set[str],
    tag: str,
    seed: int,
) -> list[_Candidate]:
    ranked: list[tuple[str, _Candidate]] = []
    for trajectory_id, rows in candidates.items():
        if trajectory_id in used_trajectories:
            continue
        chosen = min(
            rows,
            key=lambda row: _stable_hash(
                f"{tag}-decision-v1", seed, trajectory_id, row.step_index, int(row.action)
            ),
        )
        ranked.append(
            (
                _stable_hash(
                    f"{tag}-trajectory-v1", seed, trajectory_id, chosen.step_index
                ),
                chosen,
            )
        )
    ranked.sort(key=lambda item: item[0])
    selected = [row for _, row in ranked[:count]]
    if len(selected) != count:
        raise ValueError(f"Insufficient trajectories for {tag}: {len(selected)} < {count}.")
    used_trajectories.update(row.episode.trajectory_id for row in selected)
    return selected


def _training_decisions(
    candidates: Mapping[str, Sequence[_Candidate]],
    *,
    used_trajectories: set[str],
    protocol: NativeC0DevelopmentProtocol,
) -> list[_Candidate]:
    remaining = sorted(set(candidates) - used_trajectories)
    if len(remaining) != protocol.train_trajectories:
        raise ValueError(
            f"C0 expected {protocol.train_trajectories} training trajectories, "
            f"found {len(remaining)}."
        )
    base: list[_Candidate] = []
    for trajectory_id in remaining:
        base.append(
            min(
                candidates[trajectory_id],
                key=lambda row: _stable_hash(
                    "native-c0-train-base-decision-v1",
                    protocol.sampler_seed,
                    trajectory_id,
                    row.step_index,
                    int(row.action),
                ),
            )
        )
    selected_ids = {row.decision_identity for row in base}
    extras = [
        row
        for trajectory_id in remaining
        for row in candidates[trajectory_id]
        if row.decision_identity not in selected_ids
    ]
    extras.sort(
        key=lambda row: _stable_hash(
            "native-c0-train-extra-decision-v1",
            protocol.sampler_seed,
            row.episode.trajectory_id,
            row.step_index,
            int(row.action),
        )
    )
    needed = protocol.train_decisions - len(base)
    if needed < 0 or len(extras) < needed:
        raise ValueError("Insufficient eligible decisions for the frozen C0 train partition.")
    return [*base, *extras[:needed]]


def _sample_record(
    candidate: _Candidate,
    *,
    split: str,
    sample_index: int,
    seed: int,
) -> NativeHistorySample:
    prefix_ids = tuple(int(action) for action in candidate.replay_prefix)
    prefix_sha = hashlib.sha256(
        json.dumps(prefix_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    episode = candidate.episode
    return NativeHistorySample(
        split=split,
        sample_index=sample_index,
        episode_id=episode.episode_id,
        trajectory_id=episode.trajectory_id,
        scene_id=episode.scene_id,
        instruction_text=episode.instruction_text,
        instruction_sha256=hashlib.sha256(episode.instruction_text.encode("utf-8")).hexdigest(),
        action_step_index=candidate.step_index,
        target_action_id=int(candidate.action),
        target_action=candidate.action.name,
        replay_prefix_action_ids=prefix_ids,
        replay_prefix_sha256=prefix_sha,
        selection_rank_sha256=_stable_hash(
            "native-c0-partition-membership-v1",
            seed,
            split,
            episode.trajectory_id,
            candidate.step_index,
        ),
        order_rank_sha256=_stable_hash(
            "native-c0-partition-order-v1",
            seed,
            split,
            episode.trajectory_id,
            candidate.step_index,
        ),
    )


def select_native_c0_partitions(
    dataset: R2RCEDataset,
    *,
    protocol: NativeC0DevelopmentProtocol | None = None,
) -> tuple[NativeHistorySample, ...]:
    """Freeze all four train-release partitions at the physical-trajectory boundary.

    The constrained balanced diagnostic is allocated first.  Primary and checkpoint-selection
    trajectories are then sampled naturally without action quotas.  Every remaining physical
    trajectory belongs to training; only training receives additional decisions from a trajectory.
    """
    protocol = protocol or NativeC0DevelopmentProtocol()
    if dataset.release_manifest is None or dataset.release_manifest.split != "train":
        raise ValueError("Native C0 development uses the official train release only.")
    candidates = _eligible_candidates(dataset, protocol)
    expected_total_trajectories = protocol.train_trajectories + protocol.reserved_trajectories
    if len(candidates) != expected_total_trajectories:
        raise ValueError(
            f"Frozen protocol requires {expected_total_trajectories} physical trajectories; "
            f"release has {len(candidates)}."
        )

    diagnostic = _balanced_diagnostic(candidates, protocol)
    used = {row.episode.trajectory_id for row in diagnostic}
    primary = _natural_one_per_trajectory(
        candidates,
        count=protocol.primary_evaluation_decisions,
        used_trajectories=used,
        tag="native-c0-primary",
        seed=protocol.sampler_seed,
    )
    selection = _natural_one_per_trajectory(
        candidates,
        count=protocol.selection_decisions,
        used_trajectories=used,
        tag="native-c0-selection",
        seed=protocol.sampler_seed,
    )
    train = _training_decisions(candidates, used_trajectories=used, protocol=protocol)

    by_split = {
        "train": train,
        "selection": selection,
        "primary_evaluation": primary,
        "balanced_diagnostic": diagnostic,
    }
    records: list[NativeHistorySample] = []
    for split, rows in by_split.items():
        ordered = sorted(
            rows,
            key=lambda row: _stable_hash(
                "native-c0-partition-order-v1",
                protocol.sampler_seed,
                split,
                row.episode.trajectory_id,
                row.step_index,
            ),
        )
        records.extend(
            _sample_record(row, split=split, sample_index=index, seed=protocol.sampler_seed)
            for index, row in enumerate(ordered)
        )

    validate_partition_records([row.to_record() for row in records], protocol=protocol)
    return tuple(records)


def validate_partition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: NativeC0DevelopmentProtocol | None = None,
) -> dict[str, Any]:
    """Fail closed on counts, action balance, decision uniqueness, and trajectory leakage."""
    protocol = protocol or NativeC0DevelopmentProtocol()
    expected_counts = {
        "train": protocol.train_decisions,
        "selection": protocol.selection_decisions,
        "primary_evaluation": protocol.primary_evaluation_decisions,
        "balanced_diagnostic": protocol.diagnostic_decisions,
    }
    counts = Counter(str(row["split"]) for row in records)
    if dict(counts) != expected_counts:
        raise ValueError(f"C0 partition decision counts mismatch: {dict(counts)}.")
    identities = [
        (str(row["trajectory_id"]), int(row["action_step_index"])) for row in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("A C0 physical decision appears more than once across partitions.")
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        trajectory_splits[str(row["trajectory_id"])].add(str(row["split"]))
    leaked = {key: value for key, value in trajectory_splits.items() if len(value) != 1}
    if leaked:
        raise ValueError(f"C0 physical trajectories cross partitions: {list(leaked)[:5]}.")
    split_trajectory_counts = {
        split: len(
            {
                str(row["trajectory_id"])
                for row in records
                if str(row["split"]) == split
            }
        )
        for split in expected_counts
    }
    expected_trajectory_counts = {
        "train": protocol.train_trajectories,
        "selection": protocol.selection_decisions,
        "primary_evaluation": protocol.primary_evaluation_decisions,
        "balanced_diagnostic": protocol.diagnostic_decisions,
    }
    if split_trajectory_counts != expected_trajectory_counts:
        raise ValueError("C0 partition trajectory counts mismatch.")
    diagnostic_counts = Counter(
        str(row["target_action"])
        for row in records
        if str(row["split"]) == "balanced_diagnostic"
    )
    expected_actions = {
        action.name: protocol.diagnostic_decisions_per_action for action in R2RCEAction
    }
    if dict(diagnostic_counts) != expected_actions:
        raise ValueError(f"C0 diagnostic action counts mismatch: {dict(diagnostic_counts)}.")
    for split in expected_counts:
        indices = sorted(
            int(row["sample_index"]) for row in records if str(row["split"]) == split
        )
        if indices != list(range(expected_counts[split])):
            raise ValueError(f"C0 {split} sample indices are not contiguous from zero.")
    return {
        "decision_counts": expected_counts,
        "trajectory_counts": split_trajectory_counts,
        "diagnostic_action_counts": expected_actions,
        "unique_decision_count": len(set(identities)),
        "unique_trajectory_count": len(trajectory_splits),
        "records_sha256": canonical_sha256(list(records)),
    }


def training_sample_order(
    records: Sequence[Mapping[str, Any]],
    protocol: NativeC0DevelopmentProtocol | None = None,
) -> tuple[int, ...]:
    """Return the deterministic single-pass order of training sample indices."""
    protocol = protocol or NativeC0DevelopmentProtocol()
    indices = [int(row["sample_index"]) for row in records if row["split"] == "train"]
    if len(indices) != protocol.train_decisions or len(indices) != len(set(indices)):
        raise ValueError("C0 training samples differ from the unit ledger.")
    random.Random(protocol.sampler_seed).shuffle(indices)
    return tuple(indices)


def native_c0_render_contract() -> dict[str, Any]:
    """Host-path-independent simulator/sensor identity used by cache object keys."""
    return {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "history_contract": NATIVE_C0_HISTORY,
        "habitat_sim_version": HABITAT_SIM_VERSION,
        "habitat_sim_commit": HABITAT_SIM_COMMIT,
        "gpu_device_id": 0,
        "seed": 100,
        "allow_sliding": True,
        "agent_height_m": 1.5,
        "agent_radius_m": 0.1,
        "sensor_height_m": 1.25,
        "rgb_height_width": [224, 224],
        "depth_height_width": [256, 256],
        "hfov_degrees": 90.0,
        "depth_min_m": 0.0,
        "depth_max_m": 10.0,
        "forward_step_m": 0.25,
        "turn_angle_degrees": 15.0,
        "stored_modalities": ["rgb"],
        "stored_frame_count": 4,
        "stored_history_action_count": 3,
    }


def cache_object_identity(
    sample: Mapping[str, Any],
    *,
    episodes_sha256: str,
    ground_truth_sha256: str,
    scene_archive_sha256: str,
) -> dict[str, Any]:
    """Build a run-ID/split/index-independent cache identity for one physical decision."""
    return {
        "schema_version": NATIVE_C0_CACHE_SCHEMA_VERSION,
        "episodes_sha256": episodes_sha256,
        "ground_truth_sha256": ground_truth_sha256,
        "scene_archive_sha256": scene_archive_sha256,
        "scene_id": str(sample["scene_id"]),
        "trajectory_id": str(sample["trajectory_id"]),
        "action_step_index": int(sample["action_step_index"]),
        "replay_prefix_action_ids": [int(value) for value in sample["replay_prefix_action_ids"]],
        "replay_prefix_sha256": str(sample["replay_prefix_sha256"]),
        "render_contract": native_c0_render_contract(),
    }


def cache_object_key(
    sample: Mapping[str, Any],
    *,
    episodes_sha256: str,
    ground_truth_sha256: str,
    scene_archive_sha256: str,
) -> str:
    return canonical_sha256(
        cache_object_identity(
            sample,
            episodes_sha256=episodes_sha256,
            ground_truth_sha256=ground_truth_sha256,
            scene_archive_sha256=scene_archive_sha256,
        )
    )


def select_checkpoint(calibrations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose minimum finite selection CE with the frozen earlier-step tiebreak."""
    candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for row in calibrations:
        step = int(row["update_step"])
        value = float(row["summary"]["mean_restricted_action_ce"])
        if not math.isfinite(value):
            raise ValueError("C0 selection calibration contains non-finite CE.")
        candidates.append((value, step, row))
    if not candidates:
        raise ValueError("C0 checkpoint selection needs at least one calibration.")
    value, step, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "selected_update": step,
        "selected_mean_restricted_action_ce": value,
        "candidate_count": len(candidates),
        "selection_partition_opened": True,
        "primary_evaluation_opened_during_selection": False,
        "selected_calibration_sha256": canonical_sha256(dict(selected)),
        "rule": "minimum finite selection CE; exact ties choose earlier update",
    }


def scene_cluster_bootstrap_upper_bound(
    decisions: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = NATIVE_C0_SEED,
) -> dict[str, Any]:
    """Compute the frozen one-stage scene-cluster bootstrap upper 95% CE bound."""
    if not decisions or samples <= 0:
        raise ValueError("Scene-cluster bootstrap requires decisions and positive resamples.")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        value = float(row["restricted_action_ce"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Scene-cluster bootstrap received invalid restricted CE.")
        grouped[str(row["scene_id"])].append(value)
    scenes = sorted(grouped)
    if not scenes:
        raise ValueError("Scene-cluster bootstrap found no scenes.")
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(samples):
        drawn = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        values = [value for scene in drawn for value in grouped[scene]]
        replicates.append(sum(values) / len(values))
    replicates.sort()
    upper_index = max(0, math.ceil(0.95 * samples) - 1)
    observed = [float(row["restricted_action_ce"]) for row in decisions]
    scene_sizes = [len(grouped[scene]) for scene in scenes]
    return {
        "method": "one_stage_scene_cluster_bootstrap_v1",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "decision_count": len(observed),
        "scene_count": len(scenes),
        "minimum_decisions_per_scene": min(scene_sizes),
        "maximum_decisions_per_scene": max(scene_sizes),
        "mean_decisions_per_scene": sum(scene_sizes) / len(scene_sizes),
        "observed_mean_restricted_action_ce": sum(observed) / len(observed),
        "upper_95_restricted_action_ce": replicates[upper_index],
    }


def evaluate_existence_gate(
    *,
    primary_decisions: Sequence[Mapping[str, Any]],
    diagnostic_decisions: Sequence[Mapping[str, Any]],
    protocol: NativeC0DevelopmentProtocol | None = None,
) -> dict[str, Any]:
    """Apply the preregistered conjunctive existence-test pass rule."""
    protocol = protocol or NativeC0DevelopmentProtocol()
    if len(primary_decisions) != protocol.primary_evaluation_decisions:
        raise ValueError("C0 primary-evaluation decision count mismatch.")
    if len(diagnostic_decisions) != protocol.diagnostic_decisions:
        raise ValueError("C0 balanced-diagnostic decision count mismatch.")
    actions = Counter(str(row["target_action"]) for row in diagnostic_decisions)
    if actions != Counter(
        {action.name: protocol.diagnostic_decisions_per_action for action in R2RCEAction}
    ):
        raise ValueError("C0 diagnostic is not exactly balanced across official actions.")
    primary = scene_cluster_bootstrap_upper_bound(
        primary_decisions,
        samples=protocol.bootstrap_samples,
        seed=protocol.bootstrap_seed,
    )
    diagnostic = scene_cluster_bootstrap_upper_bound(
        diagnostic_decisions,
        samples=protocol.bootstrap_samples,
        seed=protocol.bootstrap_seed + 1,
    )
    stop_rows = [row for row in diagnostic_decisions if row["target_action"] == "STOP"]
    stop_correct = sum(row["predicted_action"] == "STOP" for row in stop_rows)
    stop_recall = stop_correct / len(stop_rows)
    conditions = {
        "natural_upper_95_below_blind_prior_minus_margin": (
            primary["upper_95_restricted_action_ce"] < protocol.natural_ce_threshold
        ),
        "balanced_upper_95_below_uniform_minus_margin": (
            diagnostic["upper_95_restricted_action_ce"] < protocol.balanced_ce_threshold
        ),
        "stop_recall_at_or_above_non_mute_floor": stop_recall >= protocol.stop_recall_floor,
    }
    passed = all(conditions.values())
    return {
        "status": "passed" if passed else "not_demonstrated_at_frozen_budget",
        "conditions": conditions,
        "primary_natural": {
            **primary,
            "blind_reference_ce_nats": protocol.natural_blind_prior_ce_nats,
            "required_margin_nats": protocol.required_margin_nats,
            "threshold_ce_nats": protocol.natural_ce_threshold,
        },
        "balanced_diagnostic": {
            **diagnostic,
            "uniform_reference_ce_nats": protocol.balanced_uniform_ce_nats,
            "required_margin_nats": protocol.required_margin_nats,
            "threshold_ce_nats": protocol.balanced_ce_threshold,
            "stop_count": len(stop_rows),
            "stop_correct": stop_correct,
            "stop_recall": stop_recall,
            "stop_recall_floor": protocol.stop_recall_floor,
        },
        "interpretation_if_passed": (
            "Native C0 beat the frozen blind references under this train-only existence-test "
            "budget; this is not closed-loop navigation evidence."
        ),
        "interpretation_if_failed": (
            "Native C0 was not demonstrated at this frozen rate, seed, horizon, and data budget; "
            "failure is not evidence that C0 cannot learn."
        ),
    }
