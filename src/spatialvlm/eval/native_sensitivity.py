"""Frozen primitives for the CR-025 native-Qwen sensitivity diagnostic.

The diagnostic asks whether the four legal action probabilities at a source-locked H0 decision
change when its real RGB observation is replaced by a real observation from another decision.
It is a forward-only development diagnostic, not a competence or navigation result.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode

NATIVE_SENSITIVITY_PROTOCOL_VERSION = 1
NATIVE_SENSITIVITY_SCHEMA_VERSION = 1
NATIVE_SENSITIVITY_GATE_NAME = "native_qwen_c0_visual_sensitivity_diagnostic_v1"
NATIVE_SENSITIVITY_CONDITIONS = (
    "clean",
    "paired_image_swap",
    "constant_image",
    "instruction_swap",
)
NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS = (
    "docs/NATIVE_QWEN_SENSITIVITY_DIAGNOSTIC.md",
    "scripts/check_native_sensitivity_artifacts.py",
    "scripts/native_sensitivity_sync_files.txt",
    "scripts/run_spark_native_sensitivity.py",
    "scripts/run_spark_native_sensitivity.sh",
    "scripts/spark_launch_helpers.sh",
    "scripts/start_spark_native_sensitivity.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/native_sensitivity.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "tests/test_native_qwen.py",
    "tests/test_native_sensitivity.py",
    "tests/test_native_sensitivity_artifacts.py",
    "tests/test_spark_launch_helpers.py",
)


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeSensitivityProtocol:
    """Pre-registered action-balanced development diagnostic."""

    split: str = "val_seen"
    samples_per_action: int = 4
    seed: int = 20260824
    max_replay_prefix_actions: int = 96
    insensitive_max_median_total_variation: float = 1e-4
    insensitive_max_maximum_total_variation: float = 1e-3
    sensitive_min_median_total_variation: float = 1e-2

    def __post_init__(self) -> None:
        if self.split != "val_seen":
            raise ValueError(
                "The v1 sensitivity diagnostic is frozen to val_seen development data."
            )
        if self.samples_per_action <= 1:
            raise ValueError("Sensitivity requires multiple examples per official action.")
        if self.max_replay_prefix_actions <= 0:
            raise ValueError("Replay-prefix ceiling must be positive.")
        if not (
            0.0
            <= self.insensitive_max_median_total_variation
            <= self.insensitive_max_maximum_total_variation
            < self.sensitive_min_median_total_variation
            <= 1.0
        ):
            raise ValueError("Sensitivity decision thresholds are inconsistent.")

    @property
    def sample_count(self) -> int:
        return self.samples_per_action * len(R2RCEAction)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "protocol_version": NATIVE_SENSITIVITY_PROTOCOL_VERSION,
                "gate": NATIVE_SENSITIVITY_GATE_NAME,
                "sample_count": self.sample_count,
                "samples_per_action": self.samples_per_action,
                "conditions": list(NATIVE_SENSITIVITY_CONDITIONS),
                "primary_control": "paired_image_swap",
                "development_split": "val_seen",
                "sealed_split": "val_unseen",
                "history_contract": "H0_CURRENT",
                "optimization_updates": 0,
                "decision_rule": (
                    "paired-swap median TV >= sensitive minimum => sensitive; paired-swap "
                    "median TV <= insensitive median maximum and maximum TV <= insensitive "
                    "maximum => insensitive; otherwise inconclusive"
                ),
            }
        )
        return record


@dataclass(frozen=True)
class NativeSensitivitySample:
    """One source decision and its deterministic real-observation donor."""

    sample_index: int
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_text: str
    instruction_sha256: str
    action_step_index: int
    target_action_id: int
    target_action: str
    replay_prefix_action_ids: tuple[int, ...]
    replay_prefix_sha256: str
    selection_rank_sha256: str
    donor_sample_index: int

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["replay_prefix_action_ids"] = list(self.replay_prefix_action_ids)
        return record


@dataclass(frozen=True)
class _Candidate:
    episode: R2RCEEpisode
    action: R2RCEAction
    step_index: int
    replay_prefix: tuple[R2RCEAction, ...]
    rank: str


def _canonical_episodes(dataset: R2RCEDataset, seed: int) -> dict[str, R2RCEEpisode]:
    grouped: dict[str, list[R2RCEEpisode]] = {}
    for episode in dataset.official_episodes:
        grouped.setdefault(episode.trajectory_id, []).append(episode)
    return {
        trajectory_id: min(
            episodes,
            key=lambda episode: _stable_hash(
                "native-sensitivity-canonical-instruction-v1",
                seed,
                trajectory_id,
                episode.episode_id,
                episode.instruction_text,
            ),
        )
        for trajectory_id, episodes in grouped.items()
    }


def select_native_sensitivity_samples(
    dataset: R2RCEDataset,
    protocol: NativeSensitivityProtocol | None = None,
) -> tuple[NativeSensitivitySample, ...]:
    """Select unique-trajectory, action-balanced decisions and deterministic real-image donors."""
    protocol = protocol or NativeSensitivityProtocol()
    release_manifest = getattr(dataset, "release_manifest", None)
    observed_split = getattr(release_manifest, "split", None) or getattr(dataset, "split", None)
    if observed_split != protocol.split:
        raise ValueError(
            f"Sensitivity dataset split mismatch: expected {protocol.split}, got {observed_split}."
        )
    canonical = _canonical_episodes(dataset, protocol.seed)
    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for episode in canonical.values():
        actions = dataset.ground_truth_record(episode.episode_id).actions
        for step_index, action in enumerate(actions):
            if step_index > protocol.max_replay_prefix_actions:
                continue
            prefix = actions[:step_index]
            if R2RCEAction.STOP in prefix:
                continue
            buckets[action].append(
                _Candidate(
                    episode=episode,
                    action=action,
                    step_index=step_index,
                    replay_prefix=prefix,
                    rank=_stable_hash(
                        "native-sensitivity-candidate-v1",
                        protocol.seed,
                        episode.trajectory_id,
                        episode.episode_id,
                        step_index,
                        int(action),
                    ),
                )
            )
    for candidates in buckets.values():
        candidates.sort(key=lambda candidate: candidate.rank)

    chosen: list[_Candidate] = []
    used_trajectories: set[str] = set()
    used_scenes: set[str] = set()
    for _ in range(protocol.samples_per_action):
        for action in R2RCEAction:
            eligible = [
                candidate
                for candidate in buckets[action]
                if candidate.episode.trajectory_id not in used_trajectories
            ]
            if not eligible:
                raise ValueError(
                    f"Not enough unique val_seen trajectories for {action.name} sensitivity."
                )
            novel_scene = [
                candidate for candidate in eligible if candidate.episode.scene_id not in used_scenes
            ]
            selected = (novel_scene or eligible)[0]
            chosen.append(selected)
            used_trajectories.add(selected.episode.trajectory_id)
            used_scenes.add(selected.episode.scene_id)

    donor_indices: list[int] = []
    for source_index, source in enumerate(chosen):
        donors = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(chosen)
            if candidate_index != source_index
            and candidate.action != source.action
            and candidate.episode.trajectory_id != source.episode.trajectory_id
        ]
        different_scene = [
            item for item in donors if item[1].episode.scene_id != source.episode.scene_id
        ]
        candidates = different_scene or donors
        if not candidates:
            raise ValueError(f"No valid paired-swap donor exists for sample {source_index}.")
        donor_index, _ = min(
            candidates,
            key=lambda item: _stable_hash(
                "native-sensitivity-donor-v1",
                protocol.seed,
                source.episode.trajectory_id,
                item[1].episode.trajectory_id,
                item[1].step_index,
            ),
        )
        donor_indices.append(donor_index)

    records: list[NativeSensitivitySample] = []
    for sample_index, (candidate, donor_index) in enumerate(
        zip(chosen, donor_indices, strict=True)
    ):
        prefix_ids = tuple(int(action) for action in candidate.replay_prefix)
        prefix_bytes = repr(prefix_ids).encode("ascii")
        records.append(
            NativeSensitivitySample(
                sample_index=sample_index,
                episode_id=candidate.episode.episode_id,
                trajectory_id=candidate.episode.trajectory_id,
                scene_id=candidate.episode.scene_id,
                instruction_text=candidate.episode.instruction_text,
                instruction_sha256=hashlib.sha256(
                    candidate.episode.instruction_text.encode("utf-8")
                ).hexdigest(),
                action_step_index=candidate.step_index,
                target_action_id=int(candidate.action),
                target_action=candidate.action.name,
                replay_prefix_action_ids=prefix_ids,
                replay_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
                selection_rank_sha256=candidate.rank,
                donor_sample_index=donor_index,
            )
        )
    return tuple(records)


def _condition_probabilities(decision: dict[str, Any]) -> tuple[float, float, float, float]:
    scores = decision.get("scores")
    if not isinstance(scores, dict) or set(scores) != {action.name for action in R2RCEAction}:
        raise ValueError("Decision does not contain exactly four official action scores.")
    probabilities = tuple(
        float(scores[action.name]["restricted_probability"]) for action in R2RCEAction
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("Decision contains an invalid restricted action probability.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Restricted action probabilities do not sum to one.")
    return probabilities  # type: ignore[return-value]


def total_variation(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return total-variation distance between two equal-length categorical distributions."""
    if len(first) != len(second) or not first:
        raise ValueError("Total variation requires non-empty equal-length distributions.")
    return 0.5 * sum(abs(left - right) for left, right in zip(first, second, strict=True))


def _centered_action_logits(decision: dict[str, Any]) -> tuple[float, float, float, float]:
    logits = decision.get("action_logits")
    if not isinstance(logits, dict) or set(logits) != {action.name for action in R2RCEAction}:
        raise ValueError("Decision does not contain exactly four action logits.")
    values = tuple(float(logits[action.name]) for action in R2RCEAction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Decision contains a non-finite action logit.")
    center = sum(values) / len(values)
    return tuple(value - center for value in values)  # type: ignore[return-value]


def summarize_native_sensitivity(
    decisions: list[dict[str, Any]],
    protocol: NativeSensitivityProtocol | None = None,
) -> dict[str, Any]:
    """Recompute aggregate condition metrics and the preregistered three-way verdict."""
    protocol = protocol or NativeSensitivityProtocol()
    if len(decisions) != protocol.sample_count:
        raise ValueError("Sensitivity decision count does not match the frozen protocol.")
    if {int(row.get("sample_index", -1)) for row in decisions} != set(
        range(protocol.sample_count)
    ):
        raise ValueError("Sensitivity sample index coverage is invalid.")

    per_condition: dict[str, Any] = {}
    for condition in NATIVE_SENSITIVITY_CONDITIONS:
        rows = [row["conditions"][condition] for row in decisions]
        histogram = {action.name: 0 for action in R2RCEAction}
        target_nll = []
        target_probability = []
        correct = []
        for parent, decision in zip(decisions, rows, strict=True):
            probabilities = _condition_probabilities(decision)
            predicted = decision.get("predicted_action")
            if predicted not in histogram:
                raise ValueError("Sensitivity decision predicted an invalid action.")
            histogram[predicted] += 1
            target = R2RCEAction(int(parent["target_action_id"]))
            score = decision["scores"][target.name]
            nll = float(score["token_nll"])
            if not math.isfinite(nll) or nll < 0.0:
                raise ValueError("Sensitivity decision contains an invalid target-token NLL.")
            target_nll.append(nll)
            target_probability.append(probabilities[int(target)])
            correct.append(predicted == target.name)
        stop_rows = [
            decision
            for parent, decision in zip(decisions, rows, strict=True)
            if int(parent["target_action_id"]) == int(R2RCEAction.STOP)
        ]
        by_target_action = {}
        for target_action in R2RCEAction:
            target_rows = [
                (parent, decision)
                for parent, decision in zip(decisions, rows, strict=True)
                if int(parent["target_action_id"]) == int(target_action)
            ]
            target_histogram = {action.name: 0 for action in R2RCEAction}
            target_nll_values = []
            target_probability_values = []
            for _, decision in target_rows:
                predicted = decision["predicted_action"]
                target_histogram[predicted] += 1
                target_score = decision["scores"][target_action.name]
                target_nll_values.append(float(target_score["token_nll"]))
                target_probability_values.append(
                    float(target_score["restricted_probability"])
                )
            by_target_action[target_action.name] = {
                "decision_count": len(target_rows),
                "predicted_action_histogram": target_histogram,
                "mean_target_token_nll": sum(target_nll_values) / len(target_nll_values),
                "mean_target_restricted_probability": sum(target_probability_values)
                / len(target_probability_values),
                "recall": target_histogram[target_action.name] / len(target_rows),
            }
        per_condition[condition] = {
            "decision_count": len(rows),
            "predicted_action_histogram": histogram,
            "mean_target_token_nll": sum(target_nll) / len(target_nll),
            "mean_target_restricted_probability": sum(target_probability)
            / len(target_probability),
            "descriptive_accuracy": sum(correct) / len(correct),
            "stop_recall": sum(
                row.get("predicted_action") == R2RCEAction.STOP.name for row in stop_rows
            )
            / len(stop_rows),
            "by_target_action": by_target_action,
        }

    comparisons: dict[str, Any] = {}
    clean_probabilities = [
        _condition_probabilities(row["conditions"]["clean"]) for row in decisions
    ]
    clean_logits = [_centered_action_logits(row["conditions"]["clean"]) for row in decisions]
    for condition in NATIVE_SENSITIVITY_CONDITIONS[1:]:
        condition_probabilities = [
            _condition_probabilities(row["conditions"][condition]) for row in decisions
        ]
        distances = [
            total_variation(clean, changed)
            for clean, changed in zip(clean_probabilities, condition_probabilities, strict=True)
        ]
        condition_logits = [
            _centered_action_logits(row["conditions"][condition]) for row in decisions
        ]
        centered_logit_l1 = [
            sum(abs(left - right) for left, right in zip(clean, changed, strict=True))
            / len(R2RCEAction)
            for clean, changed in zip(clean_logits, condition_logits, strict=True)
        ]
        centered_logit_max = [
            max(abs(left - right) for left, right in zip(clean, changed, strict=True))
            for clean, changed in zip(clean_logits, condition_logits, strict=True)
        ]
        changed_predictions = [
            row["conditions"]["clean"]["predicted_action"]
            != row["conditions"][condition]["predicted_action"]
            for row in decisions
        ]
        comparisons[condition] = {
            "total_variation_by_sample": distances,
            "mean_total_variation": sum(distances) / len(distances),
            "median_total_variation": median(distances),
            "maximum_total_variation": max(distances),
            "prediction_change_fraction": sum(changed_predictions) / len(changed_predictions),
            "mean_centered_action_logit_l1": sum(centered_logit_l1)
            / len(centered_logit_l1),
            "maximum_centered_action_logit_delta": max(centered_logit_max),
        }

    primary = comparisons["paired_image_swap"]
    if primary["median_total_variation"] >= protocol.sensitive_min_median_total_variation:
        verdict = "sensitive_admit_optimizer_diagnostic"
        reason = "Paired real-image swaps exceeded the preregistered median TV sensitivity floor."
    elif (
        primary["median_total_variation"]
        <= protocol.insensitive_max_median_total_variation
        and primary["maximum_total_variation"]
        <= protocol.insensitive_max_maximum_total_variation
    ):
        verdict = "insensitive_block_optimizer_diagnostic"
        reason = "Paired real-image swaps remained below both preregistered invariance ceilings."
    else:
        verdict = "inconclusive_expand_before_optimizer_diagnostic"
        reason = "Paired real-image sensitivity fell between the preregistered decision regions."
    return {
        "per_condition": per_condition,
        "clean_comparisons": comparisons,
        "verdict": verdict,
        "verdict_reason": reason,
        "primary_control": "paired_image_swap",
        "constant_image_is_ood_secondary": True,
        "metrics_are_descriptive_not_navigation_performance": True,
    }
