"""Disjoint saturation-robust native-Qwen visual-sensitivity confirmation.

Version 1 used total variation (TV) between restricted four-action probabilities as its
admission statistic.  That statistic is valid, but the observed fresh model was sufficiently
softmax-saturated that large relative action-logit movements could map to almost-zero TV.  This
version does not rewrite the v1 verdict.  It freezes a new, disjoint cohort and asks a narrower
question: does replacing the real RGB observation move the *relative* four-action logits by more
than deterministic repeat noise, consistently across every target-action stratum?

The result remains a pathway diagnostic.  It is not evidence of semantic grounding, navigation
competence, or generalization.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode

NATIVE_SENSITIVITY_V2_PROTOCOL_VERSION = 2
NATIVE_SENSITIVITY_V2_SCHEMA_VERSION = 2
NATIVE_SENSITIVITY_V2_GATE_NAME = "native_qwen_c0_visual_sensitivity_diagnostic_v2"
NATIVE_SENSITIVITY_V2_VERDICT_ADMIT = (
    "logit_sensitive_admit_optimizer_diagnostic"
)
NATIVE_SENSITIVITY_V2_CONDITIONS = (
    "clean",
    "clean_repeat",
    "paired_image_swap",
    "instruction_swap",
)
NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS = (
    "docs/NATIVE_QWEN_SENSITIVITY_DIAGNOSTIC_V2.md",
    "scripts/check_native_sensitivity_v2_artifacts.py",
    "scripts/native_sensitivity_v2_sync_files.txt",
    "scripts/run_spark_native_sensitivity_v2.py",
    "scripts/run_spark_native_sensitivity_v2.sh",
    "scripts/run_spark_native_sensitivity.py",
    "scripts/spark_launch_helpers.sh",
    "scripts/start_spark_native_sensitivity_v2.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/data/native_qwen_artifacts.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/native_sensitivity.py",
    "src/spatialvlm/eval/native_sensitivity_v2.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "tests/test_native_qwen.py",
    "tests/test_native_sensitivity_v2.py",
    "tests/test_native_sensitivity_v2_artifacts.py",
    "tests/test_spark_launch_helpers.py",
)


def _stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NativeSensitivityV2Protocol:
    """Frozen confirmatory contract derived after, but disjoint from, v1."""

    split: str = "val_seen"
    samples_per_action: int = 16
    seed: int = 20260824
    max_replay_prefix_actions: int = 96
    repeat_max_centered_logit_delta: float = 1e-6
    response_delta: float = 0.5
    response_min_fraction: float = 0.75
    response_min_per_action_median: float = 0.5
    insensitive_maximum_centered_logit_delta: float = 0.25

    def __post_init__(self) -> None:
        if self.split != "val_seen":
            raise ValueError("Sensitivity v2 is frozen to val_seen development data.")
        if self.samples_per_action < 4:
            raise ValueError("Sensitivity v2 needs at least four samples per action.")
        if self.max_replay_prefix_actions <= 0:
            raise ValueError("Replay-prefix ceiling must be positive.")
        if self.repeat_max_centered_logit_delta < 0.0:
            raise ValueError("Repeat tolerance must be non-negative.")
        if not (
            0.0
            < self.insensitive_maximum_centered_logit_delta
            < self.response_delta
            <= self.response_min_per_action_median
        ):
            raise ValueError("Sensitivity-v2 logit thresholds are inconsistent.")
        if not 0.0 < self.response_min_fraction <= 1.0:
            raise ValueError("Response fraction must lie in (0,1].")

    @property
    def sample_count(self) -> int:
        return self.samples_per_action * len(R2RCEAction)

    @property
    def forward_pass_count(self) -> int:
        return self.sample_count * len(NATIVE_SENSITIVITY_V2_CONDITIONS)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "schema_version": NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
                "protocol_version": NATIVE_SENSITIVITY_V2_PROTOCOL_VERSION,
                "gate": NATIVE_SENSITIVITY_V2_GATE_NAME,
                "sample_count": self.sample_count,
                "forward_pass_count": self.forward_pass_count,
                "conditions": list(NATIVE_SENSITIVITY_V2_CONDITIONS),
                "primary_control": "paired_image_swap",
                "negative_control": "clean_repeat",
                "development_split": "val_seen",
                "sealed_split": "val_unseen",
                "history_contract": "H0_CURRENT",
                "optimization_updates": 0,
                "selection_relation_to_v1": (
                    "exclude every v1 source trajectory, then select with a versioned v2 salt"
                ),
                "decision_rule": (
                    "admit only if clean-repeat centered-logit max is within tolerance, at least "
                    "the required fraction of paired swaps reaches response_delta, and every "
                    "target-action stratum reaches the required median; block as insensitive only "
                    "if the global paired maximum stays below the insensitive ceiling; otherwise "
                    "remain inconclusive"
                ),
                "threshold_provenance": (
                    "Frozen after v1 exposed probability saturation. A 0.5 relative-logit move "
                    "changes pairwise odds by exp(0.5) and is above the observed deterministic "
                    "repeat floor; the disjoint v2 cohort is the confirmatory sample."
                ),
            }
        )
        return record


@dataclass(frozen=True)
class NativeSensitivityV2Sample:
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
                "native-sensitivity-canonical-instruction-v2",
                seed,
                trajectory_id,
                episode.episode_id,
                episode.instruction_text,
            ),
        )
        for trajectory_id, episodes in grouped.items()
    }


def select_native_sensitivity_v2_samples(
    dataset: R2RCEDataset,
    *,
    excluded_trajectory_ids: Iterable[str],
    protocol: NativeSensitivityV2Protocol | None = None,
) -> tuple[NativeSensitivityV2Sample, ...]:
    """Select a v1-disjoint, action-balanced cohort with deterministic real-image donors."""

    protocol = protocol or NativeSensitivityV2Protocol()
    excluded = {str(value) for value in excluded_trajectory_ids}
    if not excluded:
        raise ValueError("Sensitivity v2 requires the frozen v1 trajectory exclusions.")
    release_manifest = getattr(dataset, "release_manifest", None)
    observed_split = getattr(release_manifest, "split", None) or getattr(dataset, "split", None)
    if observed_split != protocol.split:
        raise ValueError(
            f"Sensitivity-v2 split mismatch: expected {protocol.split}, got {observed_split}."
        )

    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for episode in _canonical_episodes(dataset, protocol.seed).values():
        if episode.trajectory_id in excluded:
            continue
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
                        "native-sensitivity-candidate-v2",
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
                    f"Not enough v1-disjoint val_seen trajectories for {action.name}."
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
        candidates = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(chosen)
            if candidate_index != source_index
            and candidate.action != source.action
            and candidate.episode.trajectory_id != source.episode.trajectory_id
        ]
        different_scene = [
            item for item in candidates if item[1].episode.scene_id != source.episode.scene_id
        ]
        candidates = different_scene or candidates
        if not candidates:
            raise ValueError(f"No valid v2 paired-swap donor exists for sample {source_index}.")
        donor_index, _ = min(
            candidates,
            key=lambda item: _stable_hash(
                "native-sensitivity-donor-v2",
                protocol.seed,
                source.episode.trajectory_id,
                item[1].episode.trajectory_id,
                item[1].step_index,
            ),
        )
        donor_indices.append(donor_index)

    records: list[NativeSensitivityV2Sample] = []
    for sample_index, (candidate, donor_index) in enumerate(
        zip(chosen, donor_indices, strict=True)
    ):
        prefix_ids = tuple(int(action) for action in candidate.replay_prefix)
        records.append(
            NativeSensitivityV2Sample(
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
                replay_prefix_sha256=hashlib.sha256(repr(prefix_ids).encode("ascii")).hexdigest(),
                selection_rank_sha256=candidate.rank,
                donor_sample_index=donor_index,
            )
        )
    return tuple(records)


def _probabilities(decision: Mapping[str, Any]) -> tuple[float, float, float, float]:
    scores = decision.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != {
        action.name for action in R2RCEAction
    }:
        raise ValueError("Decision does not contain exactly four official action scores.")
    values = tuple(
        float(scores[action.name]["restricted_probability"]) for action in R2RCEAction
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Decision contains invalid action probabilities.")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Restricted action probabilities do not sum to one.")
    return values  # type: ignore[return-value]


def _centered_logits(decision: Mapping[str, Any]) -> tuple[float, float, float, float]:
    logits = decision.get("action_logits")
    if not isinstance(logits, Mapping) or set(logits) != {
        action.name for action in R2RCEAction
    }:
        raise ValueError("Decision does not contain exactly four action logits.")
    values = tuple(float(logits[action.name]) for action in R2RCEAction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Decision contains non-finite action logits.")
    center = sum(values) / len(values)
    return tuple(value - center for value in values)  # type: ignore[return-value]


def _comparison(
    decisions: Sequence[Mapping[str, Any]],
    condition: str,
    protocol: NativeSensitivityV2Protocol,
) -> dict[str, Any]:
    tv_by_sample: list[float] = []
    l1_by_sample: list[float] = []
    max_by_sample: list[float] = []
    prediction_changes: list[bool] = []
    for row in decisions:
        clean = row["conditions"]["clean"]
        changed = row["conditions"][condition]
        clean_probabilities = _probabilities(clean)
        changed_probabilities = _probabilities(changed)
        tv_by_sample.append(
            0.5
            * sum(
                abs(left - right)
                for left, right in zip(
                    clean_probabilities, changed_probabilities, strict=True
                )
            )
        )
        clean_logits = _centered_logits(clean)
        changed_logits = _centered_logits(changed)
        deltas = tuple(
            abs(left - right)
            for left, right in zip(clean_logits, changed_logits, strict=True)
        )
        l1_by_sample.append(sum(deltas) / len(deltas))
        max_by_sample.append(max(deltas))
        prediction_changes.append(clean["predicted_action"] != changed["predicted_action"])

    by_target_action: dict[str, Any] = {}
    for action in R2RCEAction:
        indices = [
            index
            for index, row in enumerate(decisions)
            if int(row["target_action_id"]) == int(action)
        ]
        values = [max_by_sample[index] for index in indices]
        by_target_action[action.name] = {
            "decision_count": len(values),
            "median_maximum_centered_action_logit_delta": median(values),
            "minimum_maximum_centered_action_logit_delta": min(values),
            "response_fraction": sum(value >= protocol.response_delta for value in values)
            / len(values),
        }
    return {
        "total_variation_by_sample": tv_by_sample,
        "median_total_variation": median(tv_by_sample),
        "mean_total_variation": sum(tv_by_sample) / len(tv_by_sample),
        "maximum_total_variation": max(tv_by_sample),
        "centered_action_logit_l1_by_sample": l1_by_sample,
        "median_centered_action_logit_l1": median(l1_by_sample),
        "mean_centered_action_logit_l1": sum(l1_by_sample) / len(l1_by_sample),
        "maximum_centered_action_logit_delta_by_sample": max_by_sample,
        "median_maximum_centered_action_logit_delta": median(max_by_sample),
        "maximum_centered_action_logit_delta": max(max_by_sample),
        "response_fraction": sum(value >= protocol.response_delta for value in max_by_sample)
        / len(max_by_sample),
        "prediction_change_fraction": sum(prediction_changes) / len(prediction_changes),
        "by_target_action": by_target_action,
    }


def _condition_summary(
    decisions: Sequence[Mapping[str, Any]], condition: str
) -> dict[str, Any]:
    histogram = {action.name: 0 for action in R2RCEAction}
    correct: list[bool] = []
    target_probabilities: list[float] = []
    target_nlls: list[float] = []
    for row in decisions:
        decision = row["conditions"][condition]
        predicted = str(decision["predicted_action"])
        if predicted not in histogram:
            raise ValueError("Sensitivity-v2 decision predicted an invalid action.")
        histogram[predicted] += 1
        target = R2RCEAction(int(row["target_action_id"]))
        target_score = decision["scores"][target.name]
        probability = float(target_score["restricted_probability"])
        nll = float(target_score["token_nll"])
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError("Sensitivity-v2 decision contains invalid target NLL.")
        target_probabilities.append(probability)
        target_nlls.append(nll)
        correct.append(predicted == target.name)
    stop_rows = [
        row["conditions"][condition]
        for row in decisions
        if int(row["target_action_id"]) == int(R2RCEAction.STOP)
    ]
    return {
        "decision_count": len(decisions),
        "predicted_action_histogram": histogram,
        "descriptive_accuracy": sum(correct) / len(correct),
        "stop_recall": sum(
            row["predicted_action"] == R2RCEAction.STOP.name for row in stop_rows
        )
        / len(stop_rows),
        "mean_target_restricted_probability": sum(target_probabilities)
        / len(target_probabilities),
        "mean_target_token_nll": sum(target_nlls) / len(target_nlls),
    }


def summarize_native_sensitivity_v2(
    decisions: Sequence[Mapping[str, Any]],
    protocol: NativeSensitivityV2Protocol | None = None,
) -> dict[str, Any]:
    """Recompute the v2 negative control, saturation-robust metrics, and verdict."""

    protocol = protocol or NativeSensitivityV2Protocol()
    if len(decisions) != protocol.sample_count:
        raise ValueError("Sensitivity-v2 decision count differs from the frozen protocol.")
    if {int(row.get("sample_index", -1)) for row in decisions} != set(
        range(protocol.sample_count)
    ):
        raise ValueError("Sensitivity-v2 sample-index coverage is invalid.")
    for row in decisions:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(
            NATIVE_SENSITIVITY_V2_CONDITIONS
        ):
            raise ValueError("Sensitivity-v2 condition surface is incomplete.")

    per_condition = {
        condition: _condition_summary(decisions, condition)
        for condition in NATIVE_SENSITIVITY_V2_CONDITIONS
    }
    comparisons = {
        condition: _comparison(decisions, condition, protocol)
        for condition in NATIVE_SENSITIVITY_V2_CONDITIONS[1:]
    }
    repeat = comparisons["clean_repeat"]
    paired = comparisons["paired_image_swap"]
    repeat_passed = (
        repeat["maximum_centered_action_logit_delta"]
        <= protocol.repeat_max_centered_logit_delta
    )
    per_action_passed = all(
        record["median_maximum_centered_action_logit_delta"]
        >= protocol.response_min_per_action_median
        for record in paired["by_target_action"].values()
    )
    response_passed = (
        paired["response_fraction"] >= protocol.response_min_fraction
        and per_action_passed
    )

    if not repeat_passed:
        verdict = "invalid_nondeterministic_block_optimizer_diagnostic"
        reason = "Identical clean repeats exceeded the frozen centered-logit noise tolerance."
    elif response_passed:
        verdict = NATIVE_SENSITIVITY_V2_VERDICT_ADMIT
        reason = (
            "Disjoint paired real-image swaps exceeded the frozen response fraction and every "
            "target-action median while the identical-input control remained deterministic."
        )
    elif (
        paired["maximum_centered_action_logit_delta"]
        <= protocol.insensitive_maximum_centered_logit_delta
    ):
        verdict = "logit_insensitive_block_optimizer_diagnostic"
        reason = "Every paired real-image swap stayed below the frozen logit-invariance ceiling."
    else:
        verdict = "inconclusive_block_optimizer_diagnostic"
        reason = "The disjoint cohort fell between the frozen logit response regions."

    return {
        "per_condition": per_condition,
        "clean_comparisons": comparisons,
        "negative_control_passed": repeat_passed,
        "per_action_response_passed": per_action_passed,
        "response_fraction_passed": (
            paired["response_fraction"] >= protocol.response_min_fraction
        ),
        "verdict": verdict,
        "verdict_reason": reason,
        "primary_control": "paired_image_swap",
        "probability_tv_is_descriptive_not_admission_metric": True,
        "metrics_are_pathway_diagnostics_not_navigation_performance": True,
    }
