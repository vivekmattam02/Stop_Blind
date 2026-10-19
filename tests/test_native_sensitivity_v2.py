"""CPU contracts for the disjoint, saturation-robust sensitivity confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.native_sensitivity_v2 import (
    NATIVE_SENSITIVITY_V2_CONDITIONS,
    NATIVE_SENSITIVITY_V2_VERDICT_ADMIT,
    NativeSensitivityV2Protocol,
    select_native_sensitivity_v2_samples,
    summarize_native_sensitivity_v2,
)


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_text: str


@dataclass(frozen=True)
class _GroundTruth:
    actions: tuple[R2RCEAction, ...]


class _Dataset:
    split = "val_seen"

    def __init__(self, count: int = 160) -> None:
        self.official_episodes = tuple(
            _Episode(
                episode_id=str(index),
                trajectory_id=f"trajectory-{index}",
                scene_id=f"scene-{index % 29}",
                instruction_text=f"instruction-{index}",
            )
            for index in range(count)
        )

    def ground_truth_record(self, episode_id: str) -> _GroundTruth:
        offset = int(episode_id) % 4
        return _GroundTruth(
            actions=tuple(R2RCEAction((offset + step) % 4) for step in range(4))
        )


def _decision(logits: tuple[float, float, float, float]) -> dict:
    maximum = max(logits)
    exponentials = [pow(2.718281828459045, value - maximum) for value in logits]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    predicted = R2RCEAction(max(range(4), key=lambda index: logits[index]))
    return {
        "predicted_action": predicted.name,
        "action_logits": {
            action.name: logits[int(action)] for action in R2RCEAction
        },
        "scores": {
            action.name: {
                "token_nll": 1.0 + int(action),
                "restricted_probability": probabilities[int(action)],
            }
            for action in R2RCEAction
        },
    }


def _rows(
    *, paired_delta: float, repeat_delta: float = 0.0, samples_per_action: int = 4
) -> list[dict]:
    protocol = NativeSensitivityV2Protocol(samples_per_action=samples_per_action)
    clean_logits = (8.0, 4.0, 0.0, -4.0)
    paired_logits = (8.0 - paired_delta, 4.0 + paired_delta, 0.0, -4.0)
    repeat_logits = (8.0 - repeat_delta, 4.0 + repeat_delta, 0.0, -4.0)
    rows = []
    for sample_index in range(protocol.sample_count):
        target = R2RCEAction(sample_index % 4)
        rows.append(
            {
                "sample_index": sample_index,
                "target_action_id": int(target),
                "conditions": {
                    "clean": _decision(clean_logits),
                    "clean_repeat": _decision(repeat_logits),
                    "paired_image_swap": _decision(paired_logits),
                    "instruction_swap": _decision(paired_logits),
                },
            }
        )
    return rows


def test_selector_is_disjoint_balanced_and_deterministic() -> None:
    protocol = NativeSensitivityV2Protocol(samples_per_action=4)
    excluded = {f"trajectory-{index}" for index in range(16)}
    first = select_native_sensitivity_v2_samples(
        _Dataset(), excluded_trajectory_ids=excluded, protocol=protocol  # type: ignore[arg-type]
    )
    second = select_native_sensitivity_v2_samples(
        _Dataset(), excluded_trajectory_ids=excluded, protocol=protocol  # type: ignore[arg-type]
    )

    assert first == second
    assert len(first) == 16
    assert not ({sample.trajectory_id for sample in first} & excluded)
    assert len({sample.trajectory_id for sample in first}) == len(first)
    assert {
        action: sum(sample.target_action_id == int(action) for sample in first)
        for action in R2RCEAction
    } == {action: 4 for action in R2RCEAction}
    for sample in first:
        donor = first[sample.donor_sample_index]
        assert donor.trajectory_id != sample.trajectory_id
        assert donor.target_action_id != sample.target_action_id


def test_selector_requires_v1_exclusions() -> None:
    try:
        select_native_sensitivity_v2_samples(
            _Dataset(),
            excluded_trajectory_ids=(),
            protocol=NativeSensitivityV2Protocol(samples_per_action=4),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "v1 trajectory exclusions" in str(exc)
    else:
        raise AssertionError("Expected missing v1 exclusions to fail closed.")


def test_summary_admits_saturation_robust_logit_response() -> None:
    protocol = NativeSensitivityV2Protocol(samples_per_action=4)
    summary = summarize_native_sensitivity_v2(
        _rows(paired_delta=1.0), protocol
    )

    assert summary["verdict"] == NATIVE_SENSITIVITY_V2_VERDICT_ADMIT
    assert summary["negative_control_passed"] is True
    assert summary["per_action_response_passed"] is True
    assert set(summary["per_condition"]) == set(NATIVE_SENSITIVITY_V2_CONDITIONS)
    assert summary["clean_comparisons"]["paired_image_swap"][
        "median_maximum_centered_action_logit_delta"
    ] >= 1.0


def test_summary_blocks_nondeterministic_repeat() -> None:
    protocol = NativeSensitivityV2Protocol(samples_per_action=4)
    summary = summarize_native_sensitivity_v2(
        _rows(paired_delta=1.0, repeat_delta=0.01), protocol
    )
    assert summary["verdict"] == "invalid_nondeterministic_block_optimizer_diagnostic"


def test_summary_blocks_clear_logit_invariance() -> None:
    protocol = NativeSensitivityV2Protocol(samples_per_action=4)
    summary = summarize_native_sensitivity_v2(
        _rows(paired_delta=0.01), protocol
    )
    assert summary["verdict"] == "logit_insensitive_block_optimizer_diagnostic"


def test_summary_keeps_intermediate_response_inconclusive() -> None:
    protocol = NativeSensitivityV2Protocol(samples_per_action=4)
    summary = summarize_native_sensitivity_v2(
        _rows(paired_delta=0.3), protocol
    )
    assert summary["verdict"] == "inconclusive_block_optimizer_diagnostic"
