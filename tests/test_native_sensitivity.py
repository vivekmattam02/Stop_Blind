"""CPU tests for the source-locked CR-025 diagnostic contract."""

from __future__ import annotations

from dataclasses import dataclass

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.native_sensitivity import (
    NATIVE_SENSITIVITY_CONDITIONS,
    NativeSensitivityProtocol,
    select_native_sensitivity_samples,
    summarize_native_sensitivity,
    total_variation,
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

    def __init__(self) -> None:
        self.official_episodes = tuple(
            _Episode(
                episode_id=str(index),
                trajectory_id=f"trajectory-{index}",
                scene_id=f"scene-{index % 7}",
                instruction_text=f"instruction-{index}",
            )
            for index in range(20)
        )

    def ground_truth_record(self, episode_id: str) -> _GroundTruth:
        offset = int(episode_id) % 4
        actions = tuple(R2RCEAction((offset + step) % 4) for step in range(4))
        return _GroundTruth(actions=actions)


def _decision(predicted: R2RCEAction, probabilities: tuple[float, ...]) -> dict:
    return {
        "predicted_action": predicted.name,
        "action_logits": {
            action.name: probabilities[int(action)] * 10.0 for action in R2RCEAction
        },
        "scores": {
            action.name: {
                "token_nll": 1.0 + int(action),
                "restricted_probability": probabilities[int(action)],
            }
            for action in R2RCEAction
        },
    }


def _rows(paired_swap: tuple[float, ...]) -> list[dict]:
    protocol = NativeSensitivityProtocol(samples_per_action=2)
    clean = (0.25, 0.25, 0.25, 0.25)
    rows = []
    for sample_index in range(protocol.sample_count):
        target = R2RCEAction(sample_index % 4)
        rows.append(
            {
                "sample_index": sample_index,
                "target_action_id": int(target),
                "conditions": {
                    "clean": _decision(R2RCEAction.STOP, clean),
                    "paired_image_swap": _decision(R2RCEAction.STOP, paired_swap),
                    "constant_image": _decision(R2RCEAction.STOP, clean),
                    "instruction_swap": _decision(R2RCEAction.STOP, clean),
                },
            }
        )
    return rows


def test_selector_is_balanced_unique_and_uses_different_action_donors() -> None:
    protocol = NativeSensitivityProtocol(samples_per_action=2)
    samples = select_native_sensitivity_samples(_Dataset(), protocol)  # type: ignore[arg-type]

    assert len(samples) == 8
    assert len({sample.trajectory_id for sample in samples}) == 8
    assert {
        action: sum(sample.target_action_id == int(action) for sample in samples)
        for action in R2RCEAction
    } == {action: 2 for action in R2RCEAction}
    for sample in samples:
        donor = samples[sample.donor_sample_index]
        assert donor.trajectory_id != sample.trajectory_id
        assert donor.target_action_id != sample.target_action_id


def test_total_variation_has_expected_scale() -> None:
    assert total_variation((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert total_variation((0.5, 0.5), (0.5, 0.5)) == 0.0


def test_summary_returns_sensitive_insensitive_and_inconclusive_regions() -> None:
    protocol = NativeSensitivityProtocol(samples_per_action=2)
    sensitive = summarize_native_sensitivity(_rows((0.10, 0.30, 0.30, 0.30)), protocol)
    insensitive = summarize_native_sensitivity(_rows((0.25, 0.25, 0.25, 0.25)), protocol)
    inconclusive = summarize_native_sensitivity(
        _rows((0.249, 0.251, 0.25, 0.25)), protocol
    )

    assert sensitive["verdict"] == "sensitive_admit_optimizer_diagnostic"
    assert insensitive["verdict"] == "insensitive_block_optimizer_diagnostic"
    assert inconclusive["verdict"] == "inconclusive_expand_before_optimizer_diagnostic"
    assert set(sensitive["per_condition"]) == set(NATIVE_SENSITIVITY_CONDITIONS)


def test_summary_rejects_missing_condition() -> None:
    protocol = NativeSensitivityProtocol(samples_per_action=2)
    rows = _rows((0.25, 0.25, 0.25, 0.25))
    del rows[0]["conditions"]["paired_image_swap"]
    try:
        summarize_native_sensitivity(rows, protocol)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected missing paired control to fail closed.")
