"""Adversarial tests for the shared H0/H1/H4 temporal contract."""

from __future__ import annotations

import pytest

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window


def test_h0_hides_actions_and_exposes_only_current_observation() -> None:
    observations = ["rgb-0", "rgb-1", "rgb-2"]
    actions = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT]
    window = build_history_window(
        observations, actions, contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT]
    )

    assert [frame.observation for frame in window.frames] == ["rgb-2"]
    assert window.frames[0].previous_action_id is None
    assert window.frames[0].relative_timestep == 0
    assert window.to_record()["contract"]["explicitly_excluded_fields"] == [
        "simulator_pose",
        "goal_coordinate",
        "geodesic_distance",
        "future_action",
    ]
    assert window.to_record()["contract"]["sensor_visibility"] == (
        "defined_by_the_selected_C0-C4_arm"
    )


def test_h1_adds_only_the_action_that_led_into_current_observation() -> None:
    observations = ["rgb-0", "rgb-1", "rgb-2"]
    actions = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT]
    window = build_history_window(
        observations, actions, contract=HISTORY_CONTRACTS[HistoryMode.H1_PREVIOUS_ACTION]
    )

    assert [frame.observation for frame in window.frames] == ["rgb-2"]
    assert window.frames[0].previous_action_id == int(R2RCEAction.TURN_LEFT)
    assert window.frames[0].metadata_record()["previous_action"] == "TURN_LEFT"


def test_h4_is_chronological_and_action_aligned() -> None:
    observations = [f"rgb-{index}" for index in range(5)]
    actions = [
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_RIGHT,
    ]
    window = build_history_window(
        observations, actions, contract=HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW]
    )

    assert [frame.observation for frame in window.frames] == [
        "rgb-1",
        "rgb-2",
        "rgb-3",
        "rgb-4",
    ]
    assert [frame.relative_timestep for frame in window.frames] == [-3, -2, -1, 0]
    assert [frame.previous_action_id for frame in window.frames] == [1, 2, 1, 3]
    assert not any(frame.is_padding for frame in window.frames)


def test_h4_left_padding_repeats_initial_observation_without_inventing_actions() -> None:
    window = build_history_window(
        ["rgb-0", "rgb-1"],
        [R2RCEAction.MOVE_FORWARD],
        contract=HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW],
    )

    assert [frame.observation for frame in window.frames] == [
        "rgb-0",
        "rgb-0",
        "rgb-0",
        "rgb-1",
    ]
    assert [frame.is_padding for frame in window.frames] == [True, True, False, False]
    assert [frame.previous_action_id for frame in window.frames] == [None, None, None, 1]


def test_history_alignment_and_stop_leakage_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly one more observation"):
        build_history_window(
            ["rgb-0", "rgb-1"],
            [],
            contract=HISTORY_CONTRACTS[HistoryMode.H1_PREVIOUS_ACTION],
        )
    with pytest.raises(ValueError, match="including STOP"):
        build_history_window(
            ["rgb-0", "rgb-1"],
            [R2RCEAction.STOP],
            contract=HISTORY_CONTRACTS[HistoryMode.H1_PREVIOUS_ACTION],
        )
