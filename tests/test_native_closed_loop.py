"""Pure tests for the frozen native-C0 closed-loop timing and rollout contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
)
from spatialvlm.eval.native_closed_loop import (
    CLOSED_LOOP_STRESS_DECISIONS,
    ClosedLoopDecisionTiming,
    closed_loop_protocol_record,
    run_instrumented_r2r_ce_episode,
    select_closed_loop_scope,
    summarize_closed_loop_timings,
)
from spatialvlm.eval.r2r_ce import GroundTruthActionPolicy, R2RCERolloutState


def _episode() -> R2RCEEpisode:
    return R2RCEEpisode(
        episode_id="1",
        trajectory_id="101",
        scene_id="mp3d/example/example.glb",
        instruction_text="Walk straight to the goal.",
        instruction_tokens=(2, 3, 0),
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        goals=(R2RCEGoal(position=(2.0, 0.0, 0.0), radius=3.0),),
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        geodesic_distance=2.0,
        raw={"episode_id": 1},
    )


def _ground_truth() -> R2RCEGroundTruth:
    return R2RCEGroundTruth(
        episode_id="1",
        actions=(
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.STOP,
        ),
        forward_steps=2,
        locations=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )


class _Environment:
    def __init__(self) -> None:
        self.x = 0.0

    def _state(self, *, done: bool = False) -> R2RCERolloutState:
        return R2RCERolloutState(
            observation={"rgb": self.x},
            position=(self.x, 0.0, 0.0),
            distance_to_goal=abs(2.0 - self.x),
            done=done,
        )

    def reset(self, episode: R2RCEEpisode) -> R2RCERolloutState:
        self.x = episode.start_position[0]
        return self._state()

    def step(self, action: R2RCEAction) -> R2RCERolloutState:
        if action is R2RCEAction.MOVE_FORWARD:
            self.x += 1.0
        return self._state(done=action is R2RCEAction.STOP)

    def close(self) -> None:
        return None


def _timing(seconds: float, index: int) -> ClosedLoopDecisionTiming:
    return ClosedLoopDecisionTiming(
        episode_id="1",
        episode_index=0,
        step_index=index + 1,
        action_id=int(R2RCEAction.MOVE_FORWARD),
        action_name=R2RCEAction.MOVE_FORWARD.name,
        policy_seconds=seconds * 0.8,
        environment_seconds=seconds * 0.2,
        closed_loop_seconds=seconds,
    )


@pytest.mark.parametrize(
    ("seconds", "branch", "checkpoint_count", "episode_count"),
    [
        (0.5, "all_six_61_episodes", 6, 61),
        (0.500001, "paired_original_plus_unweighted_replication_61_episodes", 3, 61),
        (1.5, "paired_original_plus_unweighted_replication_61_episodes", 3, 61),
        (1.500001, "single_unweighted_50_episodes", 1, 50),
    ],
)
def test_latency_branch_boundaries_are_frozen(
    seconds: float, branch: str, checkpoint_count: int, episode_count: int
) -> None:
    scope = select_closed_loop_scope(seconds)
    assert scope.branch_id == branch
    assert len(scope.checkpoint_ids) == checkpoint_count
    assert scope.episode_count == episode_count


def test_medium_branch_freezes_exact_checkpoint_identities() -> None:
    scope = select_closed_loop_scope(1.0)
    assert scope.checkpoint_ids == (
        "unweighted_s20260825",
        "inverse_frequency_s20260825",
        "unweighted_s20260826",
    )


def test_every_checkpoint_freezes_its_actual_producing_gate() -> None:
    from spatialvlm.eval.native_closed_loop import FROZEN_CHECKPOINTS

    expected = {
        "unweighted": "native_qwen_c0_development_v1",
        "inverse_frequency": "native_qwen_c0_action_balance_v1",
    }
    assert all(item.checkpoint_gate == expected[item.treatment] for item in FROZEN_CHECKPOINTS)


def test_latency_summary_uses_exactly_32_decisions_and_mean_for_scope() -> None:
    records = [_timing(0.4, index) for index in range(CLOSED_LOOP_STRESS_DECISIONS)]
    summary = summarize_closed_loop_timings(records)
    assert summary["decision_count"] == 32
    assert summary["mean_seconds_per_step"] == pytest.approx(0.4)
    assert summary["scope"]["branch_id"] == "all_six_61_episodes"
    with pytest.raises(ValueError, match="exactly 32"):
        summarize_closed_loop_timings(records[:-1])


def test_latency_summary_does_not_discard_first_decision_as_warmup() -> None:
    records = [_timing(1.0, 0)] + [
        _timing(0.4, index) for index in range(1, CLOSED_LOOP_STRESS_DECISIONS)
    ]
    summary = summarize_closed_loop_timings(records)
    assert summary["mean_seconds_per_step"] == pytest.approx((1.0 + 31 * 0.4) / 32)


def test_nonfinite_or_nonpositive_latency_is_rejected() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            select_closed_loop_scope(value)


def test_instrumented_episode_matches_official_success_and_oracle_semantics() -> None:
    result = run_instrumented_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        _Environment(),
        GroundTruthActionPolicy(),
        episode_index=7,
    )
    assert result.evaluation.metrics.success == pytest.approx(1.0)
    assert result.evaluation.metrics.oracle_success == pytest.approx(1.0)
    assert result.evaluation.metrics.step_count == 3
    assert len(result.decision_timings) == 3
    assert all(record.episode_index == 7 for record in result.decision_timings)
    assert [record.action_name for record in result.decision_timings] == [
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "STOP",
    ]


def test_invalid_action_is_not_timed_as_executed_closed_loop_step() -> None:
    class _Invalid:
        def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
            del episode, ground_truth

        def act(self, observation: Mapping[str, object]) -> int:
            del observation
            return 999

    result = run_instrumented_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        _Environment(),
        _Invalid(),
        episode_index=0,
    )
    assert result.decision_timings == ()
    assert result.evaluation.metrics.invalid_action_count == 1


def test_protocol_keeps_stress_non_scientific_and_val_unseen_sealed() -> None:
    record = closed_loop_protocol_record()
    assert record["stress"]["decision_count"] == 32
    assert record["stress"]["scientific_metrics_emitted"] is False
    assert record["official_max_steps"] == 500
    assert record["chunk_length"] == 1
    assert record["split"] == "val_seen"
    assert record["sealed_split"] == "val_unseen"
    assert record["headline_metrics"] == ["oracle_success_rate", "success_rate"]


def test_launchers_bound_stress_and_do_not_expose_manual_scope_selection() -> None:
    repo = Path(__file__).resolve().parents[1]
    stress_launcher = (repo / "scripts/start_spark_native_c0_closed_loop_stress.sh").read_text()
    main_launcher = (repo / "scripts/start_spark_native_c0_closed_loop.sh").read_text()
    main_inner = (repo / "scripts/run_spark_native_c0_closed_loop.sh").read_text()
    main_runner = (repo / "scripts/run_spark_native_c0_closed_loop.py").read_text()
    assert "timeout --signal=TERM --kill-after=60s 1740s" in stress_launcher
    assert "policy_ordering_v1_verification.json" in main_launcher
    assert '--control-verification "${control_verification}"' in main_inner
    assert "--scope" not in main_launcher
    assert "--scope" not in main_inner
    assert 'parser.add_argument("--scope"' not in main_runner
