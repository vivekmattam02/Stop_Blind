"""Conformance tests for the deterministic R2R-CE rollout contract."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.eval.r2r_ce import (
    AlwaysForwardPolicy,
    GroundTruthActionPolicy,
    ImmediateStopPolicy,
    InvalidActionPolicy,
    R2RCERolloutState,
    SeededHandcraftedPolicy,
    SeededRandomPolicy,
    build_r2r_ce_artifact,
    evaluate_r2r_ce_policy,
    run_r2r_ce_episode,
    write_r2r_ce_artifact,
    write_r2r_ce_step_jsonl,
)


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


class CorridorEnvironment:
    """Small deterministic test double; it does not claim Habitat conformance."""

    def __init__(self) -> None:
        self.position_x = 0.0
        self.goal_x = 2.0
        self.step_calls = 0
        self.closed = False

    def _state(self, *, done: bool = False, collision: bool = False) -> R2RCERolloutState:
        return R2RCERolloutState(
            observation={"position_x": self.position_x},
            position=(self.position_x, 0.0, 0.0),
            distance_to_goal=abs(self.goal_x - self.position_x),
            collision=collision,
            done=done,
        )

    def reset(self, episode: R2RCEEpisode) -> R2RCERolloutState:
        self.position_x = episode.start_position[0]
        self.goal_x = episode.goals[0].position[0]
        return self._state()

    def step(self, action: R2RCEAction) -> R2RCERolloutState:
        self.step_calls += 1
        if action == R2RCEAction.MOVE_FORWARD:
            self.position_x += 1.0
        return self._state(done=action == R2RCEAction.STOP)

    def close(self) -> None:
        self.closed = True


class RuntimeGeodesicEnvironment(CorridorEnvironment):
    """Test double whose reset geodesic intentionally differs from release metadata."""

    def reset(self, episode: R2RCEEpisode) -> R2RCERolloutState:
        state = super().reset(episode)
        return R2RCERolloutState(
            observation=state.observation,
            position=state.position,
            distance_to_goal=1.5,
            collision=state.collision,
            done=state.done,
        )


def test_ground_truth_policy_passes_closed_loop_conformance_trace():
    environment = CorridorEnvironment()
    result = run_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        environment,
        GroundTruthActionPolicy(),
    )
    assert result.metrics.success == pytest.approx(1.0)
    assert result.metrics.spl == pytest.approx(1.0)
    assert result.metrics.ndtw == pytest.approx(1.0)
    assert [step.event for step in result.steps] == ["reset", "action", "action", "action"]
    assert result.steps[-1].action_name == "STOP"


def test_spl_uses_simulator_geodesic_measured_at_reset():
    result = run_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        RuntimeGeodesicEnvironment(),
        GroundTruthActionPolicy(),
    )
    assert result.trace.shortest_path_length == pytest.approx(1.5)
    assert result.metrics.path_length == pytest.approx(2.0)
    assert result.metrics.spl == pytest.approx(0.75)


def test_always_forward_can_reach_radius_but_fails_without_stop():
    result = run_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        CorridorEnvironment(),
        AlwaysForwardPolicy(),
        max_steps=2,
    )
    assert result.metrics.oracle_success == pytest.approx(1.0)
    assert result.metrics.success == pytest.approx(0.0)
    assert result.metrics.stopped is False


def test_immediate_stop_is_a_zero_motion_termination_control():
    result = run_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        CorridorEnvironment(),
        ImmediateStopPolicy(),
    )
    assert result.metrics.path_length == pytest.approx(0.0)
    assert result.metrics.step_count == 1
    assert result.metrics.success == pytest.approx(1.0)
    assert result.metrics.stopped is True


def test_invalid_action_fails_closed_without_environment_execution():
    environment = CorridorEnvironment()
    result = run_r2r_ce_episode(
        _episode(),
        _ground_truth(),
        environment,
        InvalidActionPolicy(),
    )
    assert environment.step_calls == 0
    assert result.metrics.success == pytest.approx(0.0)
    assert result.metrics.invalid_action_count == 1
    assert result.metrics.step_count == 1
    assert result.steps[-1].event == "invalid_action"
    assert result.steps[-1].invalid_output == "999"


def test_seeded_random_policy_is_reproducible_per_episode():
    policy = SeededRandomPolicy(seed=17)
    episode = _episode()
    ground_truth = _ground_truth()
    policy.reset(episode, ground_truth)
    first = [policy.act({}) for _ in range(10)]
    policy.reset(episode, ground_truth)
    second = [policy.act({}) for _ in range(10)]
    assert first == second


def test_seeded_handcrafted_policy_is_reproducible_and_has_official_shape():
    policy = SeededHandcraftedPolicy(seed=29)
    episode = _episode()
    ground_truth = _ground_truth()
    policy.reset(episode, ground_truth)
    first = [policy.act({}) for _ in range(63)]
    policy.reset(episode, ground_truth)
    second = [policy.act({}) for _ in range(63)]
    assert first == second
    stop_index = first.index(R2RCEAction.STOP)
    assert 37 <= stop_index <= 61
    assert first[: stop_index - 37] == [R2RCEAction.TURN_RIGHT] * (stop_index - 37)
    assert first[stop_index - 37 : stop_index] == [R2RCEAction.MOVE_FORWARD] * 37


def test_evaluation_run_closes_environment_and_writes_reconstructable_artifacts(tmp_path):
    environments: list[CorridorEnvironment] = []

    def environment_factory(episode: R2RCEEpisode) -> CorridorEnvironment:
        del episode
        environment = CorridorEnvironment()
        environments.append(environment)
        return environment

    episode = _episode()
    run = evaluate_r2r_ce_policy(
        [episode],
        {episode.episode_id: _ground_truth()},
        environment_factory,
        GroundTruthActionPolicy(),
    )
    assert run.aggregate.success_rate == pytest.approx(1.0)
    assert environments[0].closed is True

    manifest = R2RCEReleaseManifest(
        release_id="R2R_VLNCE_v1-3",
        split="val_seen",
        episodes_path="/licensed/val_seen.json.gz",
        episodes_sha256="a" * 64,
        source_episode_count=1,
        selected_episode_count=1,
        ground_truth_path="/licensed/val_seen_gt.json.gz",
        ground_truth_sha256="b" * 64,
        ground_truth_count=1,
    )
    artifact = build_r2r_ce_artifact(run, manifest, policy_id="ground-truth-conformance")
    artifact_path = tmp_path / "evaluation.json"
    jsonl_path = tmp_path / "steps.jsonl"
    write_r2r_ce_artifact(artifact_path, artifact)
    write_r2r_ce_step_jsonl(jsonl_path, run.results)

    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert loaded["release"]["release_id"] == "R2R_VLNCE_v1-3"
    assert loaded["protocol"]["action_mapping"] == {
        "MOVE_FORWARD": 1,
        "STOP": 0,
        "TURN_LEFT": 2,
        "TURN_RIGHT": 3,
    }
    step_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert step_rows[0]["event"] == "reset"
    assert step_rows[-1]["action_name"] == "STOP"


def test_environment_protocol_accepts_observation_mappings():
    state = R2RCERolloutState(
        observation={"rgb": "opaque-to-evaluator"},
        position=(0.0, 0.0, 0.0),
        distance_to_goal=2.0,
    )
    assert isinstance(state.observation, Mapping)
