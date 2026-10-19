"""Versioned observation-history contracts for controlled VLN experiments.

The workshop study compares history choices before it compares geometry choices.  This module
keeps that choice independent of any particular vision encoder: every arm receives a
``HistoryWindow`` built from the same chronological observations and executed actions.

No simulator pose, goal coordinate, geodesic distance, or future action is represented here.
Those values are evaluator/training metadata and must not leak into the policy input.  RGB versus
RGB-D visibility remains an arm-level choice; the temporal contract does not silently add a sensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from spatialvlm.data.datasets import R2RCEAction

HISTORY_CONTRACT_VERSION = 1


class HistoryMode(str, Enum):
    """Pre-registered temporal controls for the workshop development comparison."""

    H0_CURRENT = "H0"
    H1_PREVIOUS_ACTION = "H1"
    H4_SHORT_WINDOW = "H4"


@dataclass(frozen=True)
class HistoryContract:
    """One policy-visible temporal contract.

    ``H0`` and ``H1`` contain only the current observation.  ``H1`` additionally exposes the
    action that led into that observation.  ``H4`` contains exactly four chronological slots and
    left-pads early decisions with the initial observation.  Padding never invents an action.
    """

    mode: HistoryMode
    frame_count: int
    include_previous_actions: bool
    left_pad_initial_observation: bool

    def __post_init__(self) -> None:
        expected = {
            HistoryMode.H0_CURRENT: (1, False, False),
            HistoryMode.H1_PREVIOUS_ACTION: (1, True, False),
            HistoryMode.H4_SHORT_WINDOW: (4, True, True),
        }[self.mode]
        observed = (
            self.frame_count,
            self.include_previous_actions,
            self.left_pad_initial_observation,
        )
        if observed != expected:
            raise ValueError(
                f"History contract {self.mode.value} must use {expected}, got {observed}."
            )

    @property
    def contract_id(self) -> str:
        return f"r2r_ce_{self.mode.value.lower()}_history_v{HISTORY_CONTRACT_VERSION}"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["mode"] = self.mode.value
        record["contract_id"] = self.contract_id
        record["contract_version"] = HISTORY_CONTRACT_VERSION
        visible_fields = ["chronological_observations", "relative_timestep"]
        if self.include_previous_actions:
            visible_fields.append("previous_action_id")
        record["policy_visible_fields"] = visible_fields
        record["explicitly_excluded_fields"] = [
            "simulator_pose",
            "goal_coordinate",
            "geodesic_distance",
            "future_action",
        ]
        record["sensor_visibility"] = "defined_by_the_selected_C0-C4_arm"
        return record


HISTORY_CONTRACTS: dict[HistoryMode, HistoryContract] = {
    HistoryMode.H0_CURRENT: HistoryContract(
        mode=HistoryMode.H0_CURRENT,
        frame_count=1,
        include_previous_actions=False,
        left_pad_initial_observation=False,
    ),
    HistoryMode.H1_PREVIOUS_ACTION: HistoryContract(
        mode=HistoryMode.H1_PREVIOUS_ACTION,
        frame_count=1,
        include_previous_actions=True,
        left_pad_initial_observation=False,
    ),
    HistoryMode.H4_SHORT_WINDOW: HistoryContract(
        mode=HistoryMode.H4_SHORT_WINDOW,
        frame_count=4,
        include_previous_actions=True,
        left_pad_initial_observation=True,
    ),
}


@dataclass(frozen=True)
class HistoryFrame:
    """One chronological policy-input slot."""

    observation: Any
    source_step_index: int
    relative_timestep: int
    previous_action_id: int | None
    is_padding: bool

    def metadata_record(self) -> dict[str, Any]:
        """Return policy-visible metadata without serializing the observation itself."""
        return {
            "source_step_index": self.source_step_index,
            "relative_timestep": self.relative_timestep,
            "previous_action_id": self.previous_action_id,
            "previous_action": (
                R2RCEAction(self.previous_action_id).name
                if self.previous_action_id is not None
                else None
            ),
            "is_padding": self.is_padding,
        }


@dataclass(frozen=True)
class HistoryWindow:
    """A validated, encoder-independent history window for one action decision."""

    contract: HistoryContract
    decision_step_index: int
    frames: tuple[HistoryFrame, ...]

    def __post_init__(self) -> None:
        if self.decision_step_index < 0:
            raise ValueError("decision_step_index must be non-negative.")
        if len(self.frames) != self.contract.frame_count:
            raise ValueError(
                f"{self.contract.contract_id} requires {self.contract.frame_count} frames, "
                f"got {len(self.frames)}."
            )
        if self.frames[-1].source_step_index != self.decision_step_index:
            raise ValueError("The final history frame must be the current decision observation.")
        if self.frames[-1].is_padding:
            raise ValueError("The current decision observation cannot be padding.")
        expected_relative = tuple(range(1 - len(self.frames), 1))
        observed_relative = tuple(frame.relative_timestep for frame in self.frames)
        if observed_relative != expected_relative:
            raise ValueError(
                f"History relative timesteps must be {expected_relative}, got {observed_relative}."
            )
        if not self.contract.include_previous_actions and any(
            frame.previous_action_id is not None for frame in self.frames
        ):
            raise ValueError(f"{self.contract.mode.value} must not expose previous actions.")

    def to_record(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_record(),
            "decision_step_index": self.decision_step_index,
            "frames": [frame.metadata_record() for frame in self.frames],
        }


def _coerce_executed_actions(actions: Sequence[R2RCEAction | int]) -> tuple[R2RCEAction, ...]:
    result: list[R2RCEAction] = []
    for index, action in enumerate(actions):
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"Executed action {index} must be an official integer action ID.")
        try:
            result.append(R2RCEAction(int(action)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Executed action {index} is not an official R2R-CE action.") from exc
    if R2RCEAction.STOP in result:
        raise ValueError(
            "A decision window cannot contain an action executed after or including STOP."
        )
    return tuple(result)


def build_history_window(
    observations: Sequence[Any],
    executed_actions: Sequence[R2RCEAction | int],
    *,
    contract: HistoryContract,
) -> HistoryWindow:
    """Build one history window from a complete observation/action prefix.

    The causal alignment is strict: ``N`` executed actions must be accompanied by ``N + 1``
    observations.  The last observation is the state at which the next action will be predicted.
    This makes future-observation and target-action leakage structurally impossible.
    """
    if not observations:
        raise ValueError("At least one observation is required.")
    actions = _coerce_executed_actions(executed_actions)
    if len(observations) != len(actions) + 1:
        raise ValueError(
            "History alignment requires exactly one more observation than executed actions: "
            f"observations={len(observations)}, actions={len(actions)}."
        )

    current_step = len(actions)
    start = max(0, current_step - contract.frame_count + 1)
    source_indices = list(range(start, current_step + 1))
    padding_count = contract.frame_count - len(source_indices)
    if padding_count and not contract.left_pad_initial_observation:
        raise RuntimeError("A non-padding history contract requested unavailable prior frames.")

    frames: list[HistoryFrame] = []
    for slot in range(padding_count):
        frames.append(
            HistoryFrame(
                observation=observations[0],
                source_step_index=0,
                relative_timestep=slot - contract.frame_count + 1,
                previous_action_id=None,
                is_padding=True,
            )
        )

    for source_index in source_indices:
        previous_action_id: int | None = None
        if contract.include_previous_actions and source_index > 0:
            previous_action_id = int(actions[source_index - 1])
        frames.append(
            HistoryFrame(
                observation=observations[source_index],
                source_step_index=source_index,
                relative_timestep=len(frames) - contract.frame_count + 1,
                previous_action_id=previous_action_id,
                is_padding=False,
            )
        )

    return HistoryWindow(
        contract=contract,
        decision_step_index=current_step,
        frames=tuple(frames),
    )
