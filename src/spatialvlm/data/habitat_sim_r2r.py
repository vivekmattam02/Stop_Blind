"""Direct Habitat-Sim adapter for the locked R2R-CE v1-3 protocol.

This module intentionally does not depend on Habitat-Lab.  Habitat-Sim owns scene rendering,
discrete motion, collision filtering, and navmesh geodesics; the independent GridNav evaluator
owns STOP semantics and metrics.  The separation makes modern-versus-legacy simulator drift
measurable instead of hiding it behind a high-level environment wrapper.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spatialvlm.data.datasets import Point3D, R2RCEAction, R2RCEEpisode
from spatialvlm.eval.r2r_ce import (
    R2R_CE_DEPTH_SIZE,
    R2R_CE_FORWARD_STEP_M,
    R2R_CE_RGB_SIZE,
    R2R_CE_TURN_ANGLE_DEG,
    R2RCERolloutState,
)

HABITAT_SIM_VERSION = "0.3.3"
HABITAT_SIM_COMMIT = "acbe6f4922e68145e401e55c30f9dfea460a3f24"
_MP3D_SCAN_ID = re.compile(r"^[A-Za-z0-9]+$")


def require_habitat_sim() -> Any:
    """Import the pinned runtime lazily so CPU-only tests remain importable."""
    try:
        import habitat_sim
    except ImportError as exc:  # pragma: no cover - depends on the external Spark image
        raise ImportError(
            "Habitat-Sim is unavailable. Build and use docker/spark/habitat.Dockerfile."
        ) from exc
    return habitat_sim


def resolve_mp3d_scene_path(scene_root: str | Path, scene_id: str) -> Path:
    """Resolve one canonical ``mp3d/{scan}/{scan}.glb`` path without a recursive search."""
    root = Path(scene_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MP3D scene root is not a directory: {root}")
    raw = Path(scene_id)
    if raw.is_absolute():
        raise ValueError("R2R-CE scene_id must be dataset-relative, not absolute.")

    parts = raw.parts
    if "mp3d" in parts:
        index = parts.index("mp3d")
        if index + 1 >= len(parts):
            raise ValueError(f"R2R-CE scene_id does not include an MP3D scan ID: {scene_id!r}")
        scan_id = parts[index + 1]
    elif raw.suffix == ".glb":
        scan_id = raw.stem
    elif len(parts) == 1:
        scan_id = parts[0]
    else:
        raise ValueError(f"Cannot derive an MP3D scan ID from scene_id={scene_id!r}.")

    if not _MP3D_SCAN_ID.fullmatch(scan_id):
        raise ValueError(f"Unsafe or malformed MP3D scan ID: {scan_id!r}")

    bases = (root,) if root.name == "mp3d" else (root / "mp3d", root)
    candidates = tuple(base / scan_id / f"{scan_id}.glb" for base in bases)
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Resolved scene escapes the configured root: {resolved}")
        if resolved.is_file():
            return resolved

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Canonical MP3D scene for {scene_id!r} was not found. Expected one of: {expected}"
    )


@dataclass(frozen=True)
class HabitatSimR2RConfig:
    """Modern Habitat-Sim port of the official legacy R2R-CE sensor/action contract."""

    scene_root: str | Path
    gpu_device_id: int = 0
    seed: int = 100
    allow_sliding: bool = True
    agent_height_m: float = 1.5
    agent_radius_m: float = 0.1
    sensor_height_m: float = 1.25
    rgb_height_width: tuple[int, int] = R2R_CE_RGB_SIZE
    depth_height_width: tuple[int, int] = R2R_CE_DEPTH_SIZE
    hfov_degrees: float = 90.0
    depth_min_m: float = 0.0
    depth_max_m: float = 10.0
    forward_step_m: float = R2R_CE_FORWARD_STEP_M
    turn_angle_degrees: float = R2R_CE_TURN_ANGLE_DEG

    def __post_init__(self) -> None:
        if self.gpu_device_id < 0:
            raise ValueError("Habitat-Sim GPU device ID must be non-negative.")
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("Depth maximum must be greater than depth minimum.")
        if self.forward_step_m <= 0.0 or self.turn_angle_degrees <= 0.0:
            raise ValueError("R2R-CE movement and turn amounts must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scene_root"] = str(Path(self.scene_root).expanduser().resolve())
        return payload


def postprocess_habitat_observations(
    observations: dict[str, Any],
    config: HabitatSimR2RConfig,
) -> dict[str, np.ndarray]:
    """Expose official normalized depth and a separate metric-depth geometry stream."""
    if "rgb" not in observations or "depth" not in observations:
        raise KeyError("Habitat-Sim observations must contain both 'rgb' and 'depth'.")

    rgb = np.asarray(observations["rgb"])
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"Habitat-Sim RGB must be HWC with 3 or 4 channels, got {rgb.shape}.")
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    if rgb.shape[:2] != config.rgb_height_width:
        raise ValueError(
            f"Official R2R-CE RGB must be {config.rgb_height_width}, got {rgb.shape[:2]}."
        )

    depth_meters = np.asarray(observations["depth"], dtype=np.float32)
    if depth_meters.ndim == 3 and depth_meters.shape[-1] == 1:
        depth_meters = depth_meters[..., 0]
    if depth_meters.ndim != 2:
        raise ValueError(f"Habitat-Sim depth must be HW or HW1, got {depth_meters.shape}.")
    if depth_meters.shape != config.depth_height_width:
        raise ValueError(
            f"Official R2R-CE depth must be {config.depth_height_width}, got {depth_meters.shape}."
        )
    if not np.isfinite(depth_meters).all():
        raise ValueError("Habitat-Sim metric depth contains non-finite values.")

    depth_meters = np.ascontiguousarray(
        np.clip(depth_meters, config.depth_min_m, config.depth_max_m),
        dtype=np.float32,
    )
    depth_normalized = (depth_meters - config.depth_min_m) / (
        config.depth_max_m - config.depth_min_m
    )
    return {
        "rgb": rgb,
        "depth": np.ascontiguousarray(depth_normalized[..., None], dtype=np.float32),
        "depth_meters": depth_meters,
    }


class HabitatSimR2REnvironment:
    """Concrete direct-simulator implementation of ``R2RCEEpisodeEnvironment``."""

    _ACTION_NAMES = {
        R2RCEAction.MOVE_FORWARD: "move_forward",
        R2RCEAction.TURN_LEFT: "turn_left",
        R2RCEAction.TURN_RIGHT: "turn_right",
    }

    def __init__(self, config: HabitatSimR2RConfig) -> None:
        self.config = config
        self._habitat_sim = require_habitat_sim()
        installed_version = getattr(self._habitat_sim, "__version__", None)
        if installed_version != HABITAT_SIM_VERSION:
            raise RuntimeError(
                f"Habitat-Sim version drift: expected {HABITAT_SIM_VERSION}, "
                f"found {installed_version!r}."
            )
        self._sim: Any | None = None
        self._scene_path: Path | None = None
        self._goal_positions: tuple[Point3D, ...] = ()
        self._done = False

    def _camera_spec(
        self,
        *,
        uuid: str,
        sensor_type: Any,
        height_width: tuple[int, int],
    ) -> Any:
        spec = self._habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = list(height_width)
        spec.position = [0.0, self.config.sensor_height_m, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.hfov = self.config.hfov_degrees
        spec.sensor_subtype = self._habitat_sim.SensorSubType.PINHOLE
        return spec

    def _build_configuration(self, scene_path: Path) -> Any:
        backend = self._habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene_path)
        backend.gpu_device_id = self.config.gpu_device_id
        backend.random_seed = self.config.seed
        backend.enable_physics = False
        backend.allow_sliding = self.config.allow_sliding

        rgb = self._camera_spec(
            uuid="rgb",
            sensor_type=self._habitat_sim.SensorType.COLOR,
            height_width=self.config.rgb_height_width,
        )
        depth = self._camera_spec(
            uuid="depth",
            sensor_type=self._habitat_sim.SensorType.DEPTH,
            height_width=self.config.depth_height_width,
        )
        agent = self._habitat_sim.agent.AgentConfiguration()
        agent.height = self.config.agent_height_m
        agent.radius = self.config.agent_radius_m
        agent.sensor_specifications = [rgb, depth]
        agent.action_space = {
            "move_forward": self._habitat_sim.agent.ActionSpec(
                "move_forward",
                self._habitat_sim.agent.ActuationSpec(amount=self.config.forward_step_m),
            ),
            "turn_left": self._habitat_sim.agent.ActionSpec(
                "turn_left",
                self._habitat_sim.agent.ActuationSpec(amount=self.config.turn_angle_degrees),
            ),
            "turn_right": self._habitat_sim.agent.ActionSpec(
                "turn_right",
                self._habitat_sim.agent.ActuationSpec(amount=self.config.turn_angle_degrees),
            ),
        }
        return self._habitat_sim.Configuration(backend, [agent])

    def _ensure_scene(self, episode: R2RCEEpisode) -> None:
        scene_path = resolve_mp3d_scene_path(self.config.scene_root, episode.scene_id)
        if self._sim is not None and scene_path == self._scene_path:
            return
        self.close()
        self._sim = self._habitat_sim.Simulator(self._build_configuration(scene_path))
        self._scene_path = scene_path
        if not self._sim.pathfinder.is_loaded:
            self.close()
            raise RuntimeError(f"Habitat-Sim did not load a navmesh for {scene_path}.")

    def _agent_position(self) -> Point3D:
        if self._sim is None:
            raise RuntimeError("Habitat-Sim environment has not been reset.")
        position = np.asarray(self._sim.get_agent(0).get_state().position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise RuntimeError(f"Habitat-Sim returned an invalid agent position: {position!r}")
        return (float(position[0]), float(position[1]), float(position[2]))

    def agent_rotation_xyzw(self) -> tuple[float, float, float, float]:
        """Return the current rotation in the release's XYZW coefficient convention."""
        if self._sim is None:
            raise RuntimeError("Habitat-Sim environment has not been reset.")
        rotation = self._sim.get_agent(0).get_state().rotation
        coefficients = np.asarray(
            self._habitat_sim.utils.common.quat_to_coeffs(rotation), dtype=np.float64
        )
        if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
            raise RuntimeError(f"Habitat-Sim returned an invalid agent rotation: {coefficients!r}")
        return tuple(float(value) for value in coefficients)

    def _distance_to_goal(self, position: Point3D) -> float:
        if self._sim is None or not self._goal_positions:
            raise RuntimeError("Habitat-Sim environment has no active episode goal.")
        distances: list[float] = []
        for goal in self._goal_positions:
            path = self._habitat_sim.ShortestPath()
            path.requested_start = np.asarray(position, dtype=np.float32)
            path.requested_end = np.asarray(goal, dtype=np.float32)
            if self._sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                distances.append(float(path.geodesic_distance))
        if not distances:
            raise RuntimeError(
                f"No Habitat navmesh path exists from {position} to any active episode goal."
            )
        return min(distances)

    def _state(self, observations: dict[str, Any], *, collision: bool) -> R2RCERolloutState:
        position = self._agent_position()
        return R2RCERolloutState(
            observation=postprocess_habitat_observations(observations, self.config),
            position=position,
            distance_to_goal=self._distance_to_goal(position),
            collision=collision,
            done=self._done,
        )

    def reset(self, episode: R2RCEEpisode) -> R2RCERolloutState:
        self._ensure_scene(episode)
        assert self._sim is not None
        state = self._habitat_sim.AgentState()
        state.position = np.asarray(episode.start_position, dtype=np.float32)
        state.rotation = self._habitat_sim.utils.common.quat_from_coeffs(
            np.asarray(episode.start_rotation, dtype=np.float32)
        )
        self._sim.initialize_agent(0, state)
        self._goal_positions = tuple(goal.position for goal in episode.goals)
        self._done = False
        return self._state(self._sim.get_sensor_observations(), collision=False)

    def step(self, action: R2RCEAction) -> R2RCERolloutState:
        if self._sim is None or not self._goal_positions:
            raise RuntimeError("Call reset before stepping Habitat-Sim.")
        if self._done:
            raise RuntimeError("Cannot step a completed R2R-CE episode.")
        if action == R2RCEAction.STOP:
            self._done = True
            return self._state(self._sim.get_sensor_observations(), collision=False)
        if action not in self._ACTION_NAMES:
            raise ValueError(f"Unsupported R2R-CE action: {action!r}")
        observations = self._sim.step(self._ACTION_NAMES[action])
        collision = bool(observations.get("collided", False))
        return self._state(observations, collision=collision)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "habitat_sim_version": getattr(self._habitat_sim, "__version__", None),
            "habitat_sim_commit_expected": HABITAT_SIM_COMMIT,
            "scene_path": str(self._scene_path) if self._scene_path is not None else None,
            "navmesh_loaded": bool(self._sim is not None and self._sim.pathfinder.is_loaded),
            "config": self.config.to_dict(),
            "action_names": {action.name: name for action, name in self._ACTION_NAMES.items()},
            "stop_implementation": "task_level_no_simulator_motion",
        }

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
        self._sim = None
        self._scene_path = None
        self._goal_positions = ()
        self._done = False
