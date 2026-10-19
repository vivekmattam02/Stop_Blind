"""Dataset loaders for navigation and spatial QA corpora."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

R2R_CE_RELEASE_ID = "R2R_VLNCE_v1-3"


class R2RCEAction(IntEnum):
    """Discrete action IDs used by the official R2R-CE release."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


Point3D = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


@dataclass(frozen=True)
class R2RCEGoal:
    """One R2R-CE goal location and success radius."""

    position: Point3D
    radius: float


@dataclass(frozen=True)
class R2RCEEpisode:
    """Strictly parsed episode from an R2R_VLNCE_v1-3 split archive."""

    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_text: str
    instruction_tokens: tuple[int, ...]
    start_position: Point3D
    start_rotation: QuaternionXYZW
    goals: tuple[R2RCEGoal, ...]
    reference_path: tuple[Point3D, ...]
    geodesic_distance: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class R2RCEGroundTruth:
    """Official action/location record from ``{split}_gt.json.gz``."""

    episode_id: str
    actions: tuple[R2RCEAction, ...]
    forward_steps: int
    locations: tuple[Point3D, ...]


@dataclass(frozen=True)
class R2RCEReleaseManifest:
    """Immutable identity information for a parsed R2R-CE split."""

    release_id: str
    split: str
    episodes_path: str
    episodes_sha256: str
    source_episode_count: int
    selected_episode_count: int
    ground_truth_path: str | None
    ground_truth_sha256: str | None
    ground_truth_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _read_json_document(path: Path) -> Any:
    text = _read_text(path).strip()
    if not text:
        raise ValueError(f"Dataset file is empty: {path}")
    return json.loads(text)


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from either JSON array/object or JSONL file."""
    text = _read_text(path).strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be JSON objects.")
            records.append(row)
        return records

    parsed = json.loads(text)
    if isinstance(parsed, list):
        if not all(isinstance(x, dict) for x in parsed):
            raise ValueError("JSON list dataset must contain JSON objects.")
        return parsed
    if isinstance(parsed, dict):
        # Common pattern: {"episodes": [...]}
        if "episodes" in parsed and isinstance(parsed["episodes"], list):
            episodes = parsed["episodes"]
            if not all(isinstance(x, dict) for x in episodes):
                raise ValueError("'episodes' must contain JSON objects.")
            return episodes
        return [parsed]

    raise ValueError("Unsupported dataset JSON structure.")


@dataclass(frozen=True)
class NavSample:
    """Standardized navigation sample."""

    instruction: str
    episode_id: str
    source: str
    payload: dict[str, Any]


def _pick_instruction(record: dict[str, Any]) -> str:
    for key in ("instruction", "instructions", "command", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            nested = value.get("instruction_text")
            if isinstance(nested, str) and nested.strip():
                return nested
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
    raise ValueError("Missing instruction text in record.")


def _pick_episode_id(record: dict[str, Any], idx: int) -> str:
    value = record.get("episode_id", record.get("id", record.get("path_id", idx)))
    return str(value)


def _required(record: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in record:
        raise ValueError(f"Missing {context} field: {key}")
    return record[key]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _point3(value: Any, label: str) -> Point3D:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates.")
    return (
        _finite_float(value[0], f"{label}[0]"),
        _finite_float(value[1], f"{label}[1]"),
        _finite_float(value[2], f"{label}[2]"),
    )


def _quaternion_xyzw(value: Any, label: str) -> QuaternionXYZW:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four XYZW coefficients.")
    return (
        _finite_float(value[0], f"{label}[0]"),
        _finite_float(value[1], f"{label}[1]"),
        _finite_float(value[2], f"{label}[2]"),
        _finite_float(value[3], f"{label}[3]"),
    )


def _parse_r2r_ce_episode(record: Mapping[str, Any]) -> R2RCEEpisode:
    episode_id = str(_required(record, "episode_id", "episode"))
    trajectory_id = str(_required(record, "trajectory_id", f"episode {episode_id}"))
    scene_id = _required(record, "scene_id", f"episode {episode_id}")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError(f"Episode {episode_id} scene_id must be a non-empty string.")

    instruction = _required(record, "instruction", f"episode {episode_id}")
    if not isinstance(instruction, Mapping):
        raise ValueError(f"Episode {episode_id} instruction must be an object.")
    instruction_text = _required(instruction, "instruction_text", f"episode {episode_id}")
    if not isinstance(instruction_text, str) or not instruction_text.strip():
        raise ValueError(f"Episode {episode_id} instruction text must be non-empty.")
    token_values = instruction.get("instruction_tokens", [])
    if not isinstance(token_values, Sequence) or isinstance(token_values, str | bytes):
        raise ValueError(f"Episode {episode_id} instruction tokens must be a sequence.")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in token_values):
        raise ValueError(f"Episode {episode_id} instruction tokens must be integers.")

    goals_value = _required(record, "goals", f"episode {episode_id}")
    if not isinstance(goals_value, Sequence) or isinstance(goals_value, str | bytes):
        raise ValueError(f"Episode {episode_id} goals must be a sequence.")
    goals: list[R2RCEGoal] = []
    for index, goal in enumerate(goals_value):
        if not isinstance(goal, Mapping):
            raise ValueError(f"Episode {episode_id} goal {index} must be an object.")
        radius = _finite_float(
            _required(goal, "radius", f"episode {episode_id} goal {index}"),
            f"episode {episode_id} goal {index} radius",
        )
        if radius <= 0.0:
            raise ValueError(f"Episode {episode_id} goal {index} radius must be positive.")
        goals.append(
            R2RCEGoal(
                position=_point3(
                    _required(goal, "position", f"episode {episode_id} goal {index}"),
                    f"episode {episode_id} goal {index} position",
                ),
                radius=radius,
            )
        )
    if not goals:
        raise ValueError(f"Episode {episode_id} must contain at least one goal.")

    reference_value = _required(record, "reference_path", f"episode {episode_id}")
    if not isinstance(reference_value, Sequence) or isinstance(reference_value, str | bytes):
        raise ValueError(f"Episode {episode_id} reference_path must be a sequence.")
    reference_path = tuple(
        _point3(point, f"episode {episode_id} reference_path[{index}]")
        for index, point in enumerate(reference_value)
    )
    if not reference_path:
        raise ValueError(f"Episode {episode_id} reference_path must not be empty.")

    info = _required(record, "info", f"episode {episode_id}")
    if not isinstance(info, Mapping):
        raise ValueError(f"Episode {episode_id} info must be an object.")
    geodesic_distance = _finite_float(
        _required(info, "geodesic_distance", f"episode {episode_id} info"),
        f"episode {episode_id} geodesic_distance",
    )
    if geodesic_distance < 0.0:
        raise ValueError(f"Episode {episode_id} geodesic_distance must be non-negative.")

    return R2RCEEpisode(
        episode_id=episode_id,
        trajectory_id=trajectory_id,
        scene_id=scene_id,
        instruction_text=instruction_text,
        instruction_tokens=tuple(token_values),
        start_position=_point3(
            _required(record, "start_position", f"episode {episode_id}"),
            f"episode {episode_id} start_position",
        ),
        start_rotation=_quaternion_xyzw(
            _required(record, "start_rotation", f"episode {episode_id}"),
            f"episode {episode_id} start_rotation",
        ),
        goals=tuple(goals),
        reference_path=reference_path,
        geodesic_distance=geodesic_distance,
        raw=dict(record),
    )


def _parse_r2r_ce_ground_truth(
    episode_id: str,
    record: Mapping[str, Any],
) -> R2RCEGroundTruth:
    action_values = _required(record, "actions", f"ground truth {episode_id}")
    if not isinstance(action_values, Sequence) or isinstance(action_values, str | bytes):
        raise ValueError(f"Ground truth {episode_id} actions must be a sequence.")
    actions: list[R2RCEAction] = []
    for index, value in enumerate(action_values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Ground truth {episode_id} action {index} must be an integer.")
        try:
            actions.append(R2RCEAction(value))
        except ValueError as exc:
            raise ValueError(
                f"Ground truth {episode_id} action {index} has unsupported ID {value}."
            ) from exc
    if not actions:
        raise ValueError(f"Ground truth {episode_id} actions must not be empty.")
    if actions[-1] != R2RCEAction.STOP:
        raise ValueError(f"Ground truth {episode_id} must terminate with STOP (0).")

    forward_steps_value = _required(record, "forward_steps", f"ground truth {episode_id}")
    if isinstance(forward_steps_value, bool) or not isinstance(forward_steps_value, int):
        raise ValueError(f"Ground truth {episode_id} forward_steps must be an integer.")
    if forward_steps_value < 0:
        raise ValueError(f"Ground truth {episode_id} forward_steps must be non-negative.")
    locations_value = _required(record, "locations", f"ground truth {episode_id}")
    if not isinstance(locations_value, Sequence) or isinstance(locations_value, str | bytes):
        raise ValueError(f"Ground truth {episode_id} locations must be a sequence.")
    locations = tuple(
        _point3(point, f"ground truth {episode_id} locations[{index}]")
        for index, point in enumerate(locations_value)
    )
    if not locations:
        raise ValueError(f"Ground truth {episode_id} locations must not be empty.")
    return R2RCEGroundTruth(
        episode_id=episode_id,
        actions=tuple(actions),
        forward_steps=forward_steps_value,
        locations=locations,
    )


class _BaseNavDataset(Dataset[NavSample]):
    source_name: str = "base"

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        split: str | None = None,
        limit: int | None = None,
    ) -> None:
        rows = list(records)
        if split is not None:
            rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
        if limit is not None:
            rows = rows[:limit]
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> NavSample:
        row = self._rows[idx]
        return NavSample(
            instruction=_pick_instruction(row),
            episode_id=_pick_episode_id(row, idx),
            source=self.source_name,
            payload=row,
        )


class R2RCEDataset(_BaseNavDataset):
    """R2R-CE loader with a strict v1-3 release adapter.

    ``from_file`` remains available for legacy metadata fixtures. Scientific runs must use
    ``from_release`` so the split archive, ground-truth archive, schema, IDs, and hashes are
    validated together.
    """

    source_name = "r2r-ce"

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        split: str | None = None,
        limit: int | None = None,
        *,
        official_episodes: Sequence[R2RCEEpisode] = (),
        ground_truth: Mapping[str, R2RCEGroundTruth] | None = None,
        release_manifest: R2RCEReleaseManifest | None = None,
        instruction_vocab: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(records=records, split=split, limit=limit)
        self._official_episodes = tuple(official_episodes)
        self._episode_by_id = {episode.episode_id: episode for episode in official_episodes}
        self._ground_truth = dict(ground_truth or {})
        self.release_manifest = release_manifest
        self.instruction_vocab = dict(instruction_vocab or {})

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> R2RCEDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)

    @classmethod
    def from_release(
        cls,
        episodes_path: str | Path,
        ground_truth_path: str | Path | None,
        *,
        split: str,
        limit: int | None = None,
        release_id: str = R2R_CE_RELEASE_ID,
    ) -> R2RCEDataset:
        """Load and cross-check one official R2R_VLNCE_v1-3 split.

        Parameters
        ----------
        episodes_path:
            Official ``{split}.json.gz`` file.
        ground_truth_path:
            Official ``{split}_gt.json.gz`` file. ``None`` is allowed only for a split such as
            ``test`` that intentionally has no released ground truth.
        split:
            Explicit split label recorded in the release manifest. It is never inferred from a
            filename.
        """
        if not split.strip():
            raise ValueError("R2R-CE split must be an explicit non-empty string.")
        if limit is not None and limit < 0:
            raise ValueError("R2R-CE limit must be non-negative or None.")

        episode_file = Path(episodes_path)
        document = _read_json_document(episode_file)
        if not isinstance(document, dict):
            raise ValueError("R2R-CE split archive must contain a JSON object.")
        rows = document.get("episodes")
        vocab = document.get("instruction_vocab")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("R2R-CE split archive must contain an 'episodes' object list.")
        if not isinstance(vocab, dict):
            raise ValueError("R2R-CE split archive must contain 'instruction_vocab'.")

        parsed = tuple(_parse_r2r_ce_episode(row) for row in rows)
        episode_ids = [episode.episode_id for episode in parsed]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("R2R-CE split contains duplicate episode IDs.")

        ground_truth: dict[str, R2RCEGroundTruth] = {}
        gt_file: Path | None = None
        gt_sha256: str | None = None
        if ground_truth_path is not None:
            gt_file = Path(ground_truth_path)
            gt_document = _read_json_document(gt_file)
            if not isinstance(gt_document, dict):
                raise ValueError("R2R-CE ground-truth archive must contain a JSON object.")
            for episode_id, row in gt_document.items():
                if not isinstance(row, dict):
                    raise ValueError(f"Ground truth for episode {episode_id!r} must be an object.")
                parsed_gt = _parse_r2r_ce_ground_truth(str(episode_id), row)
                ground_truth[parsed_gt.episode_id] = parsed_gt

            missing = sorted(set(episode_ids) - set(ground_truth))
            extra = sorted(set(ground_truth) - set(episode_ids))
            if missing or extra:
                raise ValueError(
                    "R2R-CE episode/ground-truth ID mismatch: "
                    f"missing={missing[:5]}, extra={extra[:5]}."
                )

            for episode in parsed:
                gt = ground_truth[episode.episode_id]
                if math.dist(episode.start_position, gt.locations[0]) > 1e-4:
                    raise ValueError(
                        f"Ground-truth start does not match episode {episode.episode_id}."
                    )
            gt_sha256 = _sha256_file(gt_file)

        selected = parsed if limit is None else parsed[:limit]
        selected_ids = {episode.episode_id for episode in selected}
        selected_gt = {
            episode_id: row
            for episode_id, row in ground_truth.items()
            if episode_id in selected_ids
        }
        manifest = R2RCEReleaseManifest(
            release_id=release_id,
            split=split,
            episodes_path=str(episode_file.resolve()),
            episodes_sha256=_sha256_file(episode_file),
            source_episode_count=len(parsed),
            selected_episode_count=len(selected),
            ground_truth_path=str(gt_file.resolve()) if gt_file is not None else None,
            ground_truth_sha256=gt_sha256,
            ground_truth_count=len(ground_truth),
        )
        return cls(
            records=[episode.raw for episode in selected],
            official_episodes=selected,
            ground_truth=selected_gt,
            release_manifest=manifest,
            instruction_vocab=vocab,
        )

    @property
    def official_episodes(self) -> tuple[R2RCEEpisode, ...]:
        """Return typed official episodes selected for this dataset."""
        return self._official_episodes

    @property
    def ground_truth(self) -> dict[str, R2RCEGroundTruth]:
        """Return a defensive copy of ground truth keyed by string episode ID."""
        return dict(self._ground_truth)

    def episode_record(self, episode_id: str | int) -> R2RCEEpisode:
        """Look up one typed episode, failing clearly for legacy datasets."""
        key = str(episode_id)
        if key not in self._episode_by_id:
            raise KeyError(f"Unknown official R2R-CE episode ID: {key}")
        return self._episode_by_id[key]

    def ground_truth_record(self, episode_id: str | int) -> R2RCEGroundTruth:
        """Look up one official ground-truth trajectory."""
        key = str(episode_id)
        if key not in self._ground_truth:
            raise KeyError(f"No R2R-CE ground truth for episode ID: {key}")
        return self._ground_truth[key]


class RxRCEDataset(_BaseNavDataset):
    """RxR-CE loader."""

    source_name = "rxr-ce"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> RxRCEDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)


class SQA3DDataset(_BaseNavDataset):
    """SQA3D loader.

    Keeps same NavSample shape for compatibility with shared collators.
    """

    source_name = "sqa3d"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> SQA3DDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)


def build_dataset(
    name: str,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
) -> Dataset[NavSample]:
    """Factory for supported benchmark datasets."""
    key = name.strip().lower()
    if key in {"r2r", "r2r-ce", "vln-r2r"}:
        return R2RCEDataset.from_file(path, split=split, limit=limit)
    if key in {"rxr", "rxr-ce", "vln-rxr"}:
        return RxRCEDataset.from_file(path, split=split, limit=limit)
    if key in {"sqa3d"}:
        return SQA3DDataset.from_file(path, split=split, limit=limit)
    raise ValueError(f"Unsupported dataset name: {name}")


class CachedFrameDataset(Dataset):
    """Load pre-rendered .pt frames from a directory.

    Each ``.pt`` file is a dict with at minimum:
        rgb:          Tensor[3, H, W]    float32, [0,1]
        depth:        Tensor[H, W]       float32, metric metres
        instruction:  str
    Optional keys:
        intrinsics:   dict  (fx, fy, cx, cy, width, height)
        target:       str   (for SFT — reasoning + action)
        episode_id:   str
        source:       str   (r2r-ce, rxr-ce, sqa3d)
    """

    def __init__(
        self,
        frame_dir: str | Path,
        limit: int | None = None,
    ) -> None:
        self.frame_dir = Path(frame_dir)
        if not self.frame_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {self.frame_dir}")
        self._paths = sorted(self.frame_dir.glob("*.pt"))
        if limit is not None:
            self._paths = self._paths[:limit]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch as _torch

        data = _torch.load(self._paths[idx], map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in {self._paths[idx]}, got {type(data)}")
        return data


def iter_instructions(dataset: Iterable[NavSample]) -> Iterable[str]:
    """Yield instruction strings from standardized samples."""
    for sample in dataset:
        yield sample.instruction
