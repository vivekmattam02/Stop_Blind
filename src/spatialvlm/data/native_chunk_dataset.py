"""Build k-action chunk training samples from the frozen content-addressed observation cache.

No new rendering is required. Each cache object already holds the complete H4 window as a
``(4, 224, 224, 3)`` uint8 tensor plus an ``object_identity`` recording ``trajectory_id`` and
``action_step_index``. A chunk target is ``actions[t : t + chunk_size]``, which comes from the
released ground truth at no simulator cost. Extending the cache costs a measured 2.4372 s per
decision, so reusing it is worth roughly 31 hours at the scales under discussion.

**The action sequence must be resolved exactly as the frozen runner resolved it.** Ground truth is
keyed by *episode* ID, several episodes share one physical trajectory, and the frozen protocol picks
one canonical episode per trajectory with a seeded stable hash. Looking up
``ground_truth[trajectory_id]`` instead returns a different episode's actions and silently misaligns
every target. ``_canonical_episodes`` is therefore imported from the frozen module rather than
reimplemented, and every sample is checked against the cache object's own
``replay_prefix_action_ids`` before it is admitted.

Sealed surfaces are enforced here, not by convention: only the splits named in ``allowed_splits``
may be read, and the natural-outcome and balanced-diagnostic partitions are rejected by default.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.history import HistoryContract, HistoryWindow, build_history_window

# Imported from the hash-locked frozen runner so that canonical-episode selection cannot drift.
from spatialvlm.training.native_c0_development import _canonical_episodes  # noqa: PLC2701

TRAIN_SPLIT = "train"
SELECTION_SPLIT = "selection"
SEALED_SPLITS = frozenset({"primary_evaluation", "balanced_diagnostic"})
DEFAULT_ALLOWED_SPLITS = frozenset({TRAIN_SPLIT, SELECTION_SPLIT})


@dataclass(frozen=True)
class ChunkSample:
    """One chunk-supervised training example resolved against the frozen cache."""

    cache_key: str
    split: str
    sample_index: int
    trajectory_id: str
    episode_id: str
    scene_id: str
    action_step_index: int
    instruction: str
    target_actions: tuple[R2RCEAction, ...]
    rgb_frames: torch.Tensor | None
    history_action_ids: tuple[int, ...]
    object_path: Path | None = None

    def frames(self) -> torch.Tensor:
        """Return the cached RGB frames, reading them from disk if they were not retained.

        A retained tensor is returned unchanged, so the eager path is bit-identical to before.
        Under ``lazy_frames`` the loader keeps only ``object_path`` and the frames are read here;
        at 0.71 MB per sample that is the difference between 95 GB resident and 135 MB.
        """
        if self.rgb_frames is not None:
            return self.rgb_frames
        if self.object_path is None:
            raise RuntimeError(
                f"ChunkSample {self.cache_key} retained no frames and carries no object path."
            )
        return torch.load(self.object_path, map_location="cpu", weights_only=False)["rgb_frames"]

    def window(self, contract: HistoryContract) -> HistoryWindow:
        """Materialise the H4 window this sample was cached for."""
        rgb = self.frames()
        frames = [rgb[i].numpy() for i in range(rgb.shape[0])]
        return build_history_window(
            frames, [R2RCEAction(int(a)) for a in self.history_action_ids], contract=contract
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "split": self.split,
            "sample_index": self.sample_index,
            "trajectory_id": self.trajectory_id,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "action_step_index": self.action_step_index,
            "target_action_ids": [int(a) for a in self.target_actions],
            "target_action_names": [a.name for a in self.target_actions],
            "target_chunk_is_constant": len({int(a) for a in self.target_actions}) == 1,
            "history_action_ids": list(self.history_action_ids),
        }


def load_chunk_samples(
    *,
    cache_index_path: Path | str,
    object_root: Path | str,
    dataset: R2RCEDataset,
    sampler_seed: int,
    chunk_size: int,
    split: str,
    allowed_splits: frozenset[str] = DEFAULT_ALLOWED_SPLITS,
    limit: int | None = None,
    lazy_frames: bool = False,
) -> list[ChunkSample]:
    """Load every cached decision of one split and attach its k-action chunk target.

    Fails closed on a sealed split, a missing cache object, a trajectory whose canonical episode is
    unknown, a replay prefix that disagrees with the resolved ground truth, or an episode too short
    to supply a full chunk at the cached step.

    ``lazy_frames`` changes only what is *retained*, never what is *checked*: every object is still
    opened and its recorded identity still validated against resolved ground truth, but the RGB
    tensor is dropped and re-read from disk when the sample is used. This exists because the eager
    path holds 0.71 MB per sample, which is 95 GB at the 134,754-decision scale on a machine whose
    121.7 GB is shared between host and accelerator. Default is ``False``, so every prior run's
    code path is unchanged.
    """
    if split in SEALED_SPLITS:
        raise ValueError(
            f"Split {split!r} is a sealed outcome surface and must not be opened by this track."
        )
    if split not in allowed_splits:
        raise ValueError(f"Split {split!r} is not in the authorized set {sorted(allowed_splits)}.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one.")

    index = json.loads(Path(cache_index_path).read_text(encoding="utf-8"))
    entries = [entry for entry in index["entries"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"Cache index contains no entries for split {split!r}.")
    entries.sort(key=lambda entry: int(entry["sample_index"]))
    if limit is not None:
        entries = entries[:limit]

    canonical = _canonical_episodes(dataset, sampler_seed)
    root = Path(object_root)
    samples: list[ChunkSample] = []
    short_episodes = 0

    for entry in entries:
        object_file = root / entry["relative_path"]
        obj = torch.load(object_file, map_location="cpu", weights_only=False)
        identity = obj["object_identity"]
        trajectory_id = str(identity["trajectory_id"])
        step_index = int(identity["action_step_index"])

        episode = canonical.get(trajectory_id)
        if episode is None:
            raise KeyError(
                f"Cache object {entry['cache_key']} names trajectory {trajectory_id}, which has no "
                "canonical episode under the frozen sampler seed."
            )
        actions = list(dataset.ground_truth_record(episode.episode_id).actions)

        observed_prefix = [int(a) for a in identity["replay_prefix_action_ids"]]
        resolved_prefix = [int(a) for a in actions[:step_index]]
        if observed_prefix != resolved_prefix:
            raise ValueError(
                "Resolved ground truth disagrees with the cache object's replay prefix for "
                f"trajectory {trajectory_id} at step {step_index}: cache={observed_prefix}, "
                f"resolved={resolved_prefix}. The episode mapping is wrong; refusing to build "
                "misaligned targets."
            )

        if step_index + chunk_size > len(actions):
            short_episodes += 1
            continue

        samples.append(
            ChunkSample(
                cache_key=str(entry["cache_key"]),
                split=split,
                sample_index=int(entry["sample_index"]),
                trajectory_id=trajectory_id,
                episode_id=str(episode.episode_id),
                scene_id=str(identity["scene_id"]),
                action_step_index=step_index,
                instruction=episode.instruction_text,
                target_actions=tuple(
                    R2RCEAction(int(a)) for a in actions[step_index : step_index + chunk_size]
                ),
                rgb_frames=None if lazy_frames else obj["rgb_frames"],
                object_path=object_file,
                history_action_ids=tuple(int(a) for a in obj["history_action_ids"]),
            )
        )

    if not samples:
        raise ValueError(f"No {split!r} cache object could supply a full chunk of {chunk_size}.")
    return samples


def chunk_sample_summary(samples: Sequence[ChunkSample]) -> dict[str, Any]:
    """Summarise a resolved chunk sample set for the run manifest."""
    from collections import Counter

    per_action: Counter[str] = Counter()
    chunk_patterns: Counter[tuple[int, ...]] = Counter()
    constant = 0
    with_stop = 0
    for sample in samples:
        ids = tuple(int(a) for a in sample.target_actions)
        chunk_patterns[ids] += 1
        for value in ids:
            per_action[R2RCEAction(value).name] += 1
        if len(set(ids)) == 1:
            constant += 1
        if int(R2RCEAction.STOP) in ids:
            with_stop += 1
    total = len(samples)
    best = chunk_patterns.most_common(1)[0]
    return {
        "sample_count": total,
        "supervised_position_count": total * len(samples[0].target_actions),
        "distinct_chunk_patterns": len(chunk_patterns),
        "constant_chunk_count": constant,
        "constant_chunk_fraction": constant / total,
        "chunks_containing_stop": with_stop,
        "chunks_containing_stop_fraction": with_stop / total,
        "most_common_chunk": list(best[0]),
        "most_common_chunk_count": best[1],
        "most_common_chunk_fraction": best[1] / total,
        "supervised_action_histogram": dict(sorted(per_action.items())),
        "distinct_trajectories": len({s.trajectory_id for s in samples}),
        "distinct_scenes": len({s.scene_id for s in samples}),
    }
