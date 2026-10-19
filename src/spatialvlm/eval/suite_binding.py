"""Bind a frozen episode suite to the dataset release it was drawn from, and fail closed.

R2R-CE episode identifiers are unique *within* a release and are reused *across* releases. A suite
manifest that carries only episode ids therefore resolves without error against the wrong release,
and every lookup silently returns a different episode in a different scene. That is not a
hypothetical: the six rescue closed loops of 2026-09-06 resolved a ``val_seen``-derived suite
against the ``train`` release and executed 61 episodes nobody had selected. Episode ``328`` starts
at (-6.385, 0.072, -2.095) in the suite's own release and at (38.671, 0.080, 22.795) in the release
that was loaded.

Nothing in the numbers looks wrong when this happens, so the check has to be structural. Suite
entries already carry ``trajectory_id`` and ``scene_id``; this module compares them against the
loaded release entry by entry and refuses to run on the first disagreement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class _EpisodeSource(Protocol):
    """The part of the dataset API this check needs."""

    def episode_record(self, episode_id: str | int) -> Any: ...


@dataclass(frozen=True)
class SuiteBinding:
    """Provenance for one suite executed against one release, stamped into the run manifest."""

    suite_path: str
    suite_sha256: str
    suite_declared_split: str | None
    release_split: str | None
    release_id: str | None
    episodes_path: str | None
    ground_truth_path: str | None
    episode_count: int
    verified_fields: tuple[str, ...] = field(default=("episode_id", "trajectory_id", "scene_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_path": self.suite_path,
            "suite_sha256": self.suite_sha256,
            "suite_declared_split": self.suite_declared_split,
            "release_split": self.release_split,
            "release_id": self.release_id,
            "episodes_path": self.episodes_path,
            "ground_truth_path": self.ground_truth_path,
            "episode_count": self.episode_count,
            "verified_fields": list(self.verified_fields),
            "verified_by": "spatialvlm.eval.suite_binding",
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_split(suite: dict[str, Any]) -> str | None:
    """Read the split a suite claims, wherever the manifest happens to record it."""
    for key in ("split", "release_split"):
        if isinstance(suite.get(key), str):
            return suite[key]
    protocol = suite.get("protocol")
    if isinstance(protocol, dict) and isinstance(protocol.get("split"), str):
        return protocol["split"]
    return None


def verify_suite_against_release(
    suite_path: Path,
    dataset: _EpisodeSource,
    *,
    loaded_split: str | None,
    allow_split_mismatch: bool = False,
) -> SuiteBinding:
    """Check every suite entry resolves to the same episode in the loaded release.

    Raises ``SystemExit`` on the first disagreement, naming the entry and both values, rather than
    running and producing well-formed numbers for the wrong episodes.
    """
    suite_path = Path(suite_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    entries = suite["episodes"]
    declared = _declared_split(suite)

    if declared is not None and loaded_split is not None and declared != loaded_split:
        message = (
            f"Suite {suite_path} declares split {declared!r} but the loaded release is "
            f"{loaded_split!r}. Episode ids are reused across releases, so this would execute "
            f"different episodes than the suite names."
        )
        if not allow_split_mismatch:
            raise SystemExit(message)
        print(f"[suite] WARNING, running anyway on explicit request: {message}", flush=True)

    for position, entry in enumerate(entries):
        episode_id = str(entry["episode_id"])
        try:
            record = dataset.episode_record(episode_id)
        except KeyError as exc:  # noqa: PERF203 - the message matters more than the loop cost
            raise SystemExit(
                f"Suite entry {position} names episode {episode_id!r}, absent from the loaded "
                f"release. {exc}"
            ) from exc
        for attribute in ("trajectory_id", "scene_id"):
            expected = entry.get(attribute)
            if expected is None:
                continue
            found = getattr(record, attribute, None)
            if found is not None and str(found) != str(expected):
                raise SystemExit(
                    f"Suite/release mismatch at entry {position}, episode {episode_id}: the suite "
                    f"says {attribute}={expected!r}, the loaded release says {found!r}. Episode "
                    f"ids are reused across releases; this suite was not drawn from this release. "
                    f"Refusing to run."
                )

    release = getattr(dataset, "release_manifest", None)
    binding = SuiteBinding(
        suite_path=str(suite_path),
        suite_sha256=sha256_file(suite_path),
        suite_declared_split=declared,
        release_split=getattr(release, "split", loaded_split),
        release_id=getattr(release, "release_id", None),
        episodes_path=getattr(release, "episodes_path", None),
        ground_truth_path=getattr(release, "ground_truth_path", None),
        episode_count=len(entries),
    )
    print(
        f"[suite] {len(entries)} entries verified against release "
        f"split={binding.release_split!r} on {', '.join(binding.verified_fields)}",
        flush=True,
    )
    return binding
