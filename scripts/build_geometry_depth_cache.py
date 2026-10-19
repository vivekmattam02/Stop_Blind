#!/usr/bin/env python3
"""Build an RGB+metric-depth observation cache matched to the frozen native-C0 cache.

The C0 cache stores RGB only --- its contract records ``contains_depth: False`` and every object
lists ``depth`` under ``explicitly_excluded_policy_fields``. The geometry branch backprojects metric
depth into 3D points, so no geometry experiment can run against it. This builds the matched
depth-bearing cache.

Matched means the decisions are identical. The sample list is read from the frozen C0 cache manifest
rather than resampled, so every object here corresponds one-to-one with a C0 object: same episode,
same ``action_step_index``, same replay prefix, same scene. A geometry run against this cache
therefore differs from the C0 run in what the policy can *see*, not in what it is asked about.

Replay semantics reproduce ``run_spark_native_history_select._render_cache_entry`` exactly: reset,
capture, then step each prefix action capturing after each, keeping the last four frames, with the
last three prefix actions as the history. The only addition is that ``depth_meters`` is captured
alongside RGB at every frame.

Sealed surfaces are enforced: only ``train`` and ``selection`` may be built. The natural-outcome and
balanced-diagnostic partitions are refused, and ``val_unseen`` is never referenced.

Content-addressed and resumable at the object boundary: an existing object whose recorded identity
matches is reused, so an interrupted build continues rather than restarting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment

GATE = "gridnav_geometry_depth_cache_v1"
ALLOWED_SPLITS = ("train", "selection")
SEALED_SPLITS = ("primary_evaluation", "balanced_diagnostic")
HISTORY_FRAMES = 4


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _identity(sample: dict[str, Any], release: dict[str, str]) -> dict[str, Any]:
    return {
        "gate": GATE,
        "schema_version": 1,
        "scene_id": sample["scene_id"],
        "trajectory_id": sample["trajectory_id"],
        "episode_id": sample["episode_id"],
        "action_step_index": sample["action_step_index"],
        "replay_prefix_action_ids": list(sample["replay_prefix_action_ids"]),
        "replay_prefix_sha256": sample["replay_prefix_sha256"],
        "history_frames": HISTORY_FRAMES,
        **release,
    }


def _render(
    environment: HabitatSimR2REnvironment, episode: Any, sample: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    state = environment.reset(episode)
    rgb: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
    depth: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)

    def capture(observation: dict[str, Any]) -> None:
        rgb.append(np.asarray(observation["rgb"], dtype=np.uint8).copy())
        metric = observation.get("depth_meters")
        if metric is None:
            raise RuntimeError(
                "Observation carries no 'depth_meters'; this cache exists to store metric depth."
            )
        metric = np.asarray(metric, dtype=np.float32)
        if not np.isfinite(metric).all():
            raise RuntimeError("Captured metric depth contains non-finite values.")
        depth.append(metric.copy())

    capture(state.observation)
    prefix = tuple(R2RCEAction(int(v)) for v in sample["replay_prefix_action_ids"])
    for action in prefix:
        if action is R2RCEAction.STOP:
            raise RuntimeError("A cache replay prefix contains STOP.")
        state = environment.step(action)
        capture(state.observation)

    if len(rgb) != HISTORY_FRAMES or len(depth) != HISTORY_FRAMES:
        raise RuntimeError("Causal H4 entry does not contain exactly four real frames.")
    history = tuple(int(a) for a in prefix[-3:])
    if len(history) != 3:
        raise RuntimeError("Causal H4 entry does not contain three prior actions.")
    return (
        torch.from_numpy(np.stack(tuple(rgb), axis=0).copy()),
        torch.from_numpy(np.stack(tuple(depth), axis=0).copy()),
        history,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    args = parser.parse_args()

    for split in args.splits:
        if split in SEALED_SPLITS:
            raise SystemExit(f"Split {split!r} is a sealed outcome surface and must not be built.")
        if split not in ALLOWED_SPLITS:
            raise SystemExit(f"Split {split!r} is not authorized; allowed: {ALLOWED_SPLITS}.")

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    args.object_root.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    release = {
        "episodes_sha256": source["dataset"]["episodes_sha256"],
        "ground_truth_sha256": source["dataset"].get("ground_truth_sha256", ""),
    }
    samples = [s for s in source["samples"] if s["split"] in args.splits]
    # Group by scene. Habitat reloads the scene on every reset that changes it, which costs about
    # 10 s; the 3,328 authorized decisions span only 61 scenes, so scene-major order turns roughly
    # 3,328 scene loads into roughly 61. Order stays deterministic.
    samples.sort(key=lambda s: (s["scene_id"], s["split"], int(s["sample_index"])))
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No sample matched the requested splits.")

    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )

    entries: list[dict[str, Any]] = []
    rendered = reused = 0
    for index, sample in enumerate(samples):
        if (time.time() - started) / 3600.0 >= args.max_hours:
            print("[stop] wall-clock cap reached", flush=True)
            break
        identity = _identity(sample, release)
        key = _canonical_sha256(identity)
        path = args.object_root / key[:2] / f"{key}.pt"

        if path.exists():
            try:
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if existing.get("object_identity") == identity:
                    entries.append(
                        {
                            "cache_key": key,
                            "relative_path": str(path.relative_to(args.object_root)),
                            "split": sample["split"],
                            "sample_index": int(sample["sample_index"]),
                            "rgb_frames_sha256": existing["rgb_frames_sha256"],
                            "depth_frames_sha256": existing["depth_frames_sha256"],
                        }
                    )
                    reused += 1
                    continue
            except Exception:  # noqa: BLE001 - a corrupt object is simply re-rendered
                pass

        episode = dataset.episode_record(str(sample["episode_id"]))
        rgb_frames, depth_frames, history = _render(environment, episode, sample)
        payload = {
            "schema_version": 1,
            "gate": GATE,
            "cache_key": key,
            "object_identity": identity,
            "rgb_frames": rgb_frames,
            "rgb_frames_sha256": _tensor_sha256(rgb_frames),
            "depth_frames_m": depth_frames,
            "depth_frames_sha256": _tensor_sha256(depth_frames),
            "history_action_ids": history,
            "policy_visible_fields": ["rgb", "depth_meters", "previous_action_id_when_enabled"],
            "explicitly_excluded_policy_fields": ["simulator_pose", "goal_coordinate"],
            "depth_units": "metres",
            "depth_clip_m": [0.0, 10.0],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        entries.append(
            {
                "cache_key": key,
                "relative_path": str(path.relative_to(args.object_root)),
                "split": sample["split"],
                "sample_index": int(sample["sample_index"]),
                "rgb_frames_sha256": payload["rgb_frames_sha256"],
                "depth_frames_sha256": payload["depth_frames_sha256"],
            }
        )
        rendered += 1
        if rendered % 25 == 0 or index == len(samples) - 1:
            elapsed = time.time() - started
            rate = elapsed / max(rendered, 1)
            remaining = (len(samples) - index - 1) * rate
            print(
                f"[{index + 1:5d}/{len(samples)}] rendered={rendered} reused={reused} "
                f"{elapsed / 3600:.2f}h elapsed, {rate:.2f}s/decision, "
                f"~{remaining / 3600:.2f}h remaining",
                flush=True,
            )
            (args.out / "progress.json").write_text(
                json.dumps(
                    {
                        "completed": index + 1,
                        "target": len(samples),
                        "rendered": rendered,
                        "reused": reused,
                        "seconds_per_decision": rate,
                        "elapsed_seconds": elapsed,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    manifest = {
        "gate": GATE,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed" if len(entries) == len(samples) else "partial",
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": hashlib.sha256(
            args.source_manifest.read_bytes()
        ).hexdigest(),
        "splits_built": sorted({e["split"] for e in entries}),
        "sealed_splits_refused": list(SEALED_SPLITS),
        "val_unseen_referenced": False,
        "entry_count": len(entries),
        "rendered_this_invocation": rendered,
        "reused_this_invocation": reused,
        "elapsed_seconds": time.time() - started,
        "contains_depth": True,
        "depth_units": "metres",
        "history_frames": HISTORY_FRAMES,
        "replay_semantics": (
            "reset, capture, then step each replay-prefix action capturing after each; keep the "
            "last four frames and the last three prefix actions. Identical to the frozen C0 cache "
            "except that metric depth is captured alongside RGB."
        ),
        "entries": entries,
        "entry_index_sha256": _canonical_sha256(entries),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[done] status={manifest['status']} entries={len(entries)} "
        f"rendered={rendered} reused={reused} {manifest['elapsed_seconds'] / 3600:.2f}h",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
