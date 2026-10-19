#!/usr/bin/env python3
"""Expand the frozen C0 training partition to a larger decision budget, without moving any boundary.

The frozen study drew 3,072 training decisions from 2,451 physical trajectories --- roughly 1.25
decisions per trajectory --- while the released split holds far more eligible decisions inside those
same trajectories. This emits a manifest, shaped like the frozen cache manifest so the existing
cache builder consumes it unchanged, that draws a larger number of decisions from **exactly the same
2,451 trajectories**.

What stays fixed. The trajectory partition, the canonical-episode choice per trajectory, the
sampler seed, and the eligibility rules are all imported from the frozen module
``native_c0_development`` rather than reimplemented: a decision is eligible only at step index at
least ``min_decision_step``, at most ``max_replay_prefix_actions``, and with no ``STOP`` in its
replay prefix. The frozen 3,072-decision list is emitted first and in its original order, so it is a
prefix of the expanded list and any comparison against the frozen run is nested rather than
disjoint.

What changes. Only how many decisions each training trajectory contributes. Decisions are drawn
round-robin across trajectories by a seeded stable hash, so the expansion spreads across
trajectories and scenes instead of exhausting a few, and it is deterministic.

The selection, natural-outcome and balanced-diagnostic partitions are copied through untouched;
this script never adds to them and never reads their contents. ``val_unseen`` is not referenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.training.native_c0_development import (  # noqa: PLC2701
    NativeC0DevelopmentProtocol,
    _canonical_episodes,
    _eligible_candidates,
)

GATE = "gridnav_scale_sample_manifest_v1"


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rank(seed: int, trajectory_id: str, step_index: int) -> str:
    return hashlib.sha256(
        f"gridnav-scale-decision-v1|{seed}|{trajectory_id}|{step_index}".encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--target-train-decisions", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    protocol = NativeC0DevelopmentProtocol()
    frozen = source["samples"]
    frozen_train = [s for s in frozen if s["split"] == "train"]
    other = [s for s in frozen if s["split"] != "train"]
    frozen_train.sort(key=lambda s: int(s["sample_index"]))
    train_trajectories = {str(s["trajectory_id"]) for s in frozen_train}
    if len(train_trajectories) != protocol.train_trajectories:
        raise SystemExit(
            f"expected {protocol.train_trajectories} train trajectories, "
            f"found {len(train_trajectories)}"
        )

    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    canonical = _canonical_episodes(dataset, protocol.sampler_seed)
    candidates = _eligible_candidates(dataset, protocol)

    taken = {(str(s["trajectory_id"]), int(s["action_step_index"])) for s in frozen_train}
    pool: dict[str, list[Any]] = defaultdict(list)
    for trajectory_id in sorted(train_trajectories):
        rows = candidates.get(trajectory_id, [])
        fresh = [r for r in rows if (trajectory_id, r.step_index) not in taken]
        fresh.sort(key=lambda r: _rank(protocol.sampler_seed, trajectory_id, r.step_index))
        pool[trajectory_id] = fresh

    available = sum(len(v) for v in pool.values()) + len(frozen_train)
    if args.target_train_decisions > available:
        raise SystemExit(
            f"requested {args.target_train_decisions} train decisions but only {available} are "
            f"eligible inside the frozen {len(train_trajectories)} training trajectories"
        )

    expanded = list(frozen_train)
    order = sorted(train_trajectories)
    depth = 0
    while len(expanded) < args.target_train_decisions:
        progressed = False
        for trajectory_id in order:
            if len(expanded) >= args.target_train_decisions:
                break
            rows = pool[trajectory_id]
            if depth >= len(rows):
                continue
            row = rows[depth]
            episode = canonical[trajectory_id]
            prefix = [int(a) for a in row.replay_prefix]
            expanded.append(
                {
                    "split": "train",
                    "sample_index": len(expanded),
                    "episode_id": str(episode.episode_id),
                    "trajectory_id": trajectory_id,
                    "scene_id": str(episode.scene_id),
                    "instruction_text": episode.instruction_text,
                    "instruction_sha256": hashlib.sha256(
                        episode.instruction_text.encode("utf-8")
                    ).hexdigest(),
                    "action_step_index": int(row.step_index),
                    "target_action_id": int(row.action),
                    "target_action": R2RCEAction(int(row.action)).name,
                    "replay_prefix_action_ids": prefix,
                    "replay_prefix_sha256": _sha256(prefix),
                    "order_rank_sha256": _rank(
                        protocol.sampler_seed, trajectory_id, int(row.step_index)
                    ),
                    "selection_rank_sha256": "",
                }
            )
            progressed = True
        if not progressed:
            break
        depth += 1

    if len(expanded) != args.target_train_decisions:
        raise SystemExit(f"could only build {len(expanded)} of {args.target_train_decisions}")

    prefix_ok = all(
        expanded[i]["action_step_index"] == frozen_train[i]["action_step_index"]
        and expanded[i]["trajectory_id"] == frozen_train[i]["trajectory_id"]
        for i in range(len(frozen_train))
    )
    if not prefix_ok:
        raise SystemExit("expanded list does not begin with the frozen 3,072-decision list")
    if len({(s["trajectory_id"], s["action_step_index"]) for s in expanded}) != len(expanded):
        raise SystemExit("expanded list contains a duplicate decision")
    if {str(s["trajectory_id"]) for s in expanded} - train_trajectories:
        raise SystemExit("expanded list left the frozen training trajectory set")

    from collections import Counter

    actions = Counter(s["target_action"] for s in expanded)
    manifest = {
        "gate": GATE,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "dataset": source["dataset"],
        "source_manifest_sha256": hashlib.sha256(
            args.source_manifest.read_bytes()
        ).hexdigest(),
        "expansion": {
            "frozen_train_decisions": len(frozen_train),
            "expanded_train_decisions": len(expanded),
            "train_trajectories": len(train_trajectories),
            "decisions_per_trajectory": len(expanded) / len(train_trajectories),
            "frozen_list_is_a_prefix": True,
            "eligible_pool_size": available,
            "sampler_seed": protocol.sampler_seed,
            "min_decision_step": protocol.min_decision_step,
            "max_replay_prefix_actions": protocol.max_replay_prefix_actions,
            "target_action_histogram": dict(sorted(actions.items())),
            "stop_share": actions.get("STOP", 0) / len(expanded),
        },
        "partitions_copied_untouched": sorted({s["split"] for s in other}),
        "val_unseen_referenced": False,
        "samples": expanded + other,
        "sample_records_sha256": _sha256(expanded + other),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    e = manifest["expansion"]
    print(
        f"[done] {e['expanded_train_decisions']:,} train decisions from "
        f"{e['train_trajectories']:,} frozen trajectories "
        f"({e['decisions_per_trajectory']:.1f} each); pool {e['eligible_pool_size']:,}; "
        f"STOP share {e['stop_share']*100:.2f}%; frozen list is a prefix.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
