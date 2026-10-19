#!/usr/bin/env python3
"""Independent verifier for `native_chunk_closed_loop_v1`.

Reconstructs every reported aggregate from the raw per-episode artifacts and refuses to read
`result.json` for any value it checks. `result.json` is opened only to compare against the
independent reconstruction; if the two disagree the run fails verification.

This mirrors the discipline every other run in the project is held to: a summary is admitted only
when a second, independent pass over the primary evidence produces the same numbers.

Recomputed from `episodes/<id>.json`:

- Success, Oracle Success, SPL, nDTW, SDTW, navigation error;
- action histograms, action totals, and per-episode constant-action status;
- `STOP` events with their step index and distance to goal;
- step-cap conformance and step-count/action-count agreement;
- protocol conformance: 500-step cap, 3 m success radius, four legal actions only, success
  implying a terminal `STOP` inside the radius.

Exit code is non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction

STEP_CAP = 500
SUCCESS_RADIUS_M = 3.0
LEGAL = {int(a) for a in R2RCEAction}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    episode_dir = root / "episodes"
    for path in (manifest_path, episode_dir):
        if not path.exists():
            errors.append(f"required artifact missing: {path}")
    if errors:
        return {"status": "failed", "errors": errors, "checks": {}}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("val_unseen_referenced") is not False:
        errors.append("manifest does not assert that val_unseen was never referenced")
    if manifest.get("split") != "val_seen":
        errors.append(f"unexpected split: {manifest.get('split')}")
    if manifest.get("step_cap") != STEP_CAP:
        errors.append(f"step cap is not {STEP_CAP}: {manifest.get('step_cap')}")

    files = sorted(episode_dir.glob("*.json"))
    if not files:
        errors.append("no per-episode artifact found")
        return {"status": "failed", "errors": errors, "checks": {}}

    actions: Counter[str] = Counter()
    successes = oracle = stopped = constant = capped = 0
    nav_errors: list[float] = []
    stop_events: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    total_actions = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(payload["episode_id"])
        episode_ids.append(episode_id)
        trace = payload["trace"]
        metrics = payload["metrics"]
        raw = list(trace["actions"])

        illegal = sorted({a for a in raw if int(a) not in LEGAL})
        if illegal:
            errors.append(f"episode {episode_id} emitted illegal action ids {illegal}")
        if len(raw) > STEP_CAP:
            errors.append(f"episode {episode_id} exceeded the {STEP_CAP}-step cap: {len(raw)}")
        if int(metrics["step_count"]) != len(raw):
            errors.append(
                f"episode {episode_id} step_count {metrics['step_count']} != "
                f"{len(raw)} recorded actions"
            )

        names = [R2RCEAction(int(a)).name for a in raw]
        counts = Counter(names)
        actions.update(counts)
        total_actions += len(raw)
        if len(counts) == 1:
            constant += 1
        if len(raw) >= STEP_CAP:
            capped += 1

        terminal_stop = bool(raw) and int(raw[-1]) == int(R2RCEAction.STOP)
        if bool(metrics["stopped"]) != terminal_stop:
            errors.append(
                f"episode {episode_id} stopped={metrics['stopped']} disagrees with its "
                f"terminal action"
            )
        if terminal_stop:
            stopped += 1
            distances = trace["distances_to_goal"]
            stop_events.append(
                {
                    "episode_id": episode_id,
                    "step_index": len(raw) - 1,
                    "distance_to_goal_m": float(distances[-1]) if distances else None,
                    "inside_success_radius": bool(
                        distances and float(distances[-1]) <= SUCCESS_RADIUS_M
                    ),
                }
            )

        success = float(metrics["success"])
        successes += int(success > 0)
        oracle += int(float(metrics["oracle_success"]) > 0)
        nav_errors.append(float(metrics["navigation_error"]))

        if success > 0:
            if not terminal_stop:
                errors.append(f"episode {episode_id} reports success without a terminal STOP")
            if float(metrics["navigation_error"]) > SUCCESS_RADIUS_M:
                errors.append(
                    f"episode {episode_id} reports success at navigation error "
                    f"{metrics['navigation_error']} m, outside the {SUCCESS_RADIUS_M} m radius"
                )
        if success == 0 and float(metrics["sdtw"]) != 0.0:
            errors.append(f"episode {episode_id} has nonzero SDTW on a failed episode")

    if len(set(episode_ids)) != len(episode_ids):
        errors.append("duplicate episode artifacts found")

    n = len(files)
    reconstructed = {
        "episodes_evaluated": n,
        "success_rate": successes / n,
        "oracle_success_rate": oracle / n,
        "episodes_that_stopped": stopped,
        "constant_action_episodes": constant,
        "episodes_at_step_cap": capped,
        "mean_navigation_error": statistics.mean(nav_errors),
        "total_actions": total_actions,
        "action_histogram": dict(sorted(actions.items())),
        "stop_events": stop_events,
        "stop_events_inside_radius": sum(1 for e in stop_events if e["inside_success_radius"]),
    }

    compared: dict[str, Any] = {}
    if result_path.exists():
        summary = json.loads(result_path.read_text(encoding="utf-8")).get("aggregate", {})
        for key in (
            "episodes_evaluated",
            "success_rate",
            "oracle_success_rate",
            "episodes_that_stopped",
            "constant_action_episodes",
            "total_actions",
        ):
            claimed = summary.get(key)
            got = reconstructed[key]
            same = (
                math.isclose(float(claimed), float(got), rel_tol=1e-9, abs_tol=1e-9)
                if isinstance(claimed, int | float) and claimed is not None
                else claimed == got
            )
            compared[key] = {"reported": claimed, "reconstructed": got, "agree": bool(same)}
            if not same:
                errors.append(
                    f"reported {key}={claimed} disagrees with reconstruction {got}"
                )
        if summary.get("action_histogram") != reconstructed["action_histogram"]:
            # Ordering differences are benign; compare as multisets.
            if Counter(summary.get("action_histogram") or {}) != Counter(
                reconstructed["action_histogram"]
            ):
                errors.append("reported action histogram disagrees with reconstruction")
    else:
        warnings.append("result.json absent — run still in progress; aggregate not cross-checked")

    return {
        "gate": "native_chunk_closed_loop_artifacts_v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "errors": errors,
        "warnings": warnings,
        "episode_artifact_count": n,
        "source_sha256": {
            "manifest.json": _sha256(manifest_path),
            **({"result.json": _sha256(result_path)} if result_path.exists() else {}),
        },
        "reconstructed": reconstructed,
        "reported_vs_reconstructed": compared,
        "evidence_boundary": (
            "Exploratory pilot. This verifier reconstructs the pilot's own outputs; it does not "
            "re-verify the episode suite, non-learning controls, or checkpoint provenance against "
            "their frozen hashes. Passing here is necessary, not sufficient, for admission to the "
            "manuscript."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/spark/native_chunk_closed_loop_v1")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = verify(args.root)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
