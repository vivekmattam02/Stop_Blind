#!/usr/bin/env python3
"""Score one native-C0 policy offline on the held-out decisions (PREREG_31_CHECKPOINTS.md, 7.4, 8).

The offline step restores H2 and the offline criteria of section 8: every one of the 37 policies
is scored on the study GPU, as the first job before the main rollouts, on the three held-out
surfaces that all six training runs share: 256 selection, 640 natural and 256 balanced decisions.
One process scores one policy. It loads the checkpoint with the unchanged closed-loop loader,
``NativeQwenActionScorer.from_native_c0_checkpoint``, requires all 288 trainable tensors to be
FP32, and scores each surface by calling the training function ``_evaluate`` itself, imported
from ``scripts.run_spark_native_c0_development``. The existence gate is computed with the gate's
own function, ``evaluate_existence_gate``, and K1 to K6 with ``native_sweep_criteria``.
``torch.backends.cudnn.benchmark`` is set to False first, as training did.

* ``--step smoke`` scores wt25-u192 only, before the freeze commit, so that the scorer check of
  section 7.4 has been seen to work on the study GPU before anything is frozen;
* ``--step offline`` scores any one of the 37, after the freeze commit, on the smoke's GPU type.

Every record carries the provenance the runner records (commit, preregistration, GPU type, name,
UUID and driver, node, image, job id) and is refused on the same grounds: an output directory
other than ``<source-artifact-root>/<run id>/<checkpoint id>``, a draft preregistration after the
smoke, a GPU or image other than the committed smoke result's, a changed Python file on resume.
Whether the re-scored values agree with the training records (the scorer check) is decided by the
analysis script, from these records.

Progress: ``progress.json`` is written when the checkpoint starts loading, when each surface starts
and ends, and after every 100 decisions in between (``decisions_scored`` of 1,152), so a working
scorer never leaves it unchanged for long: the kit's hang watchdog kills a program whose progress
record has not changed for 30 minutes (section 5). ``_evaluate`` is called unchanged; it reads its
samples one decision at a time from a list that writes the record as it goes (``_CountedSamples``).

Resume: the policy is the unit. An error while loading or scoring, the FP32 check included, is a
failed attempt; after three, spanning at least two job starts (two job ids or two nodes, as the
runner counts them), the policy is recorded as not scored (``result.json`` with
``policy_scored: false``), and the analysis reports H2 and the parts of section 8 that need it as
not run. A SIGTERM is an interruption, not a failed attempt. ``failure.json`` names the checkpoint
in ``checkpoint_loading`` when the error came while loading it, so the kit can tell a failed load
(a suspect node) from a failed scoring.

Exit codes: 0 completed; 1 error after the invocation started (``failure.json``); 2 refused
before anything was loaded or written; 75 out of time before scoring started; 128 + signal number
when stopped by a signal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.run_spark_native_c0_checkpoint_sweep import (
    CACHE_INDEX_PATH,
    CACHE_INDEX_SHA256,
    LOCKED_SOURCE_SHA256,
    MODEL_SNAPSHOT_FILES,
    MODEL_WEIGHT_LOCKS,
    SMOKE_RESULT_PATH,
    _check_runtime,
    adapter_dtype_record,
    attempts_exhausted,
    check_model_snapshot,
    check_output_root,
    check_preregistration_for_step,
    check_smoke_result,
    compare_manifest,
    gpu_identity,
    loader_arguments,
    memory_record,
    platform_record,
    preregistration_record,
    verify_checkpoint_source,
)
from scripts.run_spark_native_c0_closed_loop import (
    _checkpoint_path,
    _load_json,
    _sha256,
    _software_hashes,
    _timestamp,
    _write_json,
)
from scripts.run_spark_native_c0_development import _cache_evidence, _cache_maps, _evaluate
from spatialvlm.eval.native_checkpoint_allowlist import (
    LOCKED_RUN_ID,
    RUNS,
    SCORER,
    SweepCheckpoint,
    SweepRefusalError,
    allowlist_record,
    check_run_id,
    check_step_program,
    refuse,
    resolve_checkpoint,
    resolve_step_checkpoint,
    run_of,
)
from spatialvlm.eval.native_sweep_criteria import (
    SURFACE_SIZES,
    compare_policy,
    criteria,
    recorded_decisions,
    scorer_check,
    smoke_gate,
)
from spatialvlm.eval.native_sweep_lock import (
    check_code_is_from_repo,
    check_code_unchanged,
    code_record,
    code_record_sha256,
)
from spatialvlm.eval.qwen_action_scorer import NativeQwenActionScorer
from spatialvlm.training.native_c0_development import (
    NativeC0DevelopmentProtocol,
    evaluate_existence_gate,
)

SCHEMA_VERSION = 2
GATE_ID = "native_qwen_c0_sweep_offline_scoring_v2"
SCORER_PATH = "scripts/score_native_c0_offline.py"
EXIT_COMPLETED = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_RESUMABLE = 75
# The three held-out surfaces, with the cache split each one reads.
SURFACES: dict[str, str] = {
    "selection": "selection",
    "natural": "primary_evaluation",
    "balanced": "balanced_diagnostic",
}
# progress.json is written at least this often while a policy is scored (and at each surface's start
# and end); the kit's hang watchdog kills a program whose record stays unchanged for 30 minutes.
PROGRESS_EVERY_DECISIONS = 100
TRAINING_GATE_FILE = "src/spatialvlm/training/native_c0_gate.py"
TRAINING_GATE_FILE_SHA256_AT_TRAINING = (
    "026ee1ad078cb62e0648606f9f0ec1e71d2d582f611de7042ca07d42460762a6"
)


class _StopRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _raise_stop(signum: int, _frame: Any) -> None:
    # As in the runner: from the first SIGTERM or SIGINT on, both are ignored, so a second copy
    # (Slurm signals every process of the step) cannot cut short the record this stop writes.
    for other in (signal.SIGTERM, signal.SIGINT):
        signal.signal(other, signal.SIG_IGN)
    raise _StopRequested(signum)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True, help="The one policy to score.")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--smoke-result", type=Path, default=None)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-check-record", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=30.0)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--habitat-build-lock", type=Path, default=Path("/opt/gridnav/habitat-sim-commit")
    )
    parser.add_argument("--base-variant-file", type=Path, default=Path("/opt/gridnav/base-variant"))
    parser.add_argument("--previous-stop-reason", default=None)
    return parser.parse_args(argv)


@dataclass
class OfflinePreflight:
    step: str
    run_id: str
    record: SweepCheckpoint
    preregistration: dict[str, Any]
    platform: dict[str, Any]
    smoke_result: dict[str, Any] | None
    code_files: dict[str, str]
    locked_software_sha256: dict[str, str]
    model_check: dict[str, Any]
    cache: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]]
    surface_locks: dict[str, Any]
    checkpoint_lock: dict[str, Any]
    existing_manifest: dict[str, Any] | None


def surface_samples(cache_manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The three held-out surfaces in cache-manifest order, as training filtered them."""
    samples = list(cache_manifest["samples"])
    return {
        surface: [row for row in samples if row["split"] == split]
        for surface, split in SURFACES.items()
    }


def check_run_surfaces(
    source_root: Path,
    samples: Mapping[str, Sequence[Mapping[str, Any]]],
    entries: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Refuse unless all six runs scored exactly these decisions, in this order, from this cache.

    The six runs share the three surfaces (identical ``cache_sha256`` lists), so one scoring per
    policy serves every run.
    """
    expected = {
        surface: [
            (int(row["sample_index"]), entries[(row["split"], int(row["sample_index"]))]["sha256"])
            for row in rows
        ]
        for surface, rows in samples.items()
    }
    for surface, rows in expected.items():
        if len(rows) != SURFACE_SIZES[surface]:
            raise refuse(f"the cache's {surface} surface has {len(rows)} decisions.")
    for run in RUNS:
        result = _load_json(source_root / run.source_run_id / "result.json")
        recorded = {
            "natural": result["primary_evaluation"]["per_decision"],
            "balanced": result["balanced_diagnostic"]["per_decision"],
        }
        observed: dict[str, list[list[tuple[int, str]]]] = {
            "selection": [
                [(int(row["sample_index"]), row["cache_sha256"]) for row in item["per_decision"]]
                for item in result["calibrations"]
            ],
            **{
                key: [[(int(row["sample_index"]), row["cache_sha256"]) for row in value]]
                for key, value in recorded.items()
            },
        }
        for surface, lists in observed.items():
            if any(item != expected[surface] for item in lists):
                raise refuse(
                    f"{run.source_run_id} scored a {surface} surface that differs from the cache's."
                )
    return {
        surface: {
            "decision_count": len(rows),
            "cache_sha256_list": [digest for _, digest in rows],
        }
        for surface, rows in expected.items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def failed_attempts(invocations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Exits that were neither clean nor an interruption while the policy was in progress."""
    return [
        dict(line)
        for line in invocations
        if line.get("policy_in_progress_at_exit")
        and line.get("exit_code") not in (EXIT_COMPLETED, EXIT_RESUMABLE)
        and not line.get("stop_by_hand")
    ]


def preflight(args: argparse.Namespace) -> OfflinePreflight:
    """Every refusal, before a model, a cache object or an output file is touched."""
    check_step_program(args.step, SCORER)
    if args.run_id == LOCKED_RUN_ID:
        raise refuse(f"run id {LOCKED_RUN_ID} belongs to the locked closed loop.")
    check_run_id(args.step, args.run_id)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0.0:
        raise refuse("--max-hours must be finite and positive.")
    if args.step == "smoke" and args.smoke_result is not None:
        raise refuse("the smoke does not read a smoke result.")
    record = resolve_step_checkpoint(args.step, args.checkpoint_id)
    existing = check_output_root(
        args.output_root, args.source_artifact_root, args.run_id, args.checkpoint_id
    )
    if existing is not None and existing.get("step") != args.step:
        raise refuse(f"the output directory records step {existing.get('step')!r}.")
    preregistration = preregistration_record(args.preregistration)
    check_preregistration_for_step(args.step, preregistration)
    _check_runtime(args)
    model_check = check_model_snapshot(args.model_snapshot, args.model_check_record)
    platform_now = platform_record(args, gpu_identity(args.device))
    if existing is not None:
        recorded = existing.get("platform", {})
        differing = {
            key: (recorded.get(key), platform_now[key])
            for key in ("gpu_type", "gpu_name", "image_sha256")
            if recorded.get(key) != platform_now[key]
        }
        if differing:
            raise refuse(f"this invocation differs from the policy's manifest in {differing}.")
    repo = args.repo.resolve()
    smoke_result = None
    if args.step != "smoke":
        smoke_path = args.smoke_result or repo / SMOKE_RESULT_PATH
        smoke_result = check_smoke_result(smoke_path, platform_now)
    code_files = code_record(repo, entry_file=__file__, expected_entry=SCORER_PATH)
    if existing is not None:
        check_code_unchanged(existing.get("code", {}).get("files", {}), code_files)
    locked_software = _software_hashes(repo)
    if locked_software != LOCKED_SOURCE_SHA256:
        raise refuse("locked source files differ from native_c0_closed_loop_v1.")
    index_path = args.cache_artifact_root / "cache_index.json"
    if not index_path.is_file() or _sha256(index_path) != CACHE_INDEX_SHA256:
        raise refuse(f"the held-out cache index is not {CACHE_INDEX_PATH} ({CACHE_INDEX_SHA256}).")
    try:
        cache = _cache_evidence(
            argparse.Namespace(
                cache_artifact_root=args.cache_artifact_root,
                cache_object_root=args.cache_object_root,
            )
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        raise refuse(f"the held-out cache is not admitted: {error}") from error
    samples = surface_samples(cache["manifest"])
    entries, _identities = _cache_maps(cache, args.cache_object_root.expanduser().resolve())
    surface_locks = check_run_surfaces(args.source_artifact_root, samples, entries)
    checkpoint_lock = verify_checkpoint_source(
        args.source_artifact_root, args.checkpoint_archive, record
    )
    return OfflinePreflight(
        step=args.step,
        run_id=args.run_id,
        record=record,
        preregistration=preregistration,
        platform=platform_now,
        smoke_result=smoke_result,
        code_files=code_files,
        locked_software_sha256=locked_software,
        model_check=model_check,
        cache=cache,
        samples=samples,
        surface_locks=surface_locks,
        checkpoint_lock=checkpoint_lock,
        existing_manifest=existing,
    )


def build_manifest(args: argparse.Namespace, pre: OfflinePreflight) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "gate": GATE_ID,
        "run_id": pre.run_id,
        "step": pre.step,
        "checkpoint_id": pre.record.checkpoint_id,
        "recorded_at_utc": _timestamp(),
        "scope": [pre.record.to_dict()],
        "allowlist": allowlist_record(),
        "preregistration": pre.preregistration,
        "code": {"commit": pre.platform["code_commit"], "files": pre.code_files},
        "platform": pre.platform,
        "smoke_result": pre.smoke_result,
        "software_sha256": pre.locked_software_sha256,
        "training_gate_file": {
            "path": TRAINING_GATE_FILE,
            "sha256_now": pre.locked_software_sha256[TRAINING_GATE_FILE],
            "sha256_at_training": TRAINING_GATE_FILE_SHA256_AT_TRAINING,
            "note": (
                "The scorer imports the current gate file, which holds the CR-040 loader that "
                "accepts any saved checkpoint."
            ),
        },
        "scoring": {
            "function": "scripts.run_spark_native_c0_development._evaluate",
            "gate_function": "spatialvlm.training.native_c0_development.evaluate_existence_gate",
            "loader": "NativeQwenActionScorer.from_native_c0_checkpoint",
            "cudnn_benchmark": False,
            "protocol": NativeC0DevelopmentProtocol().to_record(),
            "surfaces": SURFACES,
        },
        "cache": pre.cache["record"],
        "surfaces": pre.surface_locks,
        "checkpoint_lock": pre.checkpoint_lock,
        "model_weight_locks": MODEL_WEIGHT_LOCKS,
        "model_snapshot_files": MODEL_SNAPSHOT_FILES,
        "max_hours_per_invocation": args.max_hours,
        "resume_unit": "completed policy record",
        "outcome_used_for_scope_selection": False,
    }
    return json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))


class _CountedSamples(list):
    """One surface's samples, unchanged and in order, as the list ``_evaluate`` takes. Iterating
    it calls ``tick(n)`` each time ``_evaluate`` asks for the next sample after ``n`` decisions,
    for every ``n`` that is a positive multiple of ``every``: ``_evaluate`` scores one decision
    per sample it takes, so ``tick(n)`` runs once ``n`` decisions have been scored."""

    def __init__(
        self, rows: Sequence[Mapping[str, Any]], tick: Callable[[int], None], every: int
    ) -> None:
        super().__init__(rows)
        self._tick = tick
        self._every = every

    def __iter__(self) -> Iterator[Any]:
        for done, row in enumerate(super().__iter__()):
            if done and done % self._every == 0:
                self._tick(done)
            yield row


def score_policy(
    record: SweepCheckpoint,
    *,
    pre: OfflinePreflight,
    scorer: Any,
    device: torch.device,
    object_root: Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Score one loaded policy on the three surfaces with the training ``_evaluate``.

    ``progress(surface, decisions scored on it, decisions scored in all)`` is called when each
    surface starts and ends and after every ``PROGRESS_EVERY_DECISIONS`` decisions in between.
    """
    entries, identities = _cache_maps(pre.cache, object_root)
    surfaces: dict[str, Any] = {}
    before = 0
    for surface in SURFACES:
        rows = pre.samples[surface]

        def tick(done: int, surface: str = surface, before: int = before) -> None:
            if progress is not None:
                progress(surface, done, before + done)

        tick(0)
        surfaces[surface] = _evaluate(
            model=scorer.backbone,
            processor=scorer.processor,
            codebook=scorer.codebook,
            samples=_CountedSamples(rows, tick, PROGRESS_EVERY_DECISIONS),
            entries=entries,
            identities=identities,
            device=device,
            update_step=record.update_step,
            deadline_monotonic=math.inf,
        )
        if len(surfaces[surface]["per_decision"]) != SURFACE_SIZES[surface]:
            raise RuntimeError(f"{record.checkpoint_id}: the {surface} surface is incomplete.")
        tick(len(rows))
        before += len(rows)
    gate = evaluate_existence_gate(
        primary_decisions=surfaces["natural"]["per_decision"],
        diagnostic_decisions=surfaces["balanced"]["per_decision"],
        protocol=NativeC0DevelopmentProtocol(),
    )
    values = criteria(
        selection_rows=surfaces["selection"]["per_decision"],
        natural_rows=surfaces["natural"]["per_decision"],
        balanced_rows=surfaces["balanced"]["per_decision"],
        existence_gate=gate,
    )
    return {"surfaces": surfaces, "existence_gate": gate, "criteria": values}


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        pre = preflight(args)
        output_root = args.output_root.resolve()
        manifest_path = output_root / "manifest.json"
        manifest = build_manifest(args, pre)
        if pre.existing_manifest is not None:
            compare_manifest(pre.existing_manifest, manifest, ignore_binding=False)
        else:
            _write_json(manifest_path, manifest)
    except SweepRefusalError as refusal:
        print(json.dumps({"status": "refused", "step": args.step, "reason": str(refusal)}))
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - anything before start is a refusal to start
        print(
            json.dumps(
                {
                    "status": "refused",
                    "step": args.step,
                    "reason": f"stopped before the invocation started: {error!r}",
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        return EXIT_REFUSED
    return _execute(args, pre, manifest_path, started)


def _execute(
    args: argparse.Namespace, pre: OfflinePreflight, manifest_path: Path, started: float
) -> int:
    output_root = manifest_path.parent
    manifest_sha256 = _sha256(manifest_path)
    record = pre.record
    invocations_path = output_root / "invocations.jsonl"
    progress_path = output_root / "progress.json"
    previous = _read_jsonl(invocations_path)
    recorded = {int(line["invocation_index"]) for line in previous}
    progress = _load_json(progress_path) if progress_path.is_file() else {}
    lost = progress.get("invocation_index")
    previous_stop_reason = args.previous_stop_reason
    if (
        progress.get("status") in {"loading_checkpoint", "scoring"}
        and isinstance(lost, int)
        and lost not in recorded
    ):
        # The previous invocation ended without writing its record; unless Slurm ended it
        # (a reason is given), it is a failed attempt, as the runner counts a lost invocation.
        _append_jsonl(
            invocations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "invocation_index": lost,
                "step": pre.step,
                "run_id": pre.run_id,
                "exit_code": None,
                "outcome": "ended_without_record",
                "policy_in_progress_at_exit": progress.get("policy"),
                "stop_by_hand": previous_stop_reason is not None,
                "stop_reason": previous_stop_reason,
                "provenance": progress.get("provenance"),
                "reconstructed_by_invocation": lost + 1,
            },
        )
        recorded.add(lost)
        previous_stop_reason = None
    index = max(recorded, default=0) + 1
    failure_path = output_root / "failure.json"
    if failure_path.is_file():
        archive = output_root / "failures" / f"before_invocation_{index:04d}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        failure_path.replace(archive)
    repo = args.repo.resolve()
    device = torch.device(args.device)
    object_root = args.cache_object_root.expanduser().resolve()
    started_utc = _timestamp()
    in_progress: str | None = None
    provenance = {
        "code_commit": pre.platform["code_commit"],
        "preregistration_sha256": pre.preregistration["sha256"],
        "gpu_type": pre.platform["gpu_type"],
        "gpu_name": pre.platform["gpu_name"],
        "gpu_uuid": pre.platform["gpu_uuid"],
        "gpu_driver": pre.platform["gpu_driver"],
        "node": pre.platform["node"],
        "image_sha256": pre.platform["image_sha256"],
        "job_id": pre.platform["job_id"],
        "code_record_sha256": code_record_sha256(pre.code_files),
    }
    previous_handlers = {
        signum: signal.signal(signum, _raise_stop) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def write_progress(status: str, **fields: Any) -> None:
        _write_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "invocation_index": index,
                "policy": in_progress,
                "provenance": provenance,
                "recorded_at_utc": _timestamp(),
                **fields,
            },
        )

    def scoring_progress(surface: str, surface_done: int, done: int) -> None:
        write_progress(
            "scoring",
            surface=surface,
            surface_decisions_scored=surface_done,
            decisions_scored=done,
            decisions_total=sum(SURFACE_SIZES.values()),
        )

    def finish(exit_code: int, outcome: str, **fields: Any) -> int:
        write_progress(outcome)
        _append_jsonl(
            invocations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "invocation_index": index,
                "step": pre.step,
                "run_id": pre.run_id,
                "started_at_utc": started_utc,
                "ended_at_utc": _timestamp(),
                "exit_code": exit_code,
                "outcome": outcome,
                "policy_in_progress_at_exit": in_progress,
                "previous_stop_reason": previous_stop_reason,
                "provenance": provenance,
                "memory": memory_record(),
                **fields,
            },
        )
        return exit_code

    def write_result(scored: Mapping[str, Any] | None, reason: str | None) -> None:
        final_code = code_record(repo, entry_file=__file__, expected_entry=SCORER_PATH)
        check_code_unchanged(pre.code_files, final_code)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "gate": GATE_ID,
            "step": pre.step,
            "run_id": pre.run_id,
            "checkpoint_id": record.checkpoint_id,
            "recorded_at_utc": _timestamp(),
            "manifest_sha256": manifest_sha256,
            "preregistration_sha256": pre.preregistration["sha256"],
            "code": {"commit": pre.platform["code_commit"], "files": final_code},
            "platform": pre.platform,
            "policy_scored": scored is not None,
            "not_scored_reason": reason,
            "policy_record_sha256": (
                _sha256(output_root / "policy.json") if scored is not None else None
            ),
            "criteria": None if scored is None else scored["criteria"],
            "invocation_count": index,
            "runtime": {
                "platform": platform.platform(),
                "architecture": platform.machine().lower(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            },
            "evidence_boundary": (
                "Re-scored offline values for one policy on the study GPU; whether they agree "
                "with the training-time records is the analysis script's scorer check (7.4)."
            ),
        }
        _write_json(output_root / "result.json", result)

    loading: str | None = None
    try:
        failures = failed_attempts(_read_jsonl(invocations_path))
        if attempts_exhausted(failures):
            write_result(None, f"{len(failures)} failed attempts over at least two job starts")
            return finish(EXIT_COMPLETED, "completed_not_scored")
        if (time.monotonic() - started) / 3600.0 >= args.max_hours:
            return finish(EXIT_RESUMABLE, "partial_time_budget")
        # Set before any model is built, as training did.
        torch.backends.cudnn.benchmark = False
        in_progress = record.checkpoint_id
        loading = record.checkpoint_id
        write_progress("loading_checkpoint")
        load_started = time.monotonic()
        scorer = NativeQwenActionScorer.from_native_c0_checkpoint(
            _checkpoint_path(args.checkpoint_archive, record),
            **loader_arguments(pre.checkpoint_lock, args.device),
        )
        load_seconds = time.monotonic() - load_started
        adapter = adapter_dtype_record(scorer)
        check_code_is_from_repo(repo, entry_file=__file__, expected_entry=SCORER_PATH)
        loading = None
        scoring_started = time.monotonic()
        scored = score_policy(
            record,
            pre=pre,
            scorer=scorer,
            device=device,
            object_root=object_root,
            progress=scoring_progress,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "recorded_at_utc": _timestamp(),
            "manifest_sha256": manifest_sha256,
            "step": pre.step,
            "run_id": pre.run_id,
            "invocation_index": index,
            "checkpoint_id": record.checkpoint_id,
            "checkpoint_sha256": record.checkpoint_sha256,
            "source_run_id": record.source_run_id,
            "update_step": record.update_step,
            "role": record.role,
            "legacy_checkpoint_id": record.legacy_checkpoint_id,
            "adapter": adapter,
            "model_load_seconds": load_seconds,
            "scoring_seconds": time.monotonic() - scoring_started,
            "provenance": provenance,
            **scored,
        }
        _write_json(output_root / "policy.json", payload)
        in_progress = None
        write_result(scored, None)
        criteria_values = scored["criteria"]
        print(
            f"{record.checkpoint_id} K1={criteria_values['K1_selection_ce']:.4f} "
            f"K3={criteria_values['K3_balanced_ce']:.4f} "
            f"K4={criteria_values['K4_stop_recall_tie_to_stop']}",
            flush=True,
        )
        return finish(EXIT_COMPLETED, "completed")
    except _StopRequested as stopped:
        return finish(
            128 + stopped.signum,
            "stopped_by_signal",
            stop_by_hand=True,
            stop_reason=f"signal {signal.Signals(stopped.signum).name}",
        )
    except Exception as error:  # noqa: BLE001 - every error is logged
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_infrastructure",
            "fp32_check_failed": type(error).__name__ == "AdapterDtypeError",
            "gate": GATE_ID,
            "step": pre.step,
            "run_id": pre.run_id,
            "checkpoint_id": record.checkpoint_id,
            "invocation_index": index,
            "recorded_at_utc": _timestamp(),
            "policy_in_progress": in_progress,
            "checkpoint_loading": loading,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_root / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        return finish(EXIT_ERROR, "failed", error=str(error), error_type=type(error).__name__)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _check_main(argv: Sequence[str]) -> int:
    """The scorer check of section 7.4 for one scored policy, and the smoke's two tighter bars of
    section 6 (the smoke runs it on wt25-u192). The exit code follows the check of 7.4 alone; the
    tighter bars are recorded, and decide only whether H2 and the offline criteria are used."""
    parser = argparse.ArgumentParser(description="Compare one policy with its training records.")
    parser.add_argument("--policy", type=Path, required=True, help="the policy.json to check")
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not args.policy.is_file():
            raise refuse(f"the policy record is missing: {args.policy} (nothing was scored)")
        payload = _load_json(args.policy)
        record = resolve_checkpoint(str(payload["checkpoint_id"]))
        result_path = args.source_artifact_root / record.source_run_id / "result.json"
        if _sha256(result_path) != run_of(record).source_result_sha256:
            raise refuse(f"{result_path} is not the result native_c0_closed_loop_v1 pinned.")
        recorded = recorded_decisions(
            record.update_step, record.role == "selected", _load_json(result_path)
        )
        comparisons = compare_policy(record.checkpoint_id, payload["surfaces"], recorded)
    except SweepRefusalError as refusal:
        print(json.dumps({"status": "refused", "step": "check", "reason": str(refusal)}))
        return EXIT_REFUSED
    except (OSError, ValueError, KeyError) as error:
        reason = f"the policy record or its training result cannot be read: {error!r}"
        print(json.dumps({"status": "refused", "step": "check", "reason": reason}))
        return EXIT_REFUSED
    check = scorer_check(comparisons)
    gate = smoke_gate(comparisons)
    _write_json(
        args.output, {"scorer_check": check, "smoke_gate": gate, "comparisons": comparisons}
    )
    print(
        json.dumps(
            {
                "scorer_check": {k: v for k, v in check.items() if k != "failures"},
                "smoke_gate": {k: v for k, v in gate.items() if k != "failures"},
            },
            sort_keys=True,
        )
    )
    return EXIT_COMPLETED if check["passed"] else EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["check"]:
        return _check_main(values[1:])
    return run(_arguments(values))


if __name__ == "__main__":
    sys.exit(main())
