#!/usr/bin/env python3
"""Drive one allow-listed native-C0 checkpoint in closed loop (PREREG_31_CHECKPOINTS.md, 4.3).

This runner is additive. The locked runner, ``scripts/run_spark_native_c0_closed_loop.py``, and
the 13 other files locked by ``native_c0_closed_loop_v1`` are imported, never edited or copied:
hashing, JSON writing, report paths, the resume loader, the summaries, the control comparison and
the control hashes all come from the locked runner, and the episode loop, policy, scorer, loader
and environment are called unchanged, with the same arguments.

One process drives one checkpoint of one step (section 4.3, items 1 to 5):

* ``--step smoke`` drives wt25-u192 on the eight smoke episodes of section 6; only here may an
  invocation stop after three completed episodes (``--stop-after-completed-episodes 3``);
* ``--step main`` drives any one of the 37 checkpoints on the 61 main episodes;
* ``--step confirmation`` drives one checkpoint of the carried list (a carried candidate or the
  comparator of one) on the confirmation suite.

Anything not among the 37, or whose file does not hash as section 3 records, is refused before any
model or data is loaded. Each checkpoint writes into ``<source-artifact-root>/<run id>/<checkpoint
id>`` and nowhere else; inside the container that directory is also ``--output-root``.

Every manifest, every invocation line and every episode report records the commit the code came
from, the preregistration's sha256, the GPU's type, name, UUID and driver, the node, the image's
sha256 and the job id. The kit passes the job id with Slurm's restart count (``<job>-r<n>``), so
the job id and the node name the job start an invocation belongs to. Main and confirmation refuse
a GPU name, GPU type or image that differs from the committed smoke result's, and a
preregistration that is not frozen; every resume refuses a GPU name, type or image, or a loaded
Python file, that differs from the first invocation's.

Resume is per episode, as in the locked runner: a completed episode report with its steps and
timings is the unit. On SIGTERM (preemption, the time limit, a cancellation) the runner exits at
once without a report for the episode in progress, which runs again from its first step; that is
not a failed attempt. An error is; after three failed attempts an episode is recorded as not
completed, and so is every remaining episode of a checkpoint that failed to load three times, the
FP32 check included, but only when the three span at least two job starts (two job ids or two
nodes): one faulty node cannot use up an episode's or a checkpoint's attempts within one job start.

Exit codes: 0 completed; 75 stopped at an episode boundary and resumable (time budget, or the
smoke's stop after three episodes); 1 error after the invocation started (``failure.json``); 2
refused before anything was loaded or written; 128 + signal number when stopped by a signal.

``python -m scripts.run_spark_native_c0_checkpoint_sweep model-check --model-snapshot S --output
R`` hashes the 14 snapshot files once per job and writes a record that every worker of the job
then checks by file size and modification time instead of hashing 17.5 GB again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.run_spark_native_c0_closed_loop import (
    CONTROL_MANIFEST_FILE_SHA256,
    CONTROL_SUITE_FILE_SHA256,
    CONTROL_VERIFICATION_FILE_SHA256,
    _checkpoint_path,
    _checkpoint_summary,
    _control_summary,
    _load_existing_reports,
    _load_json,
    _report_paths,
    _sha256,
    _software_hashes,
    _timestamp,
    _write_json,
    _write_jsonl,
)
from scripts.run_spark_native_c0_closed_loop import GATE_ID as LOCKED_GATE_ID
from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import (
    HABITAT_SIM_COMMIT,
    HabitatSimR2RConfig,
    HabitatSimR2REnvironment,
)
from spatialvlm.data.history import HistoryMode
from spatialvlm.eval.native_checkpoint_allowlist import (
    GPU_DEVICE_ID,
    LOCKED_RUN_ID,
    RUNNER,
    SMOKE_STOP_AFTER_COMPLETED_EPISODES,
    SUITES,
    SweepCheckpoint,
    SweepRefusalError,
    allowlist_record,
    check_run_id,
    check_step_program,
    frozen_checkpoint_for,
    is_sha256,
    refuse,
    resolve_step_checkpoint,
    run_of,
    step_record,
)
from spatialvlm.eval.native_closed_loop import (
    closed_loop_protocol_record,
    run_instrumented_r2r_ce_episode,
)
from spatialvlm.eval.native_policy import NativePolicyConfig, NativeQwenPolicy
from spatialvlm.eval.native_sweep_lock import (
    check_code_is_from_repo,
    check_code_unchanged,
    code_record,
    code_record_sha256,
)
from spatialvlm.eval.qwen_action_scorer import NativeQwenActionScorer
from spatialvlm.eval.suite_binding import verify_suite_against_release

SCHEMA_VERSION = 2
GATE_ID = "native_qwen_c0_checkpoint_sweep_v2"
EXIT_COMPLETED = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_RESUMABLE = 75
MAX_FAILED_ATTEMPTS = 3
# Section 5: the failed attempts that make an episode (or a checkpoint's load) not completed must
# come from at least this many job starts, a job start being one (job id, node) pair.
MIN_JOB_STARTS = 2
EXPECTED_TRAINABLE_TENSORS = 288
EPISODE_MAX_STEPS = 500
RESUME_UNIT = "completed episode report with linked steps and timings"
RUNNER_PATH = "scripts/run_spark_native_c0_checkpoint_sweep.py"
SMOKE_RESULT_PATH = "artifacts/torch/stopblind_smoke/smoke_result.json"
# The preregistration's first line at freeze ("Freeze" section). Before freeze it carries the
# DRAFT line instead; main and confirmation refuse a preregistration that is not frozen.
FROZEN_LINE = "Frozen: the commit that adds this line is the freeze commit."
DRAFT_LINE_PREFIX = "DRAFT. Not frozen."
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GPU_TYPE_RE = re.compile(r"^[a-z0-9_]+$")
# The kit's job id names a job start: <Slurm job id, or array job_task>-r<restart count>.
_JOB_ID_RE = re.compile(r"^[0-9]+(_[0-9]+)?-r[0-9]+$")

# Preregistration section 4.1: the 14 files locked by native_c0_closed_loop_v1, byte-identical.
LOCKED_SOURCE_SHA256: dict[str, str] = {
    "scripts/check_native_c0_closed_loop_artifacts.py": (
        "98ece4cd6188328e9e881669a38a072ac7f88436db28daf74ef8a791c951720e"
    ),
    "scripts/run_spark_native_c0_closed_loop.py": (
        "3100ba820e90e468484875283c6b8a55144a4300e77434a2b62163f5d7ef434d"
    ),
    "scripts/run_spark_native_c0_closed_loop.sh": (
        "a0b58d18434e8662483cf0eafeef8ac0363c949551d8997a27758bdbb34b41d7"
    ),
    "scripts/start_spark_native_c0_closed_loop.sh": (
        "954a970e94eac96df3f531e862983cc2dc738b4153d49bd822d5e8b5277c9e38"
    ),
    "src/spatialvlm/data/habitat_sim_r2r.py": (
        "5f4365c5a3d6f49eee70114c13b83ab66c534a7a78025078d21aa42d63c345a3"
    ),
    "src/spatialvlm/data/history.py": (
        "f7f29f1306b386b08e66ad741ebf68452ec7d86bdc5101162cd4361fe541bdf0"
    ),
    "src/spatialvlm/data/native_qwen.py": (
        "2b221c8856d25c593bd42b5971d18bc76cb766fb1bbe468be9172902e57b1f39"
    ),
    "src/spatialvlm/eval/action_decision.py": (
        "7b526f2948745440cd84aafad1191b7e0cc998f1ca918c051a999e55420ee781"
    ),
    "src/spatialvlm/eval/metrics.py": (
        "63177d682c2b5ff38135863f473615452c056f1ae880dfbbb693f131ea558949"
    ),
    "src/spatialvlm/eval/native_closed_loop.py": (
        "61cbbcaeb15233fd7247b299bc5074c0dedab419702c4dfae2dd28c8b3832e77"
    ),
    "src/spatialvlm/eval/native_policy.py": (
        "6c0b87cdb6f4aad1b27b48453b78c5a76a8b9fa1e6b139e433e065ecc00b73ca"
    ),
    "src/spatialvlm/eval/qwen_action_scorer.py": (
        "905cc4bbf29b3552b092f73028911c434b738aa992c4940a38c6636c107ea88c"
    ),
    "src/spatialvlm/eval/r2r_ce.py": (
        "96ad6b316a1c88c3ca2107754fb9c1c2f982afcc56647a2d096397d3c884de89"
    ),
    "src/spatialvlm/training/native_c0_gate.py": (
        "f813c5d9dfc3ce146e9865978ead20d0c2c35ea783b2bbe93a31a1453e44857b"
    ),
}
CLOSED_LOOP_MANIFEST_SHA256 = "9d8e272d194106b924d4eee7e46e7dc3e478e023020954283773a788368a269a"
RELEASE_EPISODES_SHA256 = "e52a19e08a2f6f936f61b021cf4a0e17ddb8685f8807c60ba6382a61e2463df0"
RELEASE_GROUND_TRUTH_SHA256 = "cf41ac4dc6bcbada202e9cf0c414b0469153a4e1c5d02a21f088edd6032299fc"
CACHE_INDEX_PATH = "artifacts/spark/native_c0_development_cache_v1/cache_index.json"
CACHE_INDEX_SHA256 = "eee53216e52774749d67d7a7d3763084b3622d6c2a8d9c18d834652f0cd304de"
# The Qwen snapshot's weight files as all six training runs recorded them (model_weight_locks in
# each run's result.json; equal in all six).
MODEL_WEIGHT_LOCKS: dict[str, dict[str, int | str]] = {
    "model-00001-of-00004.safetensors": {
        "bytes": 4902275944,
        "sha256": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    },
    "model-00002-of-00004.safetensors": {
        "bytes": 4915962496,
        "sha256": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    },
    "model-00003-of-00004.safetensors": {
        "bytes": 4999831048,
        "sha256": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    },
    "model-00004-of-00004.safetensors": {
        "bytes": 2716270024,
        "sha256": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    },
    "model.safetensors.index.json": {
        "bytes": 67759,
        "sha256": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    },
}
# Section 4.3: all 14 files of the Spark's snapshot of Qwen/Qwen3-VL-8B-Instruct at revision
# 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b, hashed on the workstation on 2026-10-04. The five
# weight files are the ones every training run recorded (MODEL_WEIGHT_LOCKS).
MODEL_SNAPSHOT_FILES: dict[str, dict[str, int | str]] = {
    "chat_template.json": {
        "bytes": 5499,
        "sha256": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
    },
    "config.json": {
        "bytes": 1474,
        "sha256": "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
    },
    "generation_config.json": {
        "bytes": 269,
        "sha256": "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
    },
    "merges.txt": {
        "bytes": 1671839,
        "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    "model-00001-of-00004.safetensors": {
        "bytes": 4902275944,
        "sha256": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    },
    "model-00002-of-00004.safetensors": {
        "bytes": 4915962496,
        "sha256": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    },
    "model-00003-of-00004.safetensors": {
        "bytes": 4999831048,
        "sha256": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    },
    "model-00004-of-00004.safetensors": {
        "bytes": 2716270024,
        "sha256": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    },
    "model.safetensors.index.json": {
        "bytes": 67759,
        "sha256": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    },
    "preprocessor_config.json": {
        "bytes": 390,
        "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    },
    "tokenizer.json": {
        "bytes": 7032403,
        "sha256": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    },
    "tokenizer_config.json": {
        "bytes": 10868,
        "sha256": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    },
    "video_preprocessor_config.json": {
        "bytes": 385,
        "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    },
    "vocab.json": {
        "bytes": 2776833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
# What a fresh output directory may hold before the first manifest: the kit's own files.
LAUNCHER_FILES = frozenset({"run.log", "launcher.pid"})
# Platform fields that may differ between invocations of one checkpoint (a requeued job lands on
# another node of the same GPU type, with a new job id and possibly a later commit whose
# difference from the freeze commit the analysis checks). The rest must not change.
PER_INVOCATION_PLATFORM_FIELDS = frozenset(
    {
        "code_commit",
        "cuda_visible_devices",
        "gpu_driver",
        "gpu_uuid",
        "home",
        "image_ref",
        "job_id",
        "node",
        "thread_environment",
        "torch_num_threads",
    }
)


class _StopRequested(BaseException):
    """Raised from a signal handler so the runner stops between Python bytecodes."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class AdapterDtypeError(RuntimeError):
    """The loaded adapter is not the 288 FP32 tensors that training saved."""


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True, help="The one checkpoint to drive.")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--code-commit", required=True, help="The commit the bundle came from.")
    parser.add_argument("--gpu-type", required=True, help="The study GPU's type, e.g. h200.")
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-sha256", required=True, help="The SIF's sha256, from the host.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--smoke-result", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-verification", type=Path, required=True)
    parser.add_argument("--closed-loop-manifest", type=Path, required=True)
    parser.add_argument("--suite-file", type=Path, default=None)
    parser.add_argument("--carried-list", type=Path, default=None)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-check-record", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=30.0)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--habitat-build-lock", type=Path, default=Path("/opt/gridnav/habitat-sim-commit")
    )
    parser.add_argument("--base-variant-file", type=Path, default=Path("/opt/gridnav/base-variant"))
    parser.add_argument("--stop-after-completed-episodes", type=int, default=None)
    parser.add_argument(
        "--previous-stop-reason",
        default=None,
        help="Why the previous invocation ended without its record, if Slurm ended it.",
    )
    return parser.parse_args(argv)


# ── provenance: preregistration, code, platform ──────────────────────────────────────────────


def preregistration_status(text: str) -> str | None:
    """ "frozen" or "draft" from the status line, the first non-blank line after the title."""
    lines = text.splitlines()
    titles = [index for index, line in enumerate(lines) if line.startswith("# ")]
    if not titles:
        return None
    for line in lines[titles[0] + 1 :]:
        status = line.strip().strip("*").strip()
        if not status:
            continue
        if status == FROZEN_LINE:
            return "frozen"
        if status.startswith(DRAFT_LINE_PREFIX):
            return "draft"
        return None
    return None


def preregistration_record(path: Path) -> dict[str, Any]:
    """The preregistration's sha256, and whether its status line is the freeze line."""
    if not path.is_file():
        raise refuse(f"the preregistration is missing: {path}")
    status = preregistration_status(path.read_text(encoding="utf-8"))
    if status is None:
        raise refuse(
            f"the preregistration {path} must open, under its title, with the DRAFT line or the "
            f"line {FROZEN_LINE!r}."
        )
    return {"path": str(path), "sha256": _sha256(path), "frozen": status == "frozen"}


def check_preregistration_for_step(step: str, record: Mapping[str, Any]) -> None:
    """Section 5: everything after the smoke runs from the freeze commit."""
    if step != "smoke" and not record["frozen"]:
        raise refuse(f"step {step} runs only from the freeze commit; the preregistration is DRAFT.")


def snapshot_file_hashes(snapshot: Path) -> dict[str, dict[str, int | str]]:
    """Size and sha256 of the 14 snapshot files (links followed, as the loader reads them)."""
    hashes: dict[str, dict[str, int | str]] = {}
    for name in MODEL_SNAPSHOT_FILES:
        path = snapshot / name
        if not path.is_file():
            raise refuse(f"the Qwen snapshot is incomplete: {name} is missing.")
        hashes[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path.resolve())}
    return hashes


def _check_snapshot_hashes(observed: Mapping[str, Mapping[str, Any]]) -> None:
    differing = sorted(
        name
        for name in MODEL_SNAPSHOT_FILES
        if dict(observed.get(name, {})) != dict(MODEL_SNAPSHOT_FILES[name])
    )
    if differing:
        raise refuse(f"the Qwen snapshot's files differ from the committed list: {differing}")


def write_model_check_record(snapshot: Path, output: Path) -> dict[str, Any]:
    """Hash the 14 files once, check them, and record their size and modification time."""
    hashes = snapshot_file_hashes(snapshot)
    _check_snapshot_hashes(hashes)
    record = {
        "schema_version": 1,
        "snapshot": str(snapshot.resolve()),
        "hashed_at_utc": _timestamp(),
        "files": {
            name: {
                **hashes[name],
                "stat_size": (snapshot / name).stat().st_size,
                "stat_mtime_ns": (snapshot / name).stat().st_mtime_ns,
            }
            for name in MODEL_SNAPSHOT_FILES
        },
    }
    _write_json(output, record)
    return record


def check_model_snapshot(snapshot: Path, record_path: Path | None = None) -> dict[str, Any]:
    """Section 4.3: every snapshot file against the committed list.

    With a job's model-check record, its hashes must equal the committed list and every file's
    size and modification time must equal the record's; otherwise all 14 files are hashed here.
    """
    if record_path is None:
        _check_snapshot_hashes(snapshot_file_hashes(snapshot))
        return {"method": "hashed by this invocation", "files": len(MODEL_SNAPSHOT_FILES)}
    if not record_path.is_file():
        raise refuse(f"the model-check record is missing: {record_path}")
    record = _load_json(record_path)
    files = record.get("files", {})
    _check_snapshot_hashes(
        {
            name: {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
            for name, item in files.items()
        }
    )
    if record.get("snapshot") != str(snapshot.resolve()):
        raise refuse(f"the model-check record is for {record.get('snapshot')}, not {snapshot}.")
    changed = []
    for name in MODEL_SNAPSHOT_FILES:
        path = snapshot / name
        if not path.is_file():
            changed.append(name)
            continue
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            files[name]["stat_size"],
            files[name]["stat_mtime_ns"],
        ):
            changed.append(name)
    if changed:
        raise refuse(f"snapshot files changed since the job's model check: {changed}")
    return {
        "method": "the job's model-check record, files unchanged since",
        "record_sha256": _sha256(record_path),
        "hashed_at_utc": record.get("hashed_at_utc"),
        "files": len(MODEL_SNAPSHOT_FILES),
    }


def gpu_identity(device: str) -> dict[str, Any]:
    """The GPU's name, UUID and driver as nvidia-smi reports them, and torch's device name."""
    if torch.device(device).type != "cuda":
        raise refuse("the sweep runs only on a GPU (--device cuda).")
    # Slurm names the job's one GPU in CUDA_VISIBLE_DEVICES; ask nvidia-smi about that GPU, so a
    # node that does not hide its other GPUs still gives one row.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    select = [f"--id={visible}"] if visible and "," not in visible else []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                *select,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise refuse(f"nvidia-smi did not answer: {error}") from error
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise refuse(f"this job must see exactly one GPU; nvidia-smi lists {len(rows)}.")
    name, uuid, driver = (part.strip() for part in rows[0].split(","))
    return {
        "gpu_name": name,
        "gpu_uuid": uuid,
        "gpu_driver": driver,
        "torch_device_name": torch.cuda.get_device_name(0),
    }


def platform_record(args: argparse.Namespace, gpu: Mapping[str, Any]) -> dict[str, Any]:
    """What this invocation ran on (section 4.3, item 2)."""
    if not _COMMIT_RE.fullmatch(args.code_commit):
        raise refuse(f"--code-commit must be a 40-character commit, not {args.code_commit!r}.")
    if not is_sha256(args.image_sha256):
        raise refuse("--image-sha256 must be the image's sha256 (64 lowercase hex digits).")
    if not _GPU_TYPE_RE.fullmatch(args.gpu_type):
        raise refuse(f"--gpu-type is not a type name: {args.gpu_type!r}")
    if not _JOB_ID_RE.fullmatch(args.job_id):
        raise refuse(
            f"--job-id must name a job start, <Slurm job id>-r<restart count>, not {args.job_id!r}."
        )
    return {
        "code_commit": args.code_commit,
        "gpu_type": args.gpu_type,
        **dict(gpu),
        "node": platform.node(),
        "architecture": platform.machine().lower(),
        "image_ref": args.image_ref,
        "image_sha256": args.image_sha256,
        "job_id": args.job_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_num_threads": torch.get_num_threads(),
        "home": os.environ.get("HOME"),
    }


def check_smoke_result(path: Path, platform_now: Mapping[str, Any]) -> dict[str, Any]:
    """Section 4.3, item 2: the GPU and image must be the committed smoke result's.

    The kit passes the copy the bundle carries; the analysis checks that every job recorded the
    sha256 of the smoke result the freeze commit holds.
    """
    if not path.is_file():
        raise refuse(f"the committed smoke result is missing: {path}")
    smoke = _load_json(path)
    if smoke.get("verdict") != "SMOKE PASS":
        raise refuse(f"the smoke result {path} is not a pass: {smoke.get('verdict')!r}")
    differing = {
        key: (smoke.get(key), platform_now[key])
        for key in ("gpu_type", "gpu_name", "image_sha256")
        if smoke.get(key) != platform_now[key]
    }
    if differing:
        raise refuse(f"this invocation differs from the committed smoke result in {differing}.")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "gpu_type": smoke.get("gpu_type"),
        "gpu_name": smoke.get("gpu_name"),
        "image_sha256": smoke.get("image_sha256"),
        "code_commit": smoke.get("code_commit"),
        "workers_per_gpu": smoke.get("workers_per_gpu"),
    }


def memory_record() -> dict[str, Any]:
    """Peak GPU memory as torch counts it, and the process's peak host memory. The pid lets the
    smoke match this process to nvidia-smi's per-process memory (Apptainer shares the host's pid
    namespace)."""
    cuda = torch.cuda.is_available()
    return {
        "pid": os.getpid(),
        "torch_max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "torch_max_memory_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


# ── preflight: every refusal happens here, before data, a model or an output file ────────────


@dataclass
class Preflight:
    """Everything checked before data or a model is loaded."""

    step: str
    run_id: str
    suite_id: str
    record: SweepCheckpoint
    preregistration: dict[str, Any]
    platform: dict[str, Any]
    smoke_result: dict[str, Any] | None
    code_files: dict[str, str]
    locked_software_sha256: dict[str, str]
    model_check: dict[str, Any]
    suite_path: Path
    binding_path: Path
    suite_sha256: str
    suite_locks: dict[str, Any]
    entries: list[dict[str, Any]]
    carried: dict[str, Any] | None
    checkpoint_lock: dict[str, Any]
    habitat_build_commit: str
    existing_manifest: dict[str, Any] | None = None

    @property
    def scope(self) -> tuple[SweepCheckpoint, ...]:
        return (self.record,)


def _refuse_missing(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            raise refuse(f"required input is missing: {path}")


def check_output_root(
    output_root: Path,
    source_artifact_root: Path,
    run_id: str,
    checkpoint_id: str,
    *,
    locked_gate: str | None = None,
) -> dict[str, Any] | None:
    """Refuse any output directory but ``<source-artifact-root>/<run id>/<checkpoint id>``.

    Inside the container the output directory and that path are bind mounts of one host
    directory, so they are compared with ``os.path.samefile``. The directory must already exist:
    the kit creates it. A directory with no manifest may hold only the kit's own files. Returns
    the existing manifest, if any.
    """
    resolved = output_root.resolve()
    locked_dir = (source_artifact_root / LOCKED_RUN_ID).resolve()
    if LOCKED_RUN_ID in resolved.parts or resolved.is_relative_to(locked_dir):
        raise refuse(f"the output directory {resolved} belongs to the locked closed loop.")
    expected = source_artifact_root / run_id / checkpoint_id
    if not output_root.is_dir():
        raise refuse(
            f"the output directory {output_root} does not exist; the kit creates it as "
            f"<source-artifact-root>/<run id>/<checkpoint id>."
        )
    if not expected.is_dir() or not os.path.samefile(output_root, expected):
        raise refuse(
            f"the output directory {output_root} is not {expected}: each checkpoint of a run "
            "writes into <source-artifact-root>/<run id>/<checkpoint id>."
        )
    manifest_path = output_root / "manifest.json"
    existing = _load_json(manifest_path) if manifest_path.is_file() else None
    if existing is None:
        leftover = sorted(
            path.name for path in output_root.iterdir() if path.name not in LAUNCHER_FILES
        )
        if leftover:
            raise refuse(f"{output_root} has no manifest but holds {leftover}.")
    else:
        if existing.get("run_id") == LOCKED_RUN_ID or (
            locked_gate is not None and existing.get("gate") == locked_gate
        ):
            raise refuse(f"{manifest_path} is the locked closed loop's manifest.")
        recorded = (existing.get("run_id"), existing.get("checkpoint_id"))
        if recorded != (run_id, checkpoint_id):
            raise refuse(
                f"{manifest_path} records {recorded}; this invocation is {(run_id, checkpoint_id)}."
            )
    result_path = output_root / "result.json"
    if result_path.is_file() and _load_json(result_path).get("status") == "completed":
        raise refuse(f"{output_root} already holds a completed result; a checkpoint runs once.")
    return existing


def _check_runtime(args: argparse.Namespace) -> str:
    """The Habitat build, the image's base and the model snapshot."""
    if not args.habitat_build_lock.is_file():
        raise refuse(f"the Habitat-Sim build lock is missing: {args.habitat_build_lock}")
    build_commit = args.habitat_build_lock.read_text(encoding="utf-8").strip()
    if build_commit != HABITAT_SIM_COMMIT:
        raise refuse(f"Habitat-Sim build-lock mismatch: {build_commit!r}.")
    if not args.base_variant_file.is_file():
        raise refuse(f"the image's base-variant record is missing: {args.base_variant_file}")
    variant = args.base_variant_file.read_text(encoding="utf-8").strip()
    if variant != "primary":
        raise refuse(
            f"the image is built on the {variant!r} base; section 4.3 allows only primary."
        )
    return build_commit


def _main_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Repeat the locked runner's control checks (its lines 380-404) and pin the 61 entries."""
    control_manifest_path = args.control_root / "suite_manifest.json"
    control_suite_path = args.control_root / "suite.json"
    _refuse_missing(
        control_manifest_path,
        control_suite_path,
        args.control_verification,
        args.closed_loop_manifest,
    )
    manifest_sha256 = _sha256(control_manifest_path)
    if manifest_sha256 != CONTROL_MANIFEST_FILE_SHA256:
        raise refuse(f"control manifest sha256 {manifest_sha256} is not the main suite's.")
    if _sha256(control_suite_path) != CONTROL_SUITE_FILE_SHA256:
        raise refuse("control suite bytes differ from the independently verified input.")
    if _sha256(args.control_verification) != CONTROL_VERIFICATION_FILE_SHA256:
        raise refuse("independent control verification bytes differ from the frozen input.")
    control_manifest = _load_json(control_manifest_path)
    control_suite = _load_json(control_suite_path)
    control_verification = _load_json(args.control_verification)
    if control_suite.get("status") != "passed":
        raise refuse("the control suite is not independently admitted as passed.")
    control_checks = control_verification.get("checks", {})
    if (
        control_verification.get("status") != "passed"
        or control_checks.get("errors") != []
        or control_checks.get("episode_reports_verified") != 854
        or control_checks.get("step_trace_hashes_verified") != 854
        or control_verification.get("source_sha256", {}).get("suite.json")
        != CONTROL_SUITE_FILE_SHA256
        or control_verification.get("source_sha256", {}).get("suite_manifest.json")
        != CONTROL_MANIFEST_FILE_SHA256
    ):
        raise refuse("the independent policy-control verification is not fully admitted.")
    entries = control_manifest["episodes"][: SUITES["main"].episode_count]
    closed_loop_sha256 = _sha256(args.closed_loop_manifest)
    if closed_loop_sha256 != CLOSED_LOOP_MANIFEST_SHA256:
        raise refuse("the native_c0_closed_loop_v1 manifest differs from the recorded one.")
    closed_loop = _load_json(args.closed_loop_manifest)
    if len(entries) != 61 or closed_loop.get("episodes") != entries:
        raise refuse("the 61 main entries differ from native_c0_closed_loop_v1's episodes.")
    return {
        "suite_path": control_manifest_path,
        "binding_path": args.closed_loop_manifest,
        "suite_sha256": manifest_sha256,
        "entries": entries,
        "locks": {
            "control_manifest_sha256": manifest_sha256,
            "control_suite_sha256": _sha256(control_suite_path),
            "control_verification_sha256": _sha256(args.control_verification),
            "closed_loop_manifest_sha256": closed_loop_sha256,
        },
    }


def _json_values(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        return [item for value in payload.values() for item in _json_values(value)]
    if isinstance(payload, list):
        return [item for value in payload for item in _json_values(value)]
    return [payload]


def _suite_file(args: argparse.Namespace, suite_id: str) -> dict[str, Any]:
    """The smoke list or the confirmation suite: the committed file, pinned by its sha256."""
    if args.suite_file is None:
        raise refuse(f"the {suite_id} suite needs --suite-file.")
    _refuse_missing(args.suite_file)
    observed = _sha256(args.suite_file)
    if observed != SUITES[suite_id].sha256:
        others = [key for key, suite in SUITES.items() if suite.sha256 == observed]
        if others:
            raise refuse(f"the {others[0]!r} suite was given where this step needs {suite_id!r}.")
        raise refuse(f"suite file sha256 {observed} is not the {suite_id} suite's.")
    suite = _load_json(args.suite_file)
    entries = suite.get("episodes")
    if suite.get("split") != "val_seen" or suite.get("suite_id") != suite_id:
        raise refuse(f"the {suite_id} suite does not declare itself and split val_seen.")
    values = set(map(str, _json_values(suite)))
    if not {RELEASE_EPISODES_SHA256, RELEASE_GROUND_TRUTH_SHA256} <= values:
        raise refuse(f"the {suite_id} suite does not name both release files' sha256.")
    required = {"index", "episode_id", "trajectory_id", "scene_id", "artifact_stem"}
    if (
        not isinstance(entries, list)
        or len(entries) != SUITES[suite_id].episode_count
        or not all(isinstance(entry, dict) and required <= set(entry) for entry in entries)
        or [entry["index"] for entry in entries] != list(range(len(entries)))
        or len({str(entry["artifact_stem"]) for entry in entries}) != len(entries)
    ):
        raise refuse(f"the {suite_id} suite does not hold well-formed entries.")
    return {
        "suite_path": args.suite_file,
        "binding_path": args.suite_file,
        "suite_sha256": observed,
        "entries": entries,
        "locks": {f"{suite_id}_suite_sha256": observed},
    }


def verify_checkpoint_source(
    source_root: Path, archive: Path, record: SweepCheckpoint
) -> dict[str, Any]:
    """The locked runner's 12 checks, read from the run's ``checkpoints[]``, plus more.

    Checks 3-6 read the run's ``checkpoints[]`` entry for the listed update instead of
    ``selection.checkpoint``; check 6 requires the listed trainable-state hash and check 12 the
    listed run-manifest hash. Added: the entry's status is ``candidate`` and its ``bytes`` equal
    the file size, and the run's ``result.json`` and ``verification.json`` hash as
    ``native_c0_closed_loop_v1`` recorded them. For the six selected checkpoints: the run's
    ``selected_update``, agreement of ``selection.checkpoint`` with the entry, and field-by-field
    equality with ``FROZEN_CHECKPOINTS``. The file's sha256 is computed here, before any model
    load.
    """
    run_root = source_root / record.source_run_id
    result_path = run_root / "result.json"
    verification_path = run_root / "verification.json"
    checkpoint_path = _checkpoint_path(archive, record)
    _refuse_missing(result_path, verification_path, checkpoint_path)
    result = _load_json(result_path)
    verification = _load_json(verification_path)
    listed = [
        entry
        for entry in result.get("checkpoints", [])
        if isinstance(entry, Mapping) and entry.get("update_step") == record.update_step
    ]
    entry: Mapping[str, Any] = listed[0] if len(listed) == 1 else {}
    checkpoint_sha256 = _sha256(checkpoint_path)
    result_sha256 = _sha256(result_path)
    verification_sha256 = _sha256(verification_path)
    run = run_of(record)
    checks = {
        "run_id": result.get("run_id") == record.source_run_id,
        "checkpoint_gate": result.get("gate") == record.checkpoint_gate,
        "one_entry_for_update": len(listed) == 1,
        "relative_path": entry.get("relative_path") == record.checkpoint_relative_path,
        "recorded_hash": entry.get("sha256") == record.checkpoint_sha256,
        "recorded_state_hash": entry.get("trainable_state_sha256") == record.trainable_state_sha256,
        "actual_hash": checkpoint_sha256 == record.checkpoint_sha256,
        "verification_passed": verification.get("status") == "passed",
        "verification_errors": verification.get("error_count") == 0,
        "verification_link": verification.get("result_sha256") == result_sha256,
        "manifest_link": verification.get("manifest_sha256") == result.get("manifest_sha256"),
        "manifest_hash": is_sha256(result.get("manifest_sha256"))
        and result.get("manifest_sha256") == record.source_manifest_sha256,
        "entry_status": entry.get("status") == "candidate",
        "entry_bytes": entry.get("bytes") == checkpoint_path.stat().st_size,
        "source_result_pinned": result_sha256 == run.source_result_sha256,
        "source_verification_pinned": verification_sha256 == run.source_verification_sha256,
    }
    selection = result.get("selection", {})
    selected = selection.get("checkpoint", {})
    is_selected_update = selection.get("selected_update") == record.update_step
    checks["role_matches_selection"] = is_selected_update == (record.role == "selected")
    if record.role == "selected":
        checks["selected_update"] = is_selected_update
        checks["selection_path"] = selected.get("relative_path") == entry.get("relative_path")
        checks["selection_hash"] = selected.get("sha256") == entry.get("sha256")
        checks["selection_state_hash"] = selected.get("trainable_state_sha256") == entry.get(
            "trainable_state_sha256"
        )
        checks["frozen_checkpoint_record"] = record.as_frozen_checkpoint() == frozen_checkpoint_for(
            record
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise refuse(f"checkpoint verification failed for {record.checkpoint_id}: {failed}")
    return {
        "checkpoint_id": record.checkpoint_id,
        "role": record.role,
        "legacy_checkpoint_id": record.legacy_checkpoint_id,
        "is_selected_update": is_selected_update,
        "archive_relative_path": record.archive_relative_path,
        "checkpoint_bytes": entry.get("bytes"),
        "source_result_sha256": result_sha256,
        "source_verification_sha256": verification_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_gate": record.checkpoint_gate,
        "checkpoint_update_step": record.update_step,
        "checkpoint_trainable_state_sha256": entry.get("trainable_state_sha256"),
        "source_manifest_sha256": result.get("manifest_sha256"),
        "checks": checks,
    }


def loader_arguments(lock: Mapping[str, Any], device: str) -> dict[str, Any]:
    """The keyword arguments the locked runner passes the loader (its lines 493-512)."""
    return {
        "device": torch.device(device),
        "expected_checkpoint_sha256": lock["checkpoint_sha256"],
        "expected_checkpoint_gate": lock["checkpoint_gate"],
        "expected_manifest_sha256": lock["source_manifest_sha256"],
        "expected_trainable_state_sha256": lock["checkpoint_trainable_state_sha256"],
        "expected_update_step": int(lock["checkpoint_update_step"]),
        "local_files_only": True,
    }


def adapter_dtype_record(scorer: Any) -> dict[str, Any]:
    """Record the loaded adapter's dtypes and require all 288 trainable tensors to be FP32."""
    named = [
        (name, parameter)
        for name, parameter in scorer.backbone.named_parameters()
        if parameter.requires_grad
    ]
    dtypes = Counter(str(parameter.dtype) for _, parameter in named)
    record = {
        "trainable_tensor_count": len(named),
        "dtypes": dict(sorted(dtypes.items())),
        "all_names_are_lora": all("lora_" in name for name, _ in named),
    }
    if len(named) != EXPECTED_TRAINABLE_TENSORS or set(dtypes) != {str(torch.float32)}:
        raise AdapterDtypeError(
            f"the loaded adapter is not {EXPECTED_TRAINABLE_TENSORS} FP32 tensors: {record}"
        )
    return record


def check_forced_stop(step: str, value: int | None, completed_at_start: int) -> None:
    """Section 6: only the smoke may stop after three completed episodes, and only before them."""
    if value is None:
        return
    if step != "smoke":
        raise refuse("--stop-after-completed-episodes is allowed only in the smoke.")
    if value != SMOKE_STOP_AFTER_COMPLETED_EPISODES:
        raise refuse(
            f"the smoke stops after {SMOKE_STOP_AFTER_COMPLETED_EPISODES} completed episodes, "
            f"not {value}."
        )
    if completed_at_start >= value:
        raise refuse(
            f"--stop-after-completed-episodes {value} is refused: {completed_at_start} episodes "
            "are completed, so the smoke resumes without it."
        )


def _same_run(existing: Mapping[str, Any], platform_now: Mapping[str, Any]) -> None:
    """Section 4.2: a resume refuses a GPU name or type, or an image, that differs."""
    recorded = existing.get("platform", {})
    differing = {
        key: (recorded.get(key), platform_now[key])
        for key in ("gpu_type", "gpu_name", "image_sha256")
        if recorded.get(key) != platform_now[key]
    }
    if differing:
        raise refuse(f"this invocation differs from the run's manifest in {differing}.")


def preflight(args: argparse.Namespace) -> Preflight:
    """Run every check that must pass before data, a model or an output file is touched."""
    step = check_step_program(args.step, RUNNER)
    check_run_id(args.step, args.run_id)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0.0:
        raise refuse("--max-hours must be finite and positive.")
    if args.stop_after_completed_episodes is not None and args.step != "smoke":
        raise refuse("--stop-after-completed-episodes is allowed only in the smoke.")
    if args.gpu_device_id != GPU_DEVICE_ID:
        raise refuse(f"the simulator runs on GPU device {GPU_DEVICE_ID} (section 4.1).")
    if args.step != "confirmation" and args.carried_list is not None:
        raise refuse(f"step {args.step} does not take a carried list.")
    if args.step == "main" and args.suite_file is not None:
        raise refuse("the main suite is read from the control manifest, not from --suite-file.")
    if args.step == "smoke" and args.smoke_result is not None:
        raise refuse("the smoke does not read a smoke result.")
    existing_manifest = check_output_root(
        args.output_root,
        args.source_artifact_root,
        args.run_id,
        args.checkpoint_id,
        locked_gate=LOCKED_GATE_ID,
    )
    if existing_manifest is not None and existing_manifest.get("step") != args.step:
        raise refuse(
            f"{args.output_root / 'manifest.json'} records step {existing_manifest.get('step')!r};"
            f" this invocation is {args.step!r}."
        )
    preregistration = preregistration_record(args.preregistration)
    check_preregistration_for_step(args.step, preregistration)
    build_commit = _check_runtime(args)
    model_check = check_model_snapshot(args.model_snapshot, args.model_check_record)
    platform_now = platform_record(args, gpu_identity(args.device))
    if existing_manifest is not None:
        _same_run(existing_manifest, platform_now)
    repo = args.repo.resolve()
    smoke_result = None
    if args.step != "smoke":
        smoke_path = args.smoke_result or repo / SMOKE_RESULT_PATH
        smoke_result = check_smoke_result(smoke_path, platform_now)
    code_files = code_record(repo, entry_file=__file__, expected_entry=RUNNER_PATH)
    if existing_manifest is not None:
        check_code_unchanged(existing_manifest.get("code", {}).get("files", {}), code_files)
    locked_software = _software_hashes(repo)
    if locked_software != LOCKED_SOURCE_SHA256:
        changed = sorted(
            k for k in LOCKED_SOURCE_SHA256 if locked_software.get(k) != LOCKED_SOURCE_SHA256[k]
        )
        raise refuse(f"locked source files differ from native_c0_closed_loop_v1: {changed}")

    carried: dict[str, Any] | None = None
    if args.step == "confirmation":
        if args.carried_list is None:
            raise refuse("the confirmation step requires the carried list.")
        _refuse_missing(args.carried_list)
        carried = {
            "path": str(args.carried_list),
            "sha256": _sha256(args.carried_list),
            "content": _load_json(args.carried_list),
        }
    record = resolve_step_checkpoint(
        args.step, args.checkpoint_id, carried=carried["content"] if carried else None
    )
    suite_id = str(step.suite_id)
    suite = _main_suite(args) if suite_id == "main" else _suite_file(args, suite_id)
    _refuse_missing(args.episodes, args.ground_truth)
    if _sha256(args.episodes) != RELEASE_EPISODES_SHA256:
        raise refuse(f"the episodes file is not the val_seen release: {args.episodes}")
    if _sha256(args.ground_truth) != RELEASE_GROUND_TRUTH_SHA256:
        raise refuse(f"the ground-truth file is not the val_seen release: {args.ground_truth}")
    checkpoint_lock = verify_checkpoint_source(
        args.source_artifact_root, args.checkpoint_archive, record
    )
    return Preflight(
        step=args.step,
        run_id=args.run_id,
        suite_id=suite_id,
        record=record,
        preregistration=preregistration,
        platform=platform_now,
        smoke_result=smoke_result,
        code_files=code_files,
        locked_software_sha256=locked_software,
        model_check=model_check,
        suite_path=suite["suite_path"],
        binding_path=suite["binding_path"],
        suite_sha256=suite["suite_sha256"],
        suite_locks=suite["locks"],
        entries=suite["entries"],
        carried=carried,
        checkpoint_lock=checkpoint_lock,
        habitat_build_commit=build_commit,
        existing_manifest=existing_manifest,
    )


def bind_suite(binding_path: Path, dataset: Any) -> dict[str, Any]:
    """Bind the suite to the loaded release with the unchanged ``verify_suite_against_release``."""
    try:
        binding = verify_suite_against_release(binding_path, dataset, loaded_split="val_seen")
    except SystemExit as exc:  # the binding module stops with SystemExit on a mismatch
        raise refuse(f"suite binding failed: {exc}") from exc
    return binding.to_dict()


def build_manifest(
    args: argparse.Namespace, pre: Preflight, binding: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return one checkpoint's manifest as JSON-normalised data."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "gate": GATE_ID,
        "run_id": pre.run_id,
        "step": pre.step,
        "suite_id": pre.suite_id,
        "checkpoint_id": pre.record.checkpoint_id,
        "recorded_at_utc": _timestamp(),
        "step_record": step_record(pre.step).to_dict(),
        "scope": [pre.record.to_dict()],
        "allowlist": allowlist_record(),
        "preregistration": pre.preregistration,
        "code": {"commit": pre.platform["code_commit"], "files": pre.code_files},
        "platform": pre.platform,
        "smoke_result": pre.smoke_result,
        "carried_list": (
            {"path": pre.carried["path"], "sha256": pre.carried["sha256"]}
            if pre.carried is not None
            else None
        ),
        "software_sha256": pre.locked_software_sha256,
        "protocol": closed_loop_protocol_record(),
        "sweep_protocol": {
            "history_mode": HistoryMode.H4_SHORT_WINDOW.value,
            "policy_config": NativePolicyConfig.for_mode(
                HistoryMode.H4_SHORT_WINDOW, chunk_length=1, step_cap=EPISODE_MAX_STEPS
            ).to_record(),
            "max_steps": EPISODE_MAX_STEPS,
            "split": "val_seen",
            "environment_config": HabitatSimR2RConfig(
                scene_root=args.scene_root, gpu_device_id=GPU_DEVICE_ID
            ).to_dict(),
            "habitat_sim_build_commit": pre.habitat_build_commit,
            "model_weight_locks": MODEL_WEIGHT_LOCKS,
            "model_snapshot_files": MODEL_SNAPSHOT_FILES,
            "one_environment_per_invocation": True,
            "model_loaded_once_per_invocation": True,
            "fresh_policy_per_episode": True,
        },
        "suite": {
            "suite_id": pre.suite_id,
            "path": str(pre.suite_path),
            "sha256": pre.suite_sha256,
            "locks": pre.suite_locks,
        },
        "suite_binding": dict(binding) if binding is not None else None,
        "release": {
            "episodes_path": str(args.episodes),
            "episodes_sha256": RELEASE_EPISODES_SHA256,
            "ground_truth_path": str(args.ground_truth),
            "ground_truth_sha256": RELEASE_GROUND_TRUTH_SHA256,
        },
        "episodes": pre.entries,
        "checkpoint_lock": pre.checkpoint_lock,
        "max_hours_per_invocation": args.max_hours,
        "resume_unit": RESUME_UNIT,
        "outcome_used_for_scope_selection": False,
    }
    return json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))


def _comparable(manifest: Mapping[str, Any], ignore_binding: bool) -> dict[str, Any]:
    payload = dict(manifest)
    for key in ("recorded_at_utc", "max_hours_per_invocation", "code"):
        payload.pop(key, None)
    if ignore_binding:
        payload.pop("suite_binding", None)
    payload["platform"] = {
        key: value
        for key, value in dict(payload.get("platform", {})).items()
        if key not in PER_INVOCATION_PLATFORM_FIELDS
    }
    return payload


def compare_manifest(
    existing: Mapping[str, Any], current: Mapping[str, Any], *, ignore_binding: bool
) -> None:
    """Refuse unless the manifests agree in everything but the per-invocation fields.

    Per invocation: the time, ``--max-hours``, the platform fields a requeue may change (node,
    GPU UUID and driver, job id, commit, thread settings) and the code record, which
    ``check_code_unchanged`` compares file by file.
    """
    old, new = _comparable(existing, ignore_binding), _comparable(current, ignore_binding)
    if old != new:
        differing = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        raise refuse(f"the output directory holds a manifest that differs in {differing}.")


# ── resume bookkeeping ────────────────────────────────────────────────────────────────────────


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


def _pair_key(pair: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not pair:
        return None
    return str(pair["checkpoint_id"]), str(pair["episode_id"])


def _counts_as_failure(line: Mapping[str, Any]) -> bool:
    """An exit that was neither clean nor resumable, and not an interruption."""
    return line.get("exit_code") not in (EXIT_COMPLETED, EXIT_RESUMABLE) and not line.get(
        "stop_by_hand"
    )


def failed_attempts(invocations: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list]:
    """Count one failed attempt per pair in progress at an exit that was neither clean nor 75.

    An interruption (a caught signal, or a lost invocation whose reason was logged because Slurm
    ended it) is not a failed attempt: the pair runs again from its first step without penalty.
    """
    failed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in invocations:
        key = _pair_key(line.get("pair_in_progress_at_exit"))
        if key is None or not _counts_as_failure(line):
            continue
        failed.setdefault(key, []).append(dict(line))
    return failed


def failed_loads(invocations: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """Count one failed load per checkpoint that was loading at a failed exit."""
    failed: dict[str, list[dict[str, Any]]] = {}
    for line in invocations:
        checkpoint = line.get("checkpoint_loading_at_exit")
        if not checkpoint or not _counts_as_failure(line):
            continue
        failed.setdefault(str(checkpoint), []).append(dict(line))
    return failed


def job_starts(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """The distinct job starts, (job id, node), that a list of invocation lines came from.

    The kit's job id carries Slurm's restart count, so a requeued start is another job start even
    on the same node. A line without provenance counts as a start of its own.
    """
    starts = set()
    for index, line in enumerate(records):
        provenance = line.get("provenance") or {}
        job, node = provenance.get("job_id"), provenance.get("node")
        starts.add((str(job), str(node)) if job or node else (f"unknown-{index}", "unknown"))
    return sorted(starts)


def attempts_exhausted(records: Sequence[Mapping[str, Any]]) -> bool:
    """Section 5: three failed attempts (or loads), over at least two job starts."""
    return len(records) >= MAX_FAILED_ATTEMPTS and len(job_starts(records)) >= MIN_JOB_STARTS


class InvocationLog:
    """Owns ``progress.json``, ``invocations.jsonl``, ``not_completed.jsonl`` and failures."""

    def __init__(
        self,
        output_root: Path,
        step: str,
        run_id: str,
        target: int,
        *,
        provenance: Mapping[str, Any],
    ) -> None:
        self.output_root = output_root
        self.step = step
        self.run_id = run_id
        self.target = target
        self.provenance = dict(provenance)
        self.invocations_path = output_root / "invocations.jsonl"
        self.not_completed_path = output_root / "not_completed.jsonl"
        self.progress_path = output_root / "progress.json"
        self.index = 0
        self.started_at_utc = _timestamp()
        self.completed_at_start = 0
        self.previous_stop_reason: str | None = None
        self.reconstructed: dict[str, Any] | None = None
        self.model_loads: list[dict[str, Any]] = []

    def start(self, completed: int, previous_stop_reason: str | None) -> None:
        """Log a previous invocation that ended without a record, then open this one."""
        lines = _read_jsonl(self.invocations_path)
        recorded = {int(line["invocation_index"]) for line in lines}
        progress = _load_json(self.progress_path) if self.progress_path.is_file() else {}
        lost_index = progress.get("invocation_index")
        status = progress.get("status")
        if (
            status in {"running", "running_episode", "loading_checkpoint"}
            and isinstance(lost_index, int)
            and lost_index not in recorded
        ):
            pair = progress.get("current_pair") if status == "running_episode" else None
            loading = progress.get("current_checkpoint") if status == "loading_checkpoint" else None
            self.reconstructed = {
                "schema_version": SCHEMA_VERSION,
                "invocation_index": lost_index,
                "step": self.step,
                "run_id": self.run_id,
                "started_at_utc": progress.get("invocation_started_at_utc"),
                "ended_at_utc": None,
                "exit_code": None,
                "outcome": "ended_without_record",
                "completed_pairs_at_start": progress.get("completed_at_invocation_start"),
                "completed_pairs_at_end": completed,
                "pair_in_progress_at_exit": pair,
                "checkpoint_loading_at_exit": loading,
                "stop_by_hand": previous_stop_reason is not None,
                "stop_reason": previous_stop_reason,
                "provenance": progress.get("provenance"),
                "reconstructed_by_invocation": lost_index + 1,
            }
            _append_jsonl(self.invocations_path, self.reconstructed)
            recorded.add(lost_index)
            previous_stop_reason = None
        failure_path = self.output_root / "failure.json"
        self.index = max(recorded, default=0) + 1
        if failure_path.is_file():
            archive = self.output_root / "failures" / f"before_invocation_{self.index:04d}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            failure_path.replace(archive)
        self.completed_at_start = completed
        self.previous_stop_reason = previous_stop_reason
        self.started_at_utc = _timestamp()

    def invocations(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.invocations_path)

    def not_completed(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row["checkpoint_id"]), str(row["episode_id"])): row
            for row in _read_jsonl(self.not_completed_path)
        }

    def update_not_completed(
        self,
        stems: Mapping[tuple[str, str], str],
        suite_id: str,
        completed: set[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Move every pair with three failed attempts, and every remaining pair of a checkpoint
        with three failed loads, into ``not_completed.jsonl``, once; in both cases the three must
        span at least two job starts (``attempts_exhausted``)."""
        existing = self.not_completed()
        invocations = self.invocations()

        def add(key: tuple[str, str], reason: str, field: str, records: list) -> None:
            row = {
                "schema_version": SCHEMA_VERSION,
                "step": self.step,
                "run_id": self.run_id,
                "suite_id": suite_id,
                "checkpoint_id": key[0],
                "episode_id": key[1],
                "artifact_stem": stems[key],
                "recorded_at_utc": _timestamp(),
                "reason": reason,
                field: records,
                "job_starts": [list(start) for start in job_starts(records)],
                "counted_as": "against the claim at stake in every test (section 5)",
            }
            _append_jsonl(self.not_completed_path, row)
            existing[key] = row

        for key, records in failed_attempts(invocations).items():
            if not attempts_exhausted(records) or key in existing or key not in stems:
                continue
            if key in completed:
                continue
            add(key, "three failed attempts of this pair", "failed_attempts", records)
        for checkpoint_id, records in failed_loads(invocations).items():
            if not attempts_exhausted(records):
                continue
            for key in stems:
                if key[0] == checkpoint_id and key not in existing and key not in completed:
                    add(key, "three failed loads of this checkpoint", "failed_loads", records)
        return existing

    def progress(
        self,
        status: str,
        *,
        completed: int,
        not_completed: int,
        checkpoint: str | None = None,
        pair: Mapping[str, Any] | None = None,
    ) -> None:
        _write_json(
            self.progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "recorded_at_utc": _timestamp(),
                "step": self.step,
                "run_id": self.run_id,
                "invocation_index": self.index,
                "invocation_started_at_utc": self.started_at_utc,
                "completed_at_invocation_start": self.completed_at_start,
                "completed_episode_runs": completed,
                "not_completed_episode_runs": not_completed,
                "target_episode_runs": self.target,
                "current_checkpoint": checkpoint,
                "current_episode": None if pair is None else pair["episode_id"],
                "current_pair": None if pair is None else dict(pair),
                "provenance": self.provenance,
            },
        )

    def finish(
        self,
        *,
        exit_code: int,
        outcome: str,
        completed: int,
        pair: Mapping[str, Any] | None,
        checkpoint_loading: str | None = None,
        stop_by_hand: bool = False,
        stop_reason: str | None = None,
        error: BaseException | None = None,
        forced_stop: int | None = None,
    ) -> None:
        _append_jsonl(
            self.invocations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "invocation_index": self.index,
                "step": self.step,
                "run_id": self.run_id,
                "started_at_utc": self.started_at_utc,
                "ended_at_utc": _timestamp(),
                "exit_code": exit_code,
                "outcome": outcome,
                "completed_pairs_at_start": self.completed_at_start,
                "completed_pairs_at_end": completed,
                "pair_in_progress_at_exit": None if pair is None else dict(pair),
                "checkpoint_loading_at_exit": checkpoint_loading,
                "stop_by_hand": stop_by_hand,
                "stop_reason": stop_reason,
                "previous_stop_reason": self.previous_stop_reason,
                "error_type": None if error is None else type(error).__name__,
                "error": None if error is None else str(error),
                "forced_stop_after_completed_episodes": forced_stop,
                "provenance": self.provenance,
                "memory": memory_record(),
                "model_loads": self.model_loads,
            },
        )


# ── the run ───────────────────────────────────────────────────────────────────────────────────


def _raise_stop(signum: int, _frame: Any) -> None:
    # Slurm signals every process of the step, and Apptainer's starter may forward a second copy.
    # From the first signal on, SIGTERM and SIGINT are ignored, so a second one cannot cut short
    # the record this stop writes; the handlers are restored when the invocation ends.
    for other in (signal.SIGTERM, signal.SIGINT):
        signal.signal(other, signal.SIG_IGN)
    raise _StopRequested(signum)


def _report(
    *,
    pre: Preflight,
    record: SweepCheckpoint,
    entry: Mapping[str, Any],
    manifest_sha256: str,
    instrumented: Any,
    policy: NativeQwenPolicy,
    started: float,
    model_load_seconds: float,
    paths: tuple[Path, Path, Path],
    attempt: int,
    invocation_index: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the steps, timings and report exactly as the locked runner's lines 547-583 do."""
    report_path, steps_path, timings_path = paths
    steps_sha256 = _write_jsonl(
        steps_path, [step.to_dict() for step in instrumented.evaluation.steps]
    )
    timing_rows = [item.to_dict() for item in instrumented.decision_timings]
    timings_sha256 = _write_jsonl(timings_path, timing_rows)
    evaluation = instrumented.evaluation.to_dict()
    evaluation.pop("steps")
    actions = [action.name for action in instrumented.evaluation.trace.actions]
    telemetry = policy.episode_record()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "recorded_at_utc": _timestamp(),
        "manifest_sha256": manifest_sha256,
        "checkpoint_id": record.checkpoint_id,
        "checkpoint_sha256": record.checkpoint_sha256,
        "episode_index": entry["index"],
        "episode_id": str(entry["episode_id"]),
        "trajectory_id": entry["trajectory_id"],
        "scene_id": entry["scene_id"],
        "duration_seconds": time.monotonic() - started,
        "model_load_seconds_this_process": model_load_seconds,
        "environment_reset_seconds": instrumented.environment_reset_seconds,
        "policy_reset_seconds": instrumented.policy_reset_seconds,
        "evaluation": evaluation,
        "action_histogram": dict(sorted(Counter(actions).items())),
        "closed_loop_seconds": [item["closed_loop_seconds"] for item in timing_rows],
        "policy_telemetry": telemetry,
        "steps_path": str(steps_path),
        "steps_sha256": steps_sha256,
        "timings_path": str(timings_path),
        "timings_sha256": timings_sha256,
        # Added by the sweep; nothing above changes meaning.
        "step": pre.step,
        "run_id": pre.run_id,
        "suite_id": pre.suite_id,
        "attempt": attempt,
        "invocation_index": invocation_index,
        "provenance": dict(provenance),
    }
    if record.legacy_checkpoint_id is not None:
        report["legacy_checkpoint_id"] = record.legacy_checkpoint_id
    _write_json(report_path, report)
    return report


def _decision_seconds(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(value for report in reports for value in report["closed_loop_seconds"])
    if not values:
        return {"decisions": 0, "median": None, "mean": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"decisions": len(values), "median": median, "mean": sum(values) / len(values)}


def _result(
    args: argparse.Namespace,
    pre: Preflight,
    manifest_path: Path,
    manifest_sha256: str,
    reports: Mapping[tuple[str, str], Mapping[str, Any]],
    not_completed: Mapping[tuple[str, str], Mapping[str, Any]],
    log: InvocationLog,
    invocation_started: float,
    last_runtime: Mapping[str, Any] | None,
    final_code: Mapping[str, str],
) -> dict[str, Any]:
    record = pre.record
    done = [
        reports[(record.checkpoint_id, str(entry["episode_id"]))]
        for entry in pre.entries
        if (record.checkpoint_id, str(entry["episode_id"])) in reports
    ]
    missing = [
        str(entry["episode_id"])
        for entry in pre.entries
        if (record.checkpoint_id, str(entry["episode_id"])) in not_completed
    ]
    invocations = log.invocations()
    model_loads = [load for line in invocations for load in line.get("model_loads", [])]
    model_loads += log.model_loads
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "gate": GATE_ID,
        "step": pre.step,
        "run_id": pre.run_id,
        "suite_id": pre.suite_id,
        "checkpoint_id": record.checkpoint_id,
        "recorded_at_utc": _timestamp(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "preregistration_sha256": pre.preregistration["sha256"],
        "code": {"commit": pre.platform["code_commit"], "files": dict(final_code)},
        "platform": pre.platform,
        "target_episode_runs": len(pre.entries),
        "completed_episode_runs": len(done),
        "not_completed_pairs": [
            {"checkpoint_id": key[0], "episode_id": key[1]} for key in sorted(not_completed)
        ],
        "checkpoint_summary": _checkpoint_summary(done) if done else None,
        "checkpoint_completion": {
            "role": record.role,
            "legacy_checkpoint_id": record.legacy_checkpoint_id,
            "is_selected_update": pre.checkpoint_lock["is_selected_update"],
            "completed_episode_count": len(done),
            "not_completed_episode_ids": missing,
            "episode_count": len(pre.entries),
        },
        "control_comparison": (
            _control_summary(
                args.control_root,
                _load_json(args.control_root / "suite_manifest.json"),
                len(pre.entries),
            )
            if pre.suite_id == "main"
            else None
        ),
        "invocation_count": log.index,
        "adapter_loads": model_loads,
        "measurements": {
            "seconds_per_decision": _decision_seconds(done),
            "memory_by_invocation": [line.get("memory") for line in invocations]
            + [memory_record()],
        },
        "combined_invocation_seconds": time.monotonic() - invocation_started,
        "runtime": {
            **(dict(last_runtime) if last_runtime else {}),
            "habitat_sim_build_commit_observed": pre.habitat_build_commit,
            "platform": platform.platform(),
            "architecture": platform.machine().lower(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "evidence_boundary": {
            "established_if_checked": (
                "This checkpoint was driven on the named suite with the locked episode loop, H4 "
                "one-step scoring and the official 500-step cap, on the recorded GPU and image."
            ),
            "not_established": (
                "Anything before the analysis script's checks; val_unseen; other backbones, "
                "budgets or GPU types."
            ),
        },
    }


def _refused(step: str, reason: str) -> int:
    print(json.dumps({"status": "refused", "step": step, "reason": reason}), flush=True)
    return EXIT_REFUSED


def run(args: argparse.Namespace) -> int:
    invocation_started = time.monotonic()
    try:
        pre = preflight(args)
        # Data is loaded only after every refusal above; nothing is written until binding passes.
        output_root = args.output_root.resolve()
        manifest_path = output_root / "manifest.json"
        unbound = build_manifest(args, pre, None)
        if pre.existing_manifest is not None:
            compare_manifest(pre.existing_manifest, unbound, ignore_binding=True)
        dataset = R2RCEDataset.from_release(
            args.episodes, args.ground_truth, split="val_seen", limit=None
        )
        binding = bind_suite(pre.binding_path, dataset)
        manifest = build_manifest(args, pre, binding)
        completed_at_start = 0
        if pre.existing_manifest is not None:
            compare_manifest(pre.existing_manifest, manifest, ignore_binding=False)
            try:
                completed_at_start = len(
                    _load_existing_reports(
                        output_root,
                        [pre.record.checkpoint_id],
                        pre.entries,
                        _sha256(manifest_path),
                    )
                )
            except RuntimeError as error:
                raise refuse(str(error)) from error
        check_forced_stop(args.step, args.stop_after_completed_episodes, completed_at_start)
        if pre.existing_manifest is None:
            _write_json(manifest_path, manifest)
    except SweepRefusalError as refusal:
        return _refused(args.step, str(refusal))
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
    return _execute(args, pre, dataset, manifest_path, invocation_started)


def _execute(
    args: argparse.Namespace,
    pre: Preflight,
    dataset: Any,
    manifest_path: Path,
    invocation_started: float,
) -> int:
    output_root = manifest_path.parent
    manifest_sha256 = _sha256(manifest_path)
    record = pre.record
    try:
        reports = _load_existing_reports(
            output_root, [record.checkpoint_id], pre.entries, manifest_sha256
        )
    except RuntimeError as error:
        return _refused(pre.step, f"Refused: {error}")
    forced = args.stop_after_completed_episodes
    stems = {
        (record.checkpoint_id, str(entry["episode_id"])): str(entry["artifact_stem"])
        for entry in pre.entries
    }
    repo = args.repo.resolve()
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
    log = InvocationLog(output_root, pre.step, pre.run_id, target=len(stems), provenance=provenance)
    log.start(len(reports), args.previous_stop_reason)
    not_completed = log.update_not_completed(stems, pre.suite_id, set(reports))
    attempts = failed_attempts(log.invocations())

    pair: dict[str, Any] | None = None
    loading: str | None = None
    previous_handlers = {
        signum: signal.signal(signum, _raise_stop) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    environment: Any = None
    last_runtime: dict[str, Any] | None = None

    def stop(status: str, exit_code: int, **fields: Any) -> int:
        log.progress(
            status,
            completed=len(reports),
            not_completed=len(not_completed),
            checkpoint=fields.pop("checkpoint", None),
            pair=None,
        )
        log.finish(exit_code=exit_code, outcome=status, completed=len(reports), pair=None, **fields)
        return exit_code

    try:
        log.progress("running", completed=len(reports), not_completed=len(not_completed))
        pending = [
            entry
            for entry in pre.entries
            if (record.checkpoint_id, str(entry["episode_id"])) not in reports
            and (record.checkpoint_id, str(entry["episode_id"])) not in not_completed
        ]
        if pending:
            environment = HabitatSimR2REnvironment(
                HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=GPU_DEVICE_ID)
            )
            loading = record.checkpoint_id
            log.progress(
                "loading_checkpoint",
                completed=len(reports),
                not_completed=len(not_completed),
                checkpoint=record.checkpoint_id,
            )
            model_started = time.monotonic()
            scorer = NativeQwenActionScorer.from_native_c0_checkpoint(
                _checkpoint_path(args.checkpoint_archive, record),
                **loader_arguments(pre.checkpoint_lock, args.device),
            )
            model_load_seconds = time.monotonic() - model_started
            adapter = adapter_dtype_record(scorer)
            log.model_loads.append(
                {
                    "checkpoint_id": record.checkpoint_id,
                    "model_load_seconds": model_load_seconds,
                    "adapter": adapter,
                }
            )
            # Loading the model imports more modules; they too must come from --repo.
            check_code_is_from_repo(repo, entry_file=__file__, expected_entry=RUNNER_PATH)
            loading = None
            for entry in pending:
                if (time.monotonic() - invocation_started) / 3600.0 >= args.max_hours:
                    return stop(
                        "partial_time_budget", EXIT_RESUMABLE, checkpoint=record.checkpoint_id
                    )
                if forced is not None and len(reports) >= forced:
                    return stop(
                        "partial_forced_stop",
                        EXIT_RESUMABLE,
                        checkpoint=record.checkpoint_id,
                        forced_stop=forced,
                    )
                episode_id = str(entry["episode_id"])
                key = (record.checkpoint_id, episode_id)
                pair = {
                    "checkpoint_id": record.checkpoint_id,
                    "episode_id": episode_id,
                    "artifact_stem": str(entry["artifact_stem"]),
                }
                log.progress(
                    "running_episode",
                    completed=len(reports),
                    not_completed=len(not_completed),
                    checkpoint=record.checkpoint_id,
                    pair=pair,
                )
                episode = dataset.episode_record(episode_id)
                ground_truth = dataset.ground_truth_record(episode_id)
                # A fresh policy wrapper per episode, as the locked runner does; the loaded scorer
                # is shared for the whole checkpoint.
                policy = NativeQwenPolicy(
                    scorer=scorer,
                    config=NativePolicyConfig.for_mode(
                        HistoryMode.H4_SHORT_WINDOW, chunk_length=1, step_cap=EPISODE_MAX_STEPS
                    ),
                )
                episode_started = time.monotonic()
                instrumented = run_instrumented_r2r_ce_episode(
                    episode,
                    ground_truth,
                    environment,
                    policy,
                    episode_index=int(entry["index"]),
                    max_steps=EPISODE_MAX_STEPS,
                )
                last_runtime = environment.runtime_metadata()
                report = _report(
                    pre=pre,
                    record=record,
                    entry=entry,
                    manifest_sha256=manifest_sha256,
                    instrumented=instrumented,
                    policy=policy,
                    started=episode_started,
                    model_load_seconds=model_load_seconds,
                    paths=_report_paths(
                        output_root, record.checkpoint_id, str(entry["artifact_stem"])
                    ),
                    attempt=len(attempts.get(key, [])) + 1,
                    invocation_index=log.index,
                    provenance=provenance,
                )
                reports[key] = report
                pair = None
                log.progress(
                    "running",
                    completed=len(reports),
                    not_completed=len(not_completed),
                    checkpoint=record.checkpoint_id,
                )
                metrics = report["evaluation"]["metrics"]
                print(
                    f"[{len(reports) + len(not_completed)}/{len(stems)}] step={pre.step} "
                    f"checkpoint={record.checkpoint_id} episode={episode_id} "
                    f"steps={metrics['step_count']} SR={metrics['success']:.0f} "
                    f"OSR={metrics['oracle_success']:.0f}",
                    flush=True,
                )
        final_code = code_record(repo, entry_file=__file__, expected_entry=RUNNER_PATH)
        check_code_unchanged(pre.code_files, final_code)
        # Inside the try, so an error while summarising is logged like any other error.
        result = _result(
            args,
            pre,
            manifest_path,
            manifest_sha256,
            reports,
            not_completed,
            log,
            invocation_started,
            last_runtime,
            final_code,
        )
        _write_json(output_root / "result.json", result)
        log.progress("completed", completed=len(reports), not_completed=len(not_completed))
        log.finish(exit_code=EXIT_COMPLETED, outcome="completed", completed=len(reports), pair=None)
    except _StopRequested as stopped:
        name = signal.Signals(stopped.signum).name
        log.progress("stopped_by_signal", completed=len(reports), not_completed=len(not_completed))
        log.finish(
            exit_code=128 + stopped.signum,
            outcome="stopped_by_signal",
            completed=len(reports),
            pair=pair,
            checkpoint_loading=loading,
            stop_by_hand=True,
            stop_reason=f"signal {name}",
        )
        return 128 + stopped.signum
    except Exception as error:  # noqa: BLE001 - every error is logged, as the locked runner does
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_infrastructure",
            "gate": GATE_ID,
            "step": pre.step,
            "run_id": pre.run_id,
            "checkpoint_id": record.checkpoint_id,
            "invocation_index": log.index,
            "recorded_at_utc": _timestamp(),
            "pair_in_progress": pair,
            "checkpoint_loading": loading,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_root / "failure.json", failure)
        log.progress("failed", completed=len(reports), not_completed=len(not_completed))
        log.finish(
            exit_code=EXIT_ERROR,
            outcome="failed",
            completed=len(reports),
            pair=pair,
            checkpoint_loading=loading,
            error=error,
        )
        log.update_not_completed(stems, pre.suite_id, set(reports))
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return EXIT_ERROR
    finally:
        if environment is not None:
            environment.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return EXIT_COMPLETED


def _model_check_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Hash the 14 snapshot files once per job.")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = write_model_check_record(args.model_snapshot, args.output)
    except SweepRefusalError as refusal:
        return _refused("model-check", str(refusal))
    print(json.dumps({"status": "passed", "files": len(record["files"])}), flush=True)
    return EXIT_COMPLETED


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["model-check"]:
        return _model_check_main(values[1:])
    return run(_arguments(values))


if __name__ == "__main__":
    sys.exit(main())
