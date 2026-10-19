#!/usr/bin/env python3
"""Run four fresh-model updates to numerically admit the native-C0 weighted objective."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from scripts.run_spark_native_c0_development import (
    _cache_evidence,
    _cache_maps,
    _load_cache,
    _load_json,
)
from scripts.run_spark_native_development_measurements import _restricted_loss
from scripts.run_spark_native_history_select import (
    EXPECTED_DOCKER_IMAGE_ID,
    _atomic_json,
    _load_model,
    _move_batch,
    _native_batch,
)
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import build_action_token_codebook, score_action_token_logits
from spatialvlm.training.native_c0_action_balance import (
    ACCUMULATION_STEPS,
    NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
    NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
    NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
    STRESS_UPDATES,
    build_action_balance_contract,
)
from spatialvlm.training.native_c0_development import NATIVE_C0_HISTORY
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    gradient_report,
    model_weight_locks,
    tensor_state_sha256,
    trainable_state_dict,
    update_report,
    validate_lora_only_trainables,
    validate_optimization_reports,
)
from spatialvlm.training.native_history_select import optimizer_tensor_diagnostics

STRESS_VERIFIER_GATE = "independent_native_qwen_c0_action_balance_stress_verification_v1"
HARD_MAX_SECONDS = 1_800.0

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_ACTION_BALANCE_PROTOCOL_DRAFT.md",
    "scripts/run_spark_native_c0_action_balance_stress.py",
    "scripts/check_native_c0_action_balance_stress_artifacts.py",
    "scripts/run_spark_native_c0_action_balance_stress.sh",
    "scripts/start_spark_native_c0_action_balance_stress.sh",
    "scripts/native_c0_action_balance_stress_sync_files.txt",
    "scripts/run_spark_native_c0_development.py",
    "scripts/run_spark_native_c0_development_cache.py",
    "scripts/run_spark_native_development_measurements.py",
    "scripts/run_spark_native_history_select.py",
    "scripts/spark_launch_helpers.sh",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/training/native_c0_action_balance.py",
    "src/spatialvlm/training/native_c0_development.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "tests/test_native_c0_action_balance.py",
    "tests/test_native_c0_action_balance_stress_artifacts.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required stress source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _development_evidence(args: argparse.Namespace) -> dict[str, Any]:
    root = args.development_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    verification_path = root / "verification.json"
    manifest = _load_json(manifest_path, "C0 development manifest")
    result = _load_json(result_path, "C0 development result")
    verification = _load_json(verification_path, "C0 development verification")
    manifest_sha = file_sha256(manifest_path)
    result_sha = file_sha256(result_path)
    verification_sha = file_sha256(verification_path)
    if (
        manifest.get("gate") != "native_qwen_c0_development_v1"
        or manifest.get("status") != "frozen"
        or result.get("status") != "not_demonstrated"
        or result.get("manifest_sha256") != manifest_sha
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("manifest_sha256") != manifest_sha
        or verification.get("result_sha256") != result_sha
    ):
        raise RuntimeError("Action-balance stress lacks linked negative C0 development evidence.")
    return {
        "manifest": manifest,
        "result": result,
        "verification": verification,
        "record": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "result_path": str(result_path),
            "result_sha256": result_sha,
            "verification_path": str(verification_path),
            "verification_sha256": verification_sha,
        },
    }


def _analysis_evidence(args: argparse.Namespace) -> dict[str, Any]:
    analysis_path = args.failure_analysis.expanduser().resolve()
    verification_path = args.failure_verification.expanduser().resolve()
    analysis = _load_json(analysis_path, "C0 failure analysis")
    verification = _load_json(verification_path, "C0 failure-analysis verification")
    return {
        "analysis": analysis,
        "verification": verification,
        "record": {
            "analysis_path": str(analysis_path),
            "analysis_sha256": file_sha256(analysis_path),
            "verification_path": str(verification_path),
            "verification_sha256": file_sha256(verification_path),
        },
    }


def expected_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the immutable numerical-stress manifest."""
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted Habitat/Qwen image.")
    development = _development_evidence(args)
    analysis = _analysis_evidence(args)
    cache = _cache_evidence(args)
    if development["manifest"].get("cache") != cache["record"]:
        raise RuntimeError("Stress cache evidence differs from the original development run.")
    contract = build_action_balance_contract(
        development_manifest=development["manifest"],
        development_result_sha256=development["record"]["result_sha256"],
        development_verification_sha256=development["record"]["verification_sha256"],
        failure_analysis=analysis["analysis"],
        failure_analysis_sha256=analysis["record"]["analysis_sha256"],
        failure_verification=analysis["verification"],
        failure_verification_sha256=analysis["record"]["verification_sha256"],
    )
    manifest = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "contract": contract,
        "samples": development["manifest"]["samples"],
        "sample_records_sha256": development["manifest"]["sample_records_sha256"],
        "development_evidence": development["record"],
        "failure_analysis_evidence": analysis["record"],
        "cache": cache["record"],
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "revision": NATIVE_C0_MODEL_REVISION,
            "base_dtype": "bf16",
            "adapter_and_optimizer_dtype": "fp32",
            "fresh_seeded_adapter": True,
            "warm_start": False,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "model_seed": 20260825,
            "weight_locks": model_weight_locks(args.model_snapshot),
        },
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "fully_offline": True,
            "hard_max_seconds": HARD_MAX_SECONDS,
        },
        "source_locks": _source_locks(args.repo.expanduser().resolve()),
        "evidence_boundary": contract["evidence_boundary"],
    }
    return manifest, cache


def _freeze(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    expected, _ = expected_manifest(args)
    if path.exists():
        if _load_json(path, "action-balance stress manifest") != expected:
            raise RuntimeError(
                "Existing action-balance stress manifest differs from reconstruction."
            )
        print(f"Reusing identical frozen action-balance stress manifest: {path}")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _per_action_contributions(microbatches: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in microbatches:
        grouped[str(row["target_action"])].append(
            float(row["weighted_restricted_action_ce"])
        )
    return {
        action: {
            "count": len(grouped[action]),
            "sum_weighted_restricted_action_ce": sum(grouped[action]),
            "mean_weighted_restricted_action_ce": (
                sum(grouped[action]) / len(grouped[action]) if grouped[action] else None
            ),
        }
        for action in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    }


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root = args.output_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, "action-balance stress manifest")
    expected, cache = expected_manifest(args)
    if manifest != expected:
        raise RuntimeError("Action-balance stress manifest changed or no longer reconstructs.")
    contract = manifest["contract"]
    weights = contract["weights"]
    prefix = [int(value) for value in contract["numerical_stress"]["sample_indices"]]
    train = [row for row in manifest["samples"] if row["split"] == "train"]
    by_index = {int(row["sample_index"]): row for row in train}
    entries, identities = _cache_maps(cache, args.cache_object_root.expanduser().resolve())

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Action-balance stress requires a BF16-capable CUDA device.")
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID,
        revision=NATIVE_C0_MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    codebook = build_action_token_codebook(processor.tokenizer)
    model_started = time.perf_counter()
    model = _load_model(device, 20260825)
    model_load_seconds = time.perf_counter() - model_started
    trainable_contract = validate_lora_only_trainables(model)
    initial_sha = tensor_state_sha256(trainable_state_dict(model))
    previous_sha = initial_sha
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-5,
        weight_decay=0.01,
    )

    events_path = root / "updates.jsonl"
    action_histogram: Counter[str] = Counter()
    with events_path.open("x", encoding="utf-8") as handle:
        for update_step in range(1, STRESS_UPDATES + 1):
            if time.perf_counter() - started >= HARD_MAX_SECONDS:
                raise TimeoutError("Action-balance numerical stress exceeded 30 minutes.")
            start = (update_step - 1) * ACCUMULATION_STEPS
            indices = prefix[start : start + ACCUMULATION_STEPS]
            optimizer.zero_grad(set_to_none=True)
            before = trainable_state_dict(model)
            before_sha = tensor_state_sha256(before)
            if before_sha != previous_sha:
                raise RuntimeError("Action-balance stress hash chain broke before an update.")
            update_started = time.perf_counter()
            microbatches: list[dict[str, Any]] = []
            for microbatch_index, sample_index in enumerate(indices):
                sample = by_index[sample_index]
                cached, entry = _load_cache(sample, entries, identities)
                batch, position = _native_batch(
                    processor=processor,
                    codebook=codebook,
                    cached=cached,
                    sample=sample,
                    arm_id=NATIVE_C0_HISTORY,
                )
                inputs = _move_batch(batch, device)
                inputs.pop("labels")
                model.train()
                outputs = model(**inputs)
                target = R2RCEAction(int(sample["target_action_id"]))
                unweighted_loss = _restricted_loss(
                    outputs.logits,
                    action_position=position,
                    codebook=codebook,
                    target_action=target,
                )
                unweighted_value = float(unweighted_loss.detach().float().cpu().item())
                weight = float(weights["weights"][target.name])
                weighted_loss = unweighted_loss.float() * weight
                weighted_value = float(weighted_loss.detach().cpu().item())
                if (
                    not math.isfinite(unweighted_value)
                    or unweighted_value < 0.0
                    or unweighted_value > 100.0
                    or not math.isfinite(weighted_value)
                    or weighted_value < 0.0
                ):
                    raise RuntimeError("Action-balance stress produced an invalid weighted loss.")
                (weighted_loss / ACCUMULATION_STEPS).backward()
                scored = score_action_token_logits(
                    outputs.logits,
                    action_token_position=position,
                    codebook=codebook,
                )
                action_histogram[target.name] += 1
                microbatches.append(
                    {
                        "microbatch_index": microbatch_index,
                        "sample_index": sample_index,
                        "episode_id": sample["episode_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "scene_id": sample["scene_id"],
                        "target_action": target.name,
                        "predicted_action": scored["predicted_action"],
                        "target_action_weight": weight,
                        "unweighted_restricted_action_ce": unweighted_value,
                        "weighted_restricted_action_ce": weighted_value,
                        "cache_key": entry["cache_key"],
                        "cache_sha256": entry["sha256"],
                        "action_token_position": position,
                        "supervised_token_count": int(batch.record["supervised_token_count"]),
                    }
                )
                del outputs, inputs, batch, unweighted_loss, weighted_loss
            gradients_before = gradient_report(model)
            if (
                gradients_before["all_gradients_finite"] is not True
                or gradients_before["nonzero_gradient_tensor_count"] <= 0
                or gradients_before["gradient_tensor_count"]
                != gradients_before["trainable_tensor_count"]
            ):
                raise RuntimeError("Action-balance stress gradients are missing or non-finite.")
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                .detach()
                .float()
                .cpu()
                .item()
            )
            gradients_after = gradient_report(model)
            if (
                not math.isfinite(preclip)
                or preclip <= 0.0
                or gradients_after["all_gradients_finite"] is not True
                or float(gradients_after["gradient_l2_norm"]) > 1.0002
            ):
                raise RuntimeError("Action-balance stress gradient clipping is invalid.")
            optimizer.step()
            updates = update_report(model, before)
            validate_optimization_reports(gradients_before, updates)
            optimizer_state = optimizer_tensor_diagnostics(optimizer)
            if any(value.get("all_finite") is not True for value in optimizer_state.values()):
                raise RuntimeError("Action-balance stress optimizer state is non-finite.")
            after_sha = tensor_state_sha256(trainable_state_dict(model))
            if after_sha == before_sha:
                raise RuntimeError("Action-balance stress changed no trainable bytes.")
            event = {
                "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
                "gate": NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
                "objective": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
                "update_step": update_step,
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + len(indices),
                "microbatch_count": len(indices),
                "accumulation_steps": ACCUMULATION_STEPS,
                "learning_rate": 2e-5,
                "mean_unweighted_restricted_action_ce": sum(
                    row["unweighted_restricted_action_ce"] for row in microbatches
                )
                / len(microbatches),
                "mean_weighted_restricted_action_ce": sum(
                    row["weighted_restricted_action_ce"] for row in microbatches
                )
                / len(microbatches),
                "per_action_weighted_contributions": _per_action_contributions(microbatches),
                "gradient_report_before_clip": gradients_before,
                "global_gradient_norm_before_clip": preclip,
                "gradient_report_after_clip": gradients_after,
                "clip_max_norm": 1.0,
                "update_report": updates,
                "optimizer_state": optimizer_state,
                "before_trainable_state_sha256": before_sha,
                "after_trainable_state_sha256": after_sha,
                "microbatches": microbatches,
                "elapsed_seconds": time.perf_counter() - update_started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            previous_sha = after_sha
            _atomic_json(
                root / "progress.json",
                {
                    "status": "running",
                    "completed_update": update_step,
                    "planned_updates": STRESS_UPDATES,
                    "observed_action_histogram": {
                        action: action_histogram[action]
                        for action in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                    "recorded_at_utc": _utc_now(),
                },
            )
            del before

    elapsed = time.perf_counter() - started
    result = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
        "run_id": args.run_id,
        "status": "passed",
        "recorded_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "updates_path": str(events_path),
        "updates_sha256": file_sha256(events_path),
        "completed_updates": STRESS_UPDATES,
        "completed_exposures": len(prefix),
        "initial_trainable_state_sha256": initial_sha,
        "final_trainable_state_sha256": previous_sha,
        "observed_action_histogram": {
            action: action_histogram[action]
            for action in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
        },
        "weights": weights,
        "trainable_contract": trainable_contract,
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "model_load_seconds": model_load_seconds,
            "elapsed_seconds": elapsed,
            "hard_max_seconds": HARD_MAX_SECONDS,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "scientific_result_emitted": False,
        "full_action_balance_run_authorized": True,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    _atomic_json(root / "result.json", result)
    _atomic_json(
        root / "progress.json",
        {
            "status": "passed",
            "completed_updates": STRESS_UPDATES,
            "completed_exposures": len(prefix),
            "elapsed_seconds": elapsed,
            "recorded_at_utc": _utc_now(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--failure-verification", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    _common(freeze)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return _freeze(args) if args.command == "freeze" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
            "gate": NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_root = getattr(args, "output_root", None)
        if isinstance(output_root, Path):
            try:
                _atomic_json(output_root.expanduser().resolve() / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
