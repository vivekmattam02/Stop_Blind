#!/usr/bin/env python3
"""Run the frozen native-Qwen C0 development existence test from a verified H4 cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from scripts.run_spark_native_c0_development_cache import (
    _load_json,
    load_content_cache_object,
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
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.eval.action_decision import (
    build_action_token_codebook,
    score_action_token_logits,
)
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_CACHE_GATE,
    NATIVE_C0_DEVELOPMENT_GATE,
    NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
    NATIVE_C0_HISTORY,
    NATIVE_C0_OBJECTIVE,
    NativeC0DevelopmentProtocol,
    canonical_sha256,
    evaluate_existence_gate,
    select_checkpoint,
    training_sample_order,
)
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
from spatialvlm.training.native_history_select import (
    capture_rng_state,
    optimizer_tensor_diagnostics,
)

CACHE_VERIFIER_GATE = "independent_native_qwen_c0_content_cache_verification_v1"
DEVELOPMENT_VERIFIER_GATE = "independent_native_qwen_c0_development_verification_v1"
_STOP_REQUESTED = False

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_DEVELOPMENT_PROTOCOL.md",
    "scripts/run_spark_native_c0_development.py",
    "scripts/run_spark_native_development_measurements.py",
    "scripts/run_spark_native_history_select.py",
    "scripts/check_native_c0_development_artifacts.py",
    "scripts/run_spark_native_c0_development.sh",
    "scripts/start_spark_native_c0_development.sh",
    "scripts/native_c0_development_sync_files.txt",
    "scripts/spark_launch_helpers.sh",
    "scripts/run_spark_native_c0_development_cache.py",
    "scripts/check_native_c0_development_cache_artifacts.py",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/training/native_c0_development.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_c0_development.py",
    "tests/test_native_c0_development_artifacts.py",
)


class PartialRun(RuntimeError):
    """A requested or timed stop after a complete optimizer-update boundary."""


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Termination requested; saving partial state after this complete update.", flush=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _cache_evidence(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = args.cache_artifact_root.expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    index_path = cache_root / "cache_index.json"
    result_path = cache_root / "result.json"
    verification_path = cache_root / "verification.json"
    manifest = _load_json(manifest_path, "C0 cache manifest")
    index = _load_json(index_path, "C0 cache index")
    result = _load_json(result_path, "C0 cache result")
    verification = _load_json(verification_path, "C0 cache verification")
    if manifest.get("gate") != NATIVE_C0_CACHE_GATE or manifest.get("status") != "frozen":
        raise RuntimeError("C0 cache manifest is not frozen.")
    if (
        index.get("gate") != NATIVE_C0_CACHE_GATE
        or index.get("status") != "passed"
        or result.get("gate") != NATIVE_C0_CACHE_GATE
        or result.get("status") != "passed"
    ):
        raise RuntimeError("C0 content cache has not passed.")
    if (
        verification.get("gate") != CACHE_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("manifest_sha256") != file_sha256(manifest_path)
        or verification.get("cache_index_sha256") != file_sha256(index_path)
        or verification.get("result_sha256") != file_sha256(result_path)
    ):
        raise RuntimeError("C0 content cache lacks linked independent verification.")
    return {
        "manifest": manifest,
        "index": index,
        "result": result,
        "record": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "cache_index_path": str(index_path),
            "cache_index_sha256": file_sha256(index_path),
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
            "verification_path": str(verification_path),
            "verification_sha256": file_sha256(verification_path),
            "object_root": str(args.cache_object_root.expanduser().resolve()),
            "content_addressed": True,
        },
    }


def _expected_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted Habitat/Qwen image.")
    protocol = NativeC0DevelopmentProtocol()
    cache = _cache_evidence(args)
    cache_manifest = cache["manifest"]
    if cache_manifest.get("protocol") != protocol.to_record():
        raise RuntimeError("C0 cache protocol differs from the development protocol.")
    if len(cache_manifest.get("samples", [])) != protocol.cached_decisions:
        raise RuntimeError("C0 cache sample surface is incomplete.")
    manifest = {
        "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
        "gate": NATIVE_C0_DEVELOPMENT_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": cache_manifest["dataset"],
        "samples": cache_manifest["samples"],
        "sample_records_sha256": cache_manifest["sample_records_sha256"],
        "partition_summary": cache_manifest["partition_summary"],
        "cache": cache["record"],
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "revision": NATIVE_C0_MODEL_REVISION,
            "base_dtype": "bf16",
            "adapter_and_optimizer_dtype": "fp32",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "fresh_seeded_adapter": True,
            "warm_start": False,
        },
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "fully_offline": True,
            "combined_hard_cap_seconds": protocol.hard_total_hours * 3600,
            "cache_elapsed_seconds_charged_to_first_attempt": float(
                cache["result"]["elapsed_seconds"]
            ),
        },
        "source_locks": _source_locks(args.repo.expanduser().resolve()),
        "evidence_boundary": {
            "established_if_passed": (
                "At the frozen train-release budget, a selected native-Qwen C0 checkpoint beat "
                "both blind CE references by the frozen margin and met the STOP non-mute floor."
            ),
            "not_established": (
                "Closed-loop navigation competence, val_seen or val_unseen performance, rate "
                "optimality, C1-C4 effects, baseline parity, or geometry claims."
            ),
        },
    }
    return manifest, cache


def _freeze(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    expected, _ = _expected_manifest(args)
    if path.exists():
        if _load_json(path, "development manifest") != expected:
            raise RuntimeError("Existing C0 development manifest differs from reconstruction.")
        print(f"Reusing identical frozen C0 development manifest: {path}")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _cache_maps(
    cache: dict[str, Any], object_root: Path
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    entries = {
        (row["split"], int(row["sample_index"])): {**row, "object_root": object_root}
        for row in cache["index"]["entries"]
    }
    identities = {
        (row["split"], int(row["sample_index"])): row["object_identity"]
        for row in cache["manifest"]["object_references"]
    }
    return entries, identities


def _load_cache(
    sample: dict[str, Any],
    entries: dict[tuple[str, int], dict[str, Any]],
    identities: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = (sample["split"], int(sample["sample_index"]))
    entry = entries[key]
    payload = load_content_cache_object(entry, entry["object_root"], identities[key])
    return payload, entry


def _decision_record(
    *,
    outputs: Any,
    position: int,
    codebook: Any,
    sample: dict[str, Any],
    entry: dict[str, Any],
    batch: Any,
) -> dict[str, Any]:
    scored = score_action_token_logits(
        outputs.logits,
        action_token_position=position,
        codebook=codebook,
    )
    target = str(sample["target_action"])
    target_score = scored["scores"][target]
    probability = float(target_score["restricted_probability"])
    if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
        raise RuntimeError("C0 evaluation produced an invalid restricted probability.")
    return {
        "split": sample["split"],
        "sample_index": int(sample["sample_index"]),
        "episode_id": str(sample["episode_id"]),
        "trajectory_id": str(sample["trajectory_id"]),
        "scene_id": str(sample["scene_id"]),
        "target_action": target,
        "predicted_action": scored["predicted_action"],
        "correct": scored["predicted_action"] == target,
        "target_token_nll": float(target_score["token_nll"]),
        "target_restricted_probability": probability,
        "restricted_action_ce": -math.log(probability),
        "action_decision": scored,
        "cache_key": entry["cache_key"],
        "cache_sha256": entry["sha256"],
        "action_token_position": position,
        "supervised_token_count": int(batch.record["supervised_token_count"]),
        "supervised_token_id": int(batch.record["supervised_token_id"]),
    }


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise ValueError("C0 decision summary requires at least one decision.")
    ce = [float(row["restricted_action_ce"]) for row in decisions]
    probability = [float(row["target_restricted_probability"]) for row in decisions]
    if not all(math.isfinite(value) for value in (*ce, *probability)):
        raise ValueError("C0 decision summary contains non-finite metrics.")
    histogram = Counter(str(row["predicted_action"]) for row in decisions)
    per_action: dict[str, dict[str, Any]] = {}
    for action in R2RCEAction:
        rows = [row for row in decisions if row["target_action"] == action.name]
        per_action[action.name] = {
            "count": len(rows),
            "mean_restricted_action_ce": (
                sum(float(row["restricted_action_ce"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "recall": (
                sum(row["predicted_action"] == action.name for row in rows) / len(rows)
                if rows
                else None
            ),
        }
    return {
        "decision_count": len(decisions),
        "scene_count": len({str(row["scene_id"]) for row in decisions}),
        "mean_restricted_action_ce": sum(ce) / len(ce),
        "mean_target_restricted_probability": sum(probability) / len(probability),
        "descriptive_accuracy": sum(bool(row["correct"]) for row in decisions) / len(decisions),
        "predicted_action_histogram": {
            action.name: histogram.get(action.name, 0) for action in R2RCEAction
        },
        "distinct_predicted_action_count": sum(
            histogram.get(action.name, 0) > 0 for action in R2RCEAction
        ),
        "per_target_action": per_action,
    }


def _evaluate(
    *,
    model: Any,
    processor: Any,
    codebook: Any,
    samples: list[dict[str, Any]],
    entries: dict[tuple[str, int], dict[str, Any]],
    identities: dict[tuple[str, int], dict[str, Any]],
    device: torch.device,
    update_step: int,
    deadline_monotonic: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if _STOP_REQUESTED:
                raise PartialRun("termination requested during evaluation")
            if time.perf_counter() >= deadline_monotonic:
                raise PartialRun(
                    "frozen 4.5-hour combined cache-plus-training cap reached during evaluation"
                )
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
            outputs = model(**inputs)
            rows.append(
                _decision_record(
                    outputs=outputs,
                    position=position,
                    codebook=codebook,
                    sample=sample,
                    entry=entry,
                    batch=batch,
                )
            )
            del outputs, inputs, batch
    return {
        "update_step": update_step,
        "summary": summarize_decisions(rows),
        "per_decision": rows,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    exposure_end: int,
    manifest_sha256: str,
    cache_index_sha256: str,
    protocol: NativeC0DevelopmentProtocol,
    partial: bool = False,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite C0 checkpoint: {path}")
    state = trainable_state_dict(model)
    payload = {
        "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
        "gate": NATIVE_C0_DEVELOPMENT_GATE,
        "status": "partial" if partial else "candidate",
        "manifest_sha256": manifest_sha256,
        "cache_index_sha256": cache_index_sha256,
        "protocol": protocol.to_record(),
        "cursor": {
            "update_step": update_step,
            "sample_exposure_end_exclusive": exposure_end,
        },
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "update_step": update_step,
        "status": payload["status"],
    }


def _load_checkpoint_state(path: Path, model: Any, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("Selected C0 checkpoint file hash changed.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or tensor_state_sha256(state) != payload.get(
        "trainable_state_sha256"
    ):
        raise RuntimeError("Selected C0 checkpoint trainable state is invalid.")
    named = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(named) != set(state):
        raise RuntimeError("Selected C0 checkpoint trainable names differ from the model.")
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = named[name]
            if parameter.shape != tensor.shape:
                raise RuntimeError(f"Selected C0 checkpoint shape mismatch for {name}.")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return payload


def _write_partial(
    *,
    output_root: Path,
    checkpoint_root: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    manifest_sha256: str,
    cache_index_sha256: str,
    protocol: NativeC0DevelopmentProtocol,
    reason: str,
) -> None:
    checkpoint = _save_checkpoint(
        checkpoint_root / f"partial_step_{update_step:04d}.pt",
        model=model,
        optimizer=optimizer,
        update_step=update_step,
        exposure_end=update_step * protocol.accumulation_steps,
        manifest_sha256=manifest_sha256,
        cache_index_sha256=cache_index_sha256,
        protocol=protocol,
        partial=True,
    )
    _atomic_json(
        output_root / "partial_result.json",
        {
            "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
            "gate": NATIVE_C0_DEVELOPMENT_GATE,
            "status": "partial",
            "reason": reason,
            "completed_updates": update_step,
            "planned_updates": protocol.planned_optimizer_updates,
            "checkpoint": checkpoint,
            "recorded_at_utc": _utc_now(),
            "scientific_pass_emitted": False,
        },
    )


def _run(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path, "C0 development manifest")
    expected, cache = _expected_manifest(args)
    if manifest != expected:
        raise RuntimeError("C0 development manifest changed or no longer reconstructs.")
    protocol = NativeC0DevelopmentProtocol()
    samples = list(manifest["samples"])
    by_split = {
        split: [row for row in samples if row["split"] == split]
        for split in ("train", "selection", "primary_evaluation", "balanced_diagnostic")
    }
    train_by_index = {int(row["sample_index"]): row for row in by_split["train"]}
    order = training_sample_order(samples, protocol)
    entries, identities = _cache_maps(cache, args.cache_object_root.expanduser().resolve())
    manifest_sha = file_sha256(manifest_path)
    cache_index_sha = cache["record"]["cache_index_sha256"]
    cache_elapsed = float(cache["result"]["elapsed_seconds"])
    hard_total_seconds = protocol.hard_total_hours * 3600
    if cache_elapsed >= hard_total_seconds:
        raise TimeoutError("Verified cache alone consumed the frozen 4.5-hour total cap.")
    deadline_monotonic = run_started + (hard_total_seconds - cache_elapsed)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native C0 development requires a BF16-capable CUDA device.")
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
    model = _load_model(device, protocol.model_seed)
    model_load_seconds = time.perf_counter() - model_started
    trainable_contract = validate_lora_only_trainables(model)
    initial_sha = tensor_state_sha256(trainable_state_dict(model))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )

    checkpoints: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    try:
        calibration_started = time.perf_counter()
        calibrations.append(
            _evaluate(
                model=model,
                processor=processor,
                codebook=codebook,
                samples=by_split["selection"],
                entries=entries,
                identities=identities,
                device=device,
                update_step=0,
                deadline_monotonic=deadline_monotonic,
            )
        )
        calibration_seconds = time.perf_counter() - calibration_started
        checkpoints.append(
            _save_checkpoint(
                checkpoint_root / "step_0000.pt",
                model=model,
                optimizer=optimizer,
                update_step=0,
                exposure_end=0,
                manifest_sha256=manifest_sha,
                cache_index_sha256=cache_index_sha,
                protocol=protocol,
            )
        )
    except PartialRun as exc:
        _write_partial(
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            model=model,
            optimizer=optimizer,
            update_step=0,
            manifest_sha256=manifest_sha,
            cache_index_sha256=cache_index_sha,
            protocol=protocol,
            reason=str(exc),
        )
        print(f"C0 development stopped partial: {exc}", file=sys.stderr)
        return 2

    events_path = output_root / "updates.jsonl"
    previous_sha = initial_sha
    completed_updates = 0
    try:
        with events_path.open("x", encoding="utf-8") as handle:
            for update_step in range(1, protocol.planned_optimizer_updates + 1):
                combined_elapsed = cache_elapsed + (time.perf_counter() - run_started)
                if _STOP_REQUESTED:
                    raise PartialRun("termination requested")
                if combined_elapsed >= hard_total_seconds:
                    raise PartialRun("frozen 4.5-hour combined cache-plus-training cap reached")
                start = (update_step - 1) * protocol.accumulation_steps
                indices = order[start : start + protocol.accumulation_steps]
                if len(indices) != protocol.accumulation_steps:
                    raise RuntimeError("C0 single-pass exposure order ended mid-update.")
                optimizer.zero_grad(set_to_none=True)
                before = trainable_state_dict(model)
                before_sha = tensor_state_sha256(before)
                if before_sha != previous_sha:
                    raise RuntimeError("C0 trainable hash chain broke before an update.")
                microbatches: list[dict[str, Any]] = []
                update_started = time.perf_counter()
                for microbatch_index, sample_index in enumerate(indices):
                    sample = train_by_index[sample_index]
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
                    loss = _restricted_loss(
                        outputs.logits,
                        action_position=position,
                        codebook=codebook,
                        target_action=R2RCEAction(int(sample["target_action_id"])),
                    )
                    value = float(loss.detach().cpu().item())
                    if (
                        not math.isfinite(value)
                        or value < 0.0
                        or value > protocol.hard_max_restricted_ce
                    ):
                        raise RuntimeError("C0 training produced invalid restricted CE.")
                    (loss / protocol.accumulation_steps).backward()
                    scored = score_action_token_logits(
                        outputs.logits,
                        action_token_position=position,
                        codebook=codebook,
                    )
                    microbatches.append(
                        {
                            "microbatch_index": microbatch_index,
                            "sample_index": sample_index,
                            "episode_id": sample["episode_id"],
                            "trajectory_id": sample["trajectory_id"],
                            "scene_id": sample["scene_id"],
                            "target_action": sample["target_action"],
                            "predicted_action": scored["predicted_action"],
                            "restricted_action_ce": value,
                            "cache_key": entry["cache_key"],
                            "cache_sha256": entry["sha256"],
                            "action_token_position": position,
                            "supervised_token_count": int(
                                batch.record["supervised_token_count"]
                            ),
                            "supervised_token_id": int(batch.record["supervised_token_id"]),
                        }
                    )
                    del outputs, loss, inputs, batch
                gradients = gradient_report(model)
                if (
                    gradients["gradient_tensor_count"] != gradients["trainable_tensor_count"]
                    or gradients["nonzero_gradient_tensor_count"] <= 0
                    or gradients["all_gradients_finite"] is not True
                ):
                    raise RuntimeError("C0 gradients are missing, zero, or non-finite.")
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        protocol.max_grad_norm,
                    )
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                if (
                    not math.isfinite(preclip)
                    or preclip <= 0.0
                    or preclip > protocol.hard_max_gradient_norm
                ):
                    raise RuntimeError("C0 global gradient norm is invalid.")
                optimizer.step()
                updates = update_report(model, before)
                validate_optimization_reports(gradients, updates)
                after_sha = tensor_state_sha256(trainable_state_dict(model))
                if after_sha == before_sha:
                    raise RuntimeError("C0 optimizer changed no trainable bytes.")
                event = {
                    "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
                    "gate": NATIVE_C0_DEVELOPMENT_GATE,
                    "history_contract": NATIVE_C0_HISTORY,
                    "objective": NATIVE_C0_OBJECTIVE,
                    "update_step": update_step,
                    "sample_exposure_start": start,
                    "sample_exposure_end_exclusive": start + len(indices),
                    "microbatch_count": len(indices),
                    "accumulation_steps": protocol.accumulation_steps,
                    "learning_rate": protocol.learning_rate,
                    "mean_restricted_action_ce": sum(
                        row["restricted_action_ce"] for row in microbatches
                    )
                    / len(microbatches),
                    "gradient_report": gradients,
                    "global_gradient_norm_before_clip": preclip,
                    "clip_max_norm": protocol.max_grad_norm,
                    "update_report": updates,
                    "before_trainable_state_sha256": before_sha,
                    "after_trainable_state_sha256": after_sha,
                    "optimizer_state": optimizer_tensor_diagnostics(optimizer),
                    "microbatches": microbatches,
                    "elapsed_seconds": time.perf_counter() - update_started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                previous_sha = after_sha
                completed_updates = update_step
                _atomic_json(
                    output_root / "progress.json",
                    {
                        "status": "running",
                        "current_stage": "training",
                        "completed_update": update_step,
                        "planned_updates": protocol.planned_optimizer_updates,
                        "combined_elapsed_seconds": cache_elapsed
                        + (time.perf_counter() - run_started),
                        "recorded_at_utc": _utc_now(),
                    },
                )
                if update_step in protocol.calibration_updates[1:]:
                    calibration_started = time.perf_counter()
                    calibrations.append(
                        _evaluate(
                            model=model,
                            processor=processor,
                            codebook=codebook,
                            samples=by_split["selection"],
                            entries=entries,
                            identities=identities,
                            device=device,
                            update_step=update_step,
                            deadline_monotonic=deadline_monotonic,
                        )
                    )
                    calibration_seconds += time.perf_counter() - calibration_started
                    checkpoints.append(
                        _save_checkpoint(
                            checkpoint_root / f"step_{update_step:04d}.pt",
                            model=model,
                            optimizer=optimizer,
                            update_step=update_step,
                            exposure_end=update_step * protocol.accumulation_steps,
                            manifest_sha256=manifest_sha,
                            cache_index_sha256=cache_index_sha,
                            protocol=protocol,
                        )
                    )
                del before
    except PartialRun as exc:
        _write_partial(
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            model=model,
            optimizer=optimizer,
            update_step=completed_updates,
            manifest_sha256=manifest_sha,
            cache_index_sha256=cache_index_sha,
            protocol=protocol,
            reason=str(exc),
        )
        print(f"C0 development stopped partial: {exc}", file=sys.stderr)
        return 2

    selection = select_checkpoint(calibrations)
    checkpoints_by_step = {int(row["update_step"]): row for row in checkpoints}
    selected_checkpoint = checkpoints_by_step[int(selection["selected_update"])]
    _load_checkpoint_state(
        Path(selected_checkpoint["path"]), model, selected_checkpoint["sha256"]
    )
    try:
        primary = _evaluate(
            model=model,
            processor=processor,
            codebook=codebook,
            samples=by_split["primary_evaluation"],
            entries=entries,
            identities=identities,
            device=device,
            update_step=int(selection["selected_update"]),
            deadline_monotonic=deadline_monotonic,
        )
        diagnostic = _evaluate(
            model=model,
            processor=processor,
            codebook=codebook,
            samples=by_split["balanced_diagnostic"],
            entries=entries,
            identities=identities,
            device=device,
            update_step=int(selection["selected_update"]),
            deadline_monotonic=deadline_monotonic,
        )
    except PartialRun as exc:
        _write_partial(
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            model=model,
            optimizer=optimizer,
            update_step=completed_updates,
            manifest_sha256=manifest_sha,
            cache_index_sha256=cache_index_sha,
            protocol=protocol,
            reason=str(exc),
        )
        print(f"C0 development stopped partial: {exc}", file=sys.stderr)
        return 2
    outcome = evaluate_existence_gate(
        primary_decisions=primary["per_decision"],
        diagnostic_decisions=diagnostic["per_decision"],
        protocol=protocol,
    )
    runtime_seconds = time.perf_counter() - run_started
    combined_seconds = cache_elapsed + runtime_seconds
    if combined_seconds > hard_total_seconds:
        raise RuntimeError("C0 completed after the frozen combined 4.5-hour cap.")
    result = {
        "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
        "gate": NATIVE_C0_DEVELOPMENT_GATE,
        "run_id": args.run_id,
        "status": "passed" if outcome["status"] == "passed" else "not_demonstrated",
        "completed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "cache": manifest["cache"],
        "protocol": protocol.to_record(),
        "updates_path": str(events_path),
        "updates_sha256": file_sha256(events_path),
        "completed_updates": completed_updates,
        "initial_trainable_state_sha256": initial_sha,
        "final_training_trainable_state_sha256": previous_sha,
        "calibrations": calibrations,
        "calibrations_sha256": canonical_sha256(calibrations),
        "checkpoints": checkpoints,
        "selection": {**selection, "checkpoint": selected_checkpoint},
        "primary_evaluation": primary,
        "balanced_diagnostic": diagnostic,
        "existence_gate": outcome,
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "action_codebook": codebook.to_record(),
        "trainable_contract": trainable_contract,
        "model_weight_locks": model_weight_locks(args.model_snapshot),
        "source_locks": manifest["source_locks"],
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "docker_image_id": EXPECTED_DOCKER_IMAGE_ID,
            "model_load_seconds": model_load_seconds,
            "selection_calibration_seconds": calibration_seconds,
            "training_and_evaluation_elapsed_seconds": runtime_seconds,
            "charged_cache_elapsed_seconds": cache_elapsed,
            "combined_elapsed_seconds": combined_seconds,
            "hard_total_seconds": hard_total_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "evidence_boundary": manifest["evidence_boundary"],
    }
    _atomic_json(output_root / "result.json", result)
    _atomic_json(
        output_root / "progress.json",
        {
            "status": result["status"],
            "completed_updates": completed_updates,
            "selected_update": selection["selected_update"],
            "existence_gate_status": outcome["status"],
            "combined_elapsed_seconds": combined_seconds,
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
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    _common(freeze)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--model-snapshot", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        return _freeze(args) if args.command == "freeze" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
            "gate": NATIVE_C0_DEVELOPMENT_GATE,
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
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
