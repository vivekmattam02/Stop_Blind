#!/usr/bin/env python3
"""Run the authorized full native-Qwen C0 action-balanced development comparison."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from scripts import run_spark_native_c0_development as base
from scripts.run_spark_native_c0_action_balance_stress import (
    STRESS_VERIFIER_GATE,
    _analysis_evidence,
    _development_evidence,
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
from spatialvlm.training.native_c0_action_balance import (
    NATIVE_C0_ACTION_BALANCE_GATE,
    NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
    NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
    NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
    build_action_balance_contract,
)
from spatialvlm.training.native_c0_action_balance_full import (
    NativeC0ActionBalanceProtocol,
    weighted_contribution_summary,
)
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_HISTORY,
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

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_ACTION_BALANCE_PROTOCOL.md",
    "scripts/run_spark_native_c0_action_balance.py",
    "scripts/check_native_c0_action_balance_artifacts.py",
    "scripts/run_spark_native_c0_action_balance.sh",
    "scripts/start_spark_native_c0_action_balance.sh",
    "scripts/native_c0_action_balance_sync_files.txt",
    "scripts/run_spark_native_c0_action_balance_stress.py",
    "scripts/check_native_c0_action_balance_stress_artifacts.py",
    "scripts/run_spark_native_c0_development.py",
    "scripts/check_native_c0_development_artifacts.py",
    "scripts/run_spark_native_c0_development_cache.py",
    "scripts/run_spark_native_development_measurements.py",
    "scripts/run_spark_native_history_select.py",
    "scripts/spark_launch_helpers.sh",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/training/native_c0_action_balance.py",
    "src/spatialvlm/training/native_c0_action_balance_full.py",
    "src/spatialvlm/training/native_c0_development.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_c0_action_balance.py",
    "tests/test_native_c0_action_balance_full.py",
    "tests/test_native_c0_action_balance_artifacts.py",
    "tests/test_native_c0_development.py",
    "tests/test_native_c0_development_artifacts.py",
)
EXPECTED_STRESS_ARTIFACT_SHA256: dict[str, str] = {
    "manifest": "6af09fa65359555520d7ffd6832bdce45b67a22f22eeb33893f8a1349d914945",
    "result": "4776a19269906f9b45397b5c09c7714485d99162be8f5df1db67e3a30caccc48",
    "updates": "7434050e9676cf89b29c6585220d17b6381660a55acb8d0d54052fdf8f017636",
    "verification": "327ffad9d2331f982b5b7a96d7ece0d9046fe137a936bdb1935fb672cbdaf9c9",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required action-balance source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return base._load_json(path, label)


def _stress_evidence(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    root = args.stress_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    updates_path = root / "updates.jsonl"
    verification_path = root / "verification.json"
    manifest = _load_json(manifest_path, "action-balance stress manifest")
    result = _load_json(result_path, "action-balance stress result")
    verification = _load_json(verification_path, "action-balance stress verification")
    manifest_sha = file_sha256(manifest_path)
    result_sha = file_sha256(result_path)
    updates_sha = file_sha256(updates_path)
    verification_sha = file_sha256(verification_path)
    observed_artifact_sha256 = {
        "manifest": manifest_sha,
        "result": result_sha,
        "updates": updates_sha,
        "verification": verification_sha,
    }
    if root.name != "native_c0_action_balance_stress_v1":
        raise RuntimeError("Full action-balance run requires the immutable v1 stress directory.")
    if observed_artifact_sha256 != EXPECTED_STRESS_ARTIFACT_SHA256:
        raise RuntimeError("Stress artifacts differ from the exact passed authorization evidence.")
    if manifest.get("contract") != contract:
        raise RuntimeError(
            "Stress contract differs from the independently reconstructed treatment."
        )
    repo = args.repo.expanduser().resolve()
    for relative, expected_sha256 in manifest.get("source_locks", {}).items():
        source_path = repo / relative
        if not source_path.is_file() or file_sha256(source_path) != expected_sha256:
            raise RuntimeError(f"Executed stress source lock changed: {relative}")
    if (
        manifest.get("gate") != NATIVE_C0_ACTION_BALANCE_STRESS_GATE
        or manifest.get("status") != "frozen"
        or result.get("gate") != NATIVE_C0_ACTION_BALANCE_STRESS_GATE
        or result.get("run_id") != root.name
        or result.get("status") != "passed"
        or result.get("manifest_sha256") != manifest_sha
        or result.get("updates_sha256") != updates_sha
        or result.get("full_action_balance_run_authorized") is not True
        or result.get("scientific_result_emitted") is not False
    ):
        raise RuntimeError("Action-balance stress result does not authorize the full run.")
    if (
        verification.get("gate") != STRESS_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("manifest_sha256") != manifest_sha
        or verification.get("result_sha256") != result_sha
        or verification.get("updates_sha256") != updates_sha
    ):
        raise RuntimeError("Action-balance stress lacks linked independent verification.")
    return {
        "stress_root": str(root),
        "manifest_sha256": manifest_sha,
        "result_sha256": result_sha,
        "updates_sha256": updates_sha,
        "verification_sha256": verification_sha,
        "completed_updates": int(result["completed_updates"]),
        "completed_exposures": int(result["completed_exposures"]),
        "full_run_authorized": True,
        "exact_artifact_identity_confirmed": True,
        "executed_source_locks_confirmed": True,
    }


def expected_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently reconstruct the immutable full action-balanced run manifest."""
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted Habitat/Qwen image.")
    protocol = NativeC0ActionBalanceProtocol()
    original_protocol = NativeC0DevelopmentProtocol()
    cache = base._cache_evidence(args)
    cache_manifest = cache["manifest"]
    if cache_manifest.get("protocol") != original_protocol.to_record():
        raise RuntimeError("Action-balance run requires the original verified C0 cache protocol.")
    if len(cache_manifest.get("samples", [])) != protocol.cached_decisions:
        raise RuntimeError("Action-balance cache sample surface is incomplete.")

    development = _development_evidence(args)
    analysis = _analysis_evidence(args)
    if (
        development["manifest"].get("samples") != cache_manifest.get("samples")
        or development["manifest"].get("sample_records_sha256")
        != cache_manifest.get("sample_records_sha256")
        or development["manifest"].get("cache") != cache["record"]
    ):
        raise RuntimeError("Action-balance inputs differ from the original frozen C0 surface.")
    contract = build_action_balance_contract(
        development_manifest=development["manifest"],
        development_result_sha256=development["record"]["result_sha256"],
        development_verification_sha256=development["record"]["verification_sha256"],
        failure_analysis=analysis["analysis"],
        failure_analysis_sha256=analysis["record"]["analysis_sha256"],
        failure_verification=analysis["verification"],
        failure_verification_sha256=analysis["record"]["verification_sha256"],
    )
    stress = _stress_evidence(args, contract)
    manifest = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "protocol": protocol.to_record(),
        "action_balance_contract": contract,
        "stress_authorization": stress,
        "development_evidence": development["record"],
        "failure_analysis_evidence": analysis["record"],
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
            "weight_locks": model_weight_locks(args.model_snapshot),
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
        "source_locks": source_locks(args.repo.expanduser().resolve()),
        "evidence_boundary": {
            "established_if_passed": (
                "This adaptive train-only class-balanced correction met the original native-C0 "
                "open-loop existence conditions at the frozen budget."
            ),
            "not_established": (
                "Independent confirmation, class-weighting superiority, closed-loop navigation "
                "competence, val_seen or val_unseen performance, or any C1-C4 geometry claim."
            ),
        },
    }
    return manifest, cache


def _freeze(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    expected, _ = expected_manifest(args)
    if path.exists():
        if _load_json(path, "action-balance manifest") != expected:
            raise RuntimeError("Existing action-balance manifest differs from reconstruction.")
        print(f"Reusing identical frozen action-balance manifest: {path}")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    exposure_end: int,
    manifest_sha256: str,
    cache_index_sha256: str,
    protocol: NativeC0ActionBalanceProtocol,
    partial: bool = False,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite action-balance checkpoint: {path}")
    state = trainable_state_dict(model)
    payload = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_GATE,
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


def _write_partial(
    *,
    output_root: Path,
    checkpoint_root: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    manifest_sha256: str,
    cache_index_sha256: str,
    protocol: NativeC0ActionBalanceProtocol,
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
            "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
            "gate": NATIVE_C0_ACTION_BALANCE_GATE,
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
    manifest = _load_json(manifest_path, "action-balance manifest")
    expected, cache = expected_manifest(args)
    if manifest != expected:
        raise RuntimeError("Action-balance manifest changed or no longer reconstructs.")
    protocol = NativeC0ActionBalanceProtocol()
    weights = manifest["action_balance_contract"]["weights"]
    samples = list(manifest["samples"])
    by_split = {
        split_name: [row for row in samples if row["split"] == split_name]
        for split_name in ("train", "selection", "primary_evaluation", "balanced_diagnostic")
    }
    train_by_index = {int(row["sample_index"]): row for row in by_split["train"]}
    order = training_sample_order(samples, protocol)
    entries, identities = base._cache_maps(cache, args.cache_object_root.expanduser().resolve())
    manifest_sha = file_sha256(manifest_path)
    cache_index_sha = cache["record"]["cache_index_sha256"]
    cache_elapsed = float(cache["result"]["elapsed_seconds"])
    hard_total_seconds = protocol.hard_total_hours * 3600
    if cache_elapsed >= hard_total_seconds:
        raise TimeoutError("Verified cache alone consumed the frozen 4.5-hour total cap.")
    deadline_monotonic = run_started + (hard_total_seconds - cache_elapsed)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native C0 action-balance run requires a BF16-capable CUDA device.")
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
            base._evaluate(
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
    except base.PartialRun as exc:
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
        print(f"Action-balance run stopped partial: {exc}", file=sys.stderr)
        return 2

    events_path = output_root / "updates.jsonl"
    previous_sha = initial_sha
    completed_updates = 0
    try:
        with events_path.open("x", encoding="utf-8") as handle:
            for update_step in range(1, protocol.planned_optimizer_updates + 1):
                combined_elapsed = cache_elapsed + (time.perf_counter() - run_started)
                if base._STOP_REQUESTED:
                    raise base.PartialRun("termination requested")
                if combined_elapsed >= hard_total_seconds:
                    raise base.PartialRun(
                        "frozen 4.5-hour combined cache-plus-training cap reached"
                    )
                start = (update_step - 1) * protocol.accumulation_steps
                indices = order[start : start + protocol.accumulation_steps]
                if len(indices) != protocol.accumulation_steps:
                    raise RuntimeError("Action-balance exposure order ended mid-update.")
                optimizer.zero_grad(set_to_none=True)
                before = trainable_state_dict(model)
                before_sha = tensor_state_sha256(before)
                if before_sha != previous_sha:
                    raise RuntimeError("Action-balance hash chain broke before an update.")
                microbatches: list[dict[str, Any]] = []
                update_started = time.perf_counter()
                for microbatch_index, sample_index in enumerate(indices):
                    sample = train_by_index[sample_index]
                    cached, entry = base._load_cache(sample, entries, identities)
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
                        or unweighted_value > protocol.hard_max_restricted_ce
                        or not math.isfinite(weighted_value)
                        or weighted_value < 0.0
                    ):
                        raise RuntimeError("Action-balance training produced an invalid loss.")
                    (weighted_loss / protocol.accumulation_steps).backward()
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
                            "target_action": target.name,
                            "predicted_action": scored["predicted_action"],
                            "target_action_weight": weight,
                            "unweighted_restricted_action_ce": unweighted_value,
                            "weighted_restricted_action_ce": weighted_value,
                            "cache_key": entry["cache_key"],
                            "cache_sha256": entry["sha256"],
                            "action_token_position": position,
                            "supervised_token_count": int(batch.record["supervised_token_count"]),
                            "supervised_token_id": int(batch.record["supervised_token_id"]),
                        }
                    )
                    del outputs, inputs, batch, unweighted_loss, weighted_loss
                gradients_before = gradient_report(model)
                if (
                    gradients_before["gradient_tensor_count"]
                    != gradients_before["trainable_tensor_count"]
                    or gradients_before["nonzero_gradient_tensor_count"] <= 0
                    or gradients_before["all_gradients_finite"] is not True
                ):
                    raise RuntimeError("Action-balance gradients are missing, zero, or non-finite.")
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
                gradients_after = gradient_report(model)
                if (
                    not math.isfinite(preclip)
                    or preclip <= 0.0
                    or preclip > protocol.hard_max_gradient_norm
                    or gradients_after["all_gradients_finite"] is not True
                    or float(gradients_after["gradient_l2_norm"]) > protocol.max_grad_norm + 2e-4
                ):
                    raise RuntimeError("Action-balance gradient clipping is invalid.")
                optimizer.step()
                updates = update_report(model, before)
                validate_optimization_reports(gradients_before, updates)
                optimizer_state = optimizer_tensor_diagnostics(optimizer)
                if any(value.get("all_finite") is not True for value in optimizer_state.values()):
                    raise RuntimeError("Action-balance optimizer state is non-finite.")
                after_sha = tensor_state_sha256(trainable_state_dict(model))
                if after_sha == before_sha:
                    raise RuntimeError("Action-balance optimizer changed no trainable bytes.")
                event = {
                    "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
                    "gate": NATIVE_C0_ACTION_BALANCE_GATE,
                    "history_contract": NATIVE_C0_HISTORY,
                    "objective": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
                    "update_step": update_step,
                    "sample_exposure_start": start,
                    "sample_exposure_end_exclusive": start + len(indices),
                    "microbatch_count": len(indices),
                    "accumulation_steps": protocol.accumulation_steps,
                    "learning_rate": protocol.learning_rate,
                    "mean_unweighted_restricted_action_ce": sum(
                        row["unweighted_restricted_action_ce"] for row in microbatches
                    )
                    / len(microbatches),
                    "mean_weighted_restricted_action_ce": sum(
                        row["weighted_restricted_action_ce"] for row in microbatches
                    )
                    / len(microbatches),
                    "per_action_weighted_contributions": weighted_contribution_summary(
                        microbatches
                    ),
                    "gradient_report_before_clip": gradients_before,
                    "global_gradient_norm_before_clip": preclip,
                    "gradient_report_after_clip": gradients_after,
                    "clip_max_norm": protocol.max_grad_norm,
                    "update_report": updates,
                    "before_trainable_state_sha256": before_sha,
                    "after_trainable_state_sha256": after_sha,
                    "optimizer_state": optimizer_state,
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
                        base._evaluate(
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
    except base.PartialRun as exc:
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
        print(f"Action-balance run stopped partial: {exc}", file=sys.stderr)
        return 2

    selection = select_checkpoint(calibrations)
    checkpoints_by_step = {int(row["update_step"]): row for row in checkpoints}
    selected_checkpoint = checkpoints_by_step[int(selection["selected_update"])]
    base._load_checkpoint_state(
        Path(selected_checkpoint["path"]), model, selected_checkpoint["sha256"]
    )
    try:
        primary = base._evaluate(
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
        diagnostic = base._evaluate(
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
    except base.PartialRun as exc:
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
        print(f"Action-balance run stopped partial: {exc}", file=sys.stderr)
        return 2

    outcome = evaluate_existence_gate(
        primary_decisions=primary["per_decision"],
        diagnostic_decisions=diagnostic["per_decision"],
        protocol=protocol,
    )
    runtime_seconds = time.perf_counter() - run_started
    combined_seconds = cache_elapsed + runtime_seconds
    if combined_seconds > hard_total_seconds:
        raise RuntimeError("Action-balance run completed after the frozen combined cap.")
    result = {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_GATE,
        "run_id": args.run_id,
        "status": "passed" if outcome["status"] == "passed" else "not_demonstrated",
        "completed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "cache": manifest["cache"],
        "protocol": protocol.to_record(),
        "action_balance_contract": manifest["action_balance_contract"],
        "stress_authorization": manifest["stress_authorization"],
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
        "adaptive_development_comparison": True,
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
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--failure-verification", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    _common(freeze)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, base._request_stop)
    signal.signal(signal.SIGINT, base._request_stop)
    try:
        return _freeze(args) if args.command == "freeze" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
            "gate": NATIVE_C0_ACTION_BALANCE_GATE,
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
