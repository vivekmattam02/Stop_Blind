#!/usr/bin/env python3
"""Run one balanced optimizer update for every new paired-seed configuration."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from scripts.run_spark_native_c0_development import _cache_maps, _load_cache
from scripts.run_spark_native_c0_seed_replication import expected_manifest, source_locks
from scripts.run_spark_native_development_measurements import _restricted_loss
from scripts.run_spark_native_history_select import (
    EXPECTED_DOCKER_IMAGE_ID,
    _atomic_json,
    _load_model,
    _move_batch,
    _native_batch,
)
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import build_action_token_codebook
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
from spatialvlm.training.native_c0_seed_replication import (
    NATIVE_C0_SEED_REPLICATION_STRESS_GATE,
    NEW_REPLICATION_SEEDS,
    TREATMENTS,
    canonical_sha256,
    replication_run_id,
    replication_training_order,
    stress_sample_indices,
)

HARD_MAX_SECONDS = 1_800.0
IMPLEMENTATION_STARTED_AT_UTC = "2026-08-25T20:35:56Z"
STRESS_DEADLINE_UTC = "2026-08-25T22:35:56Z"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _args_for(args: argparse.Namespace, treatment: str, seed: int) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "run_id": replication_run_id(treatment, seed),
            "treatment": treatment,
            "replication_seed": seed,
            "output_root": args.output_root
            / "configurations"
            / replication_run_id(treatment, seed),
            "allow_missing_replication_stress": True,
        }
    )
    return argparse.Namespace(**values)


def expected_stress_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct all four stress configurations and their fixed balanced microbatch."""
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Stress Docker image differs from the admitted image.")
    configurations = []
    cache: dict[str, Any] | None = None
    for seed in NEW_REPLICATION_SEEDS:
        for treatment in TREATMENTS:
            run_args = _args_for(args, treatment, seed)
            run_manifest, observed_cache = expected_manifest(run_args)
            cache = cache or observed_cache
            configurations.append(
                {
                    "run_id": run_args.run_id,
                    "treatment": treatment,
                    "replication_seed": seed,
                    "replication_contract": run_manifest["replication_contract"],
                    "training_order_sha256": canonical_sha256(
                        list(replication_training_order(run_manifest["samples"], seed))
                    ),
                }
            )
    assert cache is not None
    samples = list(cache["manifest"]["samples"])
    indices = stress_sample_indices(samples)
    by_index = {int(row["sample_index"]): row for row in samples if row.get("split") == "train"}
    stress_samples = [by_index[index] for index in indices]
    manifest = {
        "schema_version": 1,
        "gate": NATIVE_C0_SEED_REPLICATION_STRESS_GATE,
        "run_id": "native_c0_seed_replication_stress_v1",
        "status": "frozen",
        "implementation_started_at_utc": IMPLEMENTATION_STARTED_AT_UTC,
        "stress_deadline_utc": STRESS_DEADLINE_UTC,
        "hard_max_seconds": HARD_MAX_SECONDS,
        "configurations": configurations,
        "stress_samples": stress_samples,
        "stress_sample_indices": list(indices),
        "stress_sample_records_sha256": canonical_sha256(stress_samples),
        "source_locks": source_locks(args.repo.resolve()),
        "model_weight_locks": model_weight_locks(args.model_snapshot.resolve()),
        "evidence_boundary": {
            "pass_establishes": (
                "All four new treatment/seed combinations execute one balanced effective-batch "
                "update with finite gradients and changed trainable state."
            ),
            "pass_does_not_establish": "Learning quality or seed-robust scientific outcomes.",
        },
    }
    return manifest, cache


def _freeze(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "manifest.json"
    expected, _ = expected_stress_manifest(args)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Existing replication-stress manifest differs.")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    manifest_path = args.output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected, cache = expected_stress_manifest(args)
    if manifest != expected:
        raise RuntimeError("Replication-stress manifest changed before execution.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Replication stress requires a BF16-capable CUDA device.")
    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID,
        revision=NATIVE_C0_MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    codebook = build_action_token_codebook(processor.tokenizer)
    entries, identities = _cache_maps(cache, args.cache_object_root.resolve())
    samples = list(cache["manifest"]["samples"])
    by_index = {int(row["sample_index"]): row for row in samples if row.get("split") == "train"}
    sample_indices = list(manifest["stress_sample_indices"])
    events = []
    for config in manifest["configurations"]:
        if time.perf_counter() - started >= HARD_MAX_SECONDS:
            raise TimeoutError("Replication stress exceeded its 30-minute cap.")
        seed = int(config["replication_seed"])
        torch.cuda.reset_peak_memory_stats(device)
        model_started = time.perf_counter()
        model = _load_model(device, seed)
        model_load_seconds = time.perf_counter() - model_started
        validate_lora_only_trainables(model)
        initial_sha = tensor_state_sha256(trainable_state_dict(model))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2e-5,
            weight_decay=0.01,
        )
        optimizer.zero_grad(set_to_none=True)
        before = trainable_state_dict(model)
        microbatches = []
        weights = config["replication_contract"].get("treatment_weights")
        for sample_index in sample_indices:
            sample = by_index[sample_index]
            cached, entry = _load_cache(sample, entries, identities)
            batch, position = _native_batch(
                processor=processor,
                codebook=codebook,
                cached=cached,
                sample=sample,
                arm_id="H4_CAUSAL",
            )
            inputs = _move_batch(batch, device)
            inputs.pop("labels")
            outputs = model(**inputs)
            target = R2RCEAction(int(sample["target_action_id"]))
            loss = _restricted_loss(
                outputs.logits,
                action_position=position,
                codebook=codebook,
                target_action=target,
            )
            unweighted = float(loss.detach().cpu().item())
            weight = 1.0 if weights is None else float(weights[target.name])
            optimized = loss * weight
            value = float(optimized.detach().cpu().item())
            if not math.isfinite(value) or value < 0.0 or value > 1_000.0:
                raise RuntimeError("Replication stress produced invalid optimized CE.")
            (optimized / len(sample_indices)).backward()
            microbatches.append(
                {
                    "sample_index": sample_index,
                    "target_action": target.name,
                    "unweighted_restricted_action_ce": unweighted,
                    "target_action_weight": weight,
                    "optimized_restricted_action_ce": value,
                    "cache_key": entry["cache_key"],
                    "cache_sha256": entry["sha256"],
                    "supervised_token_count": int(batch.record["supervised_token_count"]),
                }
            )
            del outputs, inputs, batch, loss, optimized
        gradients = gradient_report(model)
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            .detach()
            .float()
            .cpu()
            .item()
        )
        gradients_after_clip = gradient_report(model)
        optimizer.step()
        updates = update_report(model, before)
        validate_optimization_reports(gradients, updates)
        final_sha = tensor_state_sha256(trainable_state_dict(model))
        if (
            final_sha == initial_sha
            or not math.isfinite(preclip)
            or preclip <= 0.0
            or gradients_after_clip["all_gradients_finite"] is not True
            or float(gradients_after_clip["gradient_l2_norm"]) > 1.0002
        ):
            raise RuntimeError("Replication stress did not produce a valid trainable update.")
        events.append(
            {
                **config,
                "initial_trainable_state_sha256": initial_sha,
                "final_trainable_state_sha256": final_sha,
                "gradient_report": gradients,
                "gradient_report_after_clip": gradients_after_clip,
                "global_gradient_norm_before_clip": preclip,
                "update_report": updates,
                "microbatches": microbatches,
                "model_load_seconds": model_load_seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        del optimizer, model, before
        gc.collect()
        torch.cuda.empty_cache()
    for seed in NEW_REPLICATION_SEEDS:
        paired = [row for row in events if row["replication_seed"] == seed]
        if len(paired) != 2 or len({row["initial_trainable_state_sha256"] for row in paired}) != 1:
            raise RuntimeError("Paired treatments do not share the same seeded initialization.")
    result = {
        "schema_version": 1,
        "gate": NATIVE_C0_SEED_REPLICATION_STRESS_GATE,
        "run_id": manifest["run_id"],
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "configuration_count": len(events),
        "events": events,
        "events_sha256": canonical_sha256(events),
        "elapsed_seconds": time.perf_counter() - started,
        "hard_max_seconds": HARD_MAX_SECONDS,
        "completed_at_utc": _utc_now(),
        "full_replication_authorized": True,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    _atomic_json(args.output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--unweighted-root", type=Path, required=True)
    parser.add_argument("--inverse-frequency-root", type=Path, required=True)
    parser.add_argument("--replication-stress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--failure-analysis", type=Path)
    parser.add_argument("--failure-verification", type=Path)
    parser.add_argument("--stress-root", type=Path)


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
            "schema_version": 1,
            "gate": NATIVE_C0_SEED_REPLICATION_STRESS_GATE,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            args.output_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(args.output_root / "failure.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
