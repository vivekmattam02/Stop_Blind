#!/usr/bin/env python3
"""Freeze and execute one preregistered paired-seed native-C0 replication run."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import run_spark_native_c0_action_balance as weighted_runner
from scripts import run_spark_native_c0_development as unweighted_runner
from scripts.run_spark_native_history_select import EXPECTED_DOCKER_IMAGE_ID, _atomic_json
from spatialvlm.training.native_c0_action_balance_full import NativeC0ActionBalanceProtocol
from spatialvlm.training.native_c0_development import NativeC0DevelopmentProtocol
from spatialvlm.training.native_c0_gate import file_sha256
from spatialvlm.training.native_c0_seed_replication import (
    NATIVE_C0_SEED_REPLICATION_GATE,
    NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
    build_replication_contract,
    replication_run_id,
)

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_SEED_REPLICATION_PROTOCOL.md",
    "docs/NATIVE_C0_DEVELOPMENT_PROTOCOL.md",
    "docs/NATIVE_C0_ACTION_BALANCE_PROTOCOL.md",
    "scripts/run_spark_native_c0_seed_replication.py",
    "scripts/check_native_c0_seed_replication_artifacts.py",
    "scripts/run_spark_native_c0_seed_replication_stress.py",
    "scripts/check_native_c0_seed_replication_stress_artifacts.py",
    "scripts/run_spark_native_c0_seed_replication.sh",
    "scripts/run_spark_native_c0_seed_replication_stress.sh",
    "scripts/run_spark_native_c0_seed_replication_pipeline.sh",
    "scripts/start_spark_native_c0_seed_replication.sh",
    "scripts/native_c0_seed_replication_sync_files.txt",
    "scripts/run_spark_native_c0_development.py",
    "scripts/run_spark_native_c0_action_balance.py",
    "scripts/check_native_c0_development_artifacts.py",
    "scripts/check_native_c0_action_balance_artifacts.py",
    "scripts/run_spark_native_c0_development_cache.py",
    "scripts/run_spark_native_development_measurements.py",
    "scripts/run_spark_native_history_select.py",
    "scripts/spark_launch_helpers.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/training/native_c0_seed_replication.py",
    "src/spatialvlm/training/native_c0_development.py",
    "src/spatialvlm/training/native_c0_action_balance.py",
    "src/spatialvlm/training/native_c0_action_balance_full.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_c0_seed_replication.py",
    "tests/test_native_c0_seed_replication_artifacts.py",
    "tests/test_native_c0_seed_replication_launchers.py",
    "tests/test_native_c0_development.py",
    "tests/test_native_c0_development_artifacts.py",
    "tests/test_native_c0_action_balance_full.py",
    "tests/test_native_c0_action_balance_artifacts.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_locks(repo: Path) -> dict[str, str]:
    """Hash every source that defines or executes the replication."""
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required replication source is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _verified_original(root: Path, label: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    verification_path = root / "verification.json"
    manifest = _load_json(manifest_path, f"{label} manifest")
    result = _load_json(result_path, f"{label} result")
    verification = _load_json(verification_path, f"{label} verification")
    manifest_sha = file_sha256(manifest_path)
    result_sha = file_sha256(result_path)
    verification_sha = file_sha256(verification_path)
    if (
        manifest.get("status") != "frozen"
        or result.get("status") not in {"passed", "not_demonstrated"}
        or result.get("manifest_sha256") != manifest_sha
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("manifest_sha256") != manifest_sha
        or verification.get("result_sha256") != result_sha
    ):
        raise RuntimeError(f"{label} does not have linked passing evidence.")
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


def _protocol(treatment: str, seed: int) -> NativeC0DevelopmentProtocol:
    cls = (
        NativeC0DevelopmentProtocol if treatment == "unweighted" else NativeC0ActionBalanceProtocol
    )
    return cls(sampler_seed=seed, model_seed=seed)


def expected_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct one replication manifest from the two immutable original runs."""
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Replication Docker image differs from the admitted image.")
    expected_run_id = replication_run_id(args.treatment, args.replication_seed)
    if args.run_id != expected_run_id:
        raise ValueError(f"Replication run ID must be {expected_run_id}.")
    unweighted = _verified_original(args.unweighted_root.resolve(), "original unweighted C0")
    weighted = _verified_original(
        args.inverse_frequency_root.resolve(), "original inverse-frequency C0"
    )
    left = unweighted["manifest"]
    right = weighted["manifest"]
    for field in ("samples", "sample_records_sha256", "partition_summary", "cache", "dataset"):
        if left.get(field) != right.get(field):
            raise RuntimeError(f"Original paired runs disagree on frozen field: {field}")
    cache_manifest_path = args.cache_artifact_root.resolve() / "manifest.json"
    cache_index_path = args.cache_artifact_root.resolve() / "cache_index.json"
    cache_result_path = args.cache_artifact_root.resolve() / "result.json"
    cache_verification_path = args.cache_artifact_root.resolve() / "verification.json"
    cache_manifest = _load_json(cache_manifest_path, "content-cache manifest")
    cache_index = _load_json(cache_index_path, "content-cache index")
    cache_result = _load_json(cache_result_path, "content-cache result")
    cache_verification = _load_json(cache_verification_path, "content-cache verification")
    if (
        cache_manifest.get("status") != "frozen"
        or cache_index.get("status") != "passed"
        or cache_result.get("status") != "passed"
        or cache_verification.get("status") != "passed"
        or cache_verification.get("error_count") != 0
        or cache_verification.get("manifest_sha256") != file_sha256(cache_manifest_path)
        or cache_verification.get("cache_index_sha256") != file_sha256(cache_index_path)
        or cache_verification.get("result_sha256") != file_sha256(cache_result_path)
    ):
        raise RuntimeError("Replication content cache lacks linked passing verification.")
    if cache_manifest.get("sample_records_sha256") != left.get("sample_records_sha256"):
        raise RuntimeError("Replication cache samples differ from the original paired runs.")
    expected_cache_hashes = {
        "manifest_sha256": file_sha256(cache_manifest_path),
        "cache_index_sha256": file_sha256(cache_index_path),
        "result_sha256": file_sha256(cache_result_path),
        "verification_sha256": file_sha256(cache_verification_path),
    }
    if any(left["cache"].get(name) != value for name, value in expected_cache_hashes.items()):
        raise RuntimeError(
            "Replication cache artifact hashes differ from the original paired runs."
        )
    weights = None
    if args.treatment == "inverse_frequency":
        weights = right["action_balance_contract"]["weights"]["weights"]
    contract = build_replication_contract(
        treatment=args.treatment,
        replication_seed=args.replication_seed,
        frozen_sample_records_sha256=str(left["sample_records_sha256"]),
        frozen_cache_manifest_sha256=file_sha256(cache_manifest_path),
        frozen_cache_index_sha256=file_sha256(cache_index_path),
        original_unweighted_result_sha256=unweighted["record"]["result_sha256"],
        original_inverse_frequency_result_sha256=weighted["record"]["result_sha256"],
        treatment_weights=weights,
    )
    replication_stress = None
    if not getattr(args, "allow_missing_replication_stress", False):
        stress_root = args.replication_stress_root.resolve()
        stress_manifest_path = stress_root / "manifest.json"
        stress_result_path = stress_root / "result.json"
        stress_verification_path = stress_root / "verification.json"
        stress_result = _load_json(stress_result_path, "replication stress result")
        stress_verification = _load_json(
            stress_verification_path, "replication stress verification"
        )
        if (
            stress_result.get("status") != "passed"
            or stress_result.get("full_replication_authorized") is not True
            or stress_verification.get("status") != "passed"
            or stress_verification.get("error_count") != 0
            or stress_verification.get("manifest_sha256") != file_sha256(stress_manifest_path)
            or stress_verification.get("result_sha256") != file_sha256(stress_result_path)
        ):
            raise RuntimeError("Full replication lacks its linked passing stress gate.")
        admitted = {
            (row.get("treatment"), row.get("replication_seed"))
            for row in stress_result.get("events", [])
        }
        if (args.treatment, args.replication_seed) not in admitted:
            raise RuntimeError("Replication stress did not admit this treatment/seed pair.")
        replication_stress = {
            "manifest_path": str(stress_manifest_path),
            "manifest_sha256": file_sha256(stress_manifest_path),
            "result_path": str(stress_result_path),
            "result_sha256": file_sha256(stress_result_path),
            "verification_path": str(stress_verification_path),
            "verification_sha256": file_sha256(stress_verification_path),
        }
    template = copy.deepcopy(left if args.treatment == "unweighted" else right)
    protocol = _protocol(args.treatment, args.replication_seed)
    template.update(
        {
            "schema_version": NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
            "replication_gate": NATIVE_C0_SEED_REPLICATION_GATE,
            "run_id": args.run_id,
            "status": "frozen",
            "protocol": protocol.to_record(),
            "replication_contract": contract,
            "original_evidence": {
                "unweighted": unweighted["record"],
                "inverse_frequency": weighted["record"],
            },
            "replication_stress_authorization": replication_stress,
            "source_locks": source_locks(args.repo.resolve()),
            "evidence_boundary": {
                "established_if_complete": (
                    "Training-seed robustness of the already adaptive paired C0 development "
                    "comparison on its unchanged train-release partitions."
                ),
                "not_established": (
                    "Independent data confirmation, val_seen or val_unseen generalization, "
                    "closed-loop navigation competence, or any C1-C4 geometry claim."
                ),
            },
        }
    )
    template["model"] = {
        **template["model"],
        "replication_seed": args.replication_seed,
        "fresh_seeded_adapter": True,
        "warm_start": False,
    }
    if args.treatment == "inverse_frequency":
        unchanged = template["action_balance_contract"]["unchanged_contract"]
        unchanged.pop("samples_and_exposure_order", None)
        unchanged["samples_and_paired_seed_exposure_order"] = True
        template["action_balance_contract"]["replication_context"] = {
            "replication_seed": args.replication_seed,
            "partition_membership_unchanged": True,
            "training_order_matches_paired_unweighted_run": True,
        }
    return template, {
        "manifest": cache_manifest,
        "index": cache_index,
        "result": cache_result,
        "verification": cache_verification,
        "record": left["cache"],
    }


def _freeze(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    expected, _ = expected_manifest(args)
    if path.exists():
        if _load_json(path, "replication manifest") != expected:
            raise RuntimeError("Existing replication manifest differs from reconstruction.")
        print(f"Reusing identical replication manifest: {path}")
        return 0
    _atomic_json(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


def _runner_context(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], Callable[[], NativeC0DevelopmentProtocol]]:
    def protocol_factory() -> NativeC0DevelopmentProtocol:
        return _protocol(args.treatment, args.replication_seed)

    module = unweighted_runner if args.treatment == "unweighted" else weighted_runner
    replacements: dict[str, Any] = {
        "NativeC0DevelopmentProtocol": protocol_factory,
    }
    if args.treatment == "unweighted":
        replacements["_expected_manifest"] = expected_manifest
        replacements["_source_locks"] = lambda repo: source_locks(Path(repo))
    else:
        replacements["NativeC0ActionBalanceProtocol"] = protocol_factory
        replacements["expected_manifest"] = expected_manifest
        replacements["source_locks"] = lambda repo: source_locks(Path(repo))
    return module, replacements, protocol_factory


def _run(args: argparse.Namespace) -> int:
    manifest_path = args.output_root.resolve() / "manifest.json"
    if _load_json(manifest_path, "replication manifest") != expected_manifest(args)[0]:
        raise RuntimeError("Replication manifest changed before execution.")
    module, replacements, _ = _runner_context(args)
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        code = int(module._run(args))
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
    if code != 0:
        return code
    result_path = args.output_root.resolve() / "result.json"
    result = _load_json(result_path, "replication result")
    manifest = _load_json(manifest_path, "replication manifest")
    result.update(
        {
            "replication_schema_version": NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
            "replication_gate": NATIVE_C0_SEED_REPLICATION_GATE,
            "replication_contract": manifest["replication_contract"],
            "original_evidence": manifest["original_evidence"],
            "replication_stress_authorization": manifest["replication_stress_authorization"],
        }
    )
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--treatment", choices=("unweighted", "inverse_frequency"), required=True)
    parser.add_argument("--replication-seed", type=int, required=True)
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
    # Retained for compatibility with the already verified core runners.
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
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, unweighted_runner._request_stop)
    signal.signal(signal.SIGINT, unweighted_runner._request_stop)
    weighted_runner.base._request_stop = unweighted_runner._request_stop
    try:
        return _freeze(args) if args.command == "freeze" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_C0_SEED_REPLICATION_SCHEMA_VERSION,
            "gate": NATIVE_C0_SEED_REPLICATION_GATE,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output = getattr(args, "output_root", None)
        if isinstance(output, Path):
            try:
                _atomic_json(output.resolve() / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
