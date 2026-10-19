#!/usr/bin/env python3
"""Aggregate all six verified native-C0 treatment/seed results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "src/spatialvlm/training/native_c0_seed_replication_analysis.py"
SPEC = importlib.util.spec_from_file_location("native_c0_seed_replication_analysis", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load native-C0 seed-replication analysis module.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SEEDS = MODULE.SEEDS
TREATMENTS = MODULE.TREATMENTS
build_analysis = MODULE.build_analysis

SOURCE_PATHS = (
    "src/spatialvlm/training/native_c0_seed_replication_analysis.py",
    "scripts/analyze_native_c0_seed_replication.py",
    "scripts/check_native_c0_seed_replication_analysis.py",
    "tests/test_native_c0_seed_replication_analysis.py",
    "docs/NATIVE_C0_SEED_REPLICATION_ANALYSIS.md",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def source_locks(repo: Path) -> dict[str, str]:
    return {relative: file_sha256(repo / relative) for relative in SOURCE_PATHS}


def run_roots(repo: Path) -> dict[int, dict[str, Path]]:
    spark = repo / "artifacts/spark"
    return {
        20260825: {
            "unweighted": spark / "native_c0_development_v1",
            "inverse_frequency": spark / "native_c0_action_balance_v1",
        },
        **{
            seed: {
                treatment: spark / f"native_c0_seed_replication_{treatment}_s{seed}_v1"
                for treatment in TREATMENTS
            }
            for seed in (20260826, 20260827)
        },
    }


def evidence_bundle(
    root: Path, *, seed: int, treatment: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {name: root / f"{name}.json" for name in ("manifest", "result", "verification")}
    paths["updates"] = root / "updates.jsonl"
    load_json(paths["manifest"], f"{seed}/{treatment} manifest")
    result = load_json(paths["result"], f"{seed}/{treatment} result")
    verification = load_json(paths["verification"], f"{seed}/{treatment} verification")
    digests = {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}
    if (
        verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("manifest_sha256") != digests["manifest_sha256"]
        or verification.get("result_sha256") != digests["result_sha256"]
        or verification.get("updates_sha256") != digests["updates_sha256"]
        or verification.get("all_checkpoint_files_fully_loaded_and_hashed") is not True
        or result.get("manifest_sha256") != digests["manifest_sha256"]
        or result.get("updates_sha256") != digests["updates_sha256"]
    ):
        raise ValueError(f"Upstream evidence is failed or hash-unlinked: {seed}/{treatment}")
    if seed == 20260825:
        expected_gate = (
            "independent_native_qwen_c0_development_verification_v1"
            if treatment == "unweighted"
            else "independent_native_qwen_c0_action_balance_verification_v1"
        )
    else:
        expected_gate = "independent_native_qwen_c0_paired_seed_replication_verification_v1"
        contract = result.get("replication_contract", {})
        if (
            verification.get("replication_seed") != seed
            or verification.get("treatment") != treatment
            or contract.get("replication_seed") != seed
            or contract.get("treatment") != treatment
        ):
            raise ValueError(f"Replication identity mismatch: {seed}/{treatment}")
    if verification.get("gate") != expected_gate:
        raise ValueError(f"Unexpected verifier gate: {seed}/{treatment}")
    return result, {
        "root": str(root),
        "run_id": str(result["run_id"]),
        "digests": digests,
        "selected_checkpoint": dict(result["selection"]["checkpoint"]),
    }


def reconstruct(repo: Path, *, recorded_at_utc: str) -> dict[str, Any]:
    roots = run_roots(repo)
    runs: dict[int, dict[str, dict[str, Any]]] = {}
    evidence: dict[str, Any] = {}
    frozen_samples = set()
    frozen_cache = set()
    for seed in SEEDS:
        runs[seed] = {}
        for treatment in TREATMENTS:
            root = roots[seed][treatment]
            result, record = evidence_bundle(root, seed=seed, treatment=treatment)
            manifest = load_json(root / "manifest.json", "manifest")
            runs[seed][treatment] = result
            evidence[f"{seed}/{treatment}"] = record
            frozen_samples.add(str(manifest["sample_records_sha256"]))
            frozen_cache.add(str(manifest["cache"]["cache_index_sha256"]))
    if len(frozen_samples) != 1 or len(frozen_cache) != 1:
        raise ValueError("Replication runs do not share one frozen sample/cache identity.")
    evidence["frozen_sample_records_sha256"] = next(iter(frozen_samples))
    evidence["frozen_cache_index_sha256"] = next(iter(frozen_cache))
    return build_analysis(
        runs=runs,
        evidence=evidence,
        source_locks=source_locks(repo),
        recorded_at_utc=recorded_at_utc,
    )


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite seed-replication analysis: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    payload = reconstruct(repo, recorded_at_utc=datetime.now(timezone.utc).isoformat())
    write_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
