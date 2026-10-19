#!/usr/bin/env python3
"""Independently verify every native-C0 paper input, value record, and output hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATE = "independent_native_qwen_c0_paper_asset_verification_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_record(repo: Path, record: dict[str, Any], label: str) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe provenance path for {label}: {relative}")
    path = repo / relative
    if not path.is_file():
        raise ValueError(f"Missing provenance file for {label}: {path}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"Provenance hash mismatch for {label}: {path}")
    return path


def _lookup(value: Any, dotted: str) -> Any:
    current = value
    for component in dotted.split("."):
        if component == "paper_values":
            continue
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"Claim-map path does not resolve: {dotted}")
        current = current[component]
    return current


def verify_paper_assets(repo: Path, provenance_path: Path) -> dict[str, Any]:
    """Verify a provenance record without importing the asset generator."""

    repo = repo.resolve()
    provenance_path = provenance_path.resolve()
    provenance = _load(provenance_path)
    errors: list[str] = []
    output_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    values: dict[str, Any] = {}
    try:
        if (
            provenance.get("schema_version") != 2
            or provenance.get("gate") != "native_qwen_c0_paper_asset_provenance_v2"
            or provenance.get("status") != "passed"
        ):
            raise ValueError("Paper provenance is not the expected passed v2 record")
        for label, record in provenance.get("sources", {}).items():
            path = _resolve_record(repo, record, f"source/{label}")
            source_hashes[label] = _sha256(path)
        values_path = _resolve_record(repo, provenance["paper_values"], "paper_values")
        values = _load(values_path)
        for label, record in provenance.get("generated_outputs", {}).items():
            path = _resolve_record(repo, record, f"output/{label}")
            output_hashes[label] = _sha256(path)

        if (
            values.get("schema_version") != 2
            or values.get("gate") != "native_qwen_c0_paper_assets_v2"
            or values.get("status") != "passed"
            or values.get("seed_replication", {}).get("seeds") != [20260825, 20260826, 20260827]
        ):
            raise ValueError("Paper value record is not the expected complete v2 record")
        counts = values["seed_replication"]["consistency_counts_out_of_three"]
        required_counts = {
            "unweighted_selected_update_160_seed_count": 3,
            "unweighted_zero_selection_stop_prediction_seed_count": 3,
            "unweighted_zero_balanced_stop_recall_seed_count": 3,
            "unweighted_natural_condition_pass_seed_count": 3,
            "inverse_frequency_natural_ce_worse_seed_count": 3,
            "inverse_frequency_balanced_ce_better_seed_count": 3,
            "inverse_frequency_balanced_accuracy_better_seed_count": 3,
            "inverse_frequency_stop_recall_better_seed_count": 3,
            "existence_gate_pass_run_count": 0,
        }
        if {key: counts.get(key) for key in required_counts} != required_counts:
            raise ValueError("Three-seed direction/count invariants changed")
        deltas = values["seed_replication"]["paired_deltas_inverse_frequency_minus_unweighted"]
        expected_means = {
            "natural_mean_ce": 0.20103423346421675,
            "balanced_mean_ce": -0.27313104791175885,
            "balanced_accuracy": 0.06510416666666667,
            "balanced_stop_recall": 0.09895833333333333,
        }
        for metric, expected in expected_means.items():
            if not math.isclose(
                float(deltas[metric]["mean"]), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Frozen paired mean changed: {metric}")
        for claim_paths in provenance.get("claim_map", {}).values():
            for dotted in claim_paths:
                _lookup(values, str(dotted))
        if len(output_hashes) != 7 or len(source_hashes) != 12:
            raise ValueError("Paper provenance source/output inventory is incomplete")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return {
        "schema_version": 2,
        "gate": GATE,
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "paper_values_sha256": provenance.get("paper_values", {}).get("sha256"),
        "source_file_count": len(source_hashes),
        "generated_output_count": len(output_hashes),
        "all_claim_map_paths_resolved": not errors,
        "all_three_preregistered_seeds_reported": (
            values.get("seed_replication", {}).get("seeds") == [20260825, 20260826, 20260827]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = verify_paper_assets(args.repo, args.provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
