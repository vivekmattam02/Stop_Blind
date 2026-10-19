"""Independent verification for checkpoint-backed action-decision evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import (
    ACTION_DECISION_METHOD,
    ACTION_DECISION_PROTOCOL_VERSION,
    ACTION_DECISION_SYSTEM_PROMPT,
    canonical_action_target,
    validate_action_decision_record,
)
from spatialvlm.training.real_batch_artifacts import REAL_BATCH_SOURCE_LOCK_PATHS
from spatialvlm.training.real_batch_gate import file_sha256

ACTION_DECISION_GATE_SCHEMA_VERSION = 1
ACTION_DECISION_GATE_NAME = "spark_qwen_action_decision_checkpoint_probe_v1"
ACTION_DECISION_PREFLIGHT_GATE_NAME = "spark_action_decision_tokenizer_preflight_v1"
ACTION_DECISION_VERIFIER_SCHEMA_VERSION = 2
ACTION_DECISION_VERIFIER_GATE_NAME = "independent_action_decision_artifact_verification_v2"
ACTION_DECISION_CHECKPOINT_SHA256 = (
    "95140f96540a63b852b417f83dde35d44440cd5b027e1ad6cbbfbb93f4457f29"
)
ACTION_DECISION_CHECKPOINT_RUN_ID = "real_batch_v2"
ACTION_DECISION_SOURCE_LOCK_PATHS = tuple(
    dict.fromkeys(
        (
            *REAL_BATCH_SOURCE_LOCK_PATHS,
            "scripts/check_action_decision_artifacts.py",
            "scripts/check_spark_action_decision_preflight.py",
            "scripts/run_spark_action_decision_gate.py",
            "scripts/run_spark_action_decision_gate.sh",
            "scripts/start_spark_action_decision_gate.sh",
            "src/spatialvlm/eval/action_decision.py",
            "src/spatialvlm/eval/action_decision_artifacts.py",
        )
    )
)


def _validate_codebook(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("protocol_version") != ACTION_DECISION_PROTOCOL_VERSION:
        errors.append("codebook protocol version mismatch")
    if record.get("method") != ACTION_DECISION_METHOD:
        errors.append("codebook method mismatch")
    prefix = record.get("shared_prefix_token_ids")
    if (
        not isinstance(prefix, list)
        or not prefix
        or not all(isinstance(item, int) for item in prefix)
    ):
        errors.append("codebook shared prefix is invalid")
        prefix = []
    if record.get("shared_prefix_token_count") != len(prefix):
        errors.append("codebook shared-prefix count mismatch")
    if record.get("action_token_offset") != len(prefix):
        errors.append("codebook action-token offset mismatch")
    actions = record.get("actions")
    if not isinstance(actions, dict) or set(actions) != {action.name for action in R2RCEAction}:
        errors.append("codebook does not contain exactly four official actions")
        return errors
    token_ids: set[int] = set()
    for action in R2RCEAction:
        row = actions[action.name]
        if not isinstance(row, dict):
            errors.append(f"{action.name}: codebook row is not a mapping")
            continue
        if row.get("action_id") != int(action):
            errors.append(f"{action.name}: codebook action ID mismatch")
        if row.get("target_text") != canonical_action_target(action):
            errors.append(f"{action.name}: codebook target text mismatch")
        if row.get("action_specific_token_count") != 1:
            errors.append(f"{action.name}: codebook action-specific count is not one")
        target_ids = row.get("target_token_ids")
        token_id = row.get("action_token_id")
        if not isinstance(token_id, int) or token_id < 0:
            errors.append(f"{action.name}: invalid codebook action token ID")
            continue
        token_ids.add(token_id)
        if not isinstance(target_ids, list) or target_ids != [*prefix, token_id]:
            errors.append(f"{action.name}: codebook target IDs do not equal prefix plus token")
    if len(token_ids) != len(R2RCEAction):
        errors.append("codebook action token IDs are not unique")
    return errors


def verify_action_decision_result(
    result: dict[str, Any],
    *,
    repo: Path,
    checkpoint: Path,
    preflight: Path,
    expected_checkpoint_sha256: str = ACTION_DECISION_CHECKPOINT_SHA256,
) -> list[str]:
    """Return all integrity errors for one action-decision gate result."""
    errors: list[str] = []
    if result.get("schema_version") != ACTION_DECISION_GATE_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if result.get("gate") != ACTION_DECISION_GATE_NAME:
        errors.append("gate name mismatch")
    if result.get("status") != "passed":
        errors.append("result status is not passed")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"action_decision_v[1-9][0-9]*", run_id) is None:
        errors.append("run_id is missing or invalid")

    source_locks = result.get("source_locks")
    if not isinstance(source_locks, dict) or not source_locks:
        errors.append("source locks are missing")
    else:
        observed_lock_paths = set(source_locks)
        expected_lock_paths = set(ACTION_DECISION_SOURCE_LOCK_PATHS)
        if observed_lock_paths != expected_lock_paths:
            missing = sorted(expected_lock_paths - observed_lock_paths)
            unexpected = sorted(observed_lock_paths - expected_lock_paths)
            errors.append(
                f"source-lock path set mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for relative, expected in source_locks.items():
            source = repo / relative
            if not source.is_file():
                errors.append(f"locked source is missing: {relative}")
            elif file_sha256(source) != expected:
                errors.append(f"locked source hash mismatch: {relative}")

    checkpoint_record = result.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        errors.append("checkpoint record is missing")
    elif not checkpoint.is_file():
        errors.append("checkpoint file is missing")
    else:
        if checkpoint.stat().st_size != checkpoint_record.get("bytes"):
            errors.append("checkpoint byte size mismatch")
        observed_checkpoint_hash = file_sha256(checkpoint)
        if observed_checkpoint_hash != checkpoint_record.get("sha256"):
            errors.append("checkpoint SHA-256 mismatch")
        if checkpoint_record.get("expected_sha256") != expected_checkpoint_sha256:
            errors.append("checkpoint expected SHA-256 is not the frozen real_batch_v2 identity")
        if observed_checkpoint_hash != expected_checkpoint_sha256:
            errors.append("checkpoint does not match the frozen real_batch_v2 SHA-256")
        if checkpoint_record.get("metadata_run_id") != ACTION_DECISION_CHECKPOINT_RUN_ID:
            errors.append("checkpoint metadata does not identify real_batch_v2")

    preflight_record: dict[str, Any] | None = None
    if not preflight.is_file():
        errors.append("tokenizer preflight file is missing")
    else:
        if file_sha256(preflight) != result.get("preflight_sha256"):
            errors.append("tokenizer preflight SHA-256 mismatch")
        try:
            preflight_record = json.loads(preflight.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("tokenizer preflight is not valid JSON")
        else:
            if preflight_record.get("schema_version") != ACTION_DECISION_GATE_SCHEMA_VERSION:
                errors.append("tokenizer preflight schema version mismatch")
            if preflight_record.get("status") != "passed":
                errors.append("tokenizer preflight status is not passed")
            if preflight_record.get("gate") != ACTION_DECISION_PREFLIGHT_GATE_NAME:
                errors.append("tokenizer preflight gate mismatch")

    codebook = result.get("codebook")
    if not isinstance(codebook, dict):
        errors.append("codebook is missing")
        codebook = {}
    else:
        errors.extend(_validate_codebook(codebook))
    if preflight_record is not None and preflight_record.get("codebook") != codebook:
        errors.append("result codebook does not match tokenizer preflight")

    if preflight_record is not None:
        candidate_rows = preflight_record.get("candidates")
        position = preflight_record.get("shared_action_token_position")
        if not isinstance(position, int) or position < 1:
            errors.append("tokenizer preflight shared action position is invalid")
        if not isinstance(candidate_rows, dict) or set(candidate_rows) != {
            action.name for action in R2RCEAction
        }:
            errors.append("tokenizer preflight candidates are incomplete")
        else:
            for action in R2RCEAction:
                row = candidate_rows[action.name]
                if not isinstance(row, dict):
                    errors.append(f"{action.name}: tokenizer preflight row is invalid")
                    continue
                if row.get("action_token_position") != position:
                    errors.append(f"{action.name}: tokenizer preflight position mismatch")
                if not isinstance(row.get("sequence_length"), int) or row["sequence_length"] <= 0:
                    errors.append(f"{action.name}: tokenizer preflight sequence length is invalid")
                if not isinstance(row.get("target_start_idx"), int) or row["target_start_idx"] < 0:
                    errors.append(f"{action.name}: tokenizer preflight target start is invalid")
        context_hash = preflight_record.get("shared_context_sha256")
        if not isinstance(context_hash, str) or re.fullmatch(r"[0-9a-f]{64}", context_hash) is None:
            errors.append("tokenizer preflight shared-context hash is invalid")

    repeatability = result.get("repeatability")
    if not isinstance(repeatability, dict):
        errors.append("repeatability record is missing")
        return errors
    first = repeatability.get("first")
    second = repeatability.get("second")
    if not isinstance(first, dict) or not isinstance(second, dict):
        errors.append("repeat decisions are missing")
        return errors
    errors.extend(
        f"first: {error}"
        for error in validate_action_decision_record(first, codebook_record=codebook)
    )
    errors.extend(
        f"second: {error}"
        for error in validate_action_decision_record(second, codebook_record=codebook)
    )
    token_contract = result.get("token_contract")
    if not isinstance(token_contract, dict):
        errors.append("token contract is missing")
    else:
        result_position = token_contract.get("action_token_position")
        target_start = token_contract.get("target_start_idx")
        action_offset = codebook.get("action_token_offset")
        if not isinstance(result_position, int) or result_position < 1:
            errors.append("result action position is invalid")
        if not isinstance(target_start, int) or target_start < 0:
            errors.append("result target start is invalid")
        if (
            isinstance(result_position, int)
            and isinstance(target_start, int)
            and isinstance(action_offset, int)
            and result_position != target_start + action_offset
        ):
            errors.append("result action position does not match target start plus codebook offset")
        if token_contract.get("system_prompt") != ACTION_DECISION_SYSTEM_PROMPT:
            errors.append("result action-decision system prompt mismatch")
        sequence_lengths = token_contract.get("candidate_sequence_lengths")
        if not isinstance(sequence_lengths, dict) or set(sequence_lengths) != {
            action.name for action in R2RCEAction
        }:
            errors.append("result candidate sequence lengths are incomplete")
        else:
            observed_lengths = list(sequence_lengths.values())
            if not all(isinstance(length, int) and length > 0 for length in observed_lengths):
                errors.append("result candidate sequence length is invalid")
            elif len(set(observed_lengths)) != 1:
                errors.append("result candidates do not have equal sequence lengths")
            elif isinstance(result_position, int) and result_position >= observed_lengths[0]:
                errors.append("result action position lies outside the candidate sequence")
        for label, decision in (("first", first), ("second", second)):
            decision_scores = decision.get("scores")
            if isinstance(decision_scores, dict):
                positions = {
                    row.get("scored_token_position")
                    for row in decision_scores.values()
                    if isinstance(row, dict)
                }
                if positions != {result_position}:
                    errors.append(f"{label}: score position does not match token contract")
    try:
        tolerance = float(repeatability.get("token_nll_tolerance", float("nan")))
    except (TypeError, ValueError):
        tolerance = float("nan")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        errors.append("repeatability tolerance is invalid")
        tolerance = -1.0
    try:
        deltas = [
            abs(
                float(first["scores"][action.name]["token_nll"])
                - float(second["scores"][action.name]["token_nll"])
            )
            for action in R2RCEAction
        ]
    except (KeyError, TypeError, ValueError):
        errors.append("repeatability token NLLs cannot be reconstructed")
        return errors
    recomputed_max_delta = max(deltas)
    try:
        recorded_max_delta = float(repeatability.get("max_token_nll_absolute_delta", float("nan")))
    except (TypeError, ValueError):
        recorded_max_delta = float("nan")
    if (
        not math.isfinite(recorded_max_delta)
        or abs(recorded_max_delta - recomputed_max_delta) > 1e-9
    ):
        errors.append("repeatability maximum NLL delta mismatch")
    if recomputed_max_delta > tolerance:
        errors.append("repeatability token NLL drift exceeds tolerance")
    predicted_match = first.get("predicted_action") == second.get("predicted_action")
    if repeatability.get("predicted_actions_match") is not predicted_match:
        errors.append("repeatability predicted-action flag mismatch")
    if not predicted_match:
        errors.append("repeat decisions selected different actions")
    source = result.get("source")
    if not isinstance(source, dict):
        errors.append("released source record is missing")
    else:
        try:
            released_action = R2RCEAction(source.get("released_target_action_id"))
        except (TypeError, ValueError):
            errors.append("released target action ID is invalid")
        else:
            if source.get("released_target_action") != released_action.name:
                errors.append("released target action name/ID mismatch")
            expected_target_match = first.get("predicted_action") == released_action.name
            if repeatability.get("prediction_matches_released_target") is not expected_target_match:
                errors.append("released-target match diagnostic is inconsistent")
    if repeatability.get("accuracy_is_pass_condition") is not False:
        errors.append("checkpoint-probe accuracy must not be a pass condition")
    return errors
