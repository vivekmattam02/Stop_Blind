"""Independent integrity checks for one-real-batch evidence artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from spatialvlm.training.real_batch_gate import (
    CANONICAL_ACTION_TEXT,
    REAL_BATCH_GATE_SCHEMA_VERSION,
    file_sha256,
)
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS

REAL_BATCH_SOURCE_LOCK_PATHS = (
    "scripts/check_real_batch_artifacts.py",
    "scripts/check_spark_real_batch_preflight.py",
    "scripts/run_spark_real_batch_gate.py",
    "scripts/run_spark_real_batch_gate.sh",
    "scripts/start_spark_real_batch_gate.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/backbone/rope_patch.py",
    "src/spatialvlm/config/model.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/preprocessing.py",
    "src/spatialvlm/data/tokenization.py",
    "src/spatialvlm/encoders/dinov2.py",
    "src/spatialvlm/encoders/projector.py",
    "src/spatialvlm/encoders/siglip.py",
    "src/spatialvlm/fusion/norm_matching.py",
    "src/spatialvlm/fusion/sva.py",
    "src/spatialvlm/geometry/backproject.py",
    "src/spatialvlm/geometry/gatr_wrapper.py",
    "src/spatialvlm/geometry/gridcell_rope3d.py",
    "src/spatialvlm/geometry/xformers_compat.py",
    "src/spatialvlm/model.py",
    "src/spatialvlm/training/real_batch_artifacts.py",
    "src/spatialvlm/training/real_batch_gate.py",
    "src/spatialvlm/training/sft.py",
    "src/spatialvlm/utils/camera.py",
)


def verify_real_batch_result(
    result: dict[str, Any],
    *,
    repo: Path,
    checkpoint: Path,
) -> list[str]:
    """Return integrity errors for a real-batch result and checkpoint."""
    errors: list[str] = []
    if result.get("schema_version") != REAL_BATCH_GATE_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if result.get("status") != "passed":
        errors.append("result status is not passed")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"real_batch_v[1-9][0-9]*", run_id) is None:
        errors.append("run_id is missing or invalid")

    source_locks = result.get("source_locks")
    if not isinstance(source_locks, dict) or not source_locks:
        errors.append("source locks are missing")
    else:
        for relative, expected in source_locks.items():
            path = repo / relative
            if not path.is_file():
                errors.append(f"locked source is missing: {relative}")
            elif file_sha256(path) != expected:
                errors.append(f"locked source hash mismatch: {relative}")

    checkpoint_record = result.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        errors.append("checkpoint record is missing")
    else:
        if not checkpoint.is_file():
            errors.append("checkpoint file is missing")
        else:
            if checkpoint.stat().st_size != checkpoint_record.get("bytes"):
                errors.append("checkpoint byte size mismatch")
            if file_sha256(checkpoint) != checkpoint_record.get("sha256"):
                errors.append("checkpoint SHA-256 mismatch")
        saved = checkpoint_record.get("trainable_state_sha256")
        reloaded = checkpoint_record.get("reloaded_state_sha256")
        after = result.get("optimization", {}).get("trainable_state_sha256_after")
        if not saved or saved != reloaded or saved != after:
            errors.append("saved/reloaded/post-update trainable hashes do not match")
        if checkpoint_record.get("perturbed_state_sha256") in {None, saved}:
            errors.append("checkpoint reload perturbation probe is invalid")
        if float(checkpoint_record.get("reload_target_loss_absolute_delta", float("inf"))) > 1e-5:
            errors.append("checkpoint reload loss delta exceeds tolerance")

    optimization = result.get("optimization", {})
    before = optimization.get("trainable_state_sha256_before")
    after = optimization.get("trainable_state_sha256_after")
    if not before or before == after:
        errors.append("optimizer update did not change the trainable-state hash")
    gradients = optimization.get("gradients_by_group", {})
    updates = optimization.get("updates_by_group", {})
    for group in FREEZE_GROUP_PATTERNS:
        grad = gradients.get(group, {})
        update = updates.get(group, {})
        if grad.get("trainable_tensor_count", 0) <= 0:
            errors.append(f"{group}: no trainable tensors")
        if grad.get("gradient_tensor_count") != grad.get("trainable_tensor_count"):
            errors.append(f"{group}: missing gradients")
        if not grad.get("all_gradients_finite") or grad.get("gradient_l2_norm", 0.0) <= 0.0:
            errors.append(f"{group}: invalid gradient norm")
        if not update.get("all_updates_finite") or update.get("update_l2_norm", 0.0) <= 0.0:
            errors.append(f"{group}: invalid update norm")

    decode = result.get("constrained_decode", {})
    scores = decode.get("scores", {})
    legal = {action.name for action in CANONICAL_ACTION_TEXT}
    if set(scores) != legal:
        errors.append("constrained decode does not contain exactly four legal actions")
    else:
        predicted = min(scores, key=lambda name: float(scores[name]["total_nll"]))
        if decode.get("predicted_action") != predicted:
            errors.append("recorded predicted action is not the minimum-total-NLL action")
    if decode.get("predicted_action") not in legal:
        errors.append("predicted action is not legal")

    return errors
