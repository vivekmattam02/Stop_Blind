"""Tests for independent action-decision artifact verification."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import (
    ACTION_DECISION_SYSTEM_PROMPT,
    build_action_token_codebook,
    score_action_token_logits,
)
from spatialvlm.eval.action_decision_artifacts import (
    ACTION_DECISION_GATE_NAME,
    ACTION_DECISION_SOURCE_LOCK_PATHS,
    verify_action_decision_result,
)
from spatialvlm.training.real_batch_gate import file_sha256


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100, 101, 10 + int(text[-1])]


def _fixture(root: Path) -> tuple[dict, Path, Path]:
    checkpoint = root / "checkpoint.pt"
    preflight = root / "preflight.json"
    checkpoint.write_bytes(b"checkpoint")
    source_locks = {}
    for relative in ACTION_DECISION_SOURCE_LOCK_PATHS:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture for {relative}\n", encoding="utf-8")
        source_locks[relative] = file_sha256(source)
    codebook = build_action_token_codebook(_Tokenizer())
    decision = score_action_token_logits(
        torch.zeros(1, 8, 32),
        action_token_position=5,
        codebook=codebook,
    )
    preflight.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "spark_action_decision_tokenizer_preflight_v1",
                "status": "passed",
                "codebook": codebook.to_record(),
                "shared_action_token_position": 5,
                "shared_context_sha256": "a" * 64,
                "candidates": {
                    action.name: {
                        "sequence_length": 8,
                        "target_start_idx": 3,
                        "supervised_token_count": 4,
                        "action_token_position": 5,
                    }
                    for action in R2RCEAction
                },
            }
        ),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "gate": ACTION_DECISION_GATE_NAME,
        "run_id": "action_decision_v1",
        "status": "passed",
        "source_locks": source_locks,
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
            "expected_sha256": file_sha256(checkpoint),
            "metadata_run_id": "real_batch_v2",
        },
        "preflight_sha256": file_sha256(preflight),
        "codebook": codebook.to_record(),
        "token_contract": {
            "action_token_position": 5,
            "target_start_idx": 3,
            "candidate_sequence_lengths": {action.name: 8 for action in R2RCEAction},
            "system_prompt": ACTION_DECISION_SYSTEM_PROMPT,
        },
        "source": {
            "released_target_action_id": 3,
            "released_target_action": "TURN_RIGHT",
        },
        "repeatability": {
            "first": decision,
            "second": decision,
            "token_nll_tolerance": 1e-6,
            "max_token_nll_absolute_delta": 0.0,
            "predicted_actions_match": True,
            "prediction_matches_released_target": False,
            "accuracy_is_pass_condition": False,
        },
    }
    return result, checkpoint, preflight


def test_independent_action_decision_checker_accepts_complete_result(tmp_path: Path) -> None:
    result, checkpoint, preflight = _fixture(tmp_path)
    assert (
        verify_action_decision_result(
            result,
            repo=tmp_path,
            checkpoint=checkpoint,
            preflight=preflight,
            expected_checkpoint_sha256=file_sha256(checkpoint),
        )
        == []
    )


def test_checker_detects_source_preflight_and_score_tampering(tmp_path: Path) -> None:
    result, checkpoint, preflight = _fixture(tmp_path)
    first_source = tmp_path / ACTION_DECISION_SOURCE_LOCK_PATHS[0]
    first_source.write_text("changed\n", encoding="utf-8")
    result["repeatability"]["first"]["predicted_action"] = "TURN_RIGHT"
    preflight.write_text("{}\n", encoding="utf-8")

    errors = verify_action_decision_result(
        result,
        repo=tmp_path,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_checkpoint_sha256=file_sha256(checkpoint),
    )
    assert any("source hash mismatch" in error for error in errors)
    assert "tokenizer preflight SHA-256 mismatch" in errors
    assert any("deterministic minimum-token-NLL" in error for error in errors)


def test_checker_fails_closed_on_malformed_repeat_scores(tmp_path: Path) -> None:
    result, checkpoint, preflight = _fixture(tmp_path)
    result["repeatability"]["second"]["scores"]["STOP"] = None

    errors = verify_action_decision_result(
        result,
        repo=tmp_path,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_checkpoint_sha256=file_sha256(checkpoint),
    )
    assert any("score must be a mapping" in error for error in errors)
    assert "repeatability token NLLs cannot be reconstructed" in errors


def test_checker_rejects_omitted_lock_and_checkpoint_identity_tamper(tmp_path: Path) -> None:
    result, checkpoint, preflight = _fixture(tmp_path)
    result["source_locks"].pop(ACTION_DECISION_SOURCE_LOCK_PATHS[-1])
    result["checkpoint"]["expected_sha256"] = "0" * 64

    errors = verify_action_decision_result(
        result,
        repo=tmp_path,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_checkpoint_sha256=file_sha256(checkpoint),
    )
    assert any("source-lock path set mismatch" in error for error in errors)
    assert "checkpoint expected SHA-256 is not the frozen real_batch_v2 identity" in errors


def test_checker_allows_instruction_dependent_absolute_position(tmp_path: Path) -> None:
    result, checkpoint, preflight = _fixture(tmp_path)
    result["token_contract"]["action_token_position"] = 7
    result["token_contract"]["target_start_idx"] = 5
    for repeat in ("first", "second"):
        for row in result["repeatability"][repeat]["scores"].values():
            row["scored_token_position"] = 7
            row["causal_logit_position"] = 6

    assert (
        verify_action_decision_result(
            result,
            repo=tmp_path,
            checkpoint=checkpoint,
            preflight=preflight,
            expected_checkpoint_sha256=file_sha256(checkpoint),
        )
        == []
    )

