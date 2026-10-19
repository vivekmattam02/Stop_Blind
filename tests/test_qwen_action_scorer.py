"""No-download tests for the model-backed native-Qwen action scorer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.eval import qwen_action_scorer as scorer_module
from spatialvlm.eval.native_policy import NativePolicyConfig, NativeQwenPolicy
from spatialvlm.eval.qwen_action_scorer import NativeQwenActionScorer


class _Tokenizer:
    eos_token_id = 2
    name_or_path = "fixture"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text.startswith("Action ID: ")
        return [100, 101, 10 + int(text[-1])]


class _ImageProcessor:
    merge_size = 2


class _Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        self.calls += 1
        image_count = sum(item.get("type") == "image" for item in messages[1]["content"])
        prefix_length = 8 + image_count
        return {
            "input_ids": torch.arange(prefix_length).unsqueeze(0),
            "attention_mask": torch.ones(1, prefix_length, dtype=torch.long),
            "pixel_values": torch.zeros(image_count, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * image_count),
        }


class _Backbone:
    def __init__(self, predicted: R2RCEAction, *, nonfinite: bool = False) -> None:
        self.predicted = predicted
        self.nonfinite = nonfinite
        self.eval_calls = 0
        self.forward_calls = 0
        self.seen_inputs = None

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, **inputs):  # noqa: ANN003
        self.forward_calls += 1
        self.seen_inputs = inputs
        sequence_length = inputs["input_ids"].shape[1]
        logits = torch.zeros(1, sequence_length, 32)
        # The four fixture action IDs are tokenizer IDs 10..13.  The probe suffix has four
        # tokens, so its action token sits at T-2 and is predicted by causal row T-3.
        logits[0, sequence_length - 3, 10 + int(self.predicted)] = 8.0
        if self.nonfinite:
            logits[0, sequence_length - 3, 0] = torch.inf
        return SimpleNamespace(logits=logits)


def _window():
    return build_history_window(
        [{"rgb": "rgb-0"}],
        [],
        contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT],
    )


def _scorer(
    predicted: R2RCEAction = R2RCEAction.TURN_LEFT,
    *,
    nonfinite: bool = False,
):
    backbone = _Backbone(predicted, nonfinite=nonfinite)
    processor = _Processor()
    scorer = NativeQwenActionScorer(
        backbone=backbone,  # type: ignore[arg-type]
        processor=processor,
        device=torch.device("cpu"),
        checkpoint_sha256="a" * 64,
    )
    return scorer, backbone, processor


def test_scores_one_action_from_shared_causal_logits() -> None:
    scorer, backbone, processor = _scorer(R2RCEAction.TURN_RIGHT)

    actions, diagnostics = scorer.score_chunk(
        window=_window(), instruction="turn toward the doorway", chunk_length=1
    )

    assert actions == (R2RCEAction.TURN_RIGHT,)
    assert processor.calls == 1
    assert backbone.forward_calls == 1
    assert "labels" not in backbone.seen_inputs
    assert diagnostics["chunk_length"] == 1
    assert diagnostics["checkpoint_sha256"] == "a" * 64
    assert diagnostics["action_decision"]["predicted_action"] == "TURN_RIGHT"
    assert diagnostics["probe_suffix_action"] == "STOP"
    assert diagnostics["processor_seconds"] >= 0.0
    assert diagnostics["model_forward_seconds"] >= 0.0


def test_rejects_four_action_chunk_before_model_forward() -> None:
    scorer, backbone, processor = _scorer()

    with pytest.raises(ValueError, match="CR-021"):
        scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)

    assert processor.calls == 0
    assert backbone.forward_calls == 0


def test_nonfinite_action_logit_row_fails_closed() -> None:
    scorer, _, _ = _scorer(nonfinite=True)

    with pytest.raises(ValueError, match="non-finite"):
        scorer.score_chunk(window=_window(), instruction="go", chunk_length=1)


def test_checkpoint_digest_is_mandatory_provenance() -> None:
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        NativeQwenActionScorer(
            backbone=_Backbone(R2RCEAction.STOP),  # type: ignore[arg-type]
            processor=_Processor(),
            device=torch.device("cpu"),
            checkpoint_sha256="not-a-digest",
        )


def test_history_contract_is_recorded() -> None:
    scorer, _, _ = _scorer()
    _, diagnostics = scorer.score_chunk(window=_window(), instruction="go", chunk_length=1)
    assert diagnostics["history_contract_id"] == "r2r_ce_h0_history_v1"


def test_scorer_satisfies_single_step_native_policy_seam() -> None:
    scorer, backbone, _ = _scorer(R2RCEAction.MOVE_FORWARD)
    policy = NativeQwenPolicy(
        scorer=scorer,
        config=NativePolicyConfig(
            history_contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT],
            chunk_length=1,
            step_cap=10,
        ),
    )
    episode = SimpleNamespace(episode_id="fixture-episode", instruction_text="go forward")

    policy.reset(episode, ground_truth={"must_not_be_read": True})
    action = policy.act({"rgb": "rgb-0"})

    assert action == R2RCEAction.MOVE_FORWARD
    assert backbone.forward_calls == 1
    record = policy.episode_record()
    assert record["steps_taken"] == 1
    assert record["decode_calls"] == 1
    assert record["config"]["chunk_length"] == 1
    assert record["decisions"][0]["diagnostics"]["checkpoint_sha256"] == "a" * 64


def test_factory_forwards_the_exact_selected_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    backbone = _Backbone(R2RCEAction.STOP)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        scorer_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _Processor(),
    )
    monkeypatch.setattr(
        scorer_module,
        "Qwen3VLBackbone",
        lambda **kwargs: backbone,
    )

    def capture_loader(path: Path, **kwargs: Any) -> dict[str, str]:
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "candidate"}

    monkeypatch.setattr(scorer_module, "load_native_c0_selected_checkpoint", capture_loader)

    scorer = NativeQwenActionScorer.from_native_c0_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_checkpoint_gate="native_qwen_c0_development_v1",
        expected_manifest_sha256="b" * 64,
        expected_trainable_state_sha256="c" * 64,
        expected_update_step=160,
    )

    assert scorer.checkpoint_sha256 == checkpoint_sha256
    assert captured["path"] == checkpoint.resolve()
    assert captured["expected_checkpoint_sha256"] == checkpoint_sha256
    assert captured["expected_gate"] == "native_qwen_c0_development_v1"
    assert captured["expected_manifest_sha256"] == "b" * 64
    assert captured["expected_trainable_state_sha256"] == "c" * 64
    assert captured["expected_update_step"] == 160


def test_factory_rejects_checkpoint_hash_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    processor_calls = []
    monkeypatch.setattr(
        scorer_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="scorer checkpoint hash"):
        NativeQwenActionScorer.from_native_c0_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            expected_checkpoint_sha256="0" * 64,
            expected_checkpoint_gate="native_qwen_c0_development_v1",
            expected_manifest_sha256="b" * 64,
            expected_trainable_state_sha256="c" * 64,
            expected_update_step=160,
        )

    assert processor_calls == []
