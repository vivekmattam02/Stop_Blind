"""Adversarial CPU tests for CR-025 sensitivity artifact verification."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

import scripts.check_native_sensitivity_artifacts as artifacts_module
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.eval.action_decision import build_action_token_codebook, score_action_token_logits
from spatialvlm.eval.native_sensitivity import (
    NATIVE_SENSITIVITY_GATE_NAME,
    NATIVE_SENSITIVITY_SCHEMA_VERSION,
    NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS,
    NativeSensitivityProtocol,
    select_native_sensitivity_samples,
    summarize_native_sensitivity,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    model_weight_locks,
)


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_text: str


@dataclass(frozen=True)
class _GroundTruth:
    actions: tuple[R2RCEAction, ...]


class _Dataset:
    split = "val_seen"
    release_manifest = None

    def __init__(self) -> None:
        self.official_episodes = tuple(
            _Episode(
                episode_id=str(index),
                trajectory_id=f"trajectory-{index}",
                scene_id=f"scene-{index % 7}",
                instruction_text=f"instruction-{index}",
            )
            for index in range(40)
        )

    def ground_truth_record(self, episode_id: str) -> _GroundTruth:
        offset = int(episode_id) % 4
        return _GroundTruth(
            actions=tuple(R2RCEAction((offset + step) % 4) for step in range(4))
        )


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100, 101, 10 + int(text[-1])]


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict[str, Path]]:
    repo = tmp_path / "repo"
    locks = {}
    for relative in NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        locks[relative] = file_sha256(path)
    episodes = tmp_path / "val_seen.json.gz"
    ground_truth = tmp_path / "val_seen_gt.json.gz"
    episodes.write_bytes(b"episodes")
    ground_truth.write_bytes(b"ground truth")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}\n', encoding="utf-8"
    )

    dataset = _Dataset()
    monkeypatch.setattr(
        artifacts_module.R2RCEDataset,
        "from_release",
        staticmethod(lambda *_args, **_kwargs: dataset),
    )
    protocol = NativeSensitivityProtocol()
    samples = [
        sample.to_record()
        for sample in select_native_sensitivity_samples(dataset, protocol)  # type: ignore[arg-type]
    ]
    manifest = {
        "schema_version": NATIVE_SENSITIVITY_SCHEMA_VERSION,
        "gate": NATIVE_SENSITIVITY_GATE_NAME,
        "run_id": "native_sensitivity_v1",
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": {
            "episodes_sha256": file_sha256(episodes),
            "ground_truth_sha256": file_sha256(ground_truth),
        },
        "samples": samples,
        "source_locks": locks,
    }
    codebook = build_action_token_codebook(_Tokenizer())
    decisions = []
    for sample in samples:
        condition_decisions = {}
        for condition in protocol.to_record()["conditions"]:
            logits = torch.zeros(1, 2, 128)
            decision = score_action_token_logits(
                logits, action_token_position=1, codebook=codebook
            )
            decision.update(
                {
                    "action_logits": {action.name: 0.0 for action in R2RCEAction},
                    "supervision_schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
                    "supervision_method": NATIVE_QWEN_SUPERVISION_METHOD,
                    "supervised_token_count": 1,
                    "supervised_token_position": 1,
                    "supervised_token_id": codebook.action_token_id(
                        R2RCEAction(sample["target_action_id"])
                    ),
                }
            )
            condition_decisions[condition] = decision
        donor = samples[sample["donor_sample_index"]]
        decisions.append(
            {
                "sample_index": sample["sample_index"],
                "episode_id": sample["episode_id"],
                "trajectory_id": sample["trajectory_id"],
                "scene_id": sample["scene_id"],
                "target_action_id": sample["target_action_id"],
                "target_action": sample["target_action"],
                "donor_sample_index": sample["donor_sample_index"],
                "donor_episode_id": donor["episode_id"],
                "donor_target_action": donor["target_action"],
                "conditions": condition_decisions,
            }
        )
    result = {
        "schema_version": NATIVE_SENSITIVITY_SCHEMA_VERSION,
        "gate": NATIVE_SENSITIVITY_GATE_NAME,
        "run_id": "native_sensitivity_v1",
        "status": "passed",
        "protocol": protocol.to_record(),
        "source_locks": locks,
        "runtime": {
            "architecture": "aarch64",
            "gpu": "NVIDIA GB10",
            "docker_image_id": (
                "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
            ),
        },
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "model_revision": NATIVE_C0_MODEL_REVISION,
            "initialization": "fresh_seeded_public_pretrained_lora",
            "optimization_updates": 0,
            "weight_locks": model_weight_locks(snapshot),
        },
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "action_codebook": codebook.to_record(),
        "decisions": decisions,
        "summary": summarize_native_sensitivity(decisions, protocol),
    }
    return result, manifest, {
        "repo": repo,
        "episodes": episodes,
        "ground_truth": ground_truth,
        "snapshot": snapshot,
    }


def _verify(result: dict, manifest: dict, paths: dict[str, Path]) -> list[str]:
    return artifacts_module.verify_native_sensitivity_result(
        result,
        manifest,
        repo=paths["repo"],
        episodes=paths["episodes"],
        ground_truth=paths["ground_truth"],
        model_snapshot=paths["snapshot"],
    )


def test_verifier_accepts_complete_sensitivity_fixture(tmp_path: Path, monkeypatch) -> None:
    result, manifest, paths = _fixture(tmp_path, monkeypatch)
    assert _verify(result, manifest, paths) == []


def test_verifier_rejects_supervision_and_summary_tamper(tmp_path: Path, monkeypatch) -> None:
    result, manifest, paths = _fixture(tmp_path, monkeypatch)
    tampered = copy.deepcopy(result)
    tampered["decisions"][0]["conditions"]["clean"]["supervised_token_count"] = 6
    tampered["summary"]["verdict"] = "sensitive_admit_optimizer_diagnostic"

    errors = _verify(tampered, manifest, paths)
    assert any("one-token objective mismatch" in error for error in errors)
    assert "sensitivity aggregate summary mismatch" in errors
