"""Tests for deterministic native-C0 failure analysis."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "src"
    / "spatialvlm"
    / "training"
    / "native_c0_failure_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("native_c0_failure_analysis_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load the pure failure-analysis module.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NATIVE_C0_DEVELOPMENT_GATE = MODULE.NATIVE_C0_DEVELOPMENT_GATE
NATIVE_C0_DEVELOPMENT_VERIFIER_GATE = MODULE.NATIVE_C0_DEVELOPMENT_VERIFIER_GATE
OFFICIAL_ACTIONS = MODULE.OFFICIAL_ACTIONS
build_failure_analysis = MODULE.build_failure_analysis
sys.path.insert(0, str(REPO_ROOT))
from scripts import analyze_native_c0_failure as analysis_script  # noqa: E402
from scripts import check_native_c0_failure_analysis as analysis_verifier  # noqa: E402


def _decision(split: str, index: int, target: str, predicted: str) -> dict:
    probability = 0.7 if target == predicted else 0.1
    remaining = (1.0 - probability) / 3.0
    probabilities = {
        action: probability if action == target else remaining for action in OFFICIAL_ACTIONS
    }
    if predicted != target:
        probabilities = {action: 0.1 for action in OFFICIAL_ACTIONS}
        probabilities[predicted] = 0.7
    scores = {
        action: {
            "restricted_probability": probabilities[action],
            "token_nll": -math.log(probabilities[action]),
        }
        for action in OFFICIAL_ACTIONS
    }
    ordered = sorted(scores[action]["token_nll"] for action in OFFICIAL_ACTIONS)
    entropy = -sum(value * math.log(value) for value in probabilities.values())
    return {
        "split": split,
        "sample_index": index,
        "episode_id": str(index),
        "trajectory_id": f"trajectory-{split}-{index}",
        "scene_id": f"scene-{index % 2}",
        "target_action": target,
        "predicted_action": predicted,
        "correct": target == predicted,
        "target_restricted_probability": probabilities[target],
        "restricted_action_ce": -math.log(probabilities[target]),
        "action_decision": {
            "scores": scores,
            "restricted_entropy_nats": entropy,
            "top_two_margin_nats": ordered[1] - ordered[0],
        },
    }


def _inputs() -> tuple[dict, dict, dict, dict]:
    train = [
        {
            "split": "train",
            "trajectory_id": f"train-{index}",
            "target_action": action,
        }
        for index, action in enumerate((*OFFICIAL_ACTIONS, *OFFICIAL_ACTIONS))
    ]
    primary_rows = [
        _decision("primary_evaluation", index, action, action)
        for index, action in enumerate(OFFICIAL_ACTIONS)
    ]
    balanced_rows = [
        _decision(
            "balanced_diagnostic",
            index,
            action,
            "MOVE_FORWARD" if action == "STOP" else action,
        )
        for index, action in enumerate(OFFICIAL_ACTIONS)
    ]
    selection_rows = [
        _decision("selection", index, action, action)
        for index, action in enumerate(OFFICIAL_ACTIONS)
    ]
    manifest = {
        "gate": NATIVE_C0_DEVELOPMENT_GATE,
        "status": "frozen",
        "samples": train,
    }
    checkpoint = {"bytes": 123, "sha256": "c" * 64, "update_step": 160}
    result = {
        "gate": NATIVE_C0_DEVELOPMENT_GATE,
        "run_id": "native_c0_development_v1",
        "status": "not_demonstrated",
        "manifest_sha256": "a" * 64,
        "updates_sha256": "u" * 64,
        "selection": {
            "selected_update": 160,
            "checkpoint": {**checkpoint, "relative_path": "step_0160.pt"},
        },
        "calibrations": [
            {"update_step": step, "per_decision": selection_rows}
            for step in (0, 32, 64, 96, 128, 160, 192)
        ],
        "primary_evaluation": {"per_decision": primary_rows},
        "balanced_diagnostic": {"per_decision": balanced_rows},
        "existence_gate": {
            "status": "not_demonstrated_at_frozen_budget",
            "conditions": {
                "natural_upper_95_below_blind_prior_minus_margin": True,
                "balanced_upper_95_below_uniform_minus_margin": False,
                "stop_recall_at_or_above_non_mute_floor": False,
            },
        },
    }
    verification = {
        "gate": NATIVE_C0_DEVELOPMENT_VERIFIER_GATE,
        "status": "passed",
        "error_count": 0,
        "errors": [],
        "manifest_sha256": "a" * 64,
        "result_sha256": "r" * 64,
        "updates_sha256": "u" * 64,
        "dataset_reconstruction_performed": True,
        "outcome_reconstruction_performed": True,
        "all_checkpoint_files_fully_loaded_and_hashed": True,
    }
    return manifest, result, verification, checkpoint


def _build() -> dict:
    manifest, result, verification, checkpoint = _inputs()
    return build_failure_analysis(
        manifest=manifest,
        result=result,
        verification=verification,
        manifest_sha256="a" * 64,
        result_sha256="r" * 64,
        updates_sha256="u" * 64,
        verification_sha256="v" * 64,
        selected_checkpoint=checkpoint,
        source_locks={"analysis.py": "s" * 64},
        recorded_at_utc="2026-08-25T00:00:00+00:00",
    )


class NativeC0FailureAnalysisTests(unittest.TestCase):
    def test_failure_analysis_reconstructs_stop_mute_confusion(self) -> None:
        analysis = _build()
        balanced = analysis["balanced_diagnostic"]
        self.assertEqual(analysis["status"], "passed")
        self.assertEqual(
            balanced["confusion_matrix_by_target"]["STOP"]["MOVE_FORWARD"], 1
        )
        self.assertEqual(balanced["per_target_action"]["STOP"]["recall"], 0.0)
        self.assertTrue(
            analysis["confirmed_descriptive_observations"]["balanced_stop_never_argmax"]
        )
        self.assertEqual(
            analysis["training_inventory"][
                "inverse_frequency_weights_normalized_to_expected_one"
            ],
            {action: 1.0 for action in OFFICIAL_ACTIONS},
        )

    def test_failure_analysis_rejects_unlinked_upstream_verifier(self) -> None:
        manifest, result, verification, checkpoint = _inputs()
        verification["result_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "verification"):
            build_failure_analysis(
                manifest=manifest,
                result=result,
                verification=verification,
                manifest_sha256="a" * 64,
                result_sha256="r" * 64,
                updates_sha256="u" * 64,
                verification_sha256="v" * 64,
                selected_checkpoint=checkpoint,
                source_locks={"analysis.py": "s" * 64},
                recorded_at_utc="2026-08-25T00:00:00+00:00",
            )

    def test_independent_verifier_reconstructs_and_detects_tamper(self) -> None:
        manifest, result, verification, _ = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = root / "result"
            result_root.mkdir()
            checkpoint = root / "step_0160.pt"
            checkpoint.write_bytes(b"selected checkpoint fixture")
            checkpoint_record = {
                "bytes": checkpoint.stat().st_size,
                "sha256": analysis_script.file_sha256(checkpoint),
                "update_step": 160,
                "relative_path": checkpoint.name,
            }
            result["selection"]["checkpoint"] = checkpoint_record

            manifest_path = result_root / "manifest.json"
            updates_path = result_root / "updates.jsonl"
            result_path = result_root / "result.json"
            verification_path = result_root / "verification.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            updates_path.write_text("{}\n", encoding="utf-8")
            result["manifest_sha256"] = analysis_script.file_sha256(manifest_path)
            result["updates_sha256"] = analysis_script.file_sha256(updates_path)
            result_path.write_text(json.dumps(result), encoding="utf-8")
            verification["manifest_sha256"] = result["manifest_sha256"]
            verification["result_sha256"] = analysis_script.file_sha256(result_path)
            verification["updates_sha256"] = result["updates_sha256"]
            verification_path.write_text(json.dumps(verification), encoding="utf-8")

            analysis = build_failure_analysis(
                manifest=manifest,
                result=result,
                verification=verification,
                manifest_sha256=result["manifest_sha256"],
                result_sha256=verification["result_sha256"],
                updates_sha256=result["updates_sha256"],
                verification_sha256=analysis_script.file_sha256(verification_path),
                selected_checkpoint=analysis_script.selected_checkpoint_record(
                    result, checkpoint
                ),
                source_locks=analysis_script.source_locks(REPO_ROOT),
                recorded_at_utc="2026-08-25T00:00:00+00:00",
            )
            analysis_path = root / "analysis.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            args = Namespace(
                repo=REPO_ROOT,
                result_root=result_root,
                selected_checkpoint=checkpoint,
                analysis=analysis_path,
            )
            self.assertEqual(analysis_verifier.verify_analysis(args), [])

            analysis["selected_update"] = 192
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            self.assertEqual(
                analysis_verifier.verify_analysis(args),
                ["failure analysis does not independently reconstruct"],
            )


if __name__ == "__main__":
    unittest.main()
