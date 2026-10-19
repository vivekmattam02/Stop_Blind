from __future__ import annotations

import copy
import importlib.util
import math
import random
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).parents[1]
        / "src/spatialvlm/training/native_c0_action_balance.py"
    )
    spec = importlib.util.spec_from_file_location("native_c0_action_balance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples():
    order = list(range(3_072))
    random.Random(20260825).shuffle(order)
    prefix = (
        ["MOVE_FORWARD"] * 44
        + ["TURN_LEFT"] * 10
        + ["TURN_RIGHT"] * 7
        + ["STOP"]
        + ["TURN_RIGHT"] * 2
    )
    remainder = (
        ["MOVE_FORWARD"] * 1_980
        + ["STOP"] * 70
        + ["TURN_LEFT"] * 527
        + ["TURN_RIGHT"] * 431
    )
    by_index = dict(zip(order, prefix + remainder))
    return [
        {"split": "train", "sample_index": index, "target_action": by_index[index]}
        for index in range(3_072)
    ]


def test_weights_are_positive_and_normalized_to_empirical_expectation_one() -> None:
    module = _module()
    record = module.reconstruct_class_weights(_samples())
    assert record["target_action_histogram"] == {
        "STOP": 71,
        "MOVE_FORWARD": 2_024,
        "TURN_LEFT": 537,
        "TURN_RIGHT": 440,
    }
    assert record["weights"]["MOVE_FORWARD"] == pytest.approx(0.3794466403162055)
    assert record["weights"]["STOP"] == pytest.approx(10.816901408450704)
    assert record["weights"]["TURN_LEFT"] == pytest.approx(1.4301675977653632)
    assert record["weights"]["TURN_RIGHT"] == pytest.approx(1.7454545454545454)
    assert record["empirical_expected_weight"] == pytest.approx(1.0)
    assert all(math.isfinite(value) and value > 0 for value in record["weights"].values())


def test_weight_one_control_is_exact_and_invalid_values_fail_closed() -> None:
    module = _module()
    unit = {"weights": {action: 1.0 for action in module.OFFICIAL_ACTIONS}}
    assert module.weighted_restricted_ce(
        1.25, target_action="STOP", weight_record=unit
    ) == 1.25
    with pytest.raises(ValueError, match="non-negative"):
        module.weighted_restricted_ce(-1.0, target_action="STOP", weight_record=unit)
    with pytest.raises(ValueError, match="official"):
        module.weighted_restricted_ce(1.0, target_action="JUMP", weight_record=unit)


def test_real_manifest_reconstructs_frozen_stress_prefix_when_available() -> None:
    module = _module()
    path = (
        Path(__file__).parents[1]
        / "artifacts/spark/native_c0_development_v1/manifest.json"
    )
    if not path.is_file():
        pytest.skip("compact C0 development manifest is not mirrored")
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    prefix = module.stress_prefix(manifest["samples"])
    assert prefix["target_action_histogram"] == module.EXPECTED_STRESS_HISTOGRAM
    assert prefix["first_stop_exposure_zero_based"] == 61
    assert prefix["first_stop_optimizer_update_one_based"] == 4
    assert prefix["first_stop_microbatch_zero_based"] == 13


def test_contract_requires_linked_verification_and_observed_support_failure() -> None:
    module = _module()
    manifest = {
        "gate": "native_qwen_c0_development_v1",
        "status": "frozen",
        "sample_records_sha256": "samples",
        "samples": _samples(),
    }
    analysis = {
        "gate": "native_qwen_c0_failure_analysis_v1",
        "status": "passed",
        "confirmed_descriptive_observations": {
            "balanced_missing_predicted_stop": True,
            "balanced_stop_never_argmax": True,
        },
    }
    verification = {
        "gate": "independent_native_qwen_c0_failure_analysis_verification_v1",
        "status": "passed",
        "error_count": 0,
        "analysis_sha256": "analysis",
    }
    contract = module.build_action_balance_contract(
        development_manifest=manifest,
        development_result_sha256="result",
        development_verification_sha256="development-verifier",
        failure_analysis=analysis,
        failure_analysis_sha256="analysis",
        failure_verification=verification,
        failure_verification_sha256="failure-verifier",
    )
    assert contract["status"] == "blocked_pending_numerical_stress"
    assert contract["changed_variable"] == "train_only_target_action_loss_weight"
    broken = copy.deepcopy(verification)
    broken["analysis_sha256"] = "wrong"
    with pytest.raises(ValueError, match="linked"):
        module.build_action_balance_contract(
            development_manifest=manifest,
            development_result_sha256="result",
            development_verification_sha256="development-verifier",
            failure_analysis=analysis,
            failure_analysis_sha256="analysis",
            failure_verification=broken,
            failure_verification_sha256="failure-verifier",
        )


def test_stress_verifier_reconstructs_weighting_clipping_and_hash_chain() -> None:
    module = _module()
    samples = _samples()
    prefix = module.stress_prefix(samples)
    weights = module.reconstruct_class_weights(samples)
    by_index = {row["sample_index"]: row for row in samples}
    events = []
    previous = "initial"
    for step in range(1, 5):
        start = (step - 1) * 16
        microbatches = []
        for offset, sample_index in enumerate(prefix["sample_indices"][start : start + 16]):
            target = by_index[sample_index]["target_action"]
            unweighted = 1.0 + offset / 100
            microbatches.append(
                {
                    "microbatch_index": offset,
                    "sample_index": sample_index,
                    "target_action": target,
                    "target_action_weight": weights["weights"][target],
                    "unweighted_restricted_action_ce": unweighted,
                    "weighted_restricted_action_ce": module.weighted_restricted_ce(
                        unweighted,
                        target_action=target,
                        weight_record=weights,
                    ),
                }
            )
        after = f"after-{step}"
        events.append(
            {
                "gate": module.NATIVE_C0_ACTION_BALANCE_STRESS_GATE,
                "objective": module.NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
                "update_step": step,
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + 16,
                "before_trainable_state_sha256": previous,
                "after_trainable_state_sha256": after,
                "microbatches": microbatches,
                "mean_unweighted_restricted_action_ce": sum(
                    row["unweighted_restricted_action_ce"] for row in microbatches
                )
                / 16,
                "mean_weighted_restricted_action_ce": sum(
                    row["weighted_restricted_action_ce"] for row in microbatches
                )
                / 16,
                "gradient_report_before_clip": {
                    "all_gradients_finite": True,
                    "gradient_l2_norm": 2.0,
                },
                "global_gradient_norm_before_clip": 2.0,
                "gradient_report_after_clip": {
                    "all_gradients_finite": True,
                    "gradient_l2_norm": 0.99995,
                },
                "clip_max_norm": 1.0,
                "optimizer_state": {
                    "exp_avg": {"all_finite": True},
                    "exp_avg_sq": {"all_finite": True},
                    "step": {"all_finite": True},
                },
            }
        )
        previous = after
    contract = {"numerical_stress": prefix, "weights": weights}
    assert module.verify_stress_records(
        contract=contract,
        events=events,
        samples=samples,
        initial_trainable_sha256="initial",
        final_trainable_sha256="after-4",
    ) == []
    events[3]["microbatches"][13]["weighted_restricted_action_ce"] += 1.0
    assert "weighted CE does not reconstruct" in " ".join(
        module.verify_stress_records(
            contract=contract,
            events=events,
            samples=samples,
            initial_trainable_sha256="initial",
            final_trainable_sha256="after-4",
        )
    )
