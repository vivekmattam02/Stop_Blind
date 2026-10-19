"""Tests for the optimizer unit ledger (cross-review CR-022).

Pure arithmetic and policy: no model, no simulator, no checkpoint. The central case is
`test_gate_a_as_frozen_is_rejected`, which reproduces the exact defect CR-022 identified.
"""

from __future__ import annotations

import pytest

from spatialvlm.training.unit_ledger import (
    REPEATED_EXPOSURE_RESHUFFLED,
    SINGLE_PASS,
    UNIT_LEDGER_SCHEMA_VERSION,
    UnitLedger,
    UnitLedgerError,
    verify_unit_ledger_record,
)


def _ledger(**overrides) -> UnitLedger:
    """A ledger that reconciles: 64 decisions, effective batch 16, one pass -> 4 updates."""
    kwargs = {
        "unique_train_decisions": 64,
        "unique_calibration_decisions": 64,
        "microbatch_size": 4,
        "accumulation_steps": 4,
        "epochs": 1,
        "planned_optimizer_updates": 4,
        "reuse_policy": SINGLE_PASS,
        "scheduler_unit": "optimizer_update",
        "calibration_updates": (0, 2, 4),
        "checkpoint_updates": (4,),
        "sampler_seed": 20260824,
    }
    kwargs.update(overrides)
    return UnitLedger(**kwargs)


def test_reconciling_ledger_constructs() -> None:
    led = _ledger()
    assert led.effective_batch == 16
    assert led.total_sample_exposures == 64
    assert led.reconciled_optimizer_updates == 4
    assert led.supports_learning_quality_claim is True


def test_gate_a_as_frozen_is_rejected() -> None:
    """CR-022 verbatim: 64 decisions, effective batch 16, measurements at step 64.

    One pass gives four updates. Declaring 64 is the defect that must not be freezable.
    """
    with pytest.raises(UnitLedgerError) as exc:
        _ledger(planned_optimizer_updates=64, calibration_updates=(0, 8, 16, 32, 64))
    message = str(exc.value)
    assert "planned_optimizer_updates=64" in message
    assert "= 4 updates" in message


def test_repeated_exposure_reconciles_when_declared() -> None:
    """The other resolution CR-022 allows: declare the 16 passes explicitly."""
    led = _ledger(
        epochs=16,
        planned_optimizer_updates=64,
        reuse_policy=REPEATED_EXPOSURE_RESHUFFLED,
        calibration_updates=(0, 8, 16, 32, 64),
        checkpoint_updates=(32, 64),
    )
    assert led.total_sample_exposures == 1024
    assert led.reconciled_optimizer_updates == 64
    # ...but it may not be reported as a learning-quality result
    assert led.supports_learning_quality_claim is False
    assert led.to_record()["interpretation"]["reportable_as"] == "numerical_range_finder_only"


def test_repeated_exposure_must_be_declared_not_inherited() -> None:
    with pytest.raises(UnitLedgerError, match="must be preregistered explicitly"):
        _ledger(epochs=16, planned_optimizer_updates=64, calibration_updates=(0, 4))


def test_single_pass_policy_contradicting_epochs_is_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="contradicts reuse_policy"):
        _ledger(reuse_policy=REPEATED_EXPOSURE_RESHUFFLED)


def test_measurement_beyond_reachable_updates_is_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="outside the reconciled range"):
        _ledger(calibration_updates=(0, 2, 8))


def test_indivisible_microbatch_is_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="not divisible by microbatch_size"):
        _ledger(unique_train_decisions=65, microbatch_size=4, accumulation_steps=4,
                planned_optimizer_updates=4)


def test_partial_final_update_requires_drop_last() -> None:
    with pytest.raises(UnitLedgerError, match="short final update"):
        _ledger(unique_train_decisions=72, planned_optimizer_updates=4,
                drop_last_partial_update=False, calibration_updates=(0, 4))


def test_unknown_scheduler_unit_is_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="scheduler_unit"):
        _ledger(scheduler_unit="steps")


def test_non_positive_quantities_are_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="positive integer"):
        _ledger(microbatch_size=0)


def test_duplicate_or_unsorted_measurement_steps_are_rejected() -> None:
    with pytest.raises(UnitLedgerError, match="strictly increasing"):
        _ledger(calibration_updates=(0, 2, 2))


def test_record_round_trips_through_the_verifier() -> None:
    record = _ledger().to_record()
    assert record["schema_version"] == UNIT_LEDGER_SCHEMA_VERSION
    assert verify_unit_ledger_record(record) == []


def test_verifier_catches_tampered_derived_values() -> None:
    """A manifest must not be able to claim units its own numbers contradict."""
    record = _ledger().to_record()
    record["derived"]["reconciled_optimizer_updates"] = 64
    errors = verify_unit_ledger_record(record)
    assert any("reconciled_optimizer_updates" in err for err in errors)


def test_verifier_catches_tampered_declared_values() -> None:
    record = _ledger().to_record()
    record["declared"]["planned_optimizer_updates"] = 64
    errors = verify_unit_ledger_record(record)
    assert any("does not reconcile" in err for err in errors)


def test_verifier_rejects_malformed_record() -> None:
    assert verify_unit_ledger_record({"schema_version": 999}) != []
    assert verify_unit_ledger_record({"schema_version": UNIT_LEDGER_SCHEMA_VERSION}) != []
