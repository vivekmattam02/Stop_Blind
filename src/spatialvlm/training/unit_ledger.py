"""Optimizer unit ledger: the reproducibility contract required by cross-review CR-022.

Gate A froze 64 train decisions, effective batch 16, and measurements at optimizer steps
0/8/16/32/64. Those quantities do not describe one experiment. Consuming each decision once gives
`64 / 16 = 4` optimizer updates; reaching step 64 instead requires 1,024 sample exposures, or
sixteen passes over the same 64 decisions. The protocol declared neither, so arm cost, scheduler
timing, and held-out selection were all unreproducible.

This module makes that class of error impossible to freeze. A `UnitLedger` reconciles unique
decisions, exposures, microbatches, accumulation, and accepted optimizer updates, and refuses to
construct when they disagree. Every arm in an optimizer study must share one ledger, and the ledger
is emitted verbatim into the run artifact.

It deliberately depends on nothing but the standard library. It is arithmetic and policy, not
modelling, so it is fully verifiable by reading and by `tests/test_unit_ledger.py` without a GPU,
a simulator, or a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UNIT_LEDGER_SCHEMA_VERSION",
    "ReusePolicy",
    "SchedulerUnit",
    "UnitLedger",
    "UnitLedgerError",
]

UNIT_LEDGER_SCHEMA_VERSION = 1

# Reuse policies. `SINGLE_PASS` is the only one that supports a learning-quality claim; the others
# must be preregistered and reported as numerical range finders (CR-022 verdict).
ReusePolicy = str
SINGLE_PASS: ReusePolicy = "single_pass"
REPEATED_EXPOSURE_RESHUFFLED: ReusePolicy = "repeated_exposure_reshuffled"
REPEATED_EXPOSURE_FIXED_ORDER: ReusePolicy = "repeated_exposure_fixed_order"
_REUSE_POLICIES = frozenset(
    {SINGLE_PASS, REPEATED_EXPOSURE_RESHUFFLED, REPEATED_EXPOSURE_FIXED_ORDER}
)

# Scheduler unit. Warmup and decay mean different things per unit, and Gate A declared none.
SchedulerUnit = str
_SCHEDULER_UNITS = frozenset({"optimizer_update", "sample_exposure", "microbatch"})


class UnitLedgerError(ValueError):
    """Raised when declared units do not reconcile. Always fail closed; never round."""


@dataclass(frozen=True)
class UnitLedger:
    """One frozen, self-reconciling unit contract shared by every arm of a study.

    All quantities are declared explicitly rather than derived, so that a disagreement between any
    two of them is a construction-time error instead of a silent reinterpretation at analysis time.
    """

    unique_train_decisions: int
    unique_calibration_decisions: int
    microbatch_size: int
    accumulation_steps: int
    epochs: int
    planned_optimizer_updates: int
    reuse_policy: ReusePolicy
    scheduler_unit: SchedulerUnit
    calibration_updates: tuple[int, ...]
    checkpoint_updates: tuple[int, ...]
    sampler_seed: int
    drop_last_partial_update: bool = True
    notes: str = ""
    _errors: tuple[str, ...] = field(default=(), init=False, repr=False)

    # ── derived quantities ──────────────────────────────────────────────────

    @property
    def effective_batch(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    @property
    def total_sample_exposures(self) -> int:
        return self.unique_train_decisions * self.epochs

    @property
    def total_microbatches(self) -> int:
        return self.total_sample_exposures // self.microbatch_size

    @property
    def reconciled_optimizer_updates(self) -> int:
        """Updates implied by the declared units, ignoring `planned_optimizer_updates`."""
        return self.total_sample_exposures // self.effective_batch

    @property
    def supports_learning_quality_claim(self) -> bool:
        """Only a single pass over unique decisions supports a learning-quality interpretation.

        Repeated exposure conflates optimization speed with memorization of a tiny partition, so
        such a study may be reported only as a numerical range finder (CR-022 verdict).
        """
        return self.reuse_policy == SINGLE_PASS and self.epochs == 1

    # ── validation ──────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        errors = self._collect_errors()
        object.__setattr__(self, "_errors", tuple(errors))
        if errors:
            raise UnitLedgerError(
                "Unit ledger does not reconcile:\n  - " + "\n  - ".join(errors)
            )

    def _collect_errors(self) -> list[str]:
        e: list[str] = []
        positives = {
            "unique_train_decisions": self.unique_train_decisions,
            "microbatch_size": self.microbatch_size,
            "accumulation_steps": self.accumulation_steps,
            "epochs": self.epochs,
            "planned_optimizer_updates": self.planned_optimizer_updates,
        }
        for name, value in positives.items():
            if not isinstance(value, int) or value < 1:
                e.append(f"{name} must be a positive integer, got {value!r}")
        if self.unique_calibration_decisions < 0:
            e.append("unique_calibration_decisions must be non-negative")
        if e:
            return e  # arithmetic below is meaningless until these hold

        if self.reuse_policy not in _REUSE_POLICIES:
            e.append(f"reuse_policy must be one of {sorted(_REUSE_POLICIES)}")
        if self.scheduler_unit not in _SCHEDULER_UNITS:
            e.append(f"scheduler_unit must be one of {sorted(_SCHEDULER_UNITS)}")

        # The CR-022 defect itself: declared updates must equal what the units imply.
        implied = self.reconciled_optimizer_updates
        if self.planned_optimizer_updates != implied:
            e.append(
                f"planned_optimizer_updates={self.planned_optimizer_updates} but "
                f"{self.unique_train_decisions} unique decisions x {self.epochs} epoch(s) = "
                f"{self.total_sample_exposures} exposures / effective batch "
                f"{self.effective_batch} = {implied} updates. Declare more epochs, change the "
                f"batch, or reduce the planned updates — do not leave them inconsistent."
            )

        if self.total_sample_exposures % self.microbatch_size:
            e.append(
                f"{self.total_sample_exposures} exposures is not divisible by microbatch_size "
                f"{self.microbatch_size}; a partial microbatch changes the effective batch silently"
            )
        remainder = self.total_sample_exposures % self.effective_batch
        if remainder and not self.drop_last_partial_update:
            e.append(
                f"{remainder} exposures remain after {implied} full updates and "
                "drop_last_partial_update is False; a short final update is not a declared unit"
            )

        # Repeated exposure must be chosen, never inherited by arithmetic accident.
        if self.epochs > 1 and self.reuse_policy == SINGLE_PASS:
            e.append(
                f"epochs={self.epochs} contradicts reuse_policy='{SINGLE_PASS}'; repeated exposure "
                "must be preregistered explicitly and reported as a range finder"
            )
        if self.epochs == 1 and self.reuse_policy != SINGLE_PASS:
            e.append(f"epochs=1 contradicts reuse_policy='{self.reuse_policy}'")

        for label, updates in (
            ("calibration_updates", self.calibration_updates),
            ("checkpoint_updates", self.checkpoint_updates),
        ):
            if list(updates) != sorted(set(updates)):
                e.append(f"{label} must be strictly increasing with no duplicates")
            for step in updates:
                if step < 0 or step > implied:
                    e.append(
                        f"{label} contains step {step}, outside the reconciled range "
                        f"0..{implied}. This is the Gate A error: measuring at an update that "
                        f"the declared units never reach."
                    )
        return e

    # ── artifact surface ────────────────────────────────────────────────────

    def to_record(self) -> dict[str, Any]:
        """Emitted verbatim into the run manifest, and re-checked by the independent verifier."""
        return {
            "schema_version": UNIT_LEDGER_SCHEMA_VERSION,
            "declared": {
                "unique_train_decisions": self.unique_train_decisions,
                "unique_calibration_decisions": self.unique_calibration_decisions,
                "microbatch_size": self.microbatch_size,
                "accumulation_steps": self.accumulation_steps,
                "epochs": self.epochs,
                "planned_optimizer_updates": self.planned_optimizer_updates,
                "reuse_policy": self.reuse_policy,
                "scheduler_unit": self.scheduler_unit,
                "sampler_seed": self.sampler_seed,
                "drop_last_partial_update": self.drop_last_partial_update,
                "calibration_updates": list(self.calibration_updates),
                "checkpoint_updates": list(self.checkpoint_updates),
                "notes": self.notes,
            },
            "derived": {
                "effective_batch": self.effective_batch,
                "total_sample_exposures": self.total_sample_exposures,
                "total_microbatches": self.total_microbatches,
                "reconciled_optimizer_updates": self.reconciled_optimizer_updates,
            },
            "interpretation": {
                "supports_learning_quality_claim": self.supports_learning_quality_claim,
                "reportable_as": (
                    "learning_quality_study"
                    if self.supports_learning_quality_claim
                    else "numerical_range_finder_only"
                ),
            },
        }


def verify_unit_ledger_record(record: dict[str, Any]) -> list[str]:
    """Independently recompute a ledger record. Returns errors; empty means it reconciles.

    Used by the run verifier so that a manifest cannot claim units its own numbers contradict,
    even if the producing process was patched between freeze and execution.
    """
    errors: list[str] = []
    if record.get("schema_version") != UNIT_LEDGER_SCHEMA_VERSION:
        errors.append("unit ledger schema version mismatch")
        return errors
    try:
        d = record["declared"]
        reported = record["derived"]
        ledger = UnitLedger(
            unique_train_decisions=d["unique_train_decisions"],
            unique_calibration_decisions=d["unique_calibration_decisions"],
            microbatch_size=d["microbatch_size"],
            accumulation_steps=d["accumulation_steps"],
            epochs=d["epochs"],
            planned_optimizer_updates=d["planned_optimizer_updates"],
            reuse_policy=d["reuse_policy"],
            scheduler_unit=d["scheduler_unit"],
            calibration_updates=tuple(d["calibration_updates"]),
            checkpoint_updates=tuple(d["checkpoint_updates"]),
            sampler_seed=d["sampler_seed"],
            drop_last_partial_update=d["drop_last_partial_update"],
            notes=d.get("notes", ""),
        )
    except UnitLedgerError as exc:
        errors.append(f"recorded ledger does not reconcile: {exc}")
        return errors
    except (KeyError, TypeError) as exc:
        errors.append(f"unit ledger record is malformed: {exc}")
        return errors

    recomputed = ledger.to_record()["derived"]
    for key, value in recomputed.items():
        if reported.get(key) != value:
            errors.append(f"derived.{key} recorded {reported.get(key)!r}, recomputed {value!r}")
    return errors
