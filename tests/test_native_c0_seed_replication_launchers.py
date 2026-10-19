from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_pipeline_freezes_exactly_two_new_seeds_and_two_treatments() -> None:
    text = _read("run_spark_native_c0_seed_replication_pipeline.sh")
    assert "for seed in 20260826 20260827" in text
    assert "for treatment in unweighted inverse_frequency" in text
    assert "native_c0_seed_replication_${treatment}_s${seed}_v1" in text


def test_pipeline_blocks_late_or_unverified_stress() -> None:
    text = _read("run_spark_native_c0_seed_replication_pipeline.sh")
    assert "deadline_epoch=1787697356" in text
    assert "ship the single-seed paper" in text
    assert 'check_verification "${stress_dir}/verification.json"' in text
    assert '"Stress completion", result, "completed_at_utc"' in text
    assert '"Stress verification", verification, "recorded_at_utc"' in text


def test_full_runner_receives_separate_replication_seed() -> None:
    text = _read("run_spark_native_c0_seed_replication.sh")
    assert "--replication-seed" in text
    assert "--replication-stress-root" in text
    assert "--cache-object-root /cache_objects" in text


def test_stress_runs_tests_before_gpu_execution() -> None:
    text = _read("run_spark_native_c0_seed_replication_stress.sh")
    tests = text.index("python -m pytest")
    freeze = text.index("seed_replication_stress freeze")
    run = text.index("seed_replication_stress run")
    assert tests < freeze < run
