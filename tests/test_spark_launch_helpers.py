"""Executable regressions for the background Docker registration handshake."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).parents[1] / "scripts" / "spark_launch_helpers.sh"
REPO = HELPER.parents[1]


def _run_bash(
    script: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_waits_until_container_is_registered(tmp_path: Path) -> None:
    count = tmp_path / "count"
    verification = tmp_path / "verification.json"
    log = tmp_path / "run.log"
    result = _run_bash(
        f"""
        set -u
        docker() {{
          local current=0
          [[ -f {count!s} ]] && current=$(cat {count!s})
          current=$(( current + 1 ))
          printf '%s\n' "$current" > {count!s}
          (( current >= 3 ))
        }}
        sleep 10 &
        child=$!
        source {HELPER!s}
        wait_for_container_registration stage "$child" {verification!s} {log!s}
        status=$?
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        exit "$status"
        """
    )
    assert result.returncode == 0, result.stderr
    assert int(count.read_text()) >= 3


def test_accepts_stage_that_finished_with_verification(tmp_path: Path) -> None:
    verification = tmp_path / "verification.json"
    verification.write_text("{}\n", encoding="utf-8")
    result = _run_bash(
        f"""
        set -u
        docker() {{ return 1; }}
        (exit 0) &
        child=$!
        wait "$child"
        source {HELPER!s}
        wait_for_container_registration stage "$child" {verification!s} {tmp_path / "run.log"!s}
        """
    )
    assert result.returncode == 0, result.stderr


def test_fails_when_docker_client_exits_before_registration(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    log = tmp_path / "run.log"
    result = _run_bash(
        f"""
        set -u
        docker() {{ return 1; }}
        (exit 7) &
        child=$!
        wait "$child" 2>/dev/null || true
        source {HELPER!s}
        wait_for_container_registration stage "$child" {missing!s} {log!s}
        """
    )
    assert result.returncode != 0
    assert "exited before" in result.stderr


def test_registration_timeout_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    log = tmp_path / "run.log"
    result = _run_bash(
        f"""
        set -u
        docker() {{ return 1; }}
        sleep 10 &
        child=$!
        source {HELPER!s}
        wait_for_container_registration stage "$child" {missing!s} {log!s}
        status=$?
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        exit "$status"
        """,
        env={"GRIDNAV_CONTAINER_REGISTRATION_TIMEOUT_SECONDS": "1"},
    )
    assert result.returncode != 0
    assert "did not register within 1s" in result.stderr


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/start_spark_native_qwen_history_preflight.sh",
        "scripts/start_spark_native_sensitivity.sh",
        "scripts/start_spark_native_sensitivity_v2.sh",
        "scripts/start_spark_native_optimizer_stress.sh",
        "scripts/start_spark_native_optimizer_admission.sh",
        "scripts/start_spark_native_development_measurements.sh",
        "scripts/start_spark_native_c0_development_cache.sh",
        "scripts/start_spark_native_c0_development.sh",
    ),
)
def test_every_pipeline_stage_launcher_owns_registration(relative: str) -> None:
    source = (REPO / relative).read_text(encoding="utf-8")
    assert 'source "${repo}/scripts/spark_launch_helpers.sh"' in source
    assert source.index("pid=$!") < source.index("wait_for_container_registration")


def test_supervisor_waits_for_completion_not_registration() -> None:
    source = (REPO / "scripts/run_spark_native_optimizer_pipeline.sh").read_text(encoding="utf-8")
    wait_body = source.split("wait_for_gate() {", maxsplit=1)[1].split("}\n", maxsplit=1)[0]
    assert 'docker inspect "${container_name}"' in wait_body
    assert "never appeared" not in wait_body


def test_measurement_retry_is_v2_and_requires_preserved_v1_failure() -> None:
    source = (REPO / "scripts/start_spark_native_development_measurements.sh").read_text(
        encoding="utf-8"
    )
    assert "native_development_measurements_v2" in source
    assert 'native_development_measurements_v1/manifest.json"' in source
    assert 'native_development_measurements_v1/failure.json"' in source
    assert 'scripts/check_context_state.py"' in source


def test_c0_cache_launcher_requires_locked_matterport_release_evidence() -> None:
    launcher = (REPO / "scripts/start_spark_native_c0_development_cache.sh").read_text(
        encoding="utf-8"
    )
    runner = (REPO / "scripts/run_spark_native_c0_development_cache.sh").read_text(
        encoding="utf-8"
    )
    expected = "artifacts/spark/r2r_ce_data/data_bootstrap_manifest.json"
    assert expected in launcher
    assert expected in runner
    assert '--data-bootstrap-manifest "${data_bootstrap_manifest}"' in runner


@pytest.mark.parametrize(
    ("relative", "module"),
    (
        (
            "scripts/run_spark_native_sensitivity_v2.sh",
            "scripts.run_spark_native_sensitivity_v2",
        ),
        (
            "scripts/run_spark_native_sensitivity_v2.sh",
            "scripts.check_native_sensitivity_v2_artifacts",
        ),
        (
            "scripts/run_spark_native_optimizer_admission.sh",
            "scripts.run_spark_native_optimizer_admission",
        ),
        (
            "scripts/run_spark_native_optimizer_admission.sh",
            "scripts.check_native_optimizer_admission_artifacts",
        ),
        (
            "scripts/run_spark_native_development_measurements.sh",
            "scripts.run_spark_native_development_measurements",
        ),
        (
            "scripts/run_spark_native_development_measurements.sh",
            "scripts.check_native_development_measurements_artifacts",
        ),
        (
            "scripts/run_spark_native_c0_development_cache.sh",
            "scripts.run_spark_native_c0_development_cache",
        ),
        (
            "scripts/run_spark_native_c0_development_cache.sh",
            "scripts.check_native_c0_development_cache_artifacts",
        ),
        (
            "scripts/run_spark_native_c0_development.sh",
            "scripts.run_spark_native_c0_development",
        ),
        (
            "scripts/run_spark_native_c0_development.sh",
            "scripts.check_native_c0_development_artifacts",
        ),
    ),
)
def test_import_bearing_entrypoints_run_as_modules(relative: str, module: str) -> None:
    """Keep the repository root importable when a CLI imports ``scripts.*``."""
    source = (REPO / relative).read_text(encoding="utf-8")
    assert f"python -m {module}" in source
    assert f"python scripts/{module.rsplit('.', maxsplit=1)[-1]}.py" not in source
