#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_ACTION_BALANCE_STRESS_RUN_ID:?stress run ID is required}"
: "${GRIDNAV_ACTION_BALANCE_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_ACTION_BALANCE_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

development_root=/workspace/GridNav/artifacts/spark/native_c0_development_v1
analysis_root=/workspace/GridNav/artifacts/context/native_c0_failure_analysis_v1
cache_root=/workspace/GridNav/artifacts/spark/native_c0_development_cache_v1

shared_args=(
  --run-id "${GRIDNAV_ACTION_BALANCE_STRESS_RUN_ID}"
  --repo /workspace/GridNav
  --docker-image-ref "${GRIDNAV_ACTION_BALANCE_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_ACTION_BALANCE_IMAGE_ID}"
  --development-root "${development_root}"
  --failure-analysis "${analysis_root}/analysis.json"
  --failure-verification "${analysis_root}/verification.json"
  --cache-artifact-root "${cache_root}"
  --cache-object-root /cache_objects
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}"
)

python -m pytest -p no:cacheprovider \
  tests/test_native_c0_action_balance.py \
  tests/test_native_c0_action_balance_stress_artifacts.py \
  tests/test_native_c0_development.py \
  tests/test_native_qwen.py \
  tests/test_action_decision.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_c0_action_balance_stress freeze \
  "${shared_args[@]}" --output-root /artifacts
python -m scripts.run_spark_native_c0_action_balance_stress run \
  "${shared_args[@]}" --output-root /artifacts --device cuda

python -m scripts.check_native_c0_action_balance_stress_artifacts \
  "${shared_args[@]}" \
  --result-root /artifacts \
  --output /artifacts/verification.json

