#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID:?GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID must be set}"
: "${GRIDNAV_DOCKER_IMAGE_REF:?GRIDNAV_DOCKER_IMAGE_REF must be set}"
: "${GRIDNAV_DOCKER_IMAGE_ID:?GRIDNAV_DOCKER_IMAGE_ID must be set}"
: "${GRIDNAV_QWEN_REVISION:?GRIDNAV_QWEN_REVISION must be set}"
: "${GRIDNAV_QWEN_SNAPSHOT:?GRIDNAV_QWEN_SNAPSHOT must be set}"

python -m pytest \
  -p no:cacheprovider \
  -m 'not slow' \
  tests/test_datasets.py \
  tests/test_action_decision.py \
  tests/test_history.py \
  tests/test_native_qwen.py \
  tests/test_native_qwen_artifacts.py \
  tests/test_qwen_metadata_hydration.py \
  tests/test_spark_launch_helpers.py \
  tests/test_workshop_config.py

python scripts/check_native_qwen_history_preflight.py \
  --run-id "${GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID}" \
  --repo /workspace/GridNav \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --model-revision "${GRIDNAV_QWEN_REVISION}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --docker-image-ref "${GRIDNAV_DOCKER_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_DOCKER_IMAGE_ID}" \
  --local-files-only \
  --output /artifacts/result.json

python scripts/check_native_qwen_history_artifacts.py \
  --repo /workspace/GridNav \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --result /artifacts/result.json \
  --output /artifacts/verification.json
