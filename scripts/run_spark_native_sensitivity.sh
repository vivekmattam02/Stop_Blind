#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_SENSITIVITY_RUN_ID:?run ID is required}"
: "${GRIDNAV_NATIVE_SENSITIVITY_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_SENSITIVITY_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets

python -m pytest -p no:cacheprovider \
  tests/test_native_qwen.py \
  tests/test_native_qwen_artifacts.py \
  tests/test_action_decision.py \
  tests/test_native_sensitivity.py \
  tests/test_native_sensitivity_artifacts.py \
  tests/test_spark_launch_helpers.py

python scripts/run_spark_native_sensitivity.py \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --output-root /artifacts \
  --run-id "${GRIDNAV_NATIVE_SENSITIVITY_RUN_ID}" \
  --docker-image-ref "${GRIDNAV_NATIVE_SENSITIVITY_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_NATIVE_SENSITIVITY_IMAGE_ID}" \
  --device cuda

python scripts/check_native_sensitivity_artifacts.py \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --manifest /artifacts/manifest.json \
  --result /artifacts/result.json \
  --output /artifacts/verification.json
