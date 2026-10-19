#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_SENSITIVITY_V2_RUN_ID:?run ID is required}"
: "${GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
v1_root=/workspace/GridNav/artifacts/spark/native_sensitivity_v1
processor_root=/workspace/GridNav/artifacts/spark/native_qwen_history_preflight_v5

python -m pytest -p no:cacheprovider \
  tests/test_native_qwen.py \
  tests/test_native_qwen_artifacts.py \
  tests/test_action_decision.py \
  tests/test_native_sensitivity.py \
  tests/test_native_sensitivity_artifacts.py \
  tests/test_native_sensitivity_v2.py \
  tests/test_native_sensitivity_v2_artifacts.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_sensitivity_v2 \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --processor-result "${processor_root}/result.json" \
  --processor-verification "${processor_root}/verification.json" \
  --v1-manifest "${v1_root}/manifest.json" \
  --v1-result "${v1_root}/result.json" \
  --v1-verification "${v1_root}/verification.json" \
  --output-root /artifacts \
  --run-id "${GRIDNAV_NATIVE_SENSITIVITY_V2_RUN_ID}" \
  --docker-image-ref "${GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_ID}" \
  --device cuda

python -m scripts.check_native_sensitivity_v2_artifacts \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --processor-result "${processor_root}/result.json" \
  --processor-verification "${processor_root}/verification.json" \
  --v1-manifest "${v1_root}/manifest.json" \
  --v1-result "${v1_root}/result.json" \
  --v1-verification "${v1_root}/verification.json" \
  --manifest /artifacts/manifest.json \
  --result /artifacts/result.json \
  --output /artifacts/verification.json
