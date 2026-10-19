#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_MEASURE_RUN_ID:?run ID is required}"
: "${GRIDNAV_NATIVE_MEASURE_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_MEASURE_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
processor_root=/workspace/GridNav/artifacts/spark/native_qwen_history_preflight_v5
optimizer_root=/workspace/GridNav/artifacts/spark/native_optimizer_admission_v1
prior_root=/workspace/GridNav/artifacts/spark/native_development_measurements_v1

common_args=(
  --run-id "${GRIDNAV_NATIVE_MEASURE_RUN_ID}"
  --repo /workspace/GridNav
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --docker-image-ref "${GRIDNAV_NATIVE_MEASURE_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_NATIVE_MEASURE_IMAGE_ID}"
  --prior-manifest "${prior_root}/manifest.json"
  --prior-failure "${prior_root}/failure.json"
  --processor-result "${processor_root}/result.json"
  --processor-verification "${processor_root}/verification.json"
  --optimizer-suite "${optimizer_root}/suite.json"
  --optimizer-verification "${optimizer_root}/verification.json"
  --output-root /artifacts
)

python -m pytest -p no:cacheprovider \
  tests/test_native_development_measurements.py \
  tests/test_native_development_measurements_artifacts.py \
  tests/test_unit_ledger.py \
  tests/test_native_history_select.py \
  tests/test_native_history_select_artifacts.py \
  tests/test_native_qwen.py \
  tests/test_action_decision.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_development_measurements prepare \
  "${common_args[@]}"

python -m scripts.run_spark_native_development_measurements run \
  "${common_args[@]}" \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --cache-root /checkpoints/cache \
  --checkpoint-root /checkpoints \
  --device cuda

python -m scripts.check_native_development_measurements_artifacts \
  --run-id "${GRIDNAV_NATIVE_MEASURE_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --docker-image-ref "${GRIDNAV_NATIVE_MEASURE_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_NATIVE_MEASURE_IMAGE_ID}" \
  --prior-manifest "${prior_root}/manifest.json" \
  --prior-failure "${prior_root}/failure.json" \
  --processor-result "${processor_root}/result.json" \
  --processor-verification "${processor_root}/verification.json" \
  --optimizer-suite "${optimizer_root}/suite.json" \
  --optimizer-verification "${optimizer_root}/verification.json" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --result-root /artifacts \
  --cache-root /checkpoints/cache \
  --checkpoint-root /checkpoints \
  --output /artifacts/verification.json
