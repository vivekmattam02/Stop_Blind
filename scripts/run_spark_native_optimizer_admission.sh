#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_OPT_RUN_ID:?run ID is required}"
: "${GRIDNAV_NATIVE_OPT_PROFILE:?profile is required}"
: "${GRIDNAV_NATIVE_OPT_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_OPT_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

if [[ "${GRIDNAV_NATIVE_OPT_PROFILE}" != "stress" && "${GRIDNAV_NATIVE_OPT_PROFILE}" != "main" ]]; then
  echo "GRIDNAV_NATIVE_OPT_PROFILE must be stress or main." >&2
  exit 2
fi

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
processor_root=/workspace/GridNav/artifacts/spark/native_qwen_history_preflight_v5
sensitivity_root=/workspace/GridNav/artifacts/spark/native_sensitivity_v2
stress_root=/workspace/GridNav/artifacts/spark/native_optimizer_stress_v1

common_args=(
  --run-id "${GRIDNAV_NATIVE_OPT_RUN_ID}"
  --profile "${GRIDNAV_NATIVE_OPT_PROFILE}"
  --repo /workspace/GridNav
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --docker-image-ref "${GRIDNAV_NATIVE_OPT_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_NATIVE_OPT_IMAGE_ID}"
  --processor-result "${processor_root}/result.json"
  --processor-verification "${processor_root}/verification.json"
  --sensitivity-result "${sensitivity_root}/result.json"
  --sensitivity-verification "${sensitivity_root}/verification.json"
)
if [[ "${GRIDNAV_NATIVE_OPT_PROFILE}" == "main" ]]; then
  common_args+=(
    --stress-suite "${stress_root}/suite.json"
    --stress-verification "${stress_root}/verification.json"
  )
fi

python -m pytest -p no:cacheprovider \
  tests/test_unit_ledger.py \
  tests/test_native_optimizer_admission.py \
  tests/test_native_optimizer_admission_artifacts.py \
  tests/test_native_qwen.py \
  tests/test_action_decision.py \
  tests/test_native_sensitivity_v2.py \
  tests/test_native_sensitivity_v2_artifacts.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_optimizer_admission prepare \
  "${common_args[@]}" \
  --output /artifacts/manifest.json

python -m scripts.run_spark_native_optimizer_admission run \
  "${common_args[@]}" \
  --manifest /artifacts/manifest.json \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --cache-root /checkpoints/cache \
  --checkpoint-root /checkpoints/arms \
  --output-root /artifacts \
  --device cuda

python -m scripts.check_native_optimizer_admission_artifacts \
  "${common_args[@]}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --result-root /artifacts \
  --cache-root /checkpoints/cache \
  --output /artifacts/verification.json
