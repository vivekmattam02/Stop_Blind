#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_C0_RUN_ID:?development run ID is required}"
: "${GRIDNAV_NATIVE_C0_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_C0_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
measurements_root=/workspace/GridNav/artifacts/spark/native_development_measurements_v2
cache_root=/workspace/GridNav/artifacts/spark/native_c0_development_cache_v1

shared_args=(
  --run-id "${GRIDNAV_NATIVE_C0_RUN_ID}"
  --repo /workspace/GridNav
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --docker-image-ref "${GRIDNAV_NATIVE_C0_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_NATIVE_C0_IMAGE_ID}"
  --measurements-result "${measurements_root}/result.json"
  --measurements-verification "${measurements_root}/verification.json"
  --cache-artifact-root "${cache_root}"
  --cache-object-root /cache_objects
)

python -m pytest -p no:cacheprovider \
  tests/test_native_c0_development.py \
  tests/test_native_c0_development_artifacts.py \
  tests/test_native_c0_development_cache_artifacts.py \
  tests/test_native_qwen.py \
  tests/test_action_decision.py \
  tests/test_unit_ledger.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_c0_development freeze \
  "${shared_args[@]}" --output-root /artifacts
python -m scripts.run_spark_native_c0_development run \
  "${shared_args[@]}" \
  --output-root /artifacts \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --checkpoint-root /checkpoints \
  --device cuda

python -m scripts.check_native_c0_development_artifacts \
  "${shared_args[@]}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --result-root /artifacts \
  --checkpoint-root /checkpoints \
  --output /artifacts/verification.json
