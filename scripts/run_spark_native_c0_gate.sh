#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_C0_RUN_ID:?GRIDNAV_NATIVE_C0_RUN_ID must be set}"
: "${GRIDNAV_DOCKER_IMAGE_REF:?GRIDNAV_DOCKER_IMAGE_REF must be set}"
: "${GRIDNAV_DOCKER_IMAGE_ID:?GRIDNAV_DOCKER_IMAGE_ID must be set}"
: "${GRIDNAV_QWEN_REVISION:?GRIDNAV_QWEN_REVISION must be set}"
: "${GRIDNAV_QWEN_SNAPSHOT:?GRIDNAV_QWEN_SNAPSHOT must be set}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
processor_result=/processor_parent/result.json
processor_verification=/processor_parent/verification.json
processor_reverification=/artifacts/processor_reverification.json
checkpoint=/checkpoints/trainable_step1.pt

python -m pytest \
  -p no:cacheprovider \
  -m 'not slow' \
  tests/test_datasets.py \
  tests/test_habitat_sim_r2r.py \
  tests/test_action_decision.py \
  tests/test_history.py \
  tests/test_native_qwen.py \
  tests/test_native_qwen_artifacts.py \
  tests/test_qwen3_vl.py \
  tests/test_native_c0_gate.py \
  tests/test_native_c0_artifacts.py

python scripts/check_native_qwen_history_artifacts.py \
  --repo /workspace/GridNav \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --result "${processor_result}" \
  --output "${processor_reverification}"

python scripts/run_spark_native_c0_gate.py \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --processor-result "${processor_result}" \
  --processor-verification "${processor_verification}" \
  --processor-reverification "${processor_reverification}" \
  --checkpoint "${checkpoint}" \
  --output /artifacts/result.json \
  --run-id "${GRIDNAV_NATIVE_C0_RUN_ID}" \
  --docker-image-ref "${GRIDNAV_DOCKER_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_DOCKER_IMAGE_ID}" \
  --device cuda

python scripts/check_native_c0_artifacts.py \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --processor-result "${processor_result}" \
  --processor-verification "${processor_verification}" \
  --processor-reverification "${processor_reverification}" \
  --checkpoint "${checkpoint}" \
  --result /artifacts/result.json \
  --output /artifacts/verification.json

echo "Native-Qwen C0 one-real-batch gate PASSED independent verification."
