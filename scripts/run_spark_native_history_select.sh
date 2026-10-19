#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_HISTORY_RUN_ID:?GRIDNAV_NATIVE_HISTORY_RUN_ID must be set}"
resume="${GRIDNAV_NATIVE_HISTORY_RESUME:-0}"
stress_only="${GRIDNAV_NATIVE_HISTORY_STRESS_ONLY:-0}"
image_ref="${GRIDNAV_NATIVE_HISTORY_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
image_id="${GRIDNAV_NATIVE_HISTORY_IMAGE_ID:?GRIDNAV_NATIVE_HISTORY_IMAGE_ID must be set}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
model_snapshot="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"

python -m pytest \
  -p no:cacheprovider \
  -m 'not slow' \
  tests/test_action_decision.py \
  tests/test_datasets.py \
  tests/test_habitat_sim_r2r.py \
  tests/test_history.py \
  tests/test_native_qwen.py \
  tests/test_native_qwen_artifacts.py \
  tests/test_native_c0_gate.py \
  tests/test_native_c0_artifacts.py \
  tests/test_native_c0_artifacts_v2.py \
  tests/test_native_history_select.py \
  tests/test_native_history_select_artifacts.py \
  tests/test_qwen3_vl.py

if [[ "${resume}" == "0" ]]; then
  echo "Running the bounded real three-arm stress gate before the overnight suite."
  python scripts/run_spark_native_history_select.py prepare \
    --run-id native_history_select_stress_v1 \
    --repo /workspace/GridNav \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --docker-image-ref "${image_ref}" \
    --docker-image-id "${image_id}" \
    --planned-updates 4 \
    --train-samples-per-action 1 \
    --calibration-samples-per-action 1 \
    --checkpoint-interval 2 \
    --max-hours 0.5 \
    --output /artifacts/preflight_stress/manifest.json

  python scripts/run_spark_native_history_select.py run \
    --run-id native_history_select_stress_v1 \
    --repo /workspace/GridNav \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --docker-image-ref "${image_ref}" \
    --docker-image-id "${image_id}" \
    --planned-updates 4 \
    --train-samples-per-action 1 \
    --calibration-samples-per-action 1 \
    --checkpoint-interval 2 \
    --max-hours 0.5 \
    --manifest /artifacts/preflight_stress/manifest.json \
    --scene-root "${scene_root}" \
    --model-snapshot "${model_snapshot}" \
    --cache-root /checkpoints/preflight_stress/cache \
    --checkpoint-root /checkpoints/preflight_stress/arms \
    --output-root /artifacts/preflight_stress \
    --device cuda

  python scripts/check_native_history_select_artifacts.py \
    --repo /workspace/GridNav \
    --suite /artifacts/preflight_stress/suite.json \
    --manifest /artifacts/preflight_stress/manifest.json \
    --cache-manifest /checkpoints/preflight_stress/cache/cache_manifest.json \
    --checkpoint-root /checkpoints/preflight_stress/arms \
    --model-snapshot "${model_snapshot}" \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --output /artifacts/preflight_stress/verification.json

  if [[ "${stress_only}" == "1" ]]; then
    echo "Bounded native-history diagnostic stress PASSED; main training was intentionally not started."
    exit 0
  fi

  python scripts/run_spark_native_history_select.py prepare \
    --run-id "${GRIDNAV_NATIVE_HISTORY_RUN_ID}" \
    --repo /workspace/GridNav \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --docker-image-ref "${image_ref}" \
    --docker-image-id "${image_id}" \
    --output /artifacts/manifest.json
fi

resume_args=()
if [[ "${resume}" == "1" ]]; then
  resume_args+=(--resume)
fi

python scripts/run_spark_native_history_select.py run \
  --run-id "${GRIDNAV_NATIVE_HISTORY_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --docker-image-ref "${image_ref}" \
  --docker-image-id "${image_id}" \
  --manifest /artifacts/manifest.json \
  --scene-root "${scene_root}" \
  --model-snapshot "${model_snapshot}" \
  --cache-root /checkpoints/cache \
  --checkpoint-root /checkpoints/arms \
  --output-root /artifacts \
  --device cuda \
  "${resume_args[@]}"

python scripts/check_native_history_select_artifacts.py \
  --repo /workspace/GridNav \
  --suite /artifacts/suite.json \
  --manifest /artifacts/manifest.json \
  --cache-manifest /checkpoints/cache/cache_manifest.json \
  --checkpoint-root /checkpoints/arms \
  --model-snapshot "${model_snapshot}" \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --output /artifacts/verification.json

echo "Native-Qwen train-only H0/H1/H4 suite PASSED independent verification."
