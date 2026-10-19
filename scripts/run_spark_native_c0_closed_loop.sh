#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_CLOSED_LOOP_RUN_ID:?run ID is required}"
: "${GRIDNAV_CLOSED_LOOP_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_CLOSED_LOOP_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"
: "${GRIDNAV_CLOSED_LOOP_MAX_HOURS:?per-invocation time budget is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
control_root=/workspace/GridNav/artifacts/spark/policy_ordering_v1
control_verification=/workspace/GridNav/artifacts/context/policy_ordering_v1_verification.json
stress_root=/workspace/GridNav/artifacts/spark/native_c0_closed_loop_stress_v3

python -m pytest -p no:cacheprovider \
  tests/test_r2r_ce_evaluator.py \
  tests/test_native_policy.py \
  tests/test_qwen_action_scorer.py \
  tests/test_native_closed_loop.py

set +e
python -m scripts.run_spark_native_c0_closed_loop \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --control-root "${control_root}" \
  --control-verification "${control_verification}" \
  --stress-root "${stress_root}" \
  --checkpoint-archive /checkpoint_archive \
  --source-artifact-root /workspace/GridNav/artifacts/spark \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --output-root /artifacts \
  --docker-image-ref "${GRIDNAV_CLOSED_LOOP_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_CLOSED_LOOP_IMAGE_ID}" \
  --run-id "${GRIDNAV_CLOSED_LOOP_RUN_ID}" \
  --max-hours "${GRIDNAV_CLOSED_LOOP_MAX_HOURS}" \
  --device cuda
runner_status=$?
set -e

if [[ "${runner_status}" -eq 75 ]]; then
  echo "Closed-loop suite reached its invocation budget and is safely resumable."
  exit 75
fi
if [[ "${runner_status}" -ne 0 ]]; then
  exit "${runner_status}"
fi

python -m scripts.check_native_c0_closed_loop_artifacts \
  --repo /workspace/GridNav \
  --result-root /artifacts \
  --control-root "${control_root}" \
  --control-verification "${control_verification}" \
  --stress-root "${stress_root}" \
  --checkpoint-archive /checkpoint_archive \
  --source-artifact-root /workspace/GridNav/artifacts/spark \
  --output /artifacts/verification.json
