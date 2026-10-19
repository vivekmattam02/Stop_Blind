#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_CLOSED_LOOP_STRESS_RUN_ID:?run ID is required}"
: "${GRIDNAV_CLOSED_LOOP_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_CLOSED_LOOP_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
control_manifest=/workspace/GridNav/artifacts/spark/policy_ordering_v1/suite_manifest.json
source_root=/workspace/GridNav/artifacts/spark/native_c0_development_v1

python -m pytest -p no:cacheprovider \
  tests/test_r2r_ce_evaluator.py \
  tests/test_native_policy.py \
  tests/test_native_c0_gate.py \
  tests/test_qwen_action_scorer.py \
  tests/test_native_closed_loop.py

timeout --signal=TERM 1800s python -m scripts.run_spark_native_c0_closed_loop_stress \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --scene-root "${scene_root}" \
  --control-manifest "${control_manifest}" \
  --source-result "${source_root}/result.json" \
  --source-verification "${source_root}/verification.json" \
  --checkpoint /selected_checkpoint.pt \
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
  --output-root /artifacts \
  --docker-image-ref "${GRIDNAV_CLOSED_LOOP_IMAGE_REF}" \
  --docker-image-id "${GRIDNAV_CLOSED_LOOP_IMAGE_ID}" \
  --run-id "${GRIDNAV_CLOSED_LOOP_STRESS_RUN_ID}" \
  --device cuda \
  --hard-max-seconds 1800

python -m scripts.check_native_c0_closed_loop_stress_artifacts \
  --repo /workspace/GridNav \
  --control-manifest "${control_manifest}" \
  --source-result "${source_root}/result.json" \
  --source-verification "${source_root}/verification.json" \
  --checkpoint /selected_checkpoint.pt \
  --manifest /artifacts/manifest.json \
  --result /artifacts/result.json \
  --timings /artifacts/timings.jsonl \
  --output /artifacts/verification.json
