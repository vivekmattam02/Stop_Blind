#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_REPLICATION_RUN_ID:?replication run ID is required}"
: "${GRIDNAV_REPLICATION_TREATMENT:?replication treatment is required}"
: "${GRIDNAV_REPLICATION_SEED:?replication seed is required}"
: "${GRIDNAV_REPLICATION_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_REPLICATION_IMAGE_ID:?image ID is required}"
: "${GRIDNAV_QWEN_SNAPSHOT:?Qwen snapshot is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
measurements=/workspace/GridNav/artifacts/spark/native_development_measurements_v2
cache=/workspace/GridNav/artifacts/spark/native_c0_development_cache_v1
unweighted=/workspace/GridNav/artifacts/spark/native_c0_development_v1
weighted=/workspace/GridNav/artifacts/spark/native_c0_action_balance_v1

shared=(
  --run-id "${GRIDNAV_REPLICATION_RUN_ID}"
  --treatment "${GRIDNAV_REPLICATION_TREATMENT}"
  --replication-seed "${GRIDNAV_REPLICATION_SEED}"
  --repo /workspace/GridNav
  --docker-image-ref "${GRIDNAV_REPLICATION_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_REPLICATION_IMAGE_ID}"
  --cache-artifact-root "${cache}"
  --cache-object-root /cache_objects
  --unweighted-root "${unweighted}"
  --inverse-frequency-root "${weighted}"
  --replication-stress-root /workspace/GridNav/artifacts/spark/native_c0_seed_replication_stress_v1
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}"
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --measurements-result "${measurements}/result.json"
  --measurements-verification "${measurements}/verification.json"
  --development-root "${unweighted}"
  --failure-analysis /workspace/GridNav/artifacts/context/native_c0_failure_analysis_v1/analysis.json
  --failure-verification /workspace/GridNav/artifacts/context/native_c0_failure_analysis_v1/verification.json
  --stress-root /workspace/GridNav/artifacts/spark/native_c0_action_balance_stress_v1
)

python -m scripts.run_spark_native_c0_seed_replication freeze \
  "${shared[@]}" --output-root /artifacts
python -m scripts.run_spark_native_c0_seed_replication run \
  "${shared[@]}" --output-root /artifacts --checkpoint-root /checkpoints --device cuda
python -m scripts.check_native_c0_seed_replication_artifacts \
  "${shared[@]}" --result-root /artifacts --checkpoint-root /checkpoints \
  --output /artifacts/verification.json
