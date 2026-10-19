#!/usr/bin/env bash
set -euo pipefail

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
  --repo /workspace/GridNav
  --docker-image-ref "${GRIDNAV_REPLICATION_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_REPLICATION_IMAGE_ID}"
  --cache-artifact-root "${cache}"
  --cache-object-root /cache_objects
  --unweighted-root "${unweighted}"
  --inverse-frequency-root "${weighted}"
  --replication-stress-root /artifacts
  --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}"
  --output-root /artifacts
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --measurements-result "${measurements}/result.json"
  --measurements-verification "${measurements}/verification.json"
  --development-root "${unweighted}"
  --failure-analysis /workspace/GridNav/artifacts/context/native_c0_failure_analysis_v1/analysis.json
  --failure-verification /workspace/GridNav/artifacts/context/native_c0_failure_analysis_v1/verification.json
  --stress-root /workspace/GridNav/artifacts/spark/native_c0_action_balance_stress_v1
)

python -m pytest -p no:cacheprovider \
  tests/test_native_c0_seed_replication.py \
  tests/test_native_c0_seed_replication_artifacts.py \
  tests/test_native_c0_seed_replication_launchers.py \
  tests/test_native_c0_development.py \
  tests/test_native_c0_development_artifacts.py \
  tests/test_native_c0_action_balance_full.py \
  tests/test_native_c0_action_balance_artifacts.py

python -m scripts.run_spark_native_c0_seed_replication_stress freeze "${shared[@]}"
python -m scripts.run_spark_native_c0_seed_replication_stress run "${shared[@]}" --device cuda
python -m scripts.check_native_c0_seed_replication_stress_artifacts \
  "${shared[@]}" --result-root /artifacts --output /artifacts/verification.json
