#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_NATIVE_C0_CACHE_RUN_ID:?cache run ID is required}"
: "${GRIDNAV_NATIVE_C0_IMAGE_REF:?image reference is required}"
: "${GRIDNAV_NATIVE_C0_IMAGE_ID:?image ID is required}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
measurements_root=/workspace/GridNav/artifacts/spark/native_development_measurements_v2
data_bootstrap_manifest=/workspace/GridNav/artifacts/spark/r2r_ce_data/data_bootstrap_manifest.json

shared_args=(
  --run-id "${GRIDNAV_NATIVE_C0_CACHE_RUN_ID}"
  --repo /workspace/GridNav
  --episodes "${episodes}"
  --ground-truth "${ground_truth}"
  --docker-image-ref "${GRIDNAV_NATIVE_C0_IMAGE_REF}"
  --docker-image-id "${GRIDNAV_NATIVE_C0_IMAGE_ID}"
  --measurements-result "${measurements_root}/result.json"
  --measurements-verification "${measurements_root}/verification.json"
  --data-bootstrap-manifest "${data_bootstrap_manifest}"
)

python -m pytest -p no:cacheprovider \
  tests/test_native_c0_development.py \
  tests/test_native_c0_development_cache_artifacts.py \
  tests/test_unit_ledger.py \
  tests/test_spark_launch_helpers.py

python -m scripts.run_spark_native_c0_development_cache freeze \
  "${shared_args[@]}" --output-root /artifacts
python -m scripts.run_spark_native_c0_development_cache run \
  "${shared_args[@]}" \
  --output-root /artifacts \
  --scene-root "${scene_root}" \
  --object-root /cache_objects

python -m scripts.check_native_c0_development_cache_artifacts \
  "${shared_args[@]}" \
  --result-root /artifacts \
  --object-root /cache_objects \
  --output /artifacts/verification.json
