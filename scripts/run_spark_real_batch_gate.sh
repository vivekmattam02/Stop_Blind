#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_REAL_BATCH_RUN_ID:?GRIDNAV_REAL_BATCH_RUN_ID must be set}"

python -m pytest \
  -p no:cacheprovider \
  -m 'not slow' \
  tests/test_tokenization.py \
  tests/test_preprocessing.py \
  tests/test_camera.py \
  tests/test_encoders.py \
  tests/test_backproject.py \
  tests/test_gridcell_rope3d.py \
  tests/test_gatr_wrapper.py \
  tests/test_norm_matching.py \
  tests/test_sva.py \
  tests/test_sva_attention_probe.py \
  tests/test_qwen3_vl.py \
  tests/test_rope_patch.py \
  tests/test_collation.py \
  tests/test_model_integration.py \
  tests/test_sft.py \
  tests/test_freeze_groups.py \
  tests/test_real_batch_gate.py \
  tests/test_real_batch_artifact_check.py

python scripts/check_spark_real_batch_preflight.py \
  --output /artifacts/preflight.json

python scripts/run_spark_real_batch_gate.py \
  --run-id "${GRIDNAV_REAL_BATCH_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --scene-root /datasets/GridNav/scene_datasets \
  --checkpoint /checkpoints/trainable_step1.pt \
  --output /artifacts/result.json

python scripts/check_real_batch_artifacts.py \
  --repo /workspace/GridNav \
  --result /artifacts/result.json \
  --checkpoint /checkpoints/trainable_step1.pt \
  --output /artifacts/verification.json
