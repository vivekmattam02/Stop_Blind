#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_ACTION_DECISION_RUN_ID:?GRIDNAV_ACTION_DECISION_RUN_ID must be set}"
: "${GRIDNAV_ACTION_DECISION_CHECKPOINT_SHA256:?checkpoint SHA-256 must be set}"

python -m pytest \
  -p no:cacheprovider \
  -m 'not slow' \
  tests/test_action_decision.py \
  tests/test_action_decision_artifacts.py \
  tests/test_tokenization.py \
  tests/test_real_batch_gate.py

python scripts/check_spark_action_decision_preflight.py \
  --output /artifacts/preflight.json

python scripts/run_spark_action_decision_gate.py \
  --run-id "${GRIDNAV_ACTION_DECISION_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --scene-root /datasets/GridNav/scene_datasets \
  --checkpoint /datasets/GridNav/checkpoints/real_batch_v2/trainable_step1.pt \
  --expected-checkpoint-sha256 "${GRIDNAV_ACTION_DECISION_CHECKPOINT_SHA256}" \
  --preflight /artifacts/preflight.json \
  --output /artifacts/result.json

python scripts/check_action_decision_artifacts.py \
  --repo /workspace/GridNav \
  --result /artifacts/result.json \
  --checkpoint /datasets/GridNav/checkpoints/real_batch_v2/trainable_step1.pt \
  --preflight /artifacts/preflight.json \
  --output /artifacts/verification.json
