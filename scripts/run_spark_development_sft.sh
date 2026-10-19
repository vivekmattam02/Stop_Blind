#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_DEVELOPMENT_SFT_RUN_ID:?GRIDNAV_DEVELOPMENT_SFT_RUN_ID must be set}"
resume="${GRIDNAV_DEVELOPMENT_SFT_RESUME:-0}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets
pilot_manifest=/pilot_artifacts/manifest.json
pilot_suite=/pilot_artifacts/suite.json
pilot_verification=/pilot_artifacts/verification_v2.json
warmstart=/warmstart/step_0008.pt

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
  tests/test_model_integration.py \
  tests/test_sft.py \
  tests/test_freeze_groups.py \
  tests/test_action_decision.py \
  tests/test_stability_pilot.py \
  tests/test_development_sft.py \
  tests/test_development_sft_artifacts.py

if [[ "${resume}" == "0" ]]; then
  python scripts/check_spark_real_batch_preflight.py \
    --output /artifacts/processor_preflight.json

  python scripts/check_spark_action_decision_preflight.py \
    --output /artifacts/action_preflight.json

  python scripts/run_spark_development_sft.py prepare \
    --run-id "${GRIDNAV_DEVELOPMENT_SFT_RUN_ID}" \
    --repo /workspace/GridNav \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --pilot-manifest "${pilot_manifest}" \
    --pilot-suite "${pilot_suite}" \
    --pilot-verification "${pilot_verification}" \
    --warmstart-checkpoint "${warmstart}" \
    --processor-preflight /artifacts/processor_preflight.json \
    --action-preflight /artifacts/action_preflight.json \
    --output /artifacts/manifest.json
fi

resume_args=()
if [[ "${resume}" == "1" ]]; then
  resume_args+=(--resume)
fi

python scripts/run_spark_development_sft.py train \
  --run-id "${GRIDNAV_DEVELOPMENT_SFT_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --pilot-manifest "${pilot_manifest}" \
  --pilot-suite "${pilot_suite}" \
  --pilot-verification "${pilot_verification}" \
  --warmstart-checkpoint "${warmstart}" \
  --manifest /artifacts/manifest.json \
  --scene-root "${scene_root}" \
  --cache-root /checkpoints/cache \
  --checkpoint-root /checkpoints/snapshots \
  --output-root /artifacts \
  --device cuda \
  "${resume_args[@]}"

python scripts/check_development_sft_artifacts.py \
  --repo /workspace/GridNav \
  --result /artifacts/result.json \
  --manifest /artifacts/manifest.json \
  --cache-manifest /checkpoints/cache/cache_manifest.json \
  --checkpoint-root /checkpoints/snapshots \
  --warmstart-checkpoint "${warmstart}" \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --output /artifacts/verification.json

echo "Bounded development SFT run PASSED independent verification."
