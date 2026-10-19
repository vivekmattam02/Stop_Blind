#!/usr/bin/env bash
set -euo pipefail

: "${GRIDNAV_STABILITY_PILOT_RUN_ID:?GRIDNAV_STABILITY_PILOT_RUN_ID must be set}"

episodes=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
ground_truth=/datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
scene_root=/datasets/GridNav/scene_datasets

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
  tests/test_stability_pilot_artifacts.py

python scripts/check_spark_real_batch_preflight.py \
  --output /artifacts/processor_preflight.json

python scripts/check_spark_action_decision_preflight.py \
  --output /artifacts/action_preflight.json

python scripts/run_spark_stability_pilot.py prepare \
  --run-id "${GRIDNAV_STABILITY_PILOT_RUN_ID}" \
  --repo /workspace/GridNav \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --processor-preflight /artifacts/processor_preflight.json \
  --action-preflight /artifacts/action_preflight.json \
  --output /artifacts/manifest.json

arm_specs=(
  "lr_5em06 0.000005"
  "lr_1em05 0.000010"
  "lr_5em05 0.000050"
)

for specification in "${arm_specs[@]}"; do
  read -r arm_id learning_rate <<<"${specification}"
  mkdir -p "/artifacts/arms/${arm_id}"
  python scripts/run_spark_stability_pilot.py arm \
    --run-id "${GRIDNAV_STABILITY_PILOT_RUN_ID}" \
    --arm-id "${arm_id}" \
    --learning-rate "${learning_rate}" \
    --repo /workspace/GridNav \
    --manifest /artifacts/manifest.json \
    --episodes "${episodes}" \
    --ground-truth "${ground_truth}" \
    --scene-root "${scene_root}" \
    --checkpoint-root /checkpoints \
    --output "/artifacts/arms/${arm_id}/result.json"
done

summary_status=0
verification_status=0
python scripts/run_spark_stability_pilot.py summarize \
  --manifest /artifacts/manifest.json \
  --arms-root /artifacts/arms \
  --output /artifacts/suite.json || summary_status=$?

python scripts/check_stability_pilot_artifacts.py \
  --repo /workspace/GridNav \
  --suite /artifacts/suite.json \
  --manifest /artifacts/manifest.json \
  --arms-root /artifacts/arms \
  --checkpoint-root /checkpoints \
  --episodes "${episodes}" \
  --ground-truth "${ground_truth}" \
  --output /artifacts/verification.json || verification_status=$?

if [[ "${summary_status}" -ne 0 || "${verification_status}" -ne 0 ]]; then
  echo "Optimization/stability pilot completed with a failed scientific gate." >&2
  exit 1
fi

echo "Optimization/stability pilot PASSED."
