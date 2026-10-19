#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
image_ref=gridnav-spark-habitat:hsim0.3.3
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
cache_objects="${data_root}/checkpoints/native_c0_h4_content_cache_v1"
stress_dir="${repo}/artifacts/spark/native_c0_seed_replication_stress_v1"
deadline_epoch=1787697356

required_paths=(
  "${snapshot}/config.json"
  "${cache_objects}"
  "${repo}/artifacts/spark/native_c0_development_cache_v1/verification.json"
  "${repo}/artifacts/spark/native_c0_development_v1/verification.json"
  "${repo}/artifacts/spark/native_c0_action_balance_v1/verification.json"
  "${repo}/artifacts/spark/native_development_measurements_v2/verification.json"
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required paired-seed input is missing: ${path}" >&2
    exit 1
  fi
done

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Seed-replication Docker image identity mismatch." >&2
  exit 1
fi

check_verification() {
  python3 - "$1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("error_count") != 0:
    raise SystemExit(f"Verification is not passing: {sys.argv[1]}")
PY
}

run_container() {
  local name="$1" artifacts="$2" checkpoints="$3" command="$4"
  mkdir -p "${artifacts}" "${checkpoints}"
  docker run --rm --gpus=all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name "${name}" \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
    --mount "type=bind,src=${artifacts},dst=/artifacts" \
    --mount "type=bind,src=${checkpoints},dst=/checkpoints" \
    --mount "type=bind,src=${cache_objects},dst=/cache_objects,readonly" \
    --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
    --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
    --env HF_HOME=/cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HOME=/tmp \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env GRIDNAV_REPLICATION_IMAGE_REF="${image_ref}" \
    --env GRIDNAV_REPLICATION_IMAGE_ID="${observed_image_id}" \
    --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
    --workdir /workspace/GridNav \
    "${image_ref}" bash -lc "${command}"
}

if [[ -f "${stress_dir}/verification.json" ]]; then
  check_verification "${stress_dir}/verification.json"
  echo "Reusing independently verified paired-seed stress gate."
else
  if [[ -d "${stress_dir}" ]] && [[ -n "$(find "${stress_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Non-empty unverified stress directory must be preserved and inspected: ${stress_dir}" >&2
    exit 1
  fi
  if (( $(date +%s) >= deadline_epoch )); then
    echo "The two-hour stress-admission deadline expired; ship the single-seed paper." >&2
    exit 2
  fi
  run_container \
    gridnav-native-c0-seed-replication-stress-v1 \
    "${stress_dir}" \
    "${data_root}/checkpoints/native_c0_seed_replication_stress_v1" \
    "bash scripts/run_spark_native_c0_seed_replication_stress.sh"
  check_verification "${stress_dir}/verification.json"
fi

python3 - \
  "${stress_dir}/result.json" \
  "${stress_dir}/verification.json" \
  "${deadline_epoch}" <<'PY'
import datetime, json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
verification = json.load(open(sys.argv[2], encoding="utf-8"))
deadline = int(sys.argv[3])
for label, payload, field in (
    ("Stress completion", result, "completed_at_utc"),
    ("Stress verification", verification, "recorded_at_utc"),
):
    stamp = datetime.datetime.fromisoformat(payload[field].replace("Z", "+00:00"))
    if stamp.timestamp() >= deadline:
        raise SystemExit(f"{label} missed the frozen two-hour deadline; full runs blocked.")
PY

for seed in 20260826 20260827; do
  for treatment in unweighted inverse_frequency; do
    run_id="native_c0_seed_replication_${treatment}_s${seed}_v1"
    artifacts="${repo}/artifacts/spark/${run_id}"
    checkpoints="${data_root}/checkpoints/${run_id}"
    if [[ -f "${artifacts}/verification.json" ]]; then
      check_verification "${artifacts}/verification.json"
      echo "Reusing passed ${run_id}."
      continue
    fi
    if [[ -d "${artifacts}" ]] && [[ -n "$(find "${artifacts}" -mindepth 1 -print -quit)" ]]; then
      echo "Non-empty unverified run directory must be preserved: ${artifacts}" >&2
      exit 1
    fi
    if [[ -d "${checkpoints}" ]] && [[ -n "$(find "${checkpoints}" -mindepth 1 -print -quit)" ]]; then
      echo "Non-empty unverified checkpoint directory must be preserved: ${checkpoints}" >&2
      exit 1
    fi
    mkdir -p "${artifacts}" "${checkpoints}"
    echo "Starting ${run_id}; detailed log: ${artifacts}/run.log"
    run_container \
      "gridnav-${run_id//_/-}" \
      "${artifacts}" \
      "${checkpoints}" \
      "GRIDNAV_REPLICATION_RUN_ID=${run_id} GRIDNAV_REPLICATION_TREATMENT=${treatment} GRIDNAV_REPLICATION_SEED=${seed} bash scripts/run_spark_native_c0_seed_replication.sh" \
      >"${artifacts}/run.log" 2>&1
    check_verification "${artifacts}/verification.json"
    echo "Verified ${run_id}."
  done
done

echo "All four new paired-seed replication runs passed independent verification."
