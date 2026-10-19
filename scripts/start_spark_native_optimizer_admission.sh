#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_OPT_RUN_ID:-native_optimizer_admission_v1}"
image_ref="${GRIDNAV_NATIVE_OPT_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

if [[ "${run_id}" != "native_optimizer_admission_v1" ]]; then
  echo "The frozen main run ID is native_optimizer_admission_v1." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz" \
  "${snapshot}/model.safetensors.index.json" \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5/result.json" \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5/verification.json" \
  "${repo}/artifacts/spark/native_sensitivity_v2/result.json" \
  "${repo}/artifacts/spark/native_sensitivity_v2/verification.json" \
  "${repo}/artifacts/spark/native_optimizer_stress_v1/suite.json" \
  "${repo}/artifacts/spark/native_optimizer_stress_v1/verification.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required optimizer-admission input is missing: ${path}" >&2
    exit 1
  fi
done
observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Optimizer-admission Docker image identity mismatch." >&2
  exit 1
fi
for directory in "${output_dir}" "${checkpoint_dir}"; do
  if [[ -d "${directory}" ]] && [[ -n "$(find "${directory}" -mindepth 1 -print -quit)" ]]; then
    echo "${directory} is non-empty; preserve it and do not overwrite this frozen run." >&2
    exit 1
  fi
done
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${checkpoint_dir}"
: >"${output_dir}/run.log"
nohup docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${checkpoint_dir},dst=/checkpoints" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_OPT_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_OPT_PROFILE=main \
  --env GRIDNAV_NATIVE_OPT_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_OPT_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_optimizer_admission.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Purpose: four-arm H0 native-Qwen numerical LR range finder; not paper training."
echo "Hard suite cap: 5 hours. Use the verified stress runtime for the best ETA."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Suite: ${output_dir}/suite.json"
echo "Independent verification: ${output_dir}/verification.json"
