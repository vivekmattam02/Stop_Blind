#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_MEASURE_RUN_ID:-native_development_measurements_v2}"
image_ref="${GRIDNAV_NATIVE_MEASURE_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" \
  --output "${repo}/artifacts/context/context_state.json"

if [[ "${run_id}" != "native_development_measurements_v2" ]]; then
  echo "The corrected measurement run ID is native_development_measurements_v2." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz" \
  "${snapshot}/model.safetensors.index.json" \
  "${snapshot}/preprocessor_config.json" \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5/result.json" \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5/verification.json" \
  "${repo}/artifacts/spark/native_optimizer_admission_v1/suite.json" \
  "${repo}/artifacts/spark/native_optimizer_admission_v1/verification.json" \
  "${repo}/artifacts/spark/native_development_measurements_v1/manifest.json" \
  "${repo}/artifacts/spark/native_development_measurements_v1/failure.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required development-measurement input is missing: ${path}" >&2
    exit 1
  fi
done

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Development-measurement Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
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
  --env GRIDNAV_NATIVE_MEASURE_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_MEASURE_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_MEASURE_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_development_measurements.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Purpose: corrected source-locked train inventory plus real causal-H4 timing/memory benchmark."
echo "The failed v1 cache-complete/pre-update attempt is preserved and hash-linked."
echo "This is a bounded measurement gate, not the paper training run."
echo "Expected duration: approximately 15-45 minutes; hard cap: 1 hour."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Inventory: ${output_dir}/inventory.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Checkpoint: ${checkpoint_dir}/step_0008.pt"
