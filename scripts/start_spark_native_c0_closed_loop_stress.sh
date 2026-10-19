#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id=native_c0_closed_loop_stress_v3
image_ref=gridnav-spark-habitat:hsim0.3.3
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
checkpoint="${data_root}/checkpoints/native_c0_development_v1/step_0160.pt"
output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" --output "${repo}/artifacts/context/context_state.json"

required=(
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz"
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz"
  "${data_root}/scene_datasets"
  "${snapshot}/config.json"
  "${snapshot}/preprocessor_config.json"
  "${repo}/artifacts/spark/policy_ordering_v1/suite_manifest.json"
  "${repo}/artifacts/spark/native_c0_development_v1/result.json"
  "${repo}/artifacts/spark/native_c0_development_v1/verification.json"
  "${checkpoint}"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required closed-loop stress input is missing: ${path}" >&2
    exit 1
  fi
done

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Closed-loop Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
  exit 1
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory is non-empty; preserve it: ${output_dir}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container already exists; inspect it before relaunching: ${container_name}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
: >"${output_dir}/run.log"
nohup docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${checkpoint},dst=/selected_checkpoint.pt,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_CLOSED_LOOP_STRESS_RUN_ID="${run_id}" \
  --env GRIDNAV_CLOSED_LOOP_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_CLOSED_LOOP_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  timeout --signal=TERM --kill-after=60s 1740s \
  bash scripts/run_spark_native_c0_closed_loop_stress.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Gate: exactly 32 sequential H4 model+Habitat decisions; no scientific metrics."
echo "Hard infrastructure ceiling: 30 minutes."
echo "The verified mean latency alone selects the later checkpoint/episode scope."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
