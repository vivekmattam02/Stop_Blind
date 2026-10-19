#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID:-native_qwen_history_preflight_v5}"
image_ref="gridnav-spark:cuda13-py25.10"
qwen_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
qwen_snapshot_host="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${qwen_revision}"
qwen_snapshot_container="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${qwen_revision}"

if [[ ! "${run_id}" =~ ^native_qwen_history_preflight_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID must match native_qwen_history_preflight_vN." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi
if ! docker image inspect "${image_ref}" >/dev/null 2>&1; then
  echo "Required image ${image_ref} is missing." >&2
  exit 1
fi
image_id="$(docker image inspect --format '{{.Id}}' "${image_ref}")"
if [[ ! "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Could not resolve an immutable Docker image ID for ${image_ref}." >&2
  exit 1
fi

required_metadata=(
  chat_template.json
  config.json
  generation_config.json
  merges.txt
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)
for relative in "${required_metadata[@]}"; do
  if [[ ! -f "${qwen_snapshot_host}/${relative}" ]]; then
    echo "Frozen Qwen metadata is missing: ${qwen_snapshot_host}/${relative}" >&2
    echo "Run the versioned metadata-hydration gate before this v4 preflight." >&2
    exit 1
  fi
done

mkdir -p "${output_dir}" "${hf_cache}"

nohup docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_QWEN_HISTORY_RUN_ID="${run_id}" \
  --env GRIDNAV_DOCKER_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_DOCKER_IMAGE_ID="${image_id}" \
  --env GRIDNAV_QWEN_REVISION="${qwen_revision}" \
  --env GRIDNAV_QWEN_SNAPSHOT="${qwen_snapshot_container}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_qwen_history_preflight.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Docker image ID: ${image_id}"
echo "Expected duration: approximately 2-10 minutes; this is a processor gate, not training."
echo "Log: ${output_dir}/run.log"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
