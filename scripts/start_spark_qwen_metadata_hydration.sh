#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_QWEN_METADATA_RUN_ID:-qwen_metadata_hydration_v1}"
image_ref="gridnav-spark:cuda13-py25.10"
qwen_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

if [[ "${run_id}" != "qwen_metadata_hydration_v1" ]]; then
  echo "GRIDNAV_QWEN_METADATA_RUN_ID must be qwen_metadata_hydration_v1." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it." >&2
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

mkdir -p "${output_dir}" "${hf_cache}"

nohup docker run --rm \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  python scripts/hydrate_spark_qwen_metadata.py \
    --repo /workspace/GridNav \
    --hf-home /cache/huggingface \
    --output /artifacts/result.json \
    --run-id "${run_id}" \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --revision "${qwen_revision}" \
    --docker-image-ref "${image_ref}" \
    --docker-image-id "${image_id}" \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "This requests nine small metadata files at frozen revision ${qwen_revision}."
echo "Model weight files are excluded by the allow-list."
echo "Log: ${output_dir}/run.log"
echo "Result: ${output_dir}/result.json"
