#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_REAL_BATCH_RUN_ID:-real_batch_v1}"

if [[ ! "${run_id}" =~ ^real_batch_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_REAL_BATCH_RUN_ID must match real_batch_v<positive integer>." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}"
checkpoint_path="${checkpoint_dir}/trainable_step1.pt"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi

if [[ -e "${checkpoint_path}" ]]; then
  echo "Checkpoint ${checkpoint_path} already exists; preserve it and choose a new run ID." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${checkpoint_dir}" "${hf_cache}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

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
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_REAL_BATCH_RUN_ID="${run_id}" \
  --workdir /workspace/GridNav \
  gridnav-spark-habitat:hsim0.3.3 \
  bash scripts/run_spark_real_batch_gate.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Log: ${output_dir}/run.log"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Checkpoint: ${checkpoint_path}"
