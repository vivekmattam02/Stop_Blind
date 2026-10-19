#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_STABILITY_PILOT_RUN_ID:-stability_pilot_v1}"

if [[ ! "${run_id}" =~ ^stability_pilot_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_STABILITY_PILOT_RUN_ID must match stability_pilot_v<positive integer>." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi
if [[ -d "${checkpoint_dir}" ]] && [[ -n "$(find "${checkpoint_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory ${checkpoint_dir} is non-empty; choose a new run ID." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${checkpoint_dir}" "${hf_cache}"

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
  --env GRIDNAV_STABILITY_PILOT_RUN_ID="${run_id}" \
  --workdir /workspace/GridNav \
  gridnav-spark-habitat:hsim0.3.3 \
  bash scripts/run_spark_stability_pilot.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Expected duration: roughly 45-90 minutes; hard ceiling is two hours per arm."
echo "Log: ${output_dir}/run.log"
echo "Suite result: ${output_dir}/suite.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Checkpoints: ${checkpoint_dir}"

