#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_ACTION_DECISION_RUN_ID:-action_decision_v2}"
frozen_checkpoint_sha256="95140f96540a63b852b417f83dde35d44440cd5b027e1ad6cbbfbb93f4457f29"
expected_checkpoint_sha256="${GRIDNAV_ACTION_DECISION_CHECKPOINT_SHA256:-${frozen_checkpoint_sha256}}"

if [[ ! "${run_id}" =~ ^action_decision_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_ACTION_DECISION_RUN_ID must match action_decision_v<positive integer>." >&2
  exit 2
fi
if [[ "${expected_checkpoint_sha256}" != "${frozen_checkpoint_sha256}" ]]; then
  echo "GRIDNAV_ACTION_DECISION_CHECKPOINT_SHA256 cannot override the frozen v2 identity." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"
checkpoint_path="${data_root}/checkpoints/real_batch_v2/trainable_step1.pt"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi
if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Required real_batch_v2 checkpoint is missing: ${checkpoint_path}" >&2
  exit 1
fi
observed_checkpoint_sha256="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
if [[ "${observed_checkpoint_sha256}" != "${expected_checkpoint_sha256}" ]]; then
  echo "Required checkpoint SHA-256 does not match the frozen identity." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${hf_cache}"
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
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_ACTION_DECISION_RUN_ID="${run_id}" \
  --env GRIDNAV_ACTION_DECISION_CHECKPOINT_SHA256="${expected_checkpoint_sha256}" \
  --workdir /workspace/GridNav \
  gridnav-spark-habitat:hsim0.3.3 \
  bash scripts/run_spark_action_decision_gate.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Log: ${output_dir}/run.log"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
