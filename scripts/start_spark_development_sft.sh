#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_DEVELOPMENT_SFT_RUN_ID:-development_sft_v1}"
resume=0

if [[ "${1:-}" == "--resume" ]]; then
  resume=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/start_spark_development_sft.sh [--resume]" >&2
  exit 2
fi
if [[ ! "${run_id}" =~ ^development_sft_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_DEVELOPMENT_SFT_RUN_ID must match development_sft_v<positive integer>." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
pilot_artifacts="${repo}/artifacts/spark/stability_pilot_v1"
warmstart_dir="${data_root}/checkpoints/stability_pilot_v1/lr_5em05"
warmstart="${warmstart_dir}/step_0008.pt"
container_name="gridnav-${run_id//_/-}"

if [[ ! -f "${pilot_artifacts}/manifest.json" \
   || ! -f "${pilot_artifacts}/suite.json" \
   || ! -f "${pilot_artifacts}/verification_v2.json" ]]; then
  echo "The verified stability-pilot evidence is incomplete on Spark." >&2
  exit 1
fi
if [[ ! -f "${warmstart}" ]]; then
  echo "The selected stability-pilot checkpoint is missing: ${warmstart}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

if [[ "${resume}" == "0" ]]; then
  if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Run directory ${output_dir} is non-empty; preserve it or use --resume." >&2
    exit 1
  fi
  if [[ -d "${checkpoint_dir}" ]] \
     && [[ -n "$(find "${checkpoint_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Checkpoint directory ${checkpoint_dir} is non-empty; preserve it or use --resume." >&2
    exit 1
  fi
else
  if [[ ! -f "${output_dir}/latest.json" || ! -f "${output_dir}/manifest.json" ]]; then
    echo "Resume requires an existing manifest.json and latest.json." >&2
    exit 1
  fi
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
  --mount "type=bind,src=${pilot_artifacts},dst=/pilot_artifacts,readonly" \
  --mount "type=bind,src=${warmstart_dir},dst=/warmstart,readonly" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_DEVELOPMENT_SFT_RUN_ID="${run_id}" \
  --env GRIDNAV_DEVELOPMENT_SFT_RESUME="${resume}" \
  --workdir /workspace/GridNav \
  gridnav-spark-habitat:hsim0.3.3 \
  bash scripts/run_spark_development_sft.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Frozen maximum: 1,024 updates or pre-registered early stopping, with an 8-hour ceiling."
echo "Likely duration on the measured Spark path: about 1.5-3 hours including cache and verification."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Persistent checkpoints/cache: ${checkpoint_dir}"
