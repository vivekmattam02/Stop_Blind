#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_HISTORY_RUN_ID:-native_history_select_v1}"
image_ref="${GRIDNAV_NATIVE_HISTORY_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
resume=0
stress_only=0

for option in "$@"; do
  case "${option}" in
    --resume) resume=1 ;;
    --stress-only) stress_only=1 ;;
    *)
      echo "Usage: bash scripts/start_spark_native_history_select.sh [--resume|--stress-only]" >&2
      exit 2
      ;;
  esac
done
if [[ "${resume}" == "1" && "${stress_only}" == "1" ]]; then
  echo "--resume and --stress-only cannot be combined." >&2
  exit 2
fi
if [[ ! "${run_id}" =~ ^native_history_select_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_NATIVE_HISTORY_RUN_ID must match native_history_select_v<positive integer>." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}"
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"

if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
  exit 1
fi
if [[ ! -f "${snapshot}/model.safetensors.index.json" \
   || ! -f "${snapshot}/preprocessor_config.json" ]]; then
  echo "The exact-revision Qwen model/processor snapshot is incomplete: ${snapshot}" >&2
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
  if [[ ! -f "${output_dir}/manifest.json" || ! -d "${checkpoint_dir}" ]]; then
    echo "Resume requires the preserved manifest and checkpoint directory." >&2
    exit 1
  fi
fi

mkdir -p "${output_dir}" "${checkpoint_dir}" "${hf_cache}"
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
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_HISTORY_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_HISTORY_RESUME="${resume}" \
  --env GRIDNAV_NATIVE_HISTORY_STRESS_ONLY="${stress_only}" \
  --env GRIDNAV_NATIVE_HISTORY_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_HISTORY_IMAGE_ID="${observed_image_id}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_history_select.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Frozen workload: 3 matched arms x 2,048 distinct train-only updates."
echo "A real 12-update three-arm stress suite must pass before the overnight manifest is frozen."
echo "Calibration: 128 disjoint train-split decisions at 6 checkpoints per arm."
echo "Hard wall-time boundary: 10 hours per invocation; a partial run checkpoints and requires --resume."
echo "Expected first-invocation duration: approximately 7-10 hours on the admitted Spark path."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Suite: ${output_dir}/suite.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Persistent cache/checkpoints: ${checkpoint_dir}"
if [[ "${stress_only}" == "1" ]]; then
  echo "Diagnostic mode: the container will stop after the reduced three-arm stress; main training cannot start."
fi
