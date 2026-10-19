#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_C0_RUN_ID:-native_c0_real_batch_v1}"
image_ref="gridnav-spark-habitat:hsim0.3.3"
qwen_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
qwen_snapshot_host="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${qwen_revision}"
qwen_snapshot_container="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${qwen_revision}"
parent_dir="${repo}/artifacts/spark/native_qwen_history_preflight_v3"

if [[ ! "${run_id}" =~ ^native_c0_real_batch_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_NATIVE_C0_RUN_ID must match native_c0_real_batch_vN." >&2
  exit 2
fi

output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
checkpoint="${checkpoint_dir}/trainable_step1.pt"
container_name="gridnav-${run_id//_/-}"

if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi
if [[ -d "${checkpoint_dir}" ]] \
   && [[ -n "$(find "${checkpoint_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory ${checkpoint_dir} is non-empty; preserve it and choose a new run ID." >&2
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

expected_parent_result=bf7555659c6e8cbfdb7ed0b703440c7b5c8739ed420a83f0ec3751d82e939bba
expected_parent_verification=e56b4dfcc85e1bca9dbe9b96f2d326a8d42ce462b95aa06d0c42c21cd69fb7e5
if [[ ! -f "${parent_dir}/result.json" || ! -f "${parent_dir}/verification.json" ]]; then
  echo "Immutable native-Qwen processor parent artifacts are incomplete." >&2
  exit 1
fi
if [[ "$(sha256sum "${parent_dir}/result.json" | awk '{print $1}')" != "${expected_parent_result}" ]]; then
  echo "Immutable native-Qwen processor result is missing or changed." >&2
  exit 1
fi
if [[ "$(sha256sum "${parent_dir}/verification.json" | awk '{print $1}')" != "${expected_parent_verification}" ]]; then
  echo "Immutable native-Qwen processor verification is missing or changed." >&2
  exit 1
fi
if [[ ! -f "${qwen_snapshot_host}/model.safetensors.index.json" ]]; then
  echo "Frozen Qwen weight index is missing: ${qwen_snapshot_host}" >&2
  exit 1
fi

mkdir -p "${output_dir}" "${checkpoint_dir}"
nohup docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${checkpoint_dir},dst=/checkpoints" \
  --mount "type=bind,src=${parent_dir},dst=/processor_parent,readonly" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_C0_RUN_ID="${run_id}" \
  --env GRIDNAV_DOCKER_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_DOCKER_IMAGE_ID="${image_id}" \
  --env GRIDNAV_QWEN_REVISION="${qwen_revision}" \
  --env GRIDNAV_QWEN_SNAPSHOT="${qwen_snapshot_container}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_c0_gate.sh \
  >"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Docker image ID: ${image_id}"
echo "Expected duration: approximately 10-30 minutes; this is one real update, not a training run."
echo "Log: ${output_dir}/run.log"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Checkpoint: ${checkpoint}"
