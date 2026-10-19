#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_SENSITIVITY_RUN_ID:-native_sensitivity_v1}"
image_ref="${GRIDNAV_NATIVE_SENSITIVITY_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
processor_gate="${repo}/artifacts/spark/native_qwen_history_preflight_v5"
output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

if [[ ! "${run_id}" =~ ^native_sensitivity_v[1-9][0-9]*$ ]]; then
  echo "GRIDNAV_NATIVE_SENSITIVITY_RUN_ID must match native_sensitivity_vN." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz" \
  "${snapshot}/model.safetensors.index.json" \
  "${snapshot}/preprocessor_config.json" \
  "${processor_gate}/result.json" \
  "${processor_gate}/verification.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required sensitivity input is missing: ${path}" >&2
    exit 1
  fi
done
python3 - "${processor_gate}/result.json" "${processor_gate}/verification.json" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
verification = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if result.get("gate") != "native_qwen_c0_history_processor_preflight_v4":
    raise SystemExit("Processor result is not the corrected v4 one-token gate.")
if result.get("status") != "passed" or verification.get("status") != "passed":
    raise SystemExit("Corrected v4 processor gate has not passed both decisions.")
if verification.get("gate") != "independent_native_qwen_c0_history_artifact_verification_v4":
    raise SystemExit("Processor verification is not the corrected v4 decision.")
if verification.get("error_count") != 0:
    raise SystemExit("Corrected v4 processor verification contains errors.")
digest = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
if verification.get("result_sha256") != digest:
    raise SystemExit("Corrected v4 processor result/verification linkage mismatch.")
PY

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Sensitivity Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
  exit 1
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and choose a new run ID." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
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
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_SENSITIVITY_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_SENSITIVITY_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_SENSITIVITY_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_sensitivity.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Workload: 16 source-locked val_seen decisions x 4 forward-only conditions (64 forwards)."
echo "Primary control: paired real-image swap; no training and no val_unseen access."
echo "Expected duration: approximately 5-20 minutes, including full weight hashing twice."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
