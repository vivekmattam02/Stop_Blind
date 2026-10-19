#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_SENSITIVITY_V2_RUN_ID:-native_sensitivity_v2}"
image_ref="${GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
processor_gate="${repo}/artifacts/spark/native_qwen_history_preflight_v5"
v1_gate="${repo}/artifacts/spark/native_sensitivity_v1"
output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

if [[ "${run_id}" != "native_sensitivity_v2" ]]; then
  echo "The frozen confirmation run ID is native_sensitivity_v2." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz" \
  "${snapshot}/model.safetensors.index.json" \
  "${snapshot}/preprocessor_config.json" \
  "${processor_gate}/result.json" \
  "${processor_gate}/verification.json" \
  "${v1_gate}/manifest.json" \
  "${v1_gate}/result.json" \
  "${v1_gate}/verification.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required sensitivity-v2 input is missing: ${path}" >&2
    exit 1
  fi
done
python3 - \
  "${processor_gate}/result.json" \
  "${processor_gate}/verification.json" \
  "${v1_gate}/manifest.json" \
  "${v1_gate}/result.json" \
  "${v1_gate}/verification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

processor_result_path, processor_verification_path, v1_manifest_path, v1_result_path, v1_verification_path = map(Path, sys.argv[1:])
processor_result = json.loads(processor_result_path.read_text(encoding="utf-8"))
processor_verification = json.loads(processor_verification_path.read_text(encoding="utf-8"))
if processor_result.get("gate") != "native_qwen_c0_history_processor_preflight_v4":
    raise SystemExit("Processor result is not the corrected one-token gate.")
if processor_result.get("status") != "passed" or processor_verification.get("status") != "passed":
    raise SystemExit("Corrected processor gate has not passed both decisions.")
if processor_verification.get("error_count") != 0:
    raise SystemExit("Corrected processor verification contains errors.")
if processor_verification.get("result_sha256") != hashlib.sha256(processor_result_path.read_bytes()).hexdigest():
    raise SystemExit("Corrected processor result/verification linkage mismatch.")

v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
v1_result = json.loads(v1_result_path.read_text(encoding="utf-8"))
v1_verification = json.loads(v1_verification_path.read_text(encoding="utf-8"))
if v1_manifest.get("status") != "frozen" or v1_result.get("status") != "passed":
    raise SystemExit("V1 sensitivity discovery artifact is incomplete.")
if v1_result.get("summary", {}).get("verdict") != "inconclusive_expand_before_optimizer_diagnostic":
    raise SystemExit("V1 sensitivity verdict differs from the frozen discovery outcome.")
if v1_verification.get("status") != "passed" or v1_verification.get("error_count") != 0:
    raise SystemExit("V1 sensitivity verification did not pass.")
if v1_result.get("manifest_sha256") != hashlib.sha256(v1_manifest_path.read_bytes()).hexdigest():
    raise SystemExit("V1 sensitivity result/manifest linkage mismatch.")
if v1_verification.get("result_sha256") != hashlib.sha256(v1_result_path.read_bytes()).hexdigest():
    raise SystemExit("V1 sensitivity result/verification linkage mismatch.")
PY

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Sensitivity-v2 Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
  exit 1
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory ${output_dir} is non-empty; preserve it and do not overwrite." >&2
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
  --env GRIDNAV_NATIVE_SENSITIVITY_V2_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_SENSITIVITY_V2_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_sensitivity_v2.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Workload: 64 new v1-disjoint val_seen decisions x 4 conditions (256 forwards)."
echo "Primary metric: centered four-action-logit movement; clean repeat is the noise control."
echo "No training and no val_unseen access. Expected duration: approximately 15-45 minutes."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
