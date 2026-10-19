#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_ACTION_BALANCE_RUN_ID:-native_c0_action_balance_v1}"
image_ref="${GRIDNAV_ACTION_BALANCE_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
cache_artifact_dir="${repo}/artifacts/spark/native_c0_development_cache_v1"
cache_object_dir="${data_root}/checkpoints/native_c0_h4_content_cache_v1"
development_dir="${repo}/artifacts/spark/native_c0_development_v1"
analysis_dir="${repo}/artifacts/context/native_c0_failure_analysis_v1"
stress_dir="${repo}/artifacts/spark/native_c0_action_balance_stress_v1"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" \
  --output "${repo}/artifacts/context/context_state.json"

if [[ "${run_id}" != "native_c0_action_balance_v1" ]]; then
  echo "The frozen full action-balance run ID is native_c0_action_balance_v1." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz" \
  "${snapshot}/model.safetensors.index.json" \
  "${snapshot}/preprocessor_config.json" \
  "${cache_artifact_dir}/manifest.json" \
  "${cache_artifact_dir}/cache_index.json" \
  "${cache_artifact_dir}/result.json" \
  "${cache_artifact_dir}/verification.json" \
  "${development_dir}/manifest.json" \
  "${development_dir}/result.json" \
  "${development_dir}/verification.json" \
  "${analysis_dir}/analysis.json" \
  "${analysis_dir}/verification.json" \
  "${stress_dir}/manifest.json" \
  "${stress_dir}/result.json" \
  "${stress_dir}/updates.jsonl" \
  "${stress_dir}/verification.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required full action-balance input is missing: ${path}" >&2
    exit 1
  fi
done

python3 - "${stress_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "manifest.json"
result = root / "result.json"
updates = root / "updates.jsonl"
verification = root / "verification.json"
payload = json.loads(result.read_text(encoding="utf-8"))
checked = json.loads(verification.read_text(encoding="utf-8"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
expected = {
    "manifest": "6af09fa65359555520d7ffd6832bdce45b67a22f22eeb33893f8a1349d914945",
    "result": "4776a19269906f9b45397b5c09c7714485d99162be8f5df1db67e3a30caccc48",
    "updates": "7434050e9676cf89b29c6585220d17b6381660a55acb8d0d54052fdf8f017636",
    "verification": "327ffad9d2331f982b5b7a96d7ece0d9046fe137a936bdb1935fb672cbdaf9c9",
}
observed = {
    "manifest": sha(manifest),
    "result": sha(result),
    "updates": sha(updates),
    "verification": sha(verification),
}
if (
    observed != expected
    or payload.get("manifest_sha256") != observed["manifest"]
    or payload.get("updates_sha256") != observed["updates"]
    or payload.get("status") != "passed"
    or payload.get("full_action_balance_run_authorized") is not True
    or payload.get("scientific_result_emitted") is not False
    or checked.get("status") != "passed"
    or checked.get("error_count") != 0
    or checked.get("manifest_sha256") != observed["manifest"]
    or checked.get("result_sha256") != observed["result"]
    or checked.get("updates_sha256") != observed["updates"]
):
    raise SystemExit("The independently verified stress gate does not authorize the full run.")
print("Verified action-balance stress authorization and artifact linkage.")
PY

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Action-balance Docker image identity mismatch." >&2
  exit 1
fi
for directory in "${output_dir}" "${checkpoint_dir}"; do
  if [[ -d "${directory}" ]] && [[ -n "$(find "${directory}" -mindepth 1 -print -quit)" ]]; then
    echo "${directory} is non-empty; preserve it and do not overwrite this frozen run." >&2
    exit 1
  fi
done
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${checkpoint_dir}"
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
  --mount "type=bind,src=${cache_object_dir},dst=/cache_objects,readonly" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_ACTION_BALANCE_RUN_ID="${run_id}" \
  --env GRIDNAV_ACTION_BALANCE_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_ACTION_BALANCE_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_c0_action_balance.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Stage: one fresh 192-update adaptive action-balanced C0 comparison."
echo "Expected after cache: roughly one hour; combined cache-plus-run cap: 4.5 hours."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Checkpoints: ${checkpoint_dir}"
