#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id=native_c0_closed_loop_v1
max_hours="${GRIDNAV_CLOSED_LOOP_MAX_HOURS:-30}"
image_ref=gridnav-spark-habitat:hsim0.3.3
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
snapshot="${hf_cache}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}"
stress_dir="${repo}/artifacts/spark/native_c0_closed_loop_stress_v3"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_archive="${data_root}/checkpoints"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" --output "${repo}/artifacts/context/context_state.json"

required=(
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz"
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz"
  "${data_root}/scene_datasets"
  "${snapshot}/config.json"
  "${repo}/artifacts/spark/policy_ordering_v1/suite_manifest.json"
  "${repo}/artifacts/spark/policy_ordering_v1/suite.json"
  "${repo}/artifacts/context/policy_ordering_v1_verification.json"
  "${stress_dir}/result.json"
  "${stress_dir}/timings.jsonl"
  "${stress_dir}/verification.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required closed-loop input is missing: ${path}" >&2
    exit 1
  fi
done

python3 - "${stress_dir}/result.json" "${stress_dir}/verification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result_path, verification_path = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
verification = json.loads(verification_path.read_text(encoding="utf-8"))
checks = {
    "result_passed": result.get("status") == "passed",
    "verification_passed": verification.get("status") == "passed",
    "zero_errors": verification.get("error_count") == 0,
    "linked": verification.get("result_sha256")
    == hashlib.sha256(result_path.read_bytes()).hexdigest(),
}
if not all(checks.values()):
    raise SystemExit(f"Closed-loop stress gate is not admitted: {checks}")
scope = result["selected_scope"]
print(
    "Stress-selected scope:",
    scope["branch_id"],
    f"({len(scope['checkpoint_ids'])} checkpoints x {scope['episode_count']} episodes)",
)
PY

observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Closed-loop Docker image identity mismatch." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container already exists; inspect it before launch: ${container_name}" >&2
  exit 1
fi
if [[ -f "${output_dir}/verification.json" ]]; then
  echo "Closed-loop suite is already verified; refusing to relaunch." >&2
  exit 1
fi
if [[ -f "${output_dir}/failure.json" ]]; then
  echo "Preserved infrastructure failure requires diagnosis before resume: ${output_dir}" >&2
  exit 1
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  if [[ ! -f "${output_dir}/manifest.json" ]] || [[ ! -f "${output_dir}/progress.json" ]]; then
    echo "Non-empty output is not a recognized resumable suite: ${output_dir}" >&2
    exit 1
  fi
  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${output_dir}/progress.json")"
  if [[ "${status}" != "partial_time_budget" ]]; then
    echo "Existing suite is not in the resumable partial_time_budget state: ${status}" >&2
    exit 1
  fi
  echo "Resuming the preserved episode-boundary suite: ${output_dir}"
else
  mkdir -p "${output_dir}"
  : >"${output_dir}/run.log"
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
  --mount "type=bind,src=${checkpoint_archive},dst=/checkpoint_archive,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_CLOSED_LOOP_RUN_ID="${run_id}" \
  --env GRIDNAV_CLOSED_LOOP_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_CLOSED_LOOP_IMAGE_ID="${observed_image_id}" \
  --env GRIDNAV_CLOSED_LOOP_MAX_HOURS="${max_hours}" \
  --env GRIDNAV_QWEN_SNAPSHOT="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${revision}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_c0_closed_loop.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Scope was selected only from the passed 32-decision latency gate."
echo "Official protocol: H4 one-step scoring, 500-step cap, val_seen control episodes."
echo "val_unseen remains sealed. The suite is resumable at completed episode boundaries."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
