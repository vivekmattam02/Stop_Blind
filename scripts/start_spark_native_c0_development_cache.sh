#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
run_id="${GRIDNAV_NATIVE_C0_CACHE_RUN_ID:-native_c0_development_cache_v1}"
image_ref="${GRIDNAV_NATIVE_C0_IMAGE_REF:-gridnav-spark-habitat:hsim0.3.3}"
expected_image_id=sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf
output_dir="${repo}/artifacts/spark/${run_id}"
object_dir="${data_root}/checkpoints/native_c0_h4_content_cache_v1"
container_name="gridnav-${run_id//_/-}"
source "${repo}/scripts/spark_launch_helpers.sh"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" \
  --output "${repo}/artifacts/context/context_state.json"

if [[ "${run_id}" != "native_c0_development_cache_v1" ]]; then
  echo "The frozen cache run ID is native_c0_development_cache_v1." >&2
  exit 2
fi
for path in \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
  "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz" \
  "${repo}/artifacts/spark/native_development_measurements_v2/result.json" \
  "${repo}/artifacts/spark/native_development_measurements_v2/verification.json" \
  "${repo}/artifacts/spark/r2r_ce_data/data_bootstrap_manifest.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required native-C0 cache input is missing: ${path}" >&2
    exit 1
  fi
done
observed_image_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
if [[ "${observed_image_id}" != "${expected_image_id}" ]]; then
  echo "Native-C0 cache Docker image identity mismatch." >&2
  echo "Expected: ${expected_image_id}" >&2
  echo "Observed: ${observed_image_id}" >&2
  exit 1
fi
if [[ -f "${output_dir}/verification.json" ]] && \
   python3 - "${output_dir}/verification.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "passed" and payload.get("error_count") == 0 else 1)
PY
then
  echo "The independently verified cache is already complete: ${output_dir}" >&2
  exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before relaunching." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${object_dir}"
touch "${output_dir}/run.log"
nohup docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${object_dir},dst=/cache_objects" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GRIDNAV_NATIVE_C0_CACHE_RUN_ID="${run_id}" \
  --env GRIDNAV_NATIVE_C0_IMAGE_REF="${image_ref}" \
  --env GRIDNAV_NATIVE_C0_IMAGE_ID="${observed_image_id}" \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash scripts/run_spark_native_c0_development_cache.sh \
  >>"${output_dir}/run.log" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${output_dir}/launcher.pid"
wait_for_container_registration \
  "${container_name}" "${pid}" "${output_dir}/verification.json" "${output_dir}/run.log"
echo "Started ${container_name} with launcher PID ${pid}."
echo "Stage: separate resumable content-addressed rendering of 4,224 frozen H4 decisions."
echo "Expected first build: about 2.86 hours; completed objects are reused after interruption."
echo "Log: ${output_dir}/run.log"
echo "Progress: ${output_dir}/progress.json"
echo "Result: ${output_dir}/result.json"
echo "Independent verification: ${output_dir}/verification.json"
echo "Shared object store: ${object_dir}"
