#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
run_id="${GRIDNAV_NATIVE_HISTORY_RUN_ID:-native_history_select_v1}"
output_dir="${repo}/artifacts/spark/${run_id}"
container_name="gridnav-${run_id//_/-}"

echo "=== container ==="
docker ps --filter "name=^/${container_name}$" --format 'table {{.Names}}\t{{.Status}}'
echo "=== accelerator ==="
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,power.draw --format=csv
echo "=== progress ==="
if [[ -f "${output_dir}/progress.json" ]]; then
  python3 -m json.tool "${output_dir}/progress.json"
else
  echo "No progress.json yet; preflight tests or cache preparation may still be running."
fi
echo "=== latest log ==="
tail -n 25 "${output_dir}/run.log"
