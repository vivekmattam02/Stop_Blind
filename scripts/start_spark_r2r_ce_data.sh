#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${repo_root}/artifacts/spark/r2r_ce_data"
data_root="$HOME/datasets/GridNav"
log_path="${artifact_root}/bootstrap.log"
pid_path="${artifact_root}/bootstrap.pid"
image="gridnav-spark:cuda13-py25.10"

mkdir -p "${artifact_root}" "${data_root}"

if [[ -f "${pid_path}" ]]; then
    existing_pid="$(tr -dc '0-9' < "${pid_path}")"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Bootstrap is already running with PID ${existing_pid}."
        echo "Log: ${log_path}"
        exit 0
    fi
fi

nohup docker run --rm \
    --name gridnav-r2r-data-bootstrap \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${repo_root},dst=/workspace/GridNav,readonly" \
    --mount "type=bind,src=${data_root},dst=/datasets/GridNav" \
    --mount "type=bind,src=${artifact_root},dst=/artifacts" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --workdir /workspace/GridNav \
    "${image}" \
    python scripts/prepare_spark_r2r_ce_data.py \
    --inside-container \
    --repo-root /workspace/GridNav \
    --data-root /datasets/GridNav \
    --artifact-root /artifacts \
    --image "${image}" \
    "$@" \
    >"${log_path}" 2>&1 </dev/null &
bootstrap_pid=$!
echo "${bootstrap_pid}" > "${pid_path}"

sleep 1
if ! kill -0 "${bootstrap_pid}" 2>/dev/null; then
    echo "Bootstrap exited during startup. Inspect ${log_path}." >&2
    exit 1
fi

echo "Started GridNav data bootstrap with PID ${bootstrap_pid}."
echo "Log: ${log_path}"
echo "Manifest on success: ${artifact_root}/data_bootstrap_manifest.json"
