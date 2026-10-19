#!/usr/bin/env bash
set -euo pipefail

repo_root="${GRIDNAV_REPO_ROOT:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
image="${GRIDNAV_HABITAT_IMAGE:-gridnav-spark-habitat:hsim0.3.3}"
max_hours="${GRIDNAV_HABITAT_MAX_HOURS:-8}"
artifact_root="${repo_root}/artifacts/spark/habitat_expanded_v2"
container_name="gridnav-habitat-expanded-v2"
log_path="${artifact_root}/run.log"
pid_path="${artifact_root}/run.pid"

if [[ ! -d "${repo_root}/.git" ]]; then
    echo "GridNav repository not found at ${repo_root}" >&2
    exit 2
fi
for required_path in \
    "${data_root}/scene_datasets/mp3d" \
    "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz" \
    "${data_root}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required expanded-conformance input not found: ${required_path}" >&2
        exit 2
    fi
done
if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Pinned Habitat image ${image} is missing. Run the first-episode gate first." >&2
    exit 2
fi

mkdir -p "${artifact_root}"
container_running="$(
    docker container inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null \
        || true
)"
if [[ "${container_running}" == "true" ]]; then
    existing_pid="unknown"
    if [[ -f "${pid_path}" ]]; then
        existing_pid="$(tr -dc '0-9' < "${pid_path}")"
    fi
    echo "Expanded Habitat suite is already running with launcher PID ${existing_pid}."
    echo "Log: ${log_path}"
    echo "Progress: ${artifact_root}/progress.json"
    exit 0
fi

nohup docker run --rm --gpus=all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name "${container_name}" \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${repo_root},dst=/workspace/GridNav,readonly" \
    --mount "type=bind,src=${artifact_root},dst=/workspace/GridNav/artifacts/spark/habitat_expanded_v2" \
    --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --env HOME=/tmp \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --workdir /workspace/GridNav \
    "${image}" \
    python scripts/run_spark_habitat_expanded.py \
        --repo-root /workspace/GridNav \
        --scene-root /datasets/GridNav/scene_datasets \
        --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz \
        --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz \
        --split val_seen \
        --max-hours "${max_hours}" \
        --gpu-device-id 0 \
        --output-root /workspace/GridNav/artifacts/spark/habitat_expanded_v2 \
    >"${log_path}" 2>&1 </dev/null &
suite_pid=$!
echo "${suite_pid}" > "${pid_path}"

sleep 1
if ! kill -0 "${suite_pid}" 2>/dev/null; then
    echo "Expanded Habitat suite exited during startup. Inspect ${log_path}." >&2
    exit 1
fi

echo "Started expanded Habitat suite with PID ${suite_pid}."
echo "Maximum invocation time: ${max_hours} hours. It will finish early if all episodes complete."
echo "Log: ${log_path}"
echo "Progress: ${artifact_root}/progress.json"
echo "Final status: ${artifact_root}/suite.json"
