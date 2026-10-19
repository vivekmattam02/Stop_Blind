#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GRIDNAV_REPO_ROOT:-$HOME/Desktop/GridNav}"
DATA_ROOT="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
IMAGE="${GRIDNAV_HABITAT_IMAGE:-gridnav-spark-habitat:hsim0.3.3}"
EPISODE_COUNT="${GRIDNAV_HABITAT_EPISODES:-1}"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/spark/habitat_v2"
LOG_PATH="${ARTIFACT_ROOT}/conformance.log"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
    echo "GridNav repository not found at ${REPO_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${DATA_ROOT}/scene_datasets/mp3d" ]]; then
    echo "MP3D scene directory not found at ${DATA_ROOT}/scene_datasets/mp3d" >&2
    exit 2
fi
for required_path in \
    "${DATA_ROOT}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz" \
    "${DATA_ROOT}/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required R2R-CE file not found: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${ARTIFACT_ROOT}"
cd "${REPO_ROOT}"

{
    echo "[build] ${IMAGE}"
    docker build --progress=plain \
        --file docker/spark/habitat.Dockerfile \
        --tag "${IMAGE}" \
        .

    echo "[run] pinned Habitat-Sim ground-truth replay (${EPISODE_COUNT} episode)"
    docker run --rm --gpus=all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --name gridnav-habitat-conformance-v2 \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=${REPO_ROOT},dst=/workspace/GridNav,readonly" \
        --mount "type=bind,src=${ARTIFACT_ROOT},dst=/workspace/GridNav/artifacts/spark/habitat_v2" \
        --mount "type=bind,src=${DATA_ROOT},dst=/datasets/GridNav,readonly" \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        --env HOME=/tmp \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --workdir /workspace/GridNav \
        "${IMAGE}" \
        python scripts/check_spark_habitat_conformance.py \
            --scene-root /datasets/GridNav/scene_datasets \
            --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz \
            --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz \
            --split val_seen \
            --episode-count "${EPISODE_COUNT}" \
            --gpu-device-id 0 \
            --output artifacts/spark/habitat_v2/conformance.json \
            --steps-output artifacts/spark/habitat_v2/conformance_steps.jsonl
} 2>&1 | tee "${LOG_PATH}"

echo "Habitat conformance artifact: ${ARTIFACT_ROOT}/conformance.json"
echo "Per-step trace: ${ARTIFACT_ROOT}/conformance_steps.jsonl"
echo "Combined log: ${LOG_PATH}"
