#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
run_id="${GRIDNAV_STABILITY_PILOT_RUN_ID:-stability_pilot_v1}"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
container_name="gridnav-${run_id//_/-}-verifier-v2"

for required in manifest.json suite.json verification.json; do
  if [[ ! -f "${output_dir}/${required}" ]]; then
    echo "Required preserved artifact is missing: ${output_dir}/${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${output_dir}/arms" ]] || [[ ! -d "${checkpoint_dir}" ]]; then
  echo "Preserved arm results or checkpoints are missing." >&2
  exit 1
fi
if [[ -e "${output_dir}/verification_v2.json" ]]; then
  echo "${output_dir}/verification_v2.json already exists; preserve it and inspect manually." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before retrying." >&2
  exit 1
fi

docker run --rm --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${container_name}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${repo},dst=/workspace/GridNav,readonly" \
  --mount "type=bind,src=${output_dir},dst=/artifacts" \
  --mount "type=bind,src=${checkpoint_dir},dst=/checkpoints,readonly" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /workspace/GridNav \
  gridnav-spark-habitat:hsim0.3.3 \
  bash -c '
    python -m pytest -p no:cacheprovider \
      tests/test_stability_pilot_artifacts.py &&
    python scripts/check_stability_pilot_artifacts_v2.py \
      --repo /workspace/GridNav \
      --suite /artifacts/suite.json \
      --manifest /artifacts/manifest.json \
      --arms-root /artifacts/arms \
      --checkpoint-root /checkpoints \
      --prior-verification /artifacts/verification.json \
      --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
      --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
      --output /artifacts/verification_v2.json
  '

echo "Corrected verification written to ${output_dir}/verification_v2.json"
