#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
data_root="${GRIDNAV_DATA_ROOT:-$HOME/datasets/GridNav}"
hf_cache="${GRIDNAV_HF_CACHE:-$HOME/.cache/gridnav-huggingface}"
run_id="${GRIDNAV_NATIVE_C0_RUN_ID:-native_c0_real_batch_v1}"
image_ref="gridnav-spark-habitat:hsim0.3.3"
expected_image_id="sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
qwen_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
qwen_snapshot="/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/${qwen_revision}"
output_dir="${repo}/artifacts/spark/${run_id}"
checkpoint_dir="${data_root}/checkpoints/${run_id}"
parent_dir="${repo}/artifacts/spark/native_qwen_history_preflight_v3"
container_name="gridnav-${run_id//_/-}-verifier-v2"

for required in result.json verification.json processor_reverification.json; do
  if [[ ! -f "${output_dir}/${required}" ]]; then
    echo "Required preserved artifact is missing: ${output_dir}/${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${checkpoint_dir}/trainable_step1.pt" ]]; then
  echo "Preserved native-C0 checkpoint is missing." >&2
  exit 1
fi
if [[ -e "${output_dir}/verification_v2.json" ]]; then
  echo "${output_dir}/verification_v2.json already exists; preserve and inspect it." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} already exists; inspect it before retrying." >&2
  exit 1
fi
image_id="$(docker image inspect --format '{{.Id}}' "${image_ref}")"
if [[ "${image_id}" != "${expected_image_id}" ]]; then
  echo "Docker image identity changed: ${image_id}" >&2
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
  --mount "type=bind,src=${parent_dir},dst=/processor_parent,readonly" \
  --mount "type=bind,src=${data_root},dst=/datasets/GridNav,readonly" \
  --mount "type=bind,src=${hf_cache},dst=/cache/huggingface,readonly" \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env GRIDNAV_QWEN_SNAPSHOT="${qwen_snapshot}" \
  --env HOME=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /workspace/GridNav \
  "${image_ref}" \
  bash -c '
    python -m pytest -p no:cacheprovider tests/test_native_c0_artifacts_v2.py &&
    python scripts/check_native_c0_artifacts_v2.py \
      --repo /workspace/GridNav \
      --episodes /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
      --ground-truth /datasets/GridNav/episodes/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
      --scene-root /datasets/GridNav/scene_datasets \
      --model-snapshot "${GRIDNAV_QWEN_SNAPSHOT}" \
      --processor-result /processor_parent/result.json \
      --processor-verification /processor_parent/verification.json \
      --processor-reverification /artifacts/processor_reverification.json \
      --checkpoint /checkpoints/trainable_step1.pt \
      --result /artifacts/result.json \
      --prior-verification /artifacts/verification.json \
      --output /artifacts/verification_v2.json
  '

echo "Corrected verification written to ${output_dir}/verification_v2.json"
