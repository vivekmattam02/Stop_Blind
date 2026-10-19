#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
pipeline_dir="${repo}/artifacts/spark/native_c0_seed_replication_pipeline_v1"
pipeline_log="${pipeline_dir}/run.log"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" --output "${repo}/artifacts/context/context_state.json"

if [[ -d "${pipeline_dir}" ]] && [[ -n "$(find "${pipeline_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Pipeline directory is non-empty; preserve and inspect it: ${pipeline_dir}" >&2
  exit 1
fi
mkdir -p "${pipeline_dir}"
: >"${pipeline_log}"
nohup bash "${repo}/scripts/run_spark_native_c0_seed_replication_pipeline.sh" \
  >>"${pipeline_log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pipeline_dir}/launcher.pid"
echo "Started paired-seed replication pipeline with launcher PID ${pid}."
echo "Stress must pass before 2026-08-25T22:35:56Z or the full runs are blocked."
echo "New full runs: two seeds x two treatments, sequential, all reported."
echo "Expected after stress: about 3.5 hours for four full runs."
echo "Log: ${pipeline_log}"
