#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"
pipeline_dir="${repo}/artifacts/spark/native_optimizer_pipeline_v4"
prior_pipeline_dir="${repo}/artifacts/spark/native_optimizer_pipeline_v3"

python3 "${repo}/scripts/check_context_state.py" \
  --repo "${repo}" \
  --output "${repo}/artifacts/context/context_state.json"

if [[ -d "${pipeline_dir}" ]] && [[ -n "$(find "${pipeline_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Pipeline directory ${pipeline_dir} is non-empty; preserve it and inspect the prior attempt." >&2
  exit 1
fi
if [[ -f "${prior_pipeline_dir}/launcher.pid" ]]; then
  prior_pid="$(tr -cd '0-9' <"${prior_pipeline_dir}/launcher.pid")"
  if [[ -n "${prior_pid}" ]] \
    && kill -0 "${prior_pid}" 2>/dev/null \
    && ps -p "${prior_pid}" -o command= | grep -Fq run_spark_native_optimizer_pipeline.sh; then
    echo "The v3 supervisor is still active as PID ${prior_pid}; refusing a competing supervisor." >&2
    exit 1
  fi
fi

mkdir -p "${pipeline_dir}"
: >"${pipeline_dir}/run.log"
nohup bash "${repo}/scripts/run_spark_native_optimizer_pipeline.sh" \
  >>"${pipeline_dir}/run.log" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pipeline_dir}/launcher.pid"

echo "Started fail-closed optimizer pipeline v4 with launcher PID ${pid}."
echo "Stages: processor v5 -> disjoint sensitivity v2 -> optimizer stress -> main range finder."
echo "The supervisor reuses passed stages, joins a live named container, and preserves abandoned attempts."
echo "A failed verification or non-sensitive v2 verdict prevents every downstream stage."
echo "Expected total: roughly 1-6 hours; the main stage has a hard 5-hour cap."
echo "Pipeline log: ${pipeline_dir}/run.log"
