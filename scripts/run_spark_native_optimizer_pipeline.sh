#!/usr/bin/env bash
set -euo pipefail

repo="${GRIDNAV_REPO:-$HOME/Desktop/GridNav}"

wait_for_gate() {
  local container_name="$1"
  local verification_path="$2"
  # Launchers return only after the Docker daemon has registered the named container (or a
  # very fast stage has already written verification). This function owns only completion.
  while docker inspect "${container_name}" >/dev/null 2>&1; do
    sleep 15
  done
  if [[ ! -f "${verification_path}" ]]; then
    echo "Gate ${container_name} ended without ${verification_path}." >&2
    exit 1
  fi
}

run_or_reuse_stage() {
  local label="$1"
  local launcher="$2"
  local container_name="$3"
  local verification_path="$4"
  local stage_directory="$5"

  if [[ -f "${verification_path}" ]]; then
    echo "${label}: reusing existing independently verified attempt."
    require_passed_verification "${verification_path}"
    return 0
  fi
  if docker inspect "${container_name}" >/dev/null 2>&1; then
    echo "${label}: joining already-registered container ${container_name}."
    wait_for_gate "${container_name}" "${verification_path}"
    require_passed_verification "${verification_path}"
    return 0
  fi
  if [[ -d "${stage_directory}" ]] \
    && [[ -n "$(find "${stage_directory}" -mindepth 1 -print -quit)" ]]; then
    echo "${label}: ${stage_directory} is non-empty, but no live container or verification exists." >&2
    echo "Preserving the incomplete attempt and stopping; do not overwrite it." >&2
    exit 1
  fi

  bash "${launcher}"
  wait_for_gate "${container_name}" "${verification_path}"
  require_passed_verification "${verification_path}"
}

require_passed_verification() {
  local verification_path="$1"
  python3 - "${verification_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
if record.get("status") != "passed" or record.get("error_count") != 0:
    raise SystemExit(f"Independent verification did not pass: {path}")
print(f"Verified gate: {record.get('gate')}")
PY
}

cd "${repo}"

echo "[1/4] Corrected one-token processor preflight"
run_or_reuse_stage \
  "processor preflight" \
  "${repo}/scripts/start_spark_native_qwen_history_preflight.sh" \
  gridnav-native-qwen-history-preflight-v5 \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5/verification.json" \
  "${repo}/artifacts/spark/native_qwen_history_preflight_v5"

echo "[2/4] Disjoint saturation-robust visual-sensitivity confirmation"
run_or_reuse_stage \
  "visual sensitivity v2" \
  "${repo}/scripts/start_spark_native_sensitivity_v2.sh" \
  gridnav-native-sensitivity-v2 \
  "${repo}/artifacts/spark/native_sensitivity_v2/verification.json" \
  "${repo}/artifacts/spark/native_sensitivity_v2"
python3 - "${repo}/artifacts/spark/native_sensitivity_v2/result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
verdict = record.get("summary", {}).get("verdict")
if verdict != "logit_sensitive_admit_optimizer_diagnostic":
    raise SystemExit(f"Sensitivity-v2 verdict blocks optimizer study: {verdict}")
print(f"Sensitivity verdict: {verdict}")
PY

echo "[3/4] One-update-per-arm optimizer stress suite"
run_or_reuse_stage \
  "optimizer stress" \
  "${repo}/scripts/start_spark_native_optimizer_stress.sh" \
  gridnav-native-optimizer-stress-v1 \
  "${repo}/artifacts/spark/native_optimizer_stress_v1/verification.json" \
  "${repo}/artifacts/spark/native_optimizer_stress_v1"

echo "[4/4] Main optimizer numerical range finder"
run_or_reuse_stage \
  "optimizer range finder" \
  "${repo}/scripts/start_spark_native_optimizer_admission.sh" \
  gridnav-native-optimizer-admission-v1 \
  "${repo}/artifacts/spark/native_optimizer_admission_v1/verification.json" \
  "${repo}/artifacts/spark/native_optimizer_admission_v1"

echo "Native optimizer-admission pipeline PASSED all four independently verified stages."
