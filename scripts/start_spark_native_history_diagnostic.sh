#!/usr/bin/env bash
set -euo pipefail

# This versioned wrapper cannot enter the 2,048-update main suite. It exists only to capture
# exact multi-update native-Qwen numerical telemetry after the preserved v1 stress failure.
export GRIDNAV_NATIVE_HISTORY_RUN_ID="${GRIDNAV_NATIVE_HISTORY_DIAGNOSTIC_RUN_ID:-native_history_select_v2}"
exec bash scripts/start_spark_native_history_select.sh --stress-only
