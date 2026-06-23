#!/usr/bin/env bash
set -euo pipefail

export PAIR_MODE="${PAIR_MODE:-temporal}"
export PAIR_LABEL="${PAIR_LABEL:-control_temporal}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-_smoke_v6e_control_temporal_from_v6b_8gpu_$(date +%Y%m%d_%H%M%S)}"
else
  export RUN_NAME="${RUN_NAME:-v6e_control_temporal_from_v6b_8gpu_$(date +%Y%m%d_%H%M%S)}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fact_transition48_v6e_mined_from_v6b_8gpu.sh" "$@"
