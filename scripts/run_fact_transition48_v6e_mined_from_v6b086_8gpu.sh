#!/usr/bin/env bash
set -euo pipefail

export PAIR_MODE="${PAIR_MODE:-mined}"
export PAIR_LABEL="${PAIR_LABEL:-mined_v6b086}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-outputs/fact_tokenizer/_sweep_v6b_checkpoints_heldout_20260623_161957/patched_ckpts/fact_tokenizer_step_086000.ckpt}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-_smoke_v6e_mined_from_v6b086_8gpu_$(date +%Y%m%d_%H%M%S)}"
  export STEPS="${STEPS:-86200}"
else
  export RUN_NAME="${RUN_NAME:-v6e_mined_from_v6b086_8gpu_$(date +%Y%m%d_%H%M%S)}"
  export STEPS="${STEPS:-89000}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fact_transition48_v6e_mined_from_v6b_8gpu.sh" "$@"
