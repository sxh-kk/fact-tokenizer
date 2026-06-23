#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-100}"
STABLE_POLLS="${STABLE_POLLS:-2}"
ALERT_REPEAT="${ALERT_REPEAT:-8}"
ONCE="${ONCE:-1}"
QUIET="${QUIET:-0}"
CHECK_ONCE="${CHECK_ONCE:-0}"

usage() {
  cat <<'EOF'
Watch target GPUs and alert when they are idle.

Default:
  watch_8gpu_free.sh

Common overrides:
  GPU_IDS=0,1,2,3,4,5,6,7 INTERVAL_SEC=60 bash watch_8gpu_free.sh
  GPU_IDS=1,2,3,4,5,6,7 bash watch_8gpu_free.sh
  CHECK_ONCE=1 bash watch_8gpu_free.sh

Idle means:
  - no CUDA compute process is running on the target GPUs
  - memory.used on every target GPU is <= MEM_THRESHOLD_MIB

The memory threshold allows tiny display processes such as /usr/lib/xorg/Xorg.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --check-once)
      CHECK_ONCE=1
      ;;
    --keep-running)
      ONCE=0
      ;;
    --quiet)
      QUIET=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 127
  fi
}

split_gpu_ids() {
  local ids_csv="$1"
  local old_ifs="$IFS"
  IFS=","
  read -r -a GPU_ID_ARRAY <<< "$ids_csv"
  IFS="$old_ifs"
}

target_uuid_lines() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
    awk -F, -v ids=",$GPU_IDS," '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if (index(ids, "," $1 ",") > 0) {
          print $2
        }
      }
    '
}

target_memory_lines() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, -v ids=",$GPU_IDS," '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if (index(ids, "," $1 ",") > 0) {
          print $1 "," $2
        }
      }
    '
}

target_compute_apps() {
  local uuid_regex
  uuid_regex="$(target_uuid_lines | paste -sd'|' -)"
  if [[ -z "$uuid_regex" ]]; then
    return 0
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v uuid_regex="$uuid_regex" '
      $1 ~ uuid_regex {
        print
      }
    ' || true
}

memory_is_idle() {
  local line index used seen=0
  while IFS=, read -r index used; do
    [[ -z "${index:-}" ]] && continue
    seen=$((seen + 1))
    if (( used > MEM_THRESHOLD_MIB )); then
      return 1
    fi
  done < <(target_memory_lines)
  split_gpu_ids "$GPU_IDS"
  (( seen == ${#GPU_ID_ARRAY[@]} ))
}

is_idle() {
  local apps
  apps="$(target_compute_apps)"
  [[ -z "$apps" ]] && memory_is_idle
}

status_line() {
  local apps mem
  apps="$(target_compute_apps)"
  mem="$(target_memory_lines | tr '\n' ' ')"
  if [[ -n "$apps" ]]; then
    echo "busy compute apps: $(echo "$apps" | tr '\n' '; ') | mem: $mem"
  else
    echo "no target compute apps | mem: $mem"
  fi
}

send_alert() {
  local msg="All target GPUs are idle: GPU_IDS=$GPU_IDS at $(date '+%F %T')"
  printf '\n\a\a%s\n%s\n%s\n\n' "============================================================" "$msg" "============================================================"

  if [[ -n "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    tmux display-message -d 15000 "$msg" || true
  fi
  if [[ -n "${DISPLAY:-}" ]] && command -v notify-send >/dev/null 2>&1; then
    notify-send "GPU idle" "$msg" || true
  fi

  local i
  for ((i = 0; i < ALERT_REPEAT; i++)); do
    printf '\a'
    sleep 1
  done
}

need_command nvidia-smi

stable=0
while true; do
  if is_idle; then
    stable=$((stable + 1))
    [[ "$QUIET" == "1" ]] || echo "[$(date '+%F %T')] idle poll $stable/$STABLE_POLLS: $(status_line)"
  else
    stable=0
    [[ "$QUIET" == "1" ]] || echo "[$(date '+%F %T')] busy: $(status_line)"
  fi

  if (( CHECK_ONCE == 1 )); then
    (( stable > 0 ))
    exit $?
  fi

  if (( stable >= STABLE_POLLS )); then
    send_alert
    if (( ONCE == 1 )); then
      exit 0
    fi
    stable=0
  fi

  sleep "$INTERVAL_SEC"
done
