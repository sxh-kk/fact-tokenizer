#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PYTHON="${PYTHON:-/home/sxh/.conda/envs/fact_tokenizer/bin/python}"
RUN_NAME="${RUN_NAME:-v6b_transition48_delta_full_single_gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fact_tokenizer/${RUN_NAME}}"
TRAIN_NPZ="${TRAIN_NPZ:-outputs/fact_tokenizer/nofilter_t1p0_train_by_take_npy_mmap}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt}"
STEPS="${STEPS:-91000}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TAKE_WEIGHT_CSV="${TAKE_WEIGHT_CSV:-}"
TRANSITION_WEIGHT_CSV="${TRANSITION_WEIGHT_CSV:-}"
LIGHT_AUGMENT="${LIGHT_AUGMENT:-0}"

if [[ ! -f "$TRAIN_NPZ" && ! -d "$TRAIN_NPZ" ]]; then
  echo "Missing TRAIN_NPZ path: $TRAIN_NPZ" >&2
  exit 1
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing RESUME_CHECKPOINT: $RESUME_CHECKPOINT" >&2
  exit 1
fi

CMD=(
  "$PYTHON"
  -u
  scripts/train_fact_npz_debug.py
  --input-npz "$TRAIN_NPZ"
  --output-dir "$OUTPUT_DIR"
  --source-view-keys ego exo
  --steps "$STEPS"
  --batch-size "$PER_GPU_BATCH"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --resize 224
  --backbone dino
  --device cuda
  --model-dim 128
  --dino-dim 768
  --latent-dim 32
  --private-dim 4
  --num-latents 64
  --num-action-slots 4
  --num-private-slots 1
  --num-heads 4
  --patch-size 14
  --enc-blocks 1
  --dec-blocks 1
  --current-context-mode full
  --current-context-tokens 0
  --lr 2.0e-6
  --save-every 2000
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --discard-resume-history
  --take-grouped-batches
  --samples-per-take 8
  --vq-temperature 0.050
  --vq-beta 0.40
  --kl-weight 0.08
  --balance-weight 0.10
  --hard-usage-balance-weight 0.001
  --motion-gated-usage-weight 0.0005
  --motion-gated-usage-gamma 2.0
  --slot-balance-weight 0.001
  --slot-diversity-weight 0.003
  --assignment-entropy-weight 0.004
  --assignment-entropy-target 0.75
  --same-take-contrast-weight 0.060
  --no-private-same-take-contrast-weight 0.080
  --temporal-offset-contrast-weight 0.030
  --no-private-temporal-offset-contrast-weight 0.050
  --temporal-offset 4
  --take-uniform-weight 0.003
  --take-slot-uniform-weight 0.003
  --take-pair-uniform-weight 0.001
  --action-slot-dropout 0.10
  --action-slot-dropout-start-fraction 0.0
  --action-slot-dropout-ramp-fraction 0.05
  --action-only-weight 0.30
  --action-contrast-weight 0.22
  --no-private-contrast-weight 0.30
  --random-code-contrast-weight 0.15
  --no-private-random-code-contrast-weight 0.22
  --zero-action-contrast-weight 0.035
  --no-private-zero-action-contrast-weight 0.070
  --action-consistency-weight 0.015
  --exo-aux-multiplier 3.5
  --private-reg-weight 0.065
  --private-dropout 0.60
  --private-dropout-start-fraction 0.0
  --private-dropout-ramp-fraction 0.05
  --motion-focus-weight 0.020
  --action-only-motion-focus-weight 0.050
  --motion-contrast-weight 0.030
  --delta-focus-weight 0.018
  --action-only-delta-focus-weight 0.030
  --delta-contrast-weight 0.018
  --no-private-delta-contrast-weight 0.030
  --delta-direction-magnitude-weight 0.20
  --action-contrast-margin 0.013
  --action-aux-start-fraction 0.0
  --action-aux-ramp-fraction 0.05
  --teacher-ego-uncertainty-weight 1.5
  --teacher-disagreement-weight 2.5
  --teacher-base-bias -1.1
)

if [[ -n "$TAKE_WEIGHT_CSV" ]]; then
  if [[ ! -f "$TAKE_WEIGHT_CSV" ]]; then
    echo "Missing TAKE_WEIGHT_CSV: $TAKE_WEIGHT_CSV" >&2
    exit 1
  fi
  CMD+=(--take-weight-csv "$TAKE_WEIGHT_CSV")
fi

if [[ -n "$TRANSITION_WEIGHT_CSV" ]]; then
  if [[ ! -f "$TRANSITION_WEIGHT_CSV" ]]; then
    echo "Missing TRANSITION_WEIGHT_CSV: $TRANSITION_WEIGHT_CSV" >&2
    exit 1
  fi
  CMD+=(--transition-weight-csv "$TRANSITION_WEIGHT_CSV")
fi

if [[ "$LIGHT_AUGMENT" == "1" ]]; then
  CMD+=(--light-augment)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train_stdout.log"
