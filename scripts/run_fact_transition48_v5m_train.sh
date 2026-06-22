#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export FACT_DDP_TIMEOUT_SEC="${FACT_DDP_TIMEOUT_SEC:-3600}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

TORCHRUN="${TORCHRUN:-/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_NAME="${RUN_NAME:-v5m_transition48_from_v5f_usage_causality_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fact_tokenizer/${RUN_NAME}}"
TRAIN_NPZ="${TRAIN_NPZ:-data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-outputs/fact_tokenizer/v5f_4p0_teacher_usage_8gpu_20260617_192843/fact_tokenizer.ckpt}"

# v5f checkpoint is step 58999; 72000 runs about 13000 additional optimization steps.
STEPS="${STEPS:-72000}"
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

if [[ ! -f "$TRAIN_NPZ" ]]; then
  echo "Missing TRAIN_NPZ: $TRAIN_NPZ" >&2
  exit 1
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing RESUME_CHECKPOINT: $RESUME_CHECKPOINT" >&2
  exit 1
fi

CMD=(
  "$TORCHRUN"
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  scripts/train_fact_npz_debug.py
  --ddp
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
  --lr 4.5e-6
  --save-every 2000
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --discard-resume-history
  --take-grouped-batches
  --samples-per-take 8
  --vq-temperature 0.072
  --kl-weight 0.10
  --balance-weight 0.13
  --hard-usage-balance-weight 0.007
  --slot-balance-weight 0.003
  --assignment-entropy-weight 0.003
  --assignment-entropy-target 0.82
  --same-take-contrast-weight 0.06
  --take-uniform-weight 0.008
  --take-slot-uniform-weight 0.010
  --take-pair-uniform-weight 0.005
  --action-slot-dropout 0.16
  --action-slot-dropout-start-fraction 0.0
  --action-slot-dropout-ramp-fraction 0.05
  --action-only-weight 0.24
  --action-contrast-weight 0.22
  --no-private-contrast-weight 0.28
  --action-consistency-weight 0.025
  --exo-aux-multiplier 4.0
  --vq-beta 0.38
  --private-reg-weight 0.055
  --private-dropout 0.55
  --private-dropout-start-fraction 0.0
  --private-dropout-ramp-fraction 0.05
  --action-only-motion-focus-weight 0.050
  --motion-focus-weight 0.040
  --motion-contrast-weight 0.045
  --delta-focus-weight 0.025
  --action-only-delta-focus-weight 0.030
  --delta-contrast-weight 0.025
  --action-contrast-margin 0.008
  --action-aux-start-fraction 0.0
  --action-aux-ramp-fraction 0.05
  --teacher-ego-uncertainty-weight 2.0
  --teacher-disagreement-weight 3.0
  --teacher-base-bias -1.0
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train_stdout.log"
