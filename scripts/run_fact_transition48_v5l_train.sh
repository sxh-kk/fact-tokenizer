#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

TORCHRUN="${TORCHRUN:-/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-v5l_transition48_dense_from_v5f_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fact_tokenizer/${RUN_NAME}}"
TRAIN_NPZ="${TRAIN_NPZ:-data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-outputs/fact_tokenizer/v5f_4p0_teacher_usage_8gpu_20260617_192843/fact_tokenizer.ckpt}"

# v5f checkpoint is step 58999; 73000 runs 14000 additional optimization steps.
STEPS="${STEPS:-73000}"
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"

mkdir -p "$OUTPUT_DIR"

"$TORCHRUN" \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_fact_npz_debug.py \
  --ddp \
  --input-npz "$TRAIN_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --source-view-keys ego exo \
  --steps "$STEPS" \
  --batch-size "$PER_GPU_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --resize 224 \
  --backbone dino \
  --device cuda \
  --model-dim 128 \
  --dino-dim 768 \
  --latent-dim 32 \
  --private-dim 4 \
  --num-latents 64 \
  --num-action-slots 4 \
  --num-private-slots 1 \
  --num-heads 4 \
  --patch-size 14 \
  --enc-blocks 1 \
  --dec-blocks 1 \
  --lr 4.5e-6 \
  --save-every 2000 \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --discard-resume-history \
  --take-grouped-batches \
  --samples-per-take 8 \
  --vq-temperature 0.085 \
  --kl-weight 0.18 \
  --balance-weight 0.18 \
  --hard-usage-balance-weight 0.012 \
  --slot-balance-weight 0.006 \
  --assignment-entropy-weight 0.004 \
  --assignment-entropy-target 0.88 \
  --same-take-contrast-weight 0.08 \
  --take-uniform-weight 0.018 \
  --take-slot-uniform-weight 0.024 \
  --take-pair-uniform-weight 0.012 \
  --action-slot-dropout 0.16 \
  --action-slot-dropout-start-fraction 0.0 \
  --action-slot-dropout-ramp-fraction 0.05 \
  --action-only-weight 0.22 \
  --action-contrast-weight 0.20 \
  --no-private-contrast-weight 0.26 \
  --action-consistency-weight 0.055 \
  --exo-aux-multiplier 4.5 \
  --vq-beta 0.38 \
  --private-reg-weight 0.055 \
  --private-dropout 0.55 \
  --private-dropout-start-fraction 0.0 \
  --private-dropout-ramp-fraction 0.05 \
  --action-only-motion-focus-weight 0.045 \
  --motion-focus-weight 0.04 \
  --motion-contrast-weight 0.045 \
  --delta-focus-weight 0.025 \
  --action-only-delta-focus-weight 0.025 \
  --delta-contrast-weight 0.025 \
  --action-contrast-margin 0.008 \
  --action-aux-start-fraction 0.0 \
  --action-aux-ramp-fraction 0.05 \
  --teacher-ego-uncertainty-weight 2.0 \
  --teacher-disagreement-weight 3.0 \
  --teacher-base-bias -1.0 \
  2>&1 | tee "$OUTPUT_DIR/train_stdout.log"
