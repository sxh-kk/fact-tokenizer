#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export FACT_DDP_TIMEOUT_SEC="${FACT_DDP_TIMEOUT_SEC:-3600}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PYTHON="${PYTHON:-/home/intern02/miniconda3/envs/egoexo_fact/bin/python}"
TORCHRUN="${TORCHRUN:-/home/intern02/miniconda3/envs/egoexo_fact/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PAIR_MODE="${PAIR_MODE:-mined}"
PAIR_LABEL="${PAIR_LABEL:-$PAIR_MODE}"
TRAIN_NPZ="${TRAIN_NPZ:-data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz}"
HELDOUT_NPZ="${HELDOUT_NPZ:-data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz}"
PAIR_MAP_DIR="${PAIR_MAP_DIR:-outputs/fact_tokenizer/v6e_pair_maps}"
PAIR_MAP="${PAIR_MAP:-${PAIR_MAP_DIR}/v6e_${PAIR_LABEL}_same_take_pairs_train.npz}"
HELDOUT_PAIR_MAP="${HELDOUT_PAIR_MAP:-${PAIR_MAP_DIR}/v6e_${PAIR_LABEL}_same_take_pairs_heldout.npz}"
PAIR_REVIEW_MARKER="${PAIR_REVIEW_MARKER:-${PAIR_MAP%.npz}.reviewed}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME:-_smoke_v6e_${PAIR_LABEL}_from_v6b_8gpu_$(date +%Y%m%d_%H%M%S)}"
  STEPS="${STEPS:-90200}"
else
  RUN_NAME="${RUN_NAME:-v6e_${PAIR_LABEL}_same_take_from_v6b_8gpu_$(date +%Y%m%d_%H%M%S)}"
  STEPS="${STEPS:-95000}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fact_tokenizer/${RUN_NAME}}"

PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MINED_TOP_K="${MINED_TOP_K:-4}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  AUTO_BUILD_PAIR_MAP="${AUTO_BUILD_PAIR_MAP:-0}"
  BUILD_VISUALS="${BUILD_VISUALS:-0}"
  REQUIRE_PAIR_REVIEW="${REQUIRE_PAIR_REVIEW:-0}"
else
  AUTO_BUILD_PAIR_MAP="${AUTO_BUILD_PAIR_MAP:-1}"
  BUILD_VISUALS="${BUILD_VISUALS:-1}"
  REQUIRE_PAIR_REVIEW="${REQUIRE_PAIR_REVIEW:-1}"
fi

if [[ "$PAIR_MODE" != "mined" && "$PAIR_MODE" != "temporal" ]]; then
  echo "PAIR_MODE must be mined or temporal, got: $PAIR_MODE" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_NPZ" ]]; then
  echo "Missing TRAIN_NPZ: $TRAIN_NPZ" >&2
  exit 1
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing RESUME_CHECKPOINT: $RESUME_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$PAIR_MAP_DIR"
if [[ "$AUTO_BUILD_PAIR_MAP" == "1" && ! -f "$PAIR_MAP" ]]; then
  "$PYTHON" scripts/build_fact_same_take_hard_negative_pairs.py \
    --input-npz "$TRAIN_NPZ" \
    --output-npz "$PAIR_MAP" \
    --source-view-keys ego exo \
    --mode "$PAIR_MODE" \
    --top-k "$MINED_TOP_K" \
    --min-gap 3 \
    --max-gap 24 \
    --batch-size 32 \
    --num-workers "$NUM_WORKERS" \
    --device cuda
fi
if [[ "$AUTO_BUILD_PAIR_MAP" == "1" && "$BUILD_VISUALS" == "1" && -f "$PAIR_MAP" ]]; then
  "$PYTHON" scripts/visualize_fact_mined_pairs.py \
    --input-npz "$TRAIN_NPZ" \
    --pair-map "$PAIR_MAP" \
    --output-dir "${PAIR_MAP_DIR}/visual_${PAIR_LABEL}_train" \
    --count 300
fi
if [[ "$AUTO_BUILD_PAIR_MAP" == "1" && "$BUILD_VISUALS" == "1" && -f "$HELDOUT_NPZ" && ! -f "$HELDOUT_PAIR_MAP" ]]; then
  "$PYTHON" scripts/build_fact_same_take_hard_negative_pairs.py \
    --input-npz "$HELDOUT_NPZ" \
    --output-npz "$HELDOUT_PAIR_MAP" \
    --source-view-keys ego exo \
    --mode "$PAIR_MODE" \
    --top-k "$MINED_TOP_K" \
    --min-gap 3 \
    --max-gap 24 \
    --batch-size 32 \
    --num-workers "$NUM_WORKERS" \
    --device cuda
fi
if [[ "$AUTO_BUILD_PAIR_MAP" == "1" && "$BUILD_VISUALS" == "1" && -f "$HELDOUT_PAIR_MAP" ]]; then
  "$PYTHON" scripts/visualize_fact_mined_pairs.py \
    --input-npz "$HELDOUT_NPZ" \
    --pair-map "$HELDOUT_PAIR_MAP" \
    --output-dir "${PAIR_MAP_DIR}/visual_${PAIR_LABEL}_heldout" \
    --count 100
fi
if [[ "${DRY_RUN:-0}" != "1" && ! -f "$PAIR_MAP" ]]; then
  echo "Missing PAIR_MAP: $PAIR_MAP" >&2
  exit 1
fi
if [[ "$REQUIRE_PAIR_REVIEW" == "1" && ! -f "$PAIR_REVIEW_MARKER" ]]; then
  echo "Pair maps and visuals are ready." >&2
  echo "Review ${PAIR_MAP_DIR}/visual_${PAIR_LABEL}_train and ${PAIR_MAP_DIR}/visual_${PAIR_LABEL}_heldout." >&2
  echo "If acceptable, run: touch '$PAIR_REVIEW_MARKER'" >&2
  echo "Or bypass review for smoke only with REQUIRE_PAIR_REVIEW=0." >&2
  exit 2
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
  --current-context-mode full
  --current-context-tokens 0
  --lr 1.5e-6
  --save-every 1000
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --discard-resume-history
  --mined-negative-map "$PAIR_MAP"
  --mined-negative-top-k "$MINED_TOP_K"
  --mined-pair-batches
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
  --mined-same-take-contrast-weight 0.080
  --no-private-mined-same-take-contrast-weight 0.100
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

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train_stdout.log"
