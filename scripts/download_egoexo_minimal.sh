#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/egoexo4d}"
UID_FILE="${UID_FILE:-data/egoexo4d/fact_debug/uids.txt}"
RELEASE="${RELEASE:-v2}"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/home/sxh/.conda/envs/fact_tokenizer/bin}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be exported in the shell." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_ENV_BIN}/egoexo" \
  -o "${OUT_DIR}" \
  --release "${RELEASE}" \
  --parts metadata \
  --views ego exo \
  -y

if [[ -f "${UID_FILE}" ]]; then
  "${PYTHON_ENV_BIN}/egoexo" \
    -o "${OUT_DIR}" \
    --release "${RELEASE}" \
    --parts downscaled_takes/448 \
    --views ego exo \
    --uids $(cat "${UID_FILE}") \
    -y
else
  echo "UID file not found: ${UID_FILE}. Run scripts/select_egoexo_fact_uids.py first." >&2
fi
