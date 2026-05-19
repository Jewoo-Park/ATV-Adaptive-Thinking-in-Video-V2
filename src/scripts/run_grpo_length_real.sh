#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export REASONING_TASK_TYPE="length"
export BALANCED_STRATEGY_ROLLOUT="${BALANCED_STRATEGY_ROLLOUT:-true}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-9}"
export ROLLOUTS_PER_STRATEGY="${ROLLOUTS_PER_STRATEGY:-3}"

if [[ -n "${1:-}" && -z "${TRAIN_FILE:-}" ]]; then
  export TRAIN_FILE="$1"
fi
if [[ -z "${TRAIN_FILE:-}" ]]; then
  echo "[GRPO-LENGTH-REAL] ERROR: TRAIN_FILE is required." >&2
  echo "[GRPO-LENGTH-REAL] Example: TRAIN_FILE=/path/to/length_train.jsonl bash src/scripts/run_grpo_length_real.sh" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[GRPO-LENGTH-REAL] ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  exit 1
fi

python3 src/scripts/validate_grpo_jsonl_schema.py --input "${TRAIN_FILE}" --mode length

if [[ -z "${QWEN_PATH:-}" || -z "${QWEN_BASE_PATH:-}" ]]; then
  echo "[GRPO-LENGTH-REAL] ERROR: Set QWEN_PATH and QWEN_BASE_PATH before launch." >&2
  exit 1
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/src/r1-v/outputs/video_r1_grpo_length_real}"
export RUN_NAME="${RUN_NAME:-video_r1_grpo_length_real_$(date +%Y%m%d_%H%M%S)}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug_length.jsonl}"

# Tie/unclear groups receive no strategy bonus by default; override only for controlled ablations.
echo "[GRPO-LENGTH-REAL] REASONING_TASK_TYPE=${REASONING_TASK_TYPE}"
echo "[GRPO-LENGTH-REAL] TRAIN_FILE=${TRAIN_FILE}"
echo "[GRPO-LENGTH-REAL] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[GRPO-LENGTH-REAL] RUN_NAME=${RUN_NAME}"
echo "[GRPO-LENGTH-REAL] BALANCED_STRATEGY_ROLLOUT=${BALANCED_STRATEGY_ROLLOUT}"
echo "[GRPO-LENGTH-REAL] NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "[GRPO-LENGTH-REAL] ROLLOUTS_PER_STRATEGY=${ROLLOUTS_PER_STRATEGY}"

bash src/scripts/run_grpo_answer_only_lora.sh
