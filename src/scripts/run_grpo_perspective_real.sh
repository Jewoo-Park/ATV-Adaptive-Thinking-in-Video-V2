#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export REASONING_TASK_TYPE="perspective"
export BALANCED_STRATEGY_ROLLOUT="${BALANCED_STRATEGY_ROLLOUT:-true}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-9}"
export ROLLOUTS_PER_STRATEGY="${ROLLOUTS_PER_STRATEGY:-3}"

if [[ -n "${1:-}" && -z "${TRAIN_FILE:-}" ]]; then
  export TRAIN_FILE="$1"
fi
if [[ -z "${TRAIN_FILE:-}" ]]; then
  echo "[GRPO-PERSPECTIVE-REAL] ERROR: TRAIN_FILE is required." >&2
  echo "[GRPO-PERSPECTIVE-REAL] Example: TRAIN_FILE=/path/to/perspective_train.jsonl bash src/scripts/run_grpo_perspective_real.sh" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[GRPO-PERSPECTIVE-REAL] ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  exit 1
fi

GRPO_JSONL_SKIP_MEDIA_CHECK="${GRPO_JSONL_SKIP_MEDIA_CHECK:-true}"
SCHEMA_VALIDATE_ARGS=(--input "${TRAIN_FILE}" --mode perspective)
if [[ "${GRPO_JSONL_SKIP_MEDIA_CHECK}" == "true" || "${GRPO_JSONL_SKIP_MEDIA_CHECK}" == "1" ]]; then
  SCHEMA_VALIDATE_ARGS+=(--skip-media-check)
fi
python3 src/scripts/validate_grpo_jsonl_schema.py "${SCHEMA_VALIDATE_ARGS[@]}"

if [[ -z "${QWEN_PATH:-}" || -z "${QWEN_BASE_PATH:-}" ]]; then
  echo "[GRPO-PERSPECTIVE-REAL] ERROR: Set QWEN_PATH and QWEN_BASE_PATH before launch." >&2
  exit 1
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/src/r1-v/outputs/video_r1_grpo_perspective_real}"
export RUN_NAME="${RUN_NAME:-video_r1_grpo_perspective_real_$(date +%Y%m%d_%H%M%S)}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug_perspective.jsonl}"

# Tie/unclear groups receive no strategy bonus by default; override only for controlled ablations.
echo "[GRPO-PERSPECTIVE-REAL] REASONING_TASK_TYPE=${REASONING_TASK_TYPE}"
echo "[GRPO-PERSPECTIVE-REAL] TRAIN_FILE=${TRAIN_FILE}"
echo "[GRPO-PERSPECTIVE-REAL] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[GRPO-PERSPECTIVE-REAL] RUN_NAME=${RUN_NAME}"
echo "[GRPO-PERSPECTIVE-REAL] BALANCED_STRATEGY_ROLLOUT=${BALANCED_STRATEGY_ROLLOUT}"
echo "[GRPO-PERSPECTIVE-REAL] NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "[GRPO-PERSPECTIVE-REAL] ROLLOUTS_PER_STRATEGY=${ROLLOUTS_PER_STRATEGY}"

bash src/scripts/run_grpo_answer_only_lora.sh
