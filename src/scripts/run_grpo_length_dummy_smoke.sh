#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-data/generated_dummy_grpo_smoke/length/length_grpo_dummy.jsonl}"
if [[ ! -s "${DATA_PATH}" ]]; then
  python scripts/create_dummy_length_data.py
fi
DATA_PATH="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${DATA_PATH}")"

DEFAULT_MODEL_PATH="${REPO_ROOT}/models/qwen25vl3b_lora_merged_length"
if [[ -d "${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct-smoke" ]]; then
  DEFAULT_MODEL_PATH="${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct-smoke"
fi

export REASONING_TASK_TYPE="length"
export QWEN_PATH="${MODEL_PATH:-${QWEN_PATH:-${DEFAULT_MODEL_PATH}}}"
export QWEN_BASE_PATH="${QWEN_BASE_PATH:-${QWEN_PATH}}"
export TRAIN_FILE="${DATA_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tmp_smoke/grpo_length_dummy}"
export NUM_GPUS="${NUM_GPUS:-4}"
export MAX_STEPS="${MAX_STEPS:-50}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-3}"
export ROLLOUTS_PER_STRATEGY="${ROLLOUTS_PER_STRATEGY:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE:-1}}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-96}"
export MAX_PIXELS="${MAX_PIXELS:-50176}"
export MIN_PIXELS="${MIN_PIXELS:-50176}"
export VLLM_MAX_FRAMES="${MAX_FRAMES:-${VLLM_MAX_FRAMES:-8}}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.25}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export GRPO_EVAL_FRACTION="${GRPO_EVAL_FRACTION:-0.2}"
export SPLIT_DIR="${SPLIT_DIR:-$(dirname "${DATA_PATH}")}"
export GRPO_TRAIN_VIDEO_ONLY="${GRPO_TRAIN_VIDEO_ONLY:-true}"
export BALANCED_STRATEGY_ROLLOUT="${BALANCED_STRATEGY_ROLLOUT:-true}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug.jsonl}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export GRPO_APPLY_ROTARY_DTYPE_HOTFIX="${GRPO_APPLY_ROTARY_DTYPE_HOTFIX:-false}"

bash src/scripts/run_grpo_answer_only_lora.sh
