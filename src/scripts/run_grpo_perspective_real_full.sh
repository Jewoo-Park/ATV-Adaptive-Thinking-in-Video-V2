#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export TRAIN_FILE="/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo/video_r1_grpo_train.jsonl"
export SPLIT_DIR="/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo"
export QWEN_PATH="/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke"
export QWEN_BASE_PATH="/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke"

export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_real_9rollout}"
export RUN_NAME="${RUN_NAME:-grpo_perspective_real_9rollout}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug.jsonl}"

export NUM_GPUS="${NUM_GPUS:-4}"
export TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"
export MAX_STEPS="${MAX_STEPS:-1000}"

export REASONING_TASK_TYPE="perspective"
export BALANCED_STRATEGY_ROLLOUT="true"
export NUM_GENERATIONS="9"
export ROLLOUTS_PER_STRATEGY="3"

export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-96}"
export MAX_PIXELS="${MAX_PIXELS:-50176}"
export MIN_PIXELS="${MIN_PIXELS:-50176}"
export VLLM_MAX_FRAMES="${VLLM_MAX_FRAMES:-8}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.25}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export GRPO_APPLY_ROTARY_DTYPE_HOTFIX="${GRPO_APPLY_ROTARY_DTYPE_HOTFIX:-false}"
export ANSWER_ACCURACY_WEIGHT="${ANSWER_ACCURACY_WEIGHT:-0.65}"
export ANSWER_FORMAT_WEIGHT="${ANSWER_FORMAT_WEIGHT:-0.35}"
export STRATEGY_BONUS_SCALE="${STRATEGY_BONUS_SCALE:-0.20}"
export STRATEGY_BONUS_THRESHOLD="${STRATEGY_BONUS_THRESHOLD:-0.10}"
export TIE_BREAK_BONUS_SCALE="${TIE_BREAK_BONUS_SCALE:-0.0}"
export TEMPERATURE="${TEMPERATURE:-0.8}"

# Tie/unclear groups receive no strategy bonus by default.
bash src/scripts/run_grpo_perspective_real.sh
