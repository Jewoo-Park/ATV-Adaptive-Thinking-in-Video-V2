#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source scripts/hpc_activate_grpo.sh

echo "[MAIN-PERSPECTIVE-8F] Preflight: python/venv/package/GPU checks"
echo "[MAIN-PERSPECTIVE-8F] python=$(which python)"
python -V
python - <<'PY'
import importlib
import json
import sys
import torch

mods = ["torch", "transformers", "peft", "vllm", "deepspeed"]
versions = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        versions[m] = getattr(mod, "__version__", "unknown")
    except Exception as e:
        versions[m] = f"ERROR: {type(e).__name__}: {e}"

status = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "torch_cuda": torch.version.cuda,
    "versions": versions,
}
print("[MAIN-PERSPECTIVE-8F] runtime=", json.dumps(status, ensure_ascii=False))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("[MAIN-PERSPECTIVE-8F] ERROR: CUDA GPU is not available in this session.")
    sys.exit(2)
PY

# Main clean PERSPECTIVE GRPO run:
# - base model MUST be merged PERSPECTIVE SFT
# - fixed 8-frame train/eval setting in trainer path
# - no smoke/backbone base
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"

export REASONING_TASK_TYPE="perspective"
export QWEN_PATH="${QWEN_PATH:-/scratch/users/ntu/n2500182/models/qwen25vl3b_lora_merged_perspective}"
export QWEN_BASE_PATH="${QWEN_BASE_PATH:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-3B-Instruct}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-3B-Instruct}"

export TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/video_r1/grpo/processed/video_r1_grpo_train_strict.jsonl}"
export SPLIT_TAG="${SPLIT_TAG:-perspective_eval0p05_seed42_v3}"
export TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-${REPO_ROOT}/data/video_r1/grpo/splits/video_r1_grpo_train_strict__train_perspective_eval0p05_seed42_v3.jsonl}"
export EVAL_SPLIT_FILE="${EVAL_SPLIT_FILE:-${REPO_ROOT}/data/video_r1/grpo/splits/video_r1_grpo_train_strict__eval_perspective_eval0p05_seed42_v3.jsonl}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_perspective_sftbase_strategy_fix_main_8frames}"
export RUN_NAME="${RUN_NAME:-grpo_perspective_sftbase_strategy_fix_main_8frames}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug.jsonl}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export VLLM_MAX_FRAMES="${VLLM_MAX_FRAMES:-8}"
export VLLM_MAX_FRAMES_EVAL="${VLLM_MAX_FRAMES_EVAL:-8}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-9}"
export ROLLOUTS_PER_STRATEGY="${ROLLOUTS_PER_STRATEGY:-3}"
export BALANCED_STRATEGY_ROLLOUT="${BALANCED_STRATEGY_ROLLOUT:-true}"
export ANSWER_ACCURACY_WEIGHT="${ANSWER_ACCURACY_WEIGHT:-0.65}"
export ANSWER_FORMAT_WEIGHT="${ANSWER_FORMAT_WEIGHT:-0.35}"
export STRATEGY_BONUS_SCALE="${STRATEGY_BONUS_SCALE:-0.20}"
export STRATEGY_BONUS_THRESHOLD="${STRATEGY_BONUS_THRESHOLD:-0.10}"
export TIE_BREAK_BONUS_SCALE="${TIE_BREAK_BONUS_SCALE:-0.0}"
export TEMPERATURE="${TEMPERATURE:-0.8}"

export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export MIN_PIXELS="${MIN_PIXELS:-100352}"
export MAX_PIXELS="${MAX_PIXELS:-100352}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export VLLM_USE_MM_PROCESSOR_KWARGS="${VLLM_USE_MM_PROCESSOR_KWARGS:-true}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export STRATEGY_EVAL_EVERY_STEPS="${STRATEGY_EVAL_EVERY_STEPS:-100}"
export STRATEGY_EVAL_MAX_SAMPLES="${STRATEGY_EVAL_MAX_SAMPLES:-60}"
export STRATEGY_EVAL_BATCH_SIZE="${STRATEGY_EVAL_BATCH_SIZE:-8}"
export LOG_STRATEGY_EVAL_METRICS="${LOG_STRATEGY_EVAL_METRICS:-true}"
export VLLM_SYNC_DEBUG_FIRST_N="${VLLM_SYNC_DEBUG_FIRST_N:-1}"
export VLLM_SYNC_VALIDATE_FIRST_N="${VLLM_SYNC_VALIDATE_FIRST_N:-1}"

echo "[MAIN-PERSPECTIVE-8F] Confirmed training frame count: ${VLLM_MAX_FRAMES} frames per video."
echo "[MAIN-PERSPECTIVE-8F] Confirmed eval frame count during training: ${VLLM_MAX_FRAMES_EVAL}."
echo "[MAIN-PERSPECTIVE-8F] Base merged SFT model: ${QWEN_PATH}"
echo "[MAIN-PERSPECTIVE-8F] Clean processor/base path: ${QWEN_BASE_PATH}"
echo "[MAIN-PERSPECTIVE-8F] Output directory: ${OUTPUT_DIR}"
echo "[MAIN-PERSPECTIVE-8F] MAX_STEPS=${MAX_STEPS}, NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[MAIN-PERSPECTIVE-8F] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[MAIN-PERSPECTIVE-8F] Pixel budget: MIN_PIXELS=${MIN_PIXELS}, MAX_PIXELS=${MAX_PIXELS}"
echo "[MAIN-PERSPECTIVE-8F] DType/Attn: TORCH_DTYPE=${TORCH_DTYPE}, ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[MAIN-PERSPECTIVE-8F] VLLM_USE_MM_PROCESSOR_KWARGS=${VLLM_USE_MM_PROCESSOR_KWARGS}"
echo "[MAIN-PERSPECTIVE-8F] VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK}, VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS}"
echo "[MAIN-PERSPECTIVE-8F] Periodic eval: STRATEGY_EVAL_EVERY_STEPS=${STRATEGY_EVAL_EVERY_STEPS}, STRATEGY_EVAL_MAX_SAMPLES=${STRATEGY_EVAL_MAX_SAMPLES}, STRATEGY_EVAL_BATCH_SIZE=${STRATEGY_EVAL_BATCH_SIZE}"
echo "[MAIN-PERSPECTIVE-8F] vLLM sync diagnostics: VLLM_SYNC_DEBUG_FIRST_N=${VLLM_SYNC_DEBUG_FIRST_N}, VLLM_SYNC_VALIDATE_FIRST_N=${VLLM_SYNC_VALIDATE_FIRST_N}"

bash src/scripts/run_grpo_perspective_real.sh
