#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Re-sourcing hpc_activate runs `module purge` and can unload CUDA on an already-active GPU session.
if [[ "${VIRTUAL_ENV:-}" == *".venv_grpo"* ]]; then
  echo "[MAIN-LENGTH-8F] Reusing active venv_grpo: ${VIRTUAL_ENV}"
else
  source scripts/hpc_activate_grpo.sh
fi

echo "[MAIN-LENGTH-8F] Preflight: python/venv/package/GPU checks"
echo "[MAIN-LENGTH-8F] python=$(which python)"
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
print("[MAIN-LENGTH-8F] runtime=", json.dumps(status, ensure_ascii=False))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("[MAIN-LENGTH-8F] ERROR: CUDA GPU is not available in this session.")
    sys.exit(2)
PY

# Main clean LENGTH GRPO run:
# - base model MUST be merged LENGTH SFT
# - fixed 8-frame train/eval setting in trainer path
# - no smoke/backbone base
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"

export REASONING_TASK_TYPE="length"
export QWEN_PATH="${QWEN_PATH:-/scratch/users/ntu/n2500182/models/qwen25vl3b_lora_merged_length}"
export QWEN_BASE_PATH="${QWEN_BASE_PATH:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-3B-Instruct}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-3B-Instruct}"

export TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/video_r1/grpo/processed/video_r1_grpo_train_strict.jsonl}"
export SPLIT_TAG="${SPLIT_TAG:-length_eval0p05_seed42_v3}"
export TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-${REPO_ROOT}/data/video_r1/grpo/splits/video_r1_grpo_train_strict__train_length_eval0p05_seed42_v3.jsonl}"
export EVAL_SPLIT_FILE="${EVAL_SPLIT_FILE:-${REPO_ROOT}/data/video_r1/grpo/splits/video_r1_grpo_train_strict__eval_length_eval0p05_seed42_v3.jsonl}"

_SCRATCH_GRPO_OUT="${SCRATCH_GRPO_OUT:-/scratch/users/ntu/n2500182/grpo_outputs}"
_GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${_SCRATCH_GRPO_OUT}/outputs_0626}"
export OUTPUT_DIR="${OUTPUT_DIR:-${_GRPO_RUN_ROOT}/grpo_length_sftbase_strategy_fix_main_8frames}"
export RUN_NAME="${RUN_NAME:-grpo_length_sftbase_strategy_fix_main_8frames_outputs_0626}"
export STRATEGY_DEBUG_LOG_PATH="${STRATEGY_DEBUG_LOG_PATH:-${OUTPUT_DIR}/strategy_debug.jsonl}"
mkdir -p "${OUTPUT_DIR}"

# vLLM shares GPU 3 with 3 DDP ranks on 0–2; keep util conservative (Jun-23 success runs used scratch + ~0.25–0.4).
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.25}"
export VLLM_NUM_GPU_BLOCKS_CAP="${VLLM_NUM_GPU_BLOCKS_CAP:-2048}"
export TORCHELASTIC_ERROR_FILE="${TORCHELASTIC_ERROR_FILE:-${OUTPUT_DIR}/torch_elastic_error.json}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export VLLM_MAX_FRAMES="${VLLM_MAX_FRAMES:-8}"
export VLLM_MAX_FRAMES_EVAL="${VLLM_MAX_FRAMES_EVAL:-8}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-9}"
export ROLLOUTS_PER_STRATEGY="${ROLLOUTS_PER_STRATEGY:-3}"
export BALANCED_STRATEGY_ROLLOUT="${BALANCED_STRATEGY_ROLLOUT:-true}"
export ANSWER_ACCURACY_WEIGHT="${ANSWER_ACCURACY_WEIGHT:-0.8}"
export ANSWER_FORMAT_WEIGHT="${ANSWER_FORMAT_WEIGHT:-0.2}"
export STRATEGY_BONUS_SCALE="${STRATEGY_BONUS_SCALE:-0.20}"
export STRATEGY_BONUS_THRESHOLD="${STRATEGY_BONUS_THRESHOLD:-0.05}"
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

echo "[MAIN-LENGTH-8F] Confirmed training frame count: ${VLLM_MAX_FRAMES} frames per video."
echo "[MAIN-LENGTH-8F] Confirmed eval frame count during training: ${VLLM_MAX_FRAMES_EVAL}."
echo "[MAIN-LENGTH-8F] Base merged SFT model: ${QWEN_PATH}"
echo "[MAIN-LENGTH-8F] Clean processor/base path: ${QWEN_BASE_PATH}"
echo "[MAIN-LENGTH-8F] Output directory: ${OUTPUT_DIR}"
echo "[MAIN-LENGTH-8F] MAX_STEPS=${MAX_STEPS}, NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}, LEARNING_RATE=${LEARNING_RATE}"
echo "[MAIN-LENGTH-8F] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[MAIN-LENGTH-8F] Pixel budget: MIN_PIXELS=${MIN_PIXELS}, MAX_PIXELS=${MAX_PIXELS}"
echo "[MAIN-LENGTH-8F] DType/Attn: TORCH_DTYPE=${TORCH_DTYPE}, ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[MAIN-LENGTH-8F] VLLM_USE_MM_PROCESSOR_KWARGS=${VLLM_USE_MM_PROCESSOR_KWARGS}"
echo "[MAIN-LENGTH-8F] VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK}, VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS}"
echo "[MAIN-LENGTH-8F] Periodic eval: STRATEGY_EVAL_EVERY_STEPS=${STRATEGY_EVAL_EVERY_STEPS}, STRATEGY_EVAL_MAX_SAMPLES=${STRATEGY_EVAL_MAX_SAMPLES}, STRATEGY_EVAL_BATCH_SIZE=${STRATEGY_EVAL_BATCH_SIZE}"
echo "[MAIN-LENGTH-8F] vLLM sync diagnostics: VLLM_SYNC_DEBUG_FIRST_N=${VLLM_SYNC_DEBUG_FIRST_N}, VLLM_SYNC_VALIDATE_FIRST_N=${VLLM_SYNC_VALIDATE_FIRST_N}"

# Resume (optional): export RESUME_FROM_CHECKPOINT="${OUTPUT_DIR}/checkpoint-NNN"
# Keep QWEN_PATH as merged SFT; do not point QWEN_PATH at the GRPO adapter dir.

bash src/scripts/run_grpo_length_real.sh
