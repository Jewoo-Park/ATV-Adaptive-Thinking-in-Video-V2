#!/usr/bin/env bash
# Run the four models_0624 merged checkpoints on UVB / VideoMMMU / MMVU (full test sets).
# Uses 16 frames per sample by default.
#
# 4-GPU layout (default NUM_GPUS=4):
#   GPU 0: length_step4000   (UVB → VideoMMMU → MMVU)
#   GPU 1: perspective_step4300
#   GPU 2: length_step5000
#   GPU 3: perspective_step5000
#
# Usage (on a GPU node, after activating .venv_grpo or equivalent):
#   source scripts/hpc_activate_grpo.sh
#   bash src/scripts/run_models_0624_benchmark_test.sh
#
# Single-GPU serial run:
#   NUM_GPUS=1 bash src/scripts/run_models_0624_benchmark_test.sh
#
# Quick smoke subset:
#   BENCH_MAX_SAMPLES=20 bash src/scripts/run_models_0624_benchmark_test.sh
#
# Optional overrides:
#   MODELS_ROOT, PROCESSOR_PATH, FRAMES_PER_SAMPLE (default 16),
#   GPU_MEM_UTIL (default 0.40), BENCH_EVAL_EXTRA (default --max-model-len 8192),
#   SUMMARY_DIR, BENCH_MAX_SAMPLES, PYTHON_BIN.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# Preflight: vLLM's Qwen2.5-VL integration expects Transformers 4.x.
set +e
_tf_major="$("${PYTHON_BIN}" -c "import transformers; print(transformers.__version__.split('.')[0])" 2>/dev/null)"
_tf_minor="$("${PYTHON_BIN}" -c "import transformers; v=transformers.__version__.split('.'); print(int(v[1]) if len(v)>1 else 0)" 2>/dev/null)"
_vllm_version="$("${PYTHON_BIN}" -c "import vllm; print(getattr(vllm,'__version__',''))" 2>/dev/null)"
set -e
if [[ -n "${_tf_major}" && "${_tf_major}" -ge 5 ]]; then
  echo "[MODELS-0624] ERROR: transformers major version is ${_tf_major} (>=5). vLLM Qwen2.5-VL will likely fail." >&2
  echo "[MODELS-0624]        Fix: pip install -U 'transformers>=4.52,<5' 'huggingface-hub<1' 'tokenizers<0.23' 'numpy<2'" >&2
  exit 2
fi
if [[ -n "${_tf_major}" && "${_tf_major}" -eq 4 && -n "${_tf_minor}" && "${_tf_minor}" -ge 49 ]]; then
  if [[ -n "${_vllm_version}" && "${_vllm_version}" =~ ^0\\.7\\.2([[:space:]]|$) ]]; then
    echo "[MODELS-0624] ERROR: transformers is 4.${_tf_minor}.x (>=4.49) but vllm is ${_vllm_version}." >&2
    echo "[MODELS-0624]        Fix: pip install -U 'vllm>=0.7.3' or downgrade transformers." >&2
    exit 2
  fi
fi

_force_hf_backend=0
if [[ "${BENCH_EVAL_EXTRA:-}" =~ --backend[[:space:]]+hf ]]; then
  _force_hf_backend=1
fi

set +e
_vllm_qwen_check="$("${PYTHON_BIN}" - <<'PY'
import sys
try:
    import vllm  # noqa: F401
    import vllm.model_executor.models.qwen2_5_vl  # noqa: F401
except Exception as e:
    print(repr(e))
    sys.exit(1)
print("ok")
PY
)"
_vllm_qwen_rc=$?
set -e
if [[ "${_force_hf_backend}" -eq 0 && "${_vllm_qwen_rc}" -ne 0 ]]; then
  echo "[MODELS-0624] ERROR: vLLM Qwen2.5-VL backend failed to import." >&2
  echo "[MODELS-0624]        ${_vllm_qwen_check}" >&2
  exit 2
fi

_SCRATCH_MODELS="/scratch/users/ntu/n2500182/models"
MODELS_ROOT="${MODELS_ROOT:-/home/users/ntu/n2500182/scratch/models/models_0624}"
# If shell has MODELS_ROOT=.../scratch/models (missing models_0624), auto-fix.
if [[ ! -d "${MODELS_ROOT}/qwen25vl3b_grpo_length_step4000_merged" ]]; then
  if [[ -d "${MODELS_ROOT}/models_0624/qwen25vl3b_grpo_length_step4000_merged" ]]; then
    MODELS_ROOT="${MODELS_ROOT}/models_0624"
  elif [[ -d "/home/users/ntu/n2500182/scratch/models/models_0624/qwen25vl3b_grpo_length_step4000_merged" ]]; then
    MODELS_ROOT="/home/users/ntu/n2500182/scratch/models/models_0624"
  fi
fi

PROCESSOR_PATH="${PROCESSOR_PATH:-${_SCRATCH_MODELS}/Qwen2.5-VL-3B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-0.8}"
FRAMES_PER_SAMPLE="${FRAMES_PER_SAMPLE:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
BENCH_EVAL_EXTRA="${BENCH_EVAL_EXTRA:---max-model-len 8192}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/outputs/video_benchmark_runs/$(date +%Y%m%d_%H%M%S)_models_0624}"
mkdir -p "${SUMMARY_DIR}"

UVB_JSONL="${UVB_JSONL:-${REPO_ROOT}/data/urban_video_bench/grpo/uvb_grpo_test_strict.jsonl}"
VIDEOMMMU_JSONL="${VIDEOMMMU_JSONL:-${REPO_ROOT}/data/video_mmmu/grpo/videommmu_grpo_test_strict.jsonl}"
MMVU_JSONL="${MMVU_JSONL:-${REPO_ROOT}/data/mmvu/grpo/mmvu_grpo_test_strict.jsonl}"

declare -a MODEL_SPECS=(
  "0|length_step4000|length|${MODELS_ROOT}/qwen25vl3b_grpo_length_step4000_merged"
  "1|perspective_step4300|perspective|${MODELS_ROOT}/qwen25vl3b_grpo_perspective_step4300_merged"
  "2|length_step5000|length|${MODELS_ROOT}/qwen25vl3b_grpo_length_step5000_merged"
  "3|perspective_step5000|perspective|${MODELS_ROOT}/qwen25vl3b_grpo_perspective_step5000_merged"
)

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r _gpu _label _task model <<< "${spec}"
  if [[ ! -d "${model}" ]]; then
    echo "[MODELS-0624] ERROR: missing model dir: ${model}" >&2
    exit 1
  fi
done

_run_eval() {
  local gpu="$1"
  local label="$2"
  local script="$3"
  local model_path="$4"
  local test_file="$5"
  local reasoning_task_type="$6"
  local log="${SUMMARY_DIR}/${label}.log"

  if [[ ! -f "${test_file}" ]]; then
    echo "[MODELS-0624] SKIP ${label}: missing test file ${test_file}" | tee -a "${SUMMARY_DIR}/skipped.txt"
    return 0
  fi

  echo "================================================================================"
  echo "[MODELS-0624] ${label}"
  echo "  gpu:   ${gpu}"
  echo "  model: ${model_path}"
  echo "  test:  ${test_file}"
  echo "  task:  ${reasoning_task_type}"
  echo "  log:   ${log}"
  echo "================================================================================"

  local max_samples_args=()
  if [[ -n "${BENCH_MAX_SAMPLES:-}" ]]; then
    max_samples_args=(--max-samples "${BENCH_MAX_SAMPLES}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/${script}" \
      --model "${model_path}" \
      --test-file "${test_file}" \
      --processor-path "${PROCESSOR_PATH}" \
      --device cuda:0 \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --temperature "${BENCH_TEMPERATURE}" \
      --frames-per-sample "${FRAMES_PER_SAMPLE}" \
      --reasoning-task-type "${reasoning_task_type}" \
      "${max_samples_args[@]}" \
      ${BENCH_EVAL_EXTRA} \
      2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${label} exit_code=${rc}" >> "${SUMMARY_DIR}/exit_codes.txt"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[MODELS-0624] WARNING: ${label} failed with exit ${rc}" >&2
  fi
  return "${rc}"
}

_run_model_pipeline() {
  local gpu="$1"
  local label="$2"
  local reasoning_task_type="$3"
  local model_path="$4"

  _run_eval "${gpu}" "${label}__uvb" "src/eval/uvb_eval_only.py" \
    "${model_path}" "${UVB_JSONL}" "${reasoning_task_type}" || true
  _run_eval "${gpu}" "${label}__videommmu" "src/eval/videommmu_eval_only.py" \
    "${model_path}" "${VIDEOMMMU_JSONL}" "${reasoning_task_type}" || true
  _run_eval "${gpu}" "${label}__mmvu" "src/eval/mmvu_eval_only.py" \
    "${model_path}" "${MMVU_JSONL}" "${reasoning_task_type}" || true
}

echo "[MODELS-0624] REPO_ROOT=${REPO_ROOT}"
echo "[MODELS-0624] SUMMARY_DIR=${SUMMARY_DIR}"
echo "[MODELS-0624] MODELS_ROOT=${MODELS_ROOT}"
echo "[MODELS-0624] PROCESSOR_PATH=${PROCESSOR_PATH}"
echo "[MODELS-0624] GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[MODELS-0624] FRAMES_PER_SAMPLE=${FRAMES_PER_SAMPLE}"
echo "[MODELS-0624] BENCH_EVAL_EXTRA=${BENCH_EVAL_EXTRA}"
echo "[MODELS-0624] BENCH_MAX_SAMPLES=${BENCH_MAX_SAMPLES:-<full>}"
echo "[MODELS-0624] NUM_GPUS=${NUM_GPUS}"
echo "[MODELS-0624] PYTHON_BIN=${PYTHON_BIN}"
: > "${SUMMARY_DIR}/exit_codes.txt"

echo "[MODELS-0624] Preflight GPU memory:"
nvidia-smi || true

# vLLM can error on mixed rope_scaling legacy/modern keys. Normalize configs in-place.
model_dirs=()
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r _gpu _label _task model <<< "${spec}"
  model_dirs+=("${model}")
done
set +e
"${PYTHON_BIN}" "${REPO_ROOT}/src/scripts/fix_hf_rope_scaling_for_vllm.py" \
  "${model_dirs[@]}" \
  2>&1 | tee "${SUMMARY_DIR}/fix_rope_scaling.log"
set -e

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  echo "[MODELS-0624] 1-GPU mode: run all four model pipelines serially on GPU 0"
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r _gpu label reasoning_task_type model <<< "${spec}"
    _run_model_pipeline 0 "${label}" "${reasoning_task_type}" "${model}"
  done
else
  echo "[MODELS-0624] ${NUM_GPUS}-GPU mode: one model pipeline per GPU (UVB → VideoMMMU → MMVU)"
  pids=()
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r gpu label reasoning_task_type model <<< "${spec}"
    if [[ "${gpu}" -ge "${NUM_GPUS}" ]]; then
      echo "[MODELS-0624] SKIP ${label}: GPU ${gpu} >= NUM_GPUS=${NUM_GPUS}" >&2
      continue
    fi
    (
      _run_model_pipeline "${gpu}" "${label}" "${reasoning_task_type}" "${model}"
    ) &
    pids+=("$!")
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait "${pids[@]}" || true
  fi
fi

echo "[MODELS-0624] Done. Per-run logs under ${SUMMARY_DIR}"
echo "[MODELS-0624] Exit codes: ${SUMMARY_DIR}/exit_codes.txt"
cat "${SUMMARY_DIR}/exit_codes.txt" 2>/dev/null || true
