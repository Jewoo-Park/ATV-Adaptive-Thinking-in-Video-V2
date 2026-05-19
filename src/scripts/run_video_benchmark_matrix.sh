#!/usr/bin/env bash
# Run UVB / VideoMMMU / MMVU eval for three weight variants:
#   1) Backbone Qwen2.5-VL-7B-Instruct
#   2) SFT merged full weights (backbone + SFT LoRA merged)
#   3) SFT + GRPO merged full weights (backbone + SFT + GRPO LoRA merged)
#
# Usage (on a GPU node, after activating .venv_grpo or equivalent):
#   export MODEL_BASE=/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct
#   export MODEL_SFT_MERGED=/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length
#   export MODEL_SFT_GRPO_MERGED=/scratch/users/ntu/n2500182/models/video_r1_uvb_grpo_answer_only_lora_merged_ckpt1500
#   REASONING_TASK_TYPE=length bash src/scripts/run_video_benchmark_matrix.sh
#
# Perspective model evaluation (must set mode explicitly):
#   REASONING_TASK_TYPE=perspective bash src/scripts/run_video_benchmark_matrix.sh
#
# Optional: PROCESSOR_PATH (default = MODEL_BASE), BENCH_DEVICE (default cuda:0),
#           PYTHON_BIN, GPU_MEM_UTIL (default 0.25, matches VLLM_GPU_UTIL in GRPO training),
#           BENCH_TEMPERATURE (default 0.8), FRAMES_PER_SAMPLE (default 16), BENCH_EVAL_EXTRA.
# Eval always uses the full test JSONL (no max-samples in this launcher).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the currently activated venv python (avoids missing libpython on some clusters).
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# Preflight: vLLM's Qwen2.5-VL integration expects Transformers 4.x.
# If Transformers 5.x is installed, vLLM can fail with ImportError for Qwen2_5_VLImageProcessor.
set +e
_tf_major="$("${PYTHON_BIN}" -c "import transformers; print(transformers.__version__.split('.')[0])" 2>/dev/null)"
_tf_minor="$("${PYTHON_BIN}" -c "import transformers; v=transformers.__version__.split('.'); print(int(v[1]) if len(v)>1 else 0)" 2>/dev/null)"
_vllm_version="$("${PYTHON_BIN}" -c "import vllm; print(getattr(vllm,'__version__',''))" 2>/dev/null)"
set -e
if [[ -n "${_tf_major}" && "${_tf_major}" -ge 5 ]]; then
  echo "[BENCH-MATRIX] ERROR: transformers major version is ${_tf_major} (>=5). vLLM Qwen2.5-VL will likely fail." >&2
  echo "[BENCH-MATRIX]        Fix in your venv: pip install -U 'transformers>=4.52,<5' 'huggingface-hub<1' 'tokenizers<0.23' 'numpy<2'" >&2
  exit 2
fi

# Compatibility matrix (pragmatic):
# - vLLM 0.7.2 expects transformers < 4.49 (it imports Qwen2_5_VLImageProcessor)
# - vLLM >= 0.7.3 includes a fix for the Qwen2.5-VL image processor rename (Qwen2VLImageProcessor)
if [[ -n "${_tf_major}" && "${_tf_major}" -eq 4 && -n "${_tf_minor}" && "${_tf_minor}" -ge 49 ]]; then
  if [[ -n "${_vllm_version}" && "${_vllm_version}" =~ ^0\\.7\\.2([[:space:]]|$) ]]; then
    echo "[BENCH-MATRIX] ERROR: transformers is 4.${_tf_minor}.x (>=4.49) but vllm is ${_vllm_version}." >&2
    echo "[BENCH-MATRIX]        This combo fails for Qwen2.5-VL (image processor rename)." >&2
    echo "[BENCH-MATRIX]        Fix option A: upgrade vllm (preferred): pip install -U 'vllm>=0.7.3'" >&2
    echo "[BENCH-MATRIX]        Fix option B: downgrade transformers:    pip install -U 'transformers>=4.48,<4.49'" >&2
    exit 2
  fi
fi

# If caller forces HF backend, skip vLLM import preflight.
_force_hf_backend=0
if [[ "${BENCH_EVAL_EXTRA:-}" =~ --backend[[:space:]]+hf ]]; then
  _force_hf_backend=1
fi

# Preflight: ensure vLLM's Qwen2.5-VL model code imports with the installed Transformers.
set +e
_vllm_qwen_check="$("${PYTHON_BIN}" - <<'PY'
import sys
try:
    import vllm  # noqa: F401
    # vLLM imports its Qwen2.5-VL implementation during model inspection.
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
  echo "[BENCH-MATRIX] ERROR: vLLM Qwen2.5-VL backend failed to import in this venv." >&2
  echo "[BENCH-MATRIX]        ${_vllm_qwen_check}" >&2
  echo "[BENCH-MATRIX]        Usual fix: pip install -U vllm (or align vllm/transformers versions)." >&2
  exit 2
fi

MODEL_BASE="${MODEL_BASE:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct}"
MODEL_SFT_MERGED="${MODEL_SFT_MERGED:-/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length}"
MODEL_SFT_GRPO_MERGED="${MODEL_SFT_GRPO_MERGED:-/scratch/users/ntu/n2500182/models/video_r1_uvb_grpo_answer_only_lora_merged_ckpt1500}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MODEL_BASE}}"
BENCH_DEVICE="${BENCH_DEVICE:-cuda:0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-0.8}"
FRAMES_PER_SAMPLE="${FRAMES_PER_SAMPLE:-16}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/outputs/video_benchmark_runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${SUMMARY_DIR}"

UVB_JSONL="${UVB_JSONL:-${REPO_ROOT}/data/urban_video_bench/grpo/uvb_grpo_test_strict.jsonl}"
VIDEOMMMU_JSONL="${VIDEOMMMU_JSONL:-${REPO_ROOT}/data/video_mmmu/grpo/videommmu_grpo_test_strict.jsonl}"
MMVU_JSONL="${MMVU_JSONL:-${REPO_ROOT}/data/mmvu/grpo/mmvu_grpo_test_strict.jsonl}"
REASONING_TASK_TYPE="${REASONING_TASK_TYPE:-length}"
REASONING_TASK_TYPE="$(printf '%s' "${REASONING_TASK_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${REASONING_TASK_TYPE}" in
  length|perspective) ;;
  *)
    echo "[BENCH-MATRIX] ERROR: REASONING_TASK_TYPE must be length or perspective (got ${REASONING_TASK_TYPE})" >&2
    exit 1
    ;;
esac

PROC_ARGS=()
if [[ -n "${PROCESSOR_PATH}" ]]; then
  PROC_ARGS+=(--processor-path "${PROCESSOR_PATH}")
fi

if [[ ! -d "${MODEL_BASE}" ]]; then
  echo "[BENCH-MATRIX] ERROR: MODEL_BASE is not a directory: ${MODEL_BASE}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_SFT_MERGED}" ]]; then
  echo "[BENCH-MATRIX] ERROR: MODEL_SFT_MERGED is not a directory: ${MODEL_SFT_MERGED}" >&2
  exit 1
fi
if [[ -z "${MODEL_SFT_GRPO_MERGED}" ]]; then
  echo "[BENCH-MATRIX] ERROR: MODEL_SFT_GRPO_MERGED is unset. Set it to the merged model dir (SFT base + GRPO LoRA)." >&2
  exit 1
fi
if [[ ! -d "${MODEL_SFT_GRPO_MERGED}" ]]; then
  echo "[BENCH-MATRIX] ERROR: MODEL_SFT_GRPO_MERGED is not a directory: ${MODEL_SFT_GRPO_MERGED}" >&2
  exit 1
fi

_run_eval() {
  local label="$1"
  local script="$2"
  local model_path="$3"
  local test_file="$4"
  local log="${SUMMARY_DIR}/${label}.log"

  if [[ ! -f "${test_file}" ]]; then
    echo "[BENCH-MATRIX] SKIP ${label}: missing test file ${test_file}" | tee -a "${SUMMARY_DIR}/skipped.txt"
    return 0
  fi

  echo "================================================================================"
  echo "[BENCH-MATRIX] ${label}"
  echo "  model: ${model_path}"
  echo "  test:  ${test_file}"
  echo "  log:   ${log}"
  echo "================================================================================"

  set +e
  "${PYTHON_BIN}" "${REPO_ROOT}/${script}" \
    --model "${model_path}" \
    --test-file "${test_file}" \
    "${PROC_ARGS[@]}" \
    --device "${BENCH_DEVICE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --temperature "${BENCH_TEMPERATURE}" \
    --frames-per-sample "${FRAMES_PER_SAMPLE}" \
    --reasoning-task-type "${REASONING_TASK_TYPE}" \
    ${BENCH_EVAL_EXTRA:-} \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${label} exit_code=${rc}" >> "${SUMMARY_DIR}/exit_codes.txt"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[BENCH-MATRIX] WARNING: ${label} failed with exit ${rc}" >&2
  fi
}

echo "[BENCH-MATRIX] REPO_ROOT=${REPO_ROOT}"
echo "[BENCH-MATRIX] SUMMARY_DIR=${SUMMARY_DIR}"
echo "[BENCH-MATRIX] MODEL_BASE=${MODEL_BASE}"
echo "[BENCH-MATRIX] MODEL_SFT_MERGED=${MODEL_SFT_MERGED}"
echo "[BENCH-MATRIX] MODEL_SFT_GRPO_MERGED=${MODEL_SFT_GRPO_MERGED}"
echo "[BENCH-MATRIX] PROCESSOR_PATH=${PROCESSOR_PATH}"
echo "[BENCH-MATRIX] BENCH_DEVICE=${BENCH_DEVICE}"
echo "[BENCH-MATRIX] GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[BENCH-MATRIX] BENCH_TEMPERATURE=${BENCH_TEMPERATURE}"
echo "[BENCH-MATRIX] FRAMES_PER_SAMPLE=${FRAMES_PER_SAMPLE}"
echo "[BENCH-MATRIX] PYTHON_BIN=${PYTHON_BIN}"
echo "[BENCH-MATRIX] REASONING_TASK_TYPE=${REASONING_TASK_TYPE}"
: > "${SUMMARY_DIR}/exit_codes.txt"

# vLLM can error on mixed rope_scaling legacy/modern keys. Normalize configs in-place.
set +e
"${PYTHON_BIN}" "${REPO_ROOT}/src/scripts/fix_hf_rope_scaling_for_vllm.py" \
  "${MODEL_BASE}" "${MODEL_SFT_MERGED}" "${MODEL_SFT_GRPO_MERGED}" \
  2>&1 | tee "${SUMMARY_DIR}/fix_rope_scaling.log"
set -e

# backbone
_run_eval "base__uvb" "src/eval/uvb_eval_only.py" "${MODEL_BASE}" "${UVB_JSONL}"
_run_eval "base__videommmu" "src/eval/videommmu_eval_only.py" "${MODEL_BASE}" "${VIDEOMMMU_JSONL}"
_run_eval "base__mmvu" "src/eval/mmvu_eval_only.py" "${MODEL_BASE}" "${MMVU_JSONL}"

# SFT merged
_run_eval "sft_merged__uvb" "src/eval/uvb_eval_only.py" "${MODEL_SFT_MERGED}" "${UVB_JSONL}"
_run_eval "sft_merged__videommmu" "src/eval/videommmu_eval_only.py" "${MODEL_SFT_MERGED}" "${VIDEOMMMU_JSONL}"
_run_eval "sft_merged__mmvu" "src/eval/mmvu_eval_only.py" "${MODEL_SFT_MERGED}" "${MMVU_JSONL}"

# SFT + GRPO merged
_run_eval "sft_grpo_merged__uvb" "src/eval/uvb_eval_only.py" "${MODEL_SFT_GRPO_MERGED}" "${UVB_JSONL}"
_run_eval "sft_grpo_merged__videommmu" "src/eval/videommmu_eval_only.py" "${MODEL_SFT_GRPO_MERGED}" "${VIDEOMMMU_JSONL}"
_run_eval "sft_grpo_merged__mmvu" "src/eval/mmvu_eval_only.py" "${MODEL_SFT_GRPO_MERGED}" "${MMVU_JSONL}"

echo "[BENCH-MATRIX] Done. Per-run logs under ${SUMMARY_DIR}"
echo "[BENCH-MATRIX] Exit codes: ${SUMMARY_DIR}/exit_codes.txt"
echo "[BENCH-MATRIX] Predictions/metrics are written next to each --model dir by the eval scripts (default naming)."
