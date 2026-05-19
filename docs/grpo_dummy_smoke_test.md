# GRPO Dummy Smoke Test

This runbook prepares and validates small generated video-QA datasets for both
LENGTH and PERSPECTIVE GRPO before real data is available.

## Environment

Preferred HPC path:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
export GRPO_CUDA_MODULE="cuda/12.2.2"  # if your cluster uses modules
source scripts/hpc_activate_grpo.sh
bash src/scripts/check_environment.sh
```

If using a repo-local GRPO venv:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo/bin/activate
```

In this workspace on 2026-05-16, `.venv_grpo` was copied from an HPC install
and could not start directly because `libpython3.11.so.1.0` was not available
and the `module` command was unavailable. The completed smoke run below used
`.venv_grpo_smoke` instead.

Local fallback used for this smoke pass:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
PYTHON_BIN=python3 \
VENV_PATH="$(pwd)/.venv_grpo_smoke" \
INSTALL_TORCH_CU124=true \
INSTALL_FLASH_ATTN=false \
PIP_CACHE_DIR="$(pwd)/cache/pip" \
bash setup.sh
source .venv_grpo_smoke/bin/activate
```

The smoke wrappers default to `ATTN_IMPLEMENTATION=sdpa`, so `flash-attn` is not
required for the dummy run. For production, install the repo-pinned version on a
compatible GPU node:

```bash
pip install flash-attn==2.6.3 --no-build-isolation
```

Validated smoke runtime versions on 2026-05-16:

- Python 3.12.3
- torch 2.5.1+cu124, CUDA 12.4, CUDA available
- transformers 4.49.0.dev0
- vLLM 0.7.2
- deepspeed 0.15.4
- accelerate 1.13.0
- peft 0.14.0
- qwen-vl-utils import OK
- flash-attn missing; smoke wrappers use `ATTN_IMPLEMENTATION=sdpa`

## Dummy Data

Generate LENGTH data:

```bash
python scripts/create_dummy_length_data.py \
  --output-dir data/generated_dummy_grpo_smoke/length \
  --num-examples 180 \
  --num-frames 2
```

Generate PERSPECTIVE data:

```bash
python scripts/create_dummy_perspective_data.py \
  --output-dir data/generated_dummy_grpo_smoke/perspective \
  --num-examples 180 \
  --num-frames 2
```

Outputs:

- `data/generated_dummy_grpo_smoke/length/length_grpo_dummy.jsonl`
- `data/generated_dummy_grpo_smoke/perspective/perspective_grpo_dummy.jsonl`
- frame PNGs below each dataset's `frames/` directory

These are generated artifacts under ignored `data/`.

## CPU Validation

Run parser and balanced rollout checks:

```bash
source .venv_grpo_smoke/bin/activate

PYTHONPATH=src/r1-v/src python - <<'PY'
from open_r1.strict_answer import parse_strict_output
for task, text in [
    ("length", "<ANSWER>A</ANSWER>"),
    ("length", "<COT>reason</COT>\n<ANSWER>B</ANSWER>"),
    ("length", "<LONG_COT>long reason</LONG_COT>\n<ANSWER>C</ANSWER>"),
    ("perspective", "<ABSTRACT>category</ABSTRACT>\n<ANSWER>A</ANSWER>"),
    ("perspective", "<TEMPORAL>order</TEMPORAL>\n<ANSWER>B</ANSWER>"),
    ("perspective", "<SPATIOTEMPORAL>place over time</SPATIOTEMPORAL>\n<ANSWER>C</ANSWER>"),
]:
    assert parse_strict_output(text, task_type=task).format_ok
print("strict parser checks passed")
PY

cd src/r1-v
PYTHONPATH=src python ../scripts/dry_run_balanced_strategy.py
PYTHONPATH=src python ../scripts/verify_balanced_strategy_rollout.py
```

## GPU Smoke Tests

Defaults are conservative for 4x A100 40GB: 3 train processes plus one vLLM GPU,
batch size 1, `num_generations=3`, one rollout per strategy, 2 frames, small
pixel bounds, and 50 optimizer steps.

Run LENGTH:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate
bash src/scripts/run_grpo_length_dummy_smoke.sh
```

Run PERSPECTIVE separately:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate
bash src/scripts/run_grpo_perspective_dummy_smoke.sh
```

Useful overrides:

```bash
MODEL_PATH=/path/to/merged_model
QWEN_BASE_PATH=/path/to/Qwen2.5-VL-3B-Instruct
DATA_PATH=/path/to/train.jsonl
OUTPUT_DIR=/path/to/output
NUM_GPUS=4
MAX_STEPS=50
NUM_GENERATIONS=3
ROLLOUTS_PER_STRATEGY=1
PER_DEVICE_BATCH_SIZE=1
MAX_PIXELS=50176
MAX_FRAMES=8
VLLM_GPU_UTIL=0.25
```

Expected outputs:

- `tmp_smoke/grpo_length_dummy/training_log.txt`
- `tmp_smoke/grpo_length_dummy/strategy_debug.jsonl`
- `tmp_smoke/grpo_perspective_dummy/training_log.txt`
- `tmp_smoke/grpo_perspective_dummy/strategy_debug.jsonl`

`tmp_smoke/` is ignored by git.

## Validation Results

Completed on 2026-05-16 with 4x NVIDIA A100-PCIE-40GB:

- LENGTH dummy data: 180 JSONL rows, 2 PNG frames per row
- PERSPECTIVE dummy data: 180 JSONL rows, 2 PNG frames per row
- Strict parser check passed for `<ANSWER>A</ANSWER>`, LENGTH reasoning tags,
  and PERSPECTIVE reasoning tags
- `dry_run_balanced_strategy.py` passed
- `verify_balanced_strategy_rollout.py` passed
- Model and processor loaded from
  `models/Qwen2.5-VL-3B-Instruct-smoke`
- LENGTH GPU smoke completed 50/50 steps:
  `train_runtime=529.6257`, `train_steps_per_second=0.094`,
  `train_loss=0.00027381222989436793`
- PERSPECTIVE GPU smoke completed 50/50 steps after rejecting answer-only
  PERSPECTIVE completions:
  `train_runtime=631.6633`, `train_steps_per_second=0.079`,
  `train_loss=0.00032049637377998155`
- Strategy debug rows: 450 each for LENGTH and PERSPECTIVE

Both runs ended with a PyTorch NCCL shutdown warning about the process group not
being explicitly destroyed during normal process exit; training had already
completed all 50 steps.

## Replacing Dummy Data

For real LENGTH GRPO:

```bash
source .venv_grpo_smoke/bin/activate
REASONING_TASK_TYPE=length \
QWEN_PATH=/path/to/qwen25vl_lora_merged_length \
QWEN_BASE_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
TRAIN_FILE=/path/to/length_grpo_train.jsonl \
OUTPUT_DIR=/path/to/grpo_length_output \
bash src/scripts/run_grpo_answer_only_lora.sh
```

For real PERSPECTIVE GRPO:

```bash
source .venv_grpo_smoke/bin/activate
REASONING_TASK_TYPE=perspective \
QWEN_PATH=/path/to/qwen25vl_lora_merged_perspective \
QWEN_BASE_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
TRAIN_FILE=/path/to/perspective_grpo_train.jsonl \
OUTPUT_DIR=/path/to/grpo_perspective_output \
bash src/scripts/run_grpo_answer_only_lora.sh
```

The train JSONL schema should match the dummy files: `video_id`,
`question_id`, `question_category`, `problem`, `frames`, and `solution` with
strict ground truth like `<ANSWER>A</ANSWER>`. LENGTH rows may include
`reasoning_depth`; PERSPECTIVE rows may include `reasoning_type`.

## Notes

- Local merged 3B model directories were found at
  `models/qwen25vl3b_lora_merged_length` and
  `models/qwen25vl3b_lora_merged_perspective`, but their `model.safetensors`
  files were truncated in this workspace. The original local
  `models/Qwen2.5-VL-3B-Instruct` shards were truncated too. A clean smoke copy
  was downloaded to ignored `models/Qwen2.5-VL-3B-Instruct-smoke` and the dummy
  wrappers prefer that directory when it exists.
- To create the clean smoke copy manually:

```bash
source .venv_grpo_smoke/bin/activate
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="models/Qwen2.5-VL-3B-Instruct-smoke",
)
PY
```

- `PERSPECTIVE` strict parsing rejects answer-only `<ANSWER>X</ANSWER>`.
  Valid PERSPECTIVE completions must include one of `<ABSTRACT>`,
  `<TEMPORAL>`, or `<SPATIOTEMPORAL>` before the final answer. LENGTH direct
  answers remain valid as `<ANSWER>X</ANSWER>`.
- Keep checkpoints, logs, dummy data, venvs, caches, and model weights out of
  git.
