# GRPO Real-Data Smoke Report

Date: 2026-05-16

Repository: `/workspace/ATV-Adaptive-Thinking-in-Video-V2`

Final status: `READY_FOR_FULL_TRAINING`

## Objective

Validate that the repository is ready to run actual LENGTH and PERSPECTIVE GRPO
training with the real data now present on the server. This report covers path
discovery, schema checks, parser/reward compatibility, environment and model
loading, and short real-data GPU smoke runs. No full training run was started.

## Real Data

The real GRPO dataset found for both modes is:

```text
data/data/video_r1/grpo/video_r1_grpo_train.jsonl
```

This repository's GRPO path uses the same JSONL for LENGTH and PERSPECTIVE,
with `REASONING_TASK_TYPE` selecting the strategy prompts, parser behavior, and
reward metrics.

Dataset summary:

- Format: JSONL
- Size: 16M
- Rows: 7,299
- Keys: `frames`, `problem`, `question_category`, `question_id`, `solution`,
  `video_id`
- Frame count: 16 frame paths per row
- Answer labels observed: `A`, `B`, `C`, `D`, `E`
- Source subsets from `question_category`:
  - `LLaVA-Video-178K`: 5,096
  - `STAR`: 802
  - `NeXT-QA`: 528
  - `PerceptionTest`: 445
  - `CLEVRER`: 428

The launch script created this train/eval split:

```text
data/data/video_r1/grpo/video_r1_grpo_train_strict__train_eval0p05_seed42.jsonl
data/data/video_r1/grpo/video_r1_grpo_train_strict__eval_eval0p05_seed42.jsonl
```

Split counts:

- Train: 6,933 rows, 15M
- Eval: 366 rows, 804K
- Skipped by video-only filter: 0
- Skipped blank rows: 0
- Skipped parse failures: 0

The split files are generated artifacts under ignored `data/`.

## Schema And Media Validation

Schema checks passed:

- Required fields present and non-empty in checked rows
- JSONL parsed without malformed rows
- `solution` values use strict ground truth like `<ANSWER>C</ANSWER>`
- Answer labels are valid letters
- Option labels in `problem` match the ground-truth answer letters
- Relative frame paths resolve from `data/data/video_r1/grpo`

Media check:

- Checked the first two frame paths for every row
- Existing frame files: 14,598 / 14,598
- Missing frame files in that check: 0

Important path note: keep `SPLIT_DIR` set to the same directory as the source
JSONL:

```text
/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo
```

The dataset stores frame paths as `../processed/...`. If split files are written
elsewhere, the loader resolves frames relative to the split file and the trainer
falls back to placeholder image handling.

## Parser And Reward Compatibility

Strict parser validation passed:

- LENGTH `<ANSWER>A</ANSWER>` is valid
- PERSPECTIVE `<ANSWER>A</ANSWER>` is invalid
- PERSPECTIVE `<ABSTRACT>...</ABSTRACT><ANSWER>A</ANSWER>` is valid
- PERSPECTIVE `<TEMPORAL>...</TEMPORAL><ANSWER>B</ANSWER>` is valid
- PERSPECTIVE `<SPATIOTEMPORAL>...</SPATIOTEMPORAL><ANSWER>C</ANSWER>` is valid

The smoke runs confirmed the reward path uses the selected
`REASONING_TASK_TYPE`:

- LENGTH emitted strategy metrics for `direct`, `cot`, and `long_cot`
- PERSPECTIVE emitted strategy metrics for `abstract`, `temporal`, and
  `spatiotemporal`

## Runtime

Environment used:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate
```

Validated versions:

- Python: 3.12.3
- torch: 2.5.1+cu124
- CUDA from torch: 12.4
- CUDA available: true
- transformers: 4.49.0.dev0
- vLLM: 0.7.2
- deepspeed: 0.15.4
- accelerate: 1.13.0
- peft: 0.14.0
- qwen-vl-utils: import OK
- flash-attn: not installed

Hardware:

- 4x NVIDIA A100-PCIE-40GB
- Each GPU reported 40,960 MiB total memory

The smoke commands used `ATTN_IMPLEMENTATION=sdpa`, so missing `flash-attn` did
not block validation.

## Model

The real-data smoke runs used the same working local model path that passed the
dummy validation:

```text
models/Qwen2.5-VL-3B-Instruct-smoke
```

Model and processor load succeeded from that path:

- Processor: `Qwen2_5_VLProcessor`
- Model: `Qwen2_5_VLForConditionalGeneration`

Other local model directories were found:

```text
models/Qwen2.5-VL-3B-Instruct
models/qwen25vl3b_lora_merged_length
models/qwen25vl3b_lora_merged_perspective
```

The merged model directories were not usable as-is in this environment:

- `AutoProcessor` on `models/qwen25vl3b_lora_merged_length` failed with
  `TypeError: Qwen2_5_VLProcessor.__init__() got multiple values for argument
  'image_processor'`
- Loading `models/qwen25vl3b_lora_merged_length` weights with the known-good
  smoke processor failed with
  `safetensors_rust.SafetensorError: Error while deserializing header:
  incomplete metadata, file not fully covered`

Use `models/Qwen2.5-VL-3B-Instruct-smoke` unless the merged directories are
repaired or replaced.

## LENGTH Real-Data Smoke

Command:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate && REASONING_TASK_TYPE=length QWEN_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke QWEN_BASE_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke TRAIN_FILE=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo/video_r1_grpo_train.jsonl OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/tmp_smoke/grpo_length_realdata_check_realframes NUM_GPUS=4 TRAIN_NUM_GPUS=3 MAX_STEPS=5 NUM_GENERATIONS=3 ROLLOUTS_PER_STRATEGY=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=96 MAX_PIXELS=50176 MIN_PIXELS=50176 VLLM_MAX_FRAMES=2 VLLM_GPU_UTIL=0.25 SAVE_STEPS=1000 LOGGING_STEPS=1 ATTN_IMPLEMENTATION=sdpa GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false BALANCED_STRATEGY_ROLLOUT=true STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/tmp_smoke/grpo_length_realdata_check_realframes/strategy_debug.jsonl SPLIT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo bash src/scripts/run_grpo_answer_only_lora.sh
```

Result: passed.

Final summary:

```text
train_runtime=51.6304
train_samples_per_second=0.291
train_steps_per_second=0.097
train_loss=7.2181231985268825e-06
epoch=0.0
```

Confirmed:

- Dataset loading succeeded
- Model and processor loading succeeded
- vLLM initialized on GPU 3
- Real frame paths were used in the counted run
- Rollout generation succeeded
- Reward computation succeeded
- Optimizer steps completed through 5/5
- LENGTH strategy metrics were emitted

Outputs:

```text
tmp_smoke/grpo_length_realdata_check_realframes/
tmp_smoke/grpo_length_realdata_check_realframes/strategy_debug.jsonl
```

`strategy_debug.jsonl` rows: 45.

## PERSPECTIVE Real-Data Smoke

Command:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate && REASONING_TASK_TYPE=perspective QWEN_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke QWEN_BASE_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke TRAIN_FILE=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo/video_r1_grpo_train.jsonl OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/tmp_smoke/grpo_perspective_realdata_check_realframes NUM_GPUS=4 TRAIN_NUM_GPUS=3 MAX_STEPS=5 NUM_GENERATIONS=3 ROLLOUTS_PER_STRATEGY=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=96 MAX_PIXELS=50176 MIN_PIXELS=50176 VLLM_MAX_FRAMES=2 VLLM_GPU_UTIL=0.25 SAVE_STEPS=1000 LOGGING_STEPS=1 ATTN_IMPLEMENTATION=sdpa GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false BALANCED_STRATEGY_ROLLOUT=true STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/tmp_smoke/grpo_perspective_realdata_check_realframes/strategy_debug.jsonl SPLIT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo bash src/scripts/run_grpo_answer_only_lora.sh
```

Result: passed.

Final summary:

```text
train_runtime=69.14
train_samples_per_second=0.217
train_steps_per_second=0.072
train_loss=1.1463960981927812e-05
epoch=0.0
```

Confirmed:

- Dataset loading succeeded
- Model and processor loading succeeded
- vLLM initialized on GPU 3
- Real frame paths were used
- Rollout generation succeeded
- Reward computation succeeded with PERSPECTIVE strict parsing
- Optimizer steps completed through 5/5
- PERSPECTIVE strategy metrics were emitted

Outputs:

```text
tmp_smoke/grpo_perspective_realdata_check_realframes/
tmp_smoke/grpo_perspective_realdata_check_realframes/strategy_debug.jsonl
```

`strategy_debug.jsonl` rows: 45.

## Recommended Full-Training Commands

Start full LENGTH training by reusing the validated command and changing
`OUTPUT_DIR` and `MAX_STEPS`:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate && REASONING_TASK_TYPE=length QWEN_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke QWEN_BASE_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke TRAIN_FILE=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo/video_r1_grpo_train.jsonl OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_real NUM_GPUS=4 TRAIN_NUM_GPUS=3 MAX_STEPS=1000 NUM_GENERATIONS=3 ROLLOUTS_PER_STRATEGY=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=96 MAX_PIXELS=50176 MIN_PIXELS=50176 VLLM_MAX_FRAMES=2 VLLM_GPU_UTIL=0.25 SAVE_STEPS=100 LOGGING_STEPS=1 ATTN_IMPLEMENTATION=sdpa GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false BALANCED_STRATEGY_ROLLOUT=true STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_real/strategy_debug.jsonl SPLIT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo bash src/scripts/run_grpo_answer_only_lora.sh
```

Start full PERSPECTIVE training separately:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate && REASONING_TASK_TYPE=perspective QWEN_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke QWEN_BASE_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/models/Qwen2.5-VL-3B-Instruct-smoke TRAIN_FILE=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo/video_r1_grpo_train.jsonl OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_real NUM_GPUS=4 TRAIN_NUM_GPUS=3 MAX_STEPS=1000 NUM_GENERATIONS=3 ROLLOUTS_PER_STRATEGY=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=96 MAX_PIXELS=50176 MIN_PIXELS=50176 VLLM_MAX_FRAMES=2 VLLM_GPU_UTIL=0.25 SAVE_STEPS=100 LOGGING_STEPS=1 ATTN_IMPLEMENTATION=sdpa GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false BALANCED_STRATEGY_ROLLOUT=true STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_real/strategy_debug.jsonl SPLIT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/data/data/video_r1/grpo bash src/scripts/run_grpo_answer_only_lora.sh
```

Use `MAX_STEPS=-1` only if intentionally switching to epoch-based training.

## Warnings And Limitations

- The counted LENGTH and PERSPECTIVE smoke runs both ended with a PyTorch NCCL
  warning that the process group was not explicitly destroyed. This warning
  appeared after all requested optimizer steps completed.
- `flash-attn` is not installed in `.venv_grpo_smoke`; SDPA passed the smoke
  tests.
- The local merged LENGTH and PERSPECTIVE model directories need repair before
  they can be used directly.
- Keep `SPLIT_DIR` beside the real JSONL unless the split script or loader is
  changed to preserve the source JSONL base path for relative frame resolution.

## Git Hygiene

Generated artifacts from this validation are ignored by git:

- `data/`
- `tmp_smoke/`

No commit was made.
