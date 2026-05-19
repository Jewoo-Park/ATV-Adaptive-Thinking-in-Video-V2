# GRPO Dummy Smoke Run Report

Date: 2026-05-16

Repository: `/workspace/ATV-Adaptive-Thinking-in-Video-V2`

## Objective

Validate that both LENGTH and PERSPECTIVE GRPO experiment paths can run end to
end before real data is available, using small generated video-QA datasets with
valid frame inputs.

## Hardware

- GPUs visible: 4
- GPU type: NVIDIA A100-PCIE-40GB
- Memory per GPU: 40960 MiB

## Runtime

The repo-local `.venv_grpo` could not be used in this container because its
Python 3.11 executable required `libpython3.11.so.1.0`, and the `module`
command was unavailable. The successful smoke runs used `.venv_grpo_smoke`.

Validated versions:

- Python: 3.12.3
- torch: 2.5.1+cu124
- CUDA from torch: 12.4
- transformers: 4.49.0.dev0
- vLLM: 0.7.2
- deepspeed: 0.15.4
- accelerate: 1.13.0
- peft: 0.14.0
- qwen-vl-utils: import OK
- flash-attn: not installed

The smoke wrappers used `ATTN_IMPLEMENTATION=sdpa`, so missing `flash-attn` did
not block validation.

## Model

Model and processor load succeeded from:

```text
models/Qwen2.5-VL-3B-Instruct-smoke
```

Earlier local merged model directories existed, but the smoke run used the
clean 3B smoke copy because it was the usable model tree for this validation.

## Dummy Data

Generated datasets:

```text
data/generated_dummy_grpo_smoke/length/length_grpo_dummy.jsonl
data/generated_dummy_grpo_smoke/perspective/perspective_grpo_dummy.jsonl
```

Dataset sizes:

- LENGTH: 180 JSONL rows, 2 PNG frames per row
- PERSPECTIVE: 180 JSONL rows, 2 PNG frames per row

Split results:

- LENGTH: 150 train rows, 30 eval rows
- PERSPECTIVE: 145 train rows, 35 eval rows

All generated data is under ignored `data/`.

## CPU Validation

Passed:

- Strict parser checks for:
  - `<ANSWER>A</ANSWER>`
  - `<COT>...</COT><ANSWER>B</ANSWER>`
  - `<LONG_COT>...</LONG_COT><ANSWER>C</ANSWER>`
  - `<ABSTRACT>...</ABSTRACT><ANSWER>A</ANSWER>`
  - `<TEMPORAL>...</TEMPORAL><ANSWER>B</ANSWER>`
  - `<SPATIOTEMPORAL>...</SPATIOTEMPORAL><ANSWER>C</ANSWER>`
- `src/scripts/dry_run_balanced_strategy.py`
- `src/scripts/verify_balanced_strategy_rollout.py`

## LENGTH GPU Smoke

Command:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate
bash src/scripts/run_grpo_length_dummy_smoke.sh
```

Key settings:

- `REASONING_TASK_TYPE=length`
- `NUM_GPUS=4`
- `MAX_STEPS=50`
- `NUM_GENERATIONS=3`
- `ROLLOUTS_PER_STRATEGY=1`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_COMPLETION_LENGTH=96`
- `MAX_PIXELS=50176`
- `VLLM_MAX_FRAMES=2`
- `VLLM_GPU_UTIL=0.25`
- `BALANCED_STRATEGY_ROLLOUT=true`

Result: passed.

Final summary:

```text
train_runtime=529.6257
train_samples_per_second=0.283
train_steps_per_second=0.094
train_loss=0.00027381222989436793
epoch=1.0
```

Confirmed during run:

- Training model loaded
- Processor loaded
- vLLM initialized on GPU 3
- Rollout generation succeeded
- Reward computation succeeded
- LENGTH strategy metrics emitted for `direct`, `cot`, and `long_cot`
- Optimizer steps completed through 50/50

Outputs:

```text
tmp_smoke/grpo_length_dummy/training_log.txt
tmp_smoke/grpo_length_dummy/strategy_debug.jsonl
```

`strategy_debug.jsonl` rows: 450.

## PERSPECTIVE GPU Smoke

Command:

```bash
cd /workspace/ATV-Adaptive-Thinking-in-Video-V2
source .venv_grpo_smoke/bin/activate
bash src/scripts/run_grpo_perspective_dummy_smoke.sh
```

Key settings:

- `REASONING_TASK_TYPE=perspective`
- `NUM_GPUS=4`
- `MAX_STEPS=50`
- `NUM_GENERATIONS=3`
- `ROLLOUTS_PER_STRATEGY=1`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_COMPLETION_LENGTH=96`
- `MAX_PIXELS=50176`
- `VLLM_MAX_FRAMES=2`
- `VLLM_GPU_UTIL=0.25`
- `BALANCED_STRATEGY_ROLLOUT=true`

Result: passed.

Final summary after the parser fix:

```text
train_runtime=631.6633
train_samples_per_second=0.237
train_steps_per_second=0.079
train_loss=0.00032049637377998155
epoch=1.02
```

Confirmed during run:

- Training model loaded
- Processor loaded
- vLLM initialized on GPU 3
- Rollout generation succeeded
- Reward computation succeeded
- PERSPECTIVE strategy metrics emitted for `abstract`, `temporal`, and
  `spatiotemporal`
- Optimizer steps completed through 50/50

Outputs:

```text
tmp_smoke/grpo_perspective_dummy/training_log.txt
tmp_smoke/grpo_perspective_dummy/strategy_debug.jsonl
```

`strategy_debug.jsonl` rows: 450.

The parser-fix rerun wrote 450 debug rows: 118 with `format_ok=True` and 332
with `format_ok=False`. The false rows are expected for completions that did not
include a valid PERSPECTIVE reasoning tag.

## Warnings And Limitations

- Both GPU runs ended with a PyTorch NCCL warning that the process group was not
  explicitly destroyed during normal process exit. This occurred after all 50
  steps completed.
- `flash-attn` was not installed. The smoke configuration used SDPA and passed.
- PERSPECTIVE strict parsing now rejects answer-only `<ANSWER>X</ANSWER>`.
  Valid PERSPECTIVE completions must include `<ABSTRACT>`, `<TEMPORAL>`, or
  `<SPATIOTEMPORAL>` before the final answer. LENGTH direct answers remain
  valid as `<ANSWER>X</ANSWER>`.

## Git Hygiene

Generated artifacts are ignored:

- `data/`
- `tmp_smoke/`
- `models/`
- `cache/`
- `.venv_grpo/`
- `.venv_grpo_smoke/`
- logs and Python cache directories

No commit was made.

## Conclusion

Both LENGTH and PERSPECTIVE GRPO paths are ready to run once real datasets are
available. To switch from dummy to real data, keep the same launcher path and
replace `TRAIN_FILE`, `QWEN_PATH`, `QWEN_BASE_PATH`, and `OUTPUT_DIR` with the
real experiment values.
