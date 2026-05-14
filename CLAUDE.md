# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for video QA using `Qwen2.5-VL-7B-Instruct` as the backbone. The pipeline follows:

**SFT (LoRA) → SFT merge → GRPO (LoRA) → GRPO merge → Benchmark evaluation**

- Train set: Video-R1 (subsets: LLaVA-Video-178K, NeXT-QA, PerceptionTest, CLEVRER, STAR)
- Eval benchmarks: Urban Video Bench (UVB), VideoMMMU, MMVU (MC only)
- Note: YAML/script filenames contain `3b` (legacy naming) — **actual default is 7B model**

## Environment Setup

**SFT and GRPO use separate venvs — do not mix.**

| Purpose | One-time install | Per-session activation | Default path |
|---------|-----------------|----------------------|--------------|
| SFT | `bash scripts/run_setup_sft.sh` | `source scripts/hpc_activate_sft.sh` | `~/scratch/.venv_sft` |
| GRPO | `bash scripts/run_setup_grpo.sh` | `source scripts/hpc_activate_grpo.sh` | `~/scratch/.venv_grpo` |

- `setup.sh` at repo root is **GRPO-only**; do not use for SFT
- On HPC, always activate via `hpc_activate_*.sh` (includes module loads; bare `activate` may fail with `libpython` errors)
- Check environment: `bash src/scripts/check_environment.sh`

## Key Commands

### Data Preparation

```bash
# Video-R1 train data (full subsets)
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --num-frames 16 --sample-ratio 1 \
  --download-mode subset-directories

# UVB test set
bash src/scripts/prepare_uvb_grpo_data.sh

# VideoMMMU / MMVU
bash src/scripts/prepare_videommmu_grpo_data.sh
bash src/scripts/prepare_mmvu_grpo_data.sh

# All at once
bash src/scripts/prepare_all_grpo_data.sh
```

Use `--download-mode subset-directories` for Video-R1; `sampled-files` mode is currently unreliable.

### SFT Training (run from `sft/` directory)

```bash
# Length SFT (Direct Answer / CoT / Long CoT supervision)
SFT_MODE=length USE_VISION=true CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh

# Perspective SFT (REASONING_TYPE → REASONING → ANSWER supervision)
SFT_MODE=perspective USE_VISION=true CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh

# Resume from checkpoint
export RESUME_FROM_CHECKPOINT=/path/to/checkpoint-NNN
SFT_MODE=length USE_VISION=true bash scripts/run_train.sh
```

Prepare raw annotations → SFT JSONL:
```bash
python scripts/prepare_sft_dataset.py --mode length \
  --input data/generated_length.jsonl --output data/generated_length_strict.jsonl
```

### SFT LoRA Merge (run from `sft/` directory)

```bash
SFT_MODE=length bash scripts/run_merge.sh
# or: MERGE_STAGE=grpo SFT_MODE=length bash scripts/run_merge.sh
```

`run_merge.sh` selects the YAML config based on `MERGE_STAGE` (sft/grpo) × `SFT_MODE` (length/perspective).

### GRPO Training (run from repo root)

```bash
source scripts/hpc_activate_grpo.sh

export QWEN_PATH="/scratch/users/<USER>/models/qwen25vl7b_lora_merged_length"
export QWEN_BASE_PATH="/scratch/users/<USER>/models/Qwen2.5-VL-7B-Instruct"
export TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train_strict.jsonl"
export OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_lora"
export NUM_GPUS=2 TRAIN_NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0,1

bash src/scripts/run_grpo_answer_only_lora.sh
```

- `QWEN_PATH`: merged SFT model (full weights); `QWEN_BASE_PATH`: clean HF weights for `AutoProcessor`
- `GRPO_TEST_FILE`: optional fixed benchmark JSONL; if unset, uses a 5% eval split from train data
- `RESUME_FROM_CHECKPOINT`: path to a checkpoint dir to resume from
- Script auto-applies rotary dtype hotfix for Qwen2.5-VL; set `GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false` to skip
- Placeholder paths (`/path/to/`, `...`) cause the script to exit with an error

### Benchmark Evaluation

```bash
# Run all three benchmarks against all three model variants (base / SFT merged / GRPO merged)
export MODEL_BASE=/path/to/Qwen2.5-VL-7B-Instruct
export MODEL_SFT_MERGED=/path/to/qwen25vl7b_lora_merged_length
export MODEL_SFT_GRPO_MERGED=/path/to/grpo_merged_model
bash src/scripts/run_video_benchmark_matrix.sh

# Individual eval
python src/eval/uvb_eval_only.py \
  --model /path/to/model \
  --test-file data/urban_video_bench/grpo/uvb_grpo_test_strict.jsonl \
  --device cuda:0 --gpu-memory-utilization 0.6

python src/eval/videommmu_eval_only.py --model ... --test-file data/video_mmmu/grpo/videommmu_grpo_test_strict.jsonl ...
python src/eval/mmvu_eval_only.py --model ... --test-file data/mmvu/grpo/mmvu_grpo_test_strict.jsonl ...
```

Eval metrics: `answer_accuracy`, `answer_format_rate`, `reasoning_present_rate`.
Output files are auto-saved timestamped into the model directory.

**Compatibility note for eval venv**: requires `transformers>=4.49,<5` and `vllm>=0.7.3` (vLLM 0.7.2 + transformers≥4.49 breaks Qwen2.5-VL image processor).

## Architecture

### Data Flow

```
Raw datasets (HF)
  → prepare_*.py              # dataset-specific download, frame extraction
  → processed/*.jsonl + processed/frames/...
  → data_to_grpo.py           # unified GRPO JSONL format
  → grpo/*_strict.jsonl       # model input
```

The `_strict` suffix indicates MCQ-only rows with normalized `<ANSWER>X</ANSWER>` tags.

### Common GRPO JSONL Schema

All train and eval files share the same schema:

```json
{
  "video_id": "...",
  "question_id": 123,
  "question_category": "...",
  "problem": "Question: ...\nOptions:\nA. ...\nB. ...",
  "frames": ["../processed/frames/train/.../frame_000.jpg", ...],
  "solution": "<ANSWER>B</ANSWER>"
}
```

`frames` paths are relative to the JSONL file's directory.

### SFT Output Formats

**Length mode** (`sft_mode: length`):
- Trains `<ANSWER>`, `<COT>`, `<LONG_COT>` supervision (controlled by `reasoning_formats` in YAML)
- `drop_code_cot: true` by default

**Perspective mode** (`sft_mode: perspective`):
```xml
<REASONING_TYPE>TEMPORAL</REASONING_TYPE>
<REASONING>...</REASONING>
<ANSWER>A</ANSWER>
```

### GRPO Reward Functions

Defined in [src/r1-v/src/open_r1/grpo.py](src/r1-v/src/open_r1/grpo.py):
- `answer_accuracy`: correctness of parsed answer letter
- `answer_format`: structural validity of output tags
- `GRPO_REASONING_TASK_TYPE` env var (default `length`) selects which tag schema to parse

### Key Source Files

| File | Role |
|------|------|
| [src/r1-v/src/open_r1/grpo.py](src/r1-v/src/open_r1/grpo.py) | GRPO entry point, reward functions, `main()` |
| [src/r1-v/src/open_r1/grpo_video.py](src/r1-v/src/open_r1/grpo_video.py) | Thin alias entrypoint wrapping `grpo.py` |
| [src/r1-v/src/open_r1/trainer/grpo_trainer.py](src/r1-v/src/open_r1/trainer/grpo_trainer.py) | `Qwen2VLGRPOTrainer` — custom vLLM+DeepSpeed GRPO trainer |
| [src/r1-v/src/open_r1/strict_answer.py](src/r1-v/src/open_r1/strict_answer.py) | Output tag parsing (`parse_strict_output`) |
| [src/eval/data_to_grpo.py](src/eval/data_to_grpo.py) | Converts `processed/` → unified GRPO JSONL |
| [src/eval/grpo_data_utils.py](src/eval/grpo_data_utils.py) | Shared normalization utils (answer, problem, frame paths) |
| [src/eval/video_dataset_prep_utils.py](src/eval/video_dataset_prep_utils.py) | HF download, frame extraction utilities |
| [sft/scripts/train_sft.py](sft/scripts/train_sft.py) | SFT training loop |
| [sft/scripts/merge_lora.py](sft/scripts/merge_lora.py) | LoRA merge for both SFT and GRPO stages |

### Model Paths Convention

- SFT LoRA adapter: `sft/outputs/qwen25vl7b_lora_sft_{length|perspective}/`
- SFT merged weights: `sft/outputs/qwen25vl7b_lora_merged_{length|perspective}/`
- GRPO LoRA adapter: `src/r1-v/outputs/video_r1_uvb_grpo_answer_only_lora/`
- GRPO merged weights: configured via `export_dir` in `sft/configs/merge_lora_grpo_{length|perspective}.yaml`

Large models/data live in scratch storage (`~/scratch/`); `data/` is typically a symlink to scratch.