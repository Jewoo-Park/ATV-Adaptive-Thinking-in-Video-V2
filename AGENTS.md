# Repository Guidelines

## Project Structure & Module Organization
This repo implements a video QA research pipeline: **SFT LoRA -> SFT merge -> GRPO LoRA -> GRPO merge -> benchmark eval**. Core GRPO code lives in `src/r1-v/src/open_r1/` (`grpo.py`, `strategy.py`, `strict_answer.py`), trainers in `src/r1-v/src/open_r1/trainer/`, dataset/eval utilities in `src/eval/`, and launch scripts in `src/scripts/`. SFT assets are in `sft/` (`scripts/`, `configs/`, `data/`). Use `scripts/` only for environment setup/activation.

## Build, Test, and Development Commands
- Setup envs separately: `bash scripts/run_setup_sft.sh`, `bash scripts/run_setup_grpo.sh`
- Activate matching env: `source scripts/hpc_activate_sft.sh` or `source scripts/hpc_activate_grpo.sh`
- SFT train: `cd sft && SFT_MODE=length USE_VISION=true bash scripts/run_train.sh` (or `SFT_MODE=perspective`)
- Merge adapters: `cd sft && SFT_MODE=length bash scripts/run_merge.sh` (`MERGE_STAGE=grpo` for GRPO merge configs)
- GRPO launch: `bash src/scripts/run_grpo_answer_only_lora.sh` with `QWEN_PATH`, `QWEN_BASE_PATH`, `TRAIN_FILE`
- Eval matrix: `bash src/scripts/run_video_benchmark_matrix.sh`
- Fast checks (no long run): `bash src/scripts/check_environment.sh`, `python src/scripts/dry_run_balanced_strategy.py`, `python src/scripts/verify_balanced_strategy_rollout.py`

## Coding Style & Naming Conventions
Use Python 4-space indentation and small utility functions. Keep strict output tags unchanged: LENGTH (`<ANSWER>`, `<COT>`, `<LONG_COT>`) and PERSPECTIVE (`<ABSTRACT>`, `<TEMPORAL>`, `<SPATIOTEMPORAL>` + `<ANSWER>`). Keep mode names explicit in files/dirs (for example `*_length_*`, `*_perspective_*`).

## Testing Guidelines
No centralized pytest suite is used for pipeline validation. Prefer targeted checks: JSONL conversion via `src/eval/data_to_grpo.py`, deterministic split checks via `src/scripts/split_jsonl_train_eval.py`, and one-benchmark smoke eval (`src/eval/uvb_eval_only.py`) before full matrix runs.

## Commit & Pull Request Guidelines
Commit subjects are short and imperative, often prefixed (`feat:`, `fix:`), e.g. `feat: add balanced strategy rollout`. PRs should state changed stage (SFT/GRPO/eval), exact commands run, key env vars, and output paths used.

## Security & Configuration Tips
Never commit secrets (`HF_TOKEN`, `.env`) or cluster-private absolute paths. Keep generated artifacts untracked: `data/`, `models/`, `outputs/`, `checkpoints/`, `tmp_smoke/`, caches, logs, and weight files (`*.safetensors`, `*.bin`, `*.pt`). Replace placeholder paths (`/path/to/...`) before launch.
