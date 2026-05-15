# Repository Guidelines

## Project Structure & Module Organization

This repository implements a video QA pipeline: SFT LoRA, SFT merge, GRPO LoRA, GRPO merge, then benchmark evaluation. Core GRPO code lives in `src/r1-v/src/open_r1/`, with trainers in `src/r1-v/src/open_r1/trainer/`. Dataset preparation and benchmark entry points are in `src/eval/`. SFT training, merge scripts, YAML configs, and requirements are under `sft/`. Top-level `scripts/` contains HPC setup helpers; `src/scripts/` contains GRPO, validation, and benchmark scripts. Keep large data, weights, outputs, and logs out of Git, usually in scratch-backed `data/`, `models/`, or `outputs/` paths.

## Build, Test, and Development Commands

- `bash scripts/run_setup_sft.sh`: create the SFT environment.
- `bash scripts/run_setup_grpo.sh`: create the GRPO environment.
- `source scripts/hpc_activate_sft.sh` or `source scripts/hpc_activate_grpo.sh`: activate the correct runtime; do not mix dependencies.
- `bash src/scripts/check_environment.sh`: verify GRPO packages.
- `cd sft && SFT_MODE=length USE_VISION=true bash scripts/run_train.sh`: run length-mode SFT. Use `SFT_MODE=perspective` for perspective training.
- `cd sft && SFT_MODE=length bash scripts/run_merge.sh`: merge an SFT LoRA adapter.
- `bash src/scripts/run_grpo_answer_only_lora.sh`: run GRPO from the repository root after exporting `QWEN_PATH`, `QWEN_BASE_PATH`, and data paths.
- `bash src/scripts/run_video_benchmark_matrix.sh`: evaluate configured models on UVB, VideoMMMU, and MMVU.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, explicit argument names, and small utilities for shared dataset or parsing behavior. Preserve XML-like tags such as `<ANSWER>`, `<COT>`, and `<REASONING_TYPE>`. Keep YAML config names aligned with modes: `length`, `perspective`, `sft`, and `grpo`. Some filenames contain legacy `3b`; do not rename them casually because docs and scripts still reference those paths while defaults target Qwen2.5-VL 7B.

## Testing Guidelines

There is no centralized pytest suite. Validate changes with the nearest script-level check: `bash src/scripts/check_environment.sh`, `python src/scripts/dry_run_balanced_strategy.py`, `python src/scripts/verify_balanced_strategy_rollout.py`, or a small dataset conversion/eval run. For eval changes, test one small JSONL before the full matrix.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with `feat:` and `fix:` prefixes. Prefer `feat: add ...`, `fix: correct ...`, or concise sentence-case messages. Pull requests should describe the affected pipeline stage, list required environment variables or data paths, mention validation commands, and include key metric/log snippets for training or benchmark changes. Do not commit generated weights, datasets, logs, or scratch outputs.

## Security & Configuration Tips

Keep Hugging Face tokens, cluster paths, and private scratch locations out of committed configs unless they are placeholders. Before running training, replace `/path/to/...` examples and confirm `data/` symlinks point to the intended scratch storage.
