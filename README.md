# ATV: Adaptive Thinking in Video

Official code release for **Adaptive Thinking in Video (ATV)** — training video-language models to choose *how* to reason (reasoning length or reasoning perspective) before answering multiple-choice video questions.

**Backbone:** [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)  
**Pipeline:** SFT (LoRA) → merge → GRPO (LoRA + balanced strategy rollout) → merge → benchmark evaluation  
**Benchmarks:** [Urban Video Bench](https://github.com/DAMO-NLP-SG/UrbanVideo-Bench) (UVB), [VideoMMMU](https://videommmu.github.io/), [MMVU](https://mmvu-benchmark.github.io/)

---

## Overview

ATV studies **adaptive video reasoning** under two supervision axes:

| Task | What the model learns | Reasoning strategies (3 each) |
|------|------------------------|-------------------------------|
| **LENGTH** | How much to reason before answering | `direct`, `cot`, `long_cot` |
| **PERSPECTIVE** | Which viewpoint to reason from | `abstract`, `temporal`, `spatiotemporal` |

Training proceeds in two stages:

1. **SFT** teaches format-valid outputs with strategy-specific tags (`<COT>`, `<ABSTRACT>`, `<ANSWER>`, etc.).
2. **GRPO** refines answer quality with reinforcement learning. We use **balanced strategy rollout**: each prompt generates `num_generations=9` completions with **3 rollouts per strategy**, then applies a strategy-level reward bonus so GRPO does not collapse to a single reasoning mode.

At inference, models are evaluated on three video MCQ benchmarks with a shared JSONL schema (16 frames per video by default for the main GRPO-4000+ runs).

---

## Main Results (paper checkpoints)

Primary metric: **extractable accuracy** — last `<ANSWER>X</ANSWER>` letter match; unscorable outputs count as wrong.  
Full per-benchmark tables: [`reports/length_perspective_checkpoint_analysis.md`](reports/length_perspective_checkpoint_analysis.md).

### Macro average (UVB + VideoMMMU + MMVU)

| Checkpoint | LENGTH extractable | PERSPECTIVE extractable |
|------------|-------------------:|------------------------:|
| Base | 21.7% | 22.4% |
| SFT | 29.1% | 23.7% |
| GRPO-2000 | 26.6% | 30.3% |
| **GRPO-4000** | **30.0%** | — |
| GRPO-4300 | — | 30.9% |
| GRPO-5000 | 30.0% | **31.3%** |

**Takeaways**

- **LENGTH:** SFT mainly improves output format (+63pp format rate). GRPO-4000 is the practical sweet spot (best strict accuracy, lowest unscorable rate among late checkpoints).
- **PERSPECTIVE:** SFT alone is weak on UVB (58.8% unscorable — reasoning text without `<ANSWER>`). GRPO recovers extractability via `answer_format_reward` (~99.7% scorable on UVB).
- **Caveat:** GRPO-2000 used **8 frames**; GRPO-4000/4300/5000 benchmarks used **16 frames**.

Checkpoint naming in this repo:

| Alias | Merged model directory (on scratch) |
|-------|-------------------------------------|
| LENGTH GRPO-4000 | `qwen25vl3b_grpo_length_step4000_merged` |
| LENGTH GRPO-5000 | `qwen25vl3b_grpo_length_step5000_merged` |
| PERSPECTIVE GRPO-4300 | `qwen25vl3b_grpo_perspective_step4300_merged` |
| PERSPECTIVE GRPO-5000 | `qwen25vl3b_grpo_perspective_step5000_merged` |

---

## Repository Structure

```text
ATV-Adaptive-Thinking-in-Video-V2/
├── data/                         # symlink → large JSONL + frames (see § Data)
├── scripts/                      # HPC venv setup & activation
│   ├── hpc_activate_sft.sh
│   ├── hpc_activate_grpo.sh
│   ├── run_setup_sft.sh
│   └── run_setup_grpo.sh
├── sft/                          # Stage 1: LoRA SFT + merge
│   ├── configs/                  # train & merge YAML (length / perspective)
│   ├── scripts/
│   │   ├── train_sft.py          # SFT training
│   │   ├── merge_lora.py         # LoRA → full weights
│   │   ├── run_train.sh
│   │   └── run_merge.sh
│   └── data/prepare_/            # data-prep shell scripts (JSONL on scratch)
├── src/
│   ├── r1-v/                     # Stage 2: GRPO (open_r1 / Video-R1 lineage)
│   │   └── src/open_r1/
│   │       ├── grpo.py           # GRPO entry + reward wiring
│   │       ├── strategy.py       # balanced rollout logic
│   │       ├── strict_answer.py  # output parsers
│   │       └── trainer/
│   │           └── vllm_grpo_trainer_modified.py
│   ├── eval/                     # Benchmark inference (vLLM)
│   │   ├── uvb_eval_only.py
│   │   ├── videommmu_eval_only.py
│   │   ├── mmvu_eval_only.py
│   │   ├── strict_answer.py
│   │   └── prepare_*.py          # JSONL builders
│   └── scripts/                  # Launchers & utilities
│       ├── run_grpo_length_main_8frames.sh      # LENGTH GRPO (recommended)
│       ├── run_grpo_perspective_main_8frames.sh # PERSPECTIVE GRPO
│       ├── run_grpo_answer_only_lora.sh         # core GRPO launcher
│       ├── run_video_benchmark_matrix*.sh       # Base / SFT / GRPO eval
│       ├── run_models_0624_benchmark_test.sh    # 4-checkpoint parallel eval
│       ├── check_environment.sh
│       └── fix_hf_rope_scaling_for_vllm.py
├── reports/
│   └── length_perspective_checkpoint_analysis.md  # full result analysis
├── outputs/                      # run logs (empty by default; use scratch on HPC)
└── setup.sh                      # GRPO dependency installer
```

> **Note:** YAML filenames under `sft/configs/` retain a `qwen25vl3b` prefix from early naming. Paper experiments use **Qwen2.5-VL-3B-Instruct**. Override `model_name_or_path` / `export_dir` in YAML or via CLI flags.

---

## Installation

SFT and GRPO use **separate virtual environments**. Do not install both into one venv.

### SFT environment

```bash
cd /path/to/ATV-Adaptive-Thinking-in-Video-V2
bash scripts/run_setup_sft.sh

# GPU node + flash-attn (optional)
module load cuda/12.2.2
INSTALL_FLASH_ATTN=true bash scripts/run_setup_sft.sh
```

### GRPO environment

```bash
cd /path/to/ATV-Adaptive-Thinking-in-Video-V2
bash scripts/run_setup_grpo.sh
```

### Per session (HPC)

```bash
# SFT
source scripts/hpc_activate_sft.sh

# GRPO
export GRPO_CUDA_MODULE=cuda/12.2.2   # if needed for DeepSpeed / nvcc
source scripts/hpc_activate_grpo.sh
```

Default venv locations: `$HOME/scratch/.venv_sft` and `$HOME/scratch/.venv_grpo`.

---

## Data

Large JSONL files and video frames are **not shipped in this repo**. Point `data/` to your storage:

```bash
ln -sfn /path/to/your/data_root data
# e.g. ln -sfn $SCRATCH/GRPO_Video_2_data/data data
readlink -f data
```

### Required files

| Purpose | Path (under `data/`) |
|---------|------------------------|
| SFT length | `generated_length_strict.jsonl` (or path in train YAML) |
| SFT perspective | `video_r1_perspective_sft_granulity_strict.jsonl` |
| GRPO train | `video_r1/grpo/processed/video_r1_grpo_train_strict.jsonl` |
| UVB test | `urban_video_bench/grpo/uvb_grpo_test_strict.jsonl` |
| VideoMMMU test | `video_mmmu/grpo/videommmu_grpo_test_strict.jsonl` |
| MMVU test | `mmvu/grpo/mmvu_grpo_test_strict.jsonl` |

Prepare strict MCQ JSONL from raw sources:

```bash
cd sft
python scripts/prepare_sft_dataset.py --mode length \
  --input data/generated_length.jsonl --output data/generated_length_strict.jsonl
python scripts/prepare_sft_dataset.py --mode perspective \
  --input data/generated_granulity.jsonl --output data/video_r1_perspective_sft_granulity_strict.jsonl
```

GRPO / benchmark JSONL builders live in `src/eval/prepare_*.py` and `sft/data/prepare_/prepare_*.sh`.

### Shared JSONL schema

```json
{
  "video_id": "...",
  "question_id": "...",
  "question_category": "...",
  "problem": "<question and choices>",
  "frames": ["/path/to/frame0.png", "..."],
  "solution": "<ANSWER>B</ANSWER>"
}
```

---

## Reproduction

### Quick preflight (GPU node)

```bash
cd /path/to/ATV-Adaptive-Thinking-in-Video-V2
source scripts/hpc_activate_grpo.sh
bash src/scripts/check_environment.sh
```

### Step 1 — SFT (LoRA)

```bash
source scripts/hpc_activate_sft.sh
cd sft

# Length
SFT_MODE=length USE_VISION=true bash scripts/run_train.sh

# Perspective
SFT_MODE=perspective USE_VISION=true bash scripts/run_train.sh
```

Edit `configs/train_lora_qwen25vl3b_{length,perspective}.yaml` so `model_name_or_path` and `train_files` match your layout. Store adapters under `sft/outputs/` or scratch.

### Step 2 — SFT merge

```bash
cd sft
source ../scripts/hpc_activate_sft.sh

SFT_MODE=length MERGE_STAGE=sft bash scripts/run_merge.sh
SFT_MODE=perspective MERGE_STAGE=sft bash scripts/run_merge.sh
```

Example scratch export (recommended for ~7 GB merged weights):

```yaml
# configs/merge_lora_qwen25vl3b_length.yaml
model_name_or_path: /path/to/Qwen2.5-VL-3B-Instruct
adapter_name_or_path: ./outputs/qwen25vl3b_lora_sft_length
export_dir: /path/to/scratch/models/qwen25vl3b_lora_merged_length
remap_adapter_keys: false
```

### Step 3 — GRPO (balanced strategy rollout)

**Recommended entry points** (set hyperparameters + call the core launcher):

```bash
cd /path/to/ATV-Adaptive-Thinking-in-Video-V2
source scripts/hpc_activate_grpo.sh

# Length (defaults: 8 frames, 2000 steps, 4 GPUs)
bash src/scripts/run_grpo_length_main_8frames.sh

# Perspective
bash src/scripts/run_grpo_perspective_main_8frames.sh
```

Main LENGTH defaults (override via `export`):

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_PATH` | `.../qwen25vl3b_lora_merged_length` | SFT-merged base |
| `QWEN_BASE_PATH` | `.../Qwen2.5-VL-3B-Instruct` | clean processor tree |
| `MAX_STEPS` | `2000` | training steps |
| `NUM_GENERATIONS` | `9` | rollouts per prompt |
| `ROLLOUTS_PER_STRATEGY` | `3` | balanced slots per strategy |
| `BALANCED_STRATEGY_ROLLOUT` | `true` | enable strategy balancing |
| `VLLM_MAX_FRAMES` | `8` | frames during GRPO training |
| `ANSWER_ACCURACY_WEIGHT` | `0.8` | MCQ reward weight |
| `ANSWER_FORMAT_WEIGHT` | `0.2` | format reward weight |
| `STRATEGY_BONUS_SCALE` | `0.20` | strategy comparison bonus α |

Smoke test (1 step):

```bash
MAX_STEPS=1 SAVE_STEPS=1 LOGGING_STEPS=1 \
  bash src/scripts/run_grpo_length_main_8frames.sh
```

Resume:

```bash
export RESUME_FROM_CHECKPOINT=/path/to/outputs/.../checkpoint-1500
bash src/scripts/run_grpo_length_main_8frames.sh
```

Low-level launcher (all env vars): `src/scripts/run_grpo_answer_only_lora.sh`

### Step 4 — GRPO merge

```bash
cd sft
source ../scripts/hpc_activate_grpo.sh

python scripts/merge_lora.py \
  —config configs/merge_lora_grpo_length.yaml \
  —model-name-or-path /path/to/qwen25vl3b_lora_merged_length \
  —adapter-name-or-path /path/to/grpo_output/checkpoint-4000 \
  —export-dir /path/to/qwen25vl3b_grpo_length_step4000_merged
```

Set `MERGE_STAGE=grpo SFT_MODE=length bash scripts/run_merge.sh` after editing `adapter_name_or_path` and `export_dir` in the YAML.

### Step 5 — Benchmark evaluation

**Base / SFT / GRPO comparison** (sequential, 1 GPU):

```bash
source scripts/hpc_activate_grpo.sh
bash src/scripts/run_video_benchmark_matrix_length.sh
bash src/scripts/run_video_benchmark_matrix_perspective.sh
```

**Paper checkpoints (4 models, 4 GPUs, UVB → VideoMMMU → MMVU):**

```bash
source scripts/hpc_activate_grpo.sh
bash src/scripts/run_models_0624_benchmark_test.sh
```

Useful overrides:

```bash
FRAMES_PER_SAMPLE=16
GPU_MEM_UTIL=0.40
BENCH_EVAL_EXTRA="--max-model-len 8192"
BENCH_MAX_SAMPLES=20          # smoke subset
NUM_GPUS=1                    # serial
```

Predictions and metrics are written next to each `--model` directory by the eval scripts.

---

## Balanced Strategy Rollout (method detail)

Core implementation: [`src/r1-v/src/open_r1/strategy.py`](src/r1-v/src/open_r1/strategy.py)  
Trainer integration: [`src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py`](src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py)

1. **Slot plan** — For `num_generations=G` and `rollouts_per_strategy=k`, assign `k` slots to each of 3 strategies (LENGTH: direct/cot/long_cot; PERSPECTIVE: abstract/temporal/spatiotemporal).
2. **Prompt expansion** — Duplicate each prompt `G` times; append a strategy directive to the system prompt per slot.
3. **vLLM generation** — `n=1` per expanded prompt (not `n=G` on a single prompt).
4. **Rewards** — Base reward = weighted `answer_accuracy` + `answer_format`. Strategy bonus = deviation from group mean when margin ≥ threshold. Final reward feeds standard GRPO advantage.

Parser rules: [`src/r1-v/src/open_r1/strict_answer.py`](src/r1-v/src/open_r1/strict_answer.py) (mirrored in `src/eval/strict_answer.py` for inference).

---

## Environment Troubleshooting

| Issue | Fix |
|-------|-----|
| Transformers ≥ 5 breaks vLLM Qwen2.5-VL | `pip install 'transformers>=4.52,<5'` |
| Rotary dtype assert (bf16 vs fp32) | auto-applied by `apply_rotary_dtype_hotfix.sh`; disable with `GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false` |
| `CUDA_HOME` unset (DeepSpeed) | `module load cuda/12.2.2` before activate |
| vLLM rope_scaling error at eval | `python src/scripts/fix_hf_rope_scaling_for_vllm.py model_dir ...` |
| Workspace disk quota | store weights & logs on scratch; keep repo as code only |

```bash
bash src/scripts/check_environment.sh
GRPO_CHECK_NO_GPU=1 bash src/scripts/check_environment.sh   # login node
```

---

## Citation

```bibtex
@article{atv2025,
  title   = {Adaptive Thinking in Video},
  author  = {TODO},
  journal = {TODO},
  year    = {2025}
}
```

---

## License

This repository builds on [Video-R1](https://github.com/tulerfeng/Video-R1) / `open_r1` and uses Qwen2.5-VL. See component licenses in `src/r1-v/` and the Qwen model card. **TODO:** add project license before public release.

---

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) backbone  
- [Video-R1](https://github.com/tulerfeng/Video-R1) GRPO training stack  
- Benchmark authors: Urban Video Bench, VideoMMMU, MMVU
