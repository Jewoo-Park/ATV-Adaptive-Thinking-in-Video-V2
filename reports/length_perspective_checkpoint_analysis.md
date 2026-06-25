# LENGTH / PERSPECTIVE Checkpoint Reasoning Analysis

> Auto-generated from raw prediction jsonl via `src/scripts/analyze_checkpoint_reasoning.py`.
> Primary metric: **extractable accuracy** (lenient), not strict.

# Executive Summary

- **LENGTH best checkpoints:** macro extractable accuracy peaks at **GRPO-5000 (30.02%)** and **GRPO-4000 (29.98%)** — virtually tied. Per-benchmark, **GRPO-4000** wins VideoMMMU (33.17% vs 31.75%) and MMVU (39.20% vs 38.88%); **GRPO-4000** also has higher strict accuracy (25.98% vs 23.72%) and lower unscorable rate (3.11% vs 5.60%).
- **PERSPECTIVE best checkpoints:** **GRPO-5000 (31.34%)** macro extractable, followed by **GRPO-4300 (30.92%)** and **GRPO-2000 (30.29%)**. All GRPO checkpoints massively outperform Base/SFT on extractable accuracy.
- **PERSPECTIVE SFT is a cautionary baseline:** UVB extractable accuracy is only **12.16%** because **58.8%** of samples lack an extractable `<ANSWER>` tag (reasoning text present, answer missing). GRPO reduces UVB unscorable to **<0.5%**.
- **Base → SFT (LENGTH):** format 12.6% → 75.6%; macro extractable +7.4pp. Dominant effect is format/tag learning, not reasoning quality.
- **SFT → GRPO (LENGTH):** COT share falls from **68% → <1%**; DIRECT rises to **22–34%**; LONG_COT stays high (**49–57%**). GRPO-4000 is the practical sweet spot for strict accuracy.
- **SFT → GRPO (PERSPECTIVE):** strategy shifts to **TEMPORAL-heavy (45–71%)**; later checkpoints (4300/5000) rebalance toward **ABSTRACT+SPATIOTEMPORAL (~50% combined)** vs GRPO-2000 collapse on TEMPORAL.
- **Adaptive reasoning is weak:** UVB category-level *top strategy* ≠ *best-accuracy strategy* for most checkpoints; GRPO changes over-used strategy, not category-adaptive selection.
- **Checkpoint monotonicity fails:** LENGTH-5000 regresses format/unscorable vs 4000; PERSPECTIVE-5000 MMVU (40.96%) trails 4300 (42.08%).
- **Caveat:** GRPO-2000 used **8 frames**; GRPO-4000/4300/5000 used **16 frames**. Base/SFT are Jun-20 batches; 4000+ are Jun-24/25 `models_0624` runs.

# Evaluation Setup

## Parser / evaluator

| Component | Path | Role |
|---|---|---|
| `parse_strict_output` | `src/eval/strict_answer.py` | Valid reasoning tag + answer structure |
| `lenient_letter` | `src/scripts/comprehensive_bench_reanalysis.py` | Last `<ANSWER>X</ANSWER>` extraction |
| This script | `src/scripts/analyze_checkpoint_reasoning.py` | Aggregation & reporting |

## Metric definitions

| Metric | Definition |
|---|---|
| **Extractable accuracy** | Correct last `<ANSWER>` letter / all `n` (unscorable = wrong). |
| **Format rate** | Valid task-specific reasoning tag + strict format OK. |
| **Unscorable** | No extractable answer letter. |
| **no_valid_tag** | Scorable but invalid/missing reasoning tag. |

## Source files

| Task | Checkpoint | UVB | VideoMMMU | MMVU | Notes |
|---|---|---|---|---|---|
| length | Base | `eval_predictions_eval_1781921662.jsonl` | `videommmu_predictions_eval_1781940428.jsonl` | `mmvu_predictions_eval_1781942430.jsonl` | Qwen2.5-VL-3B-Instruct; length prompt; Jun-20 batch |
| length | SFT | `eval_predictions_eval_1781921664.jsonl` | `videommmu_predictions_eval_1781940428.jsonl` | `mmvu_predictions_eval_1781942430.jsonl` | qwen25vl3b_lora_merged_length; Jun-20 batch |
| length | GRPO-2000 | `eval_predictions_eval_1781921668.jsonl` | `videommmu_predictions_eval_1781965496.jsonl` | `mmvu_predictions_eval_1781942430.jsonl` | qwen25vl3b_grpo_length_8f_step2000_merged; 8 frames |
| length | GRPO-4000 | `eval_predictions_eval_1782305781.jsonl` | `videommmu_predictions_eval_1782360821.jsonl` | `mmvu_predictions_eval_1782357494.jsonl` | qwen25vl3b_grpo_length_step4000_merged; 16 frames |
| length | GRPO-5000 | `eval_predictions_eval_1782305781.jsonl` | `videommmu_predictions_eval_1782360821.jsonl` | `mmvu_predictions_eval_1782357494.jsonl` | qwen25vl3b_grpo_length_step5000_merged; 16 frames |
| perspective | Base | `eval_predictions_eval_1781966864.jsonl` | `videommmu_predictions_eval_1781982140.jsonl` | `mmvu_predictions_eval_1781984117.jsonl` | Qwen2.5-VL-3B-Instruct; perspective prompt; Jun-20 batch |
| perspective | SFT | `eval_predictions_eval_1781966864.jsonl` | `videommmu_predictions_eval_1781982140.jsonl` | `mmvu_predictions_eval_1781984117.jsonl` | qwen25vl3b_lora_merged_perspective; Jun-20 batch |
| perspective | GRPO-2000 | `eval_predictions_eval_1781966864.jsonl` | `videommmu_predictions_eval_1781982140.jsonl` | `mmvu_predictions_eval_1781984117.jsonl` | qwen25vl3b_grpo_perspective_8f_step2000_merged; 8 frames |
| perspective | GRPO-4300 | `eval_predictions_eval_1782305781.jsonl` | `videommmu_predictions_eval_1782355333.jsonl` | `mmvu_predictions_eval_1782357656.jsonl` | qwen25vl3b_grpo_perspective_step4300_merged; 16 frames |
| perspective | GRPO-5000 | `eval_predictions_eval_1782305781.jsonl` | `videommmu_predictions_eval_1782360821.jsonl` | `mmvu_predictions_eval_1782357494.jsonl` | qwen25vl3b_grpo_perspective_step5000_merged; 16 frames |

# LENGTH Results

## Overall

### Per benchmark

| Checkpoint | Benchmark | n | Extractable Acc | Format Rate | Unscorable | Scorable |
|---|---|---:|---:|---:|---:|---:|
| Base | UVB | 5355 | 22.39% | 13.17% | 867 (16.19%) | 4488 |
| Base | VideoMMMU | 841 | 17.48% | 8.92% | 129 (15.34%) | 712 |
| Base | MMVU | 625 | 21.44% | 12.48% | 105 (16.80%) | 520 |
| SFT | UVB | 5355 | 28.40% | 72.40% | 104 (1.94%) | 5251 |
| SFT | VideoMMMU | 841 | 28.66% | 87.75% | 16 (1.90%) | 825 |
| SFT | MMVU | 625 | 36.00% | 86.56% | 13 (2.08%) | 612 |
| GRPO-2000 | UVB | 5355 | 26.54% | 75.95% | 270 (5.04%) | 5085 |
| GRPO-2000 | VideoMMMU | 841 | 25.45% | 71.11% | 43 (5.11%) | 798 |
| GRPO-2000 | MMVU | 625 | 28.64% | 38.88% | 112 (17.92%) | 513 |
| GRPO-4000 | UVB | 5355 | 28.40% | 83.85% | 157 (2.93%) | 5198 |
| GRPO-4000 | VideoMMMU | 841 | 33.17% | 86.56% | 25 (2.97%) | 816 |
| GRPO-4000 | MMVU | 625 | 39.20% | 72.00% | 30 (4.80%) | 595 |
| GRPO-5000 | UVB | 5355 | 28.72% | 76.90% | 272 (5.08%) | 5083 |
| GRPO-5000 | VideoMMMU | 841 | 31.75% | 68.97% | 62 (7.37%) | 779 |
| GRPO-5000 | MMVU | 625 | 38.88% | 60.32% | 48 (7.68%) | 577 |

*Primary metric: extractable accuracy over all n. Unscorable samples count as incorrect.*

### Macro aggregate (UVB + VideoMMMU + MMVU)

| Checkpoint | n | Extractable Acc | Strict Acc | Format Rate | Unscorable |
|---|---:|---:|---:|---:|---:|
| Base | 6821 | 21.70% | 3.27% | 12.58% | 1101 (16.14%) |
| SFT | 6821 | 29.13% | 22.01% | 75.59% | 133 (1.95%) |
| GRPO-2000 | 6821 | 26.59% | 20.63% | 71.95% | 425 (6.23%) |
| GRPO-4000 | 6821 | 29.98% | 25.98% | 83.10% | 212 (3.11%) |
| GRPO-5000 | 6821 | 30.02% | 23.72% | 74.40% | 382 (5.60%) |

## Strategy Distribution and Accuracy

| Checkpoint | DIRECT | COT | LONG_COT | no_valid_tag | UNSCORABLE |
|---|---|---|---|---|---|
| Base | 1.2% / 19.3% | 4.6% / 26.0% | 6.7% / 27.2% | 71.3% / 25.9% | 16.1% / 0.0% |
| SFT | 0.0% / 0.0% | 68.0% / 29.3% | 7.6% / 27.5% | 22.5% / 31.7% | 1.9% / 0.0% |
| GRPO-2000 | 14.2% / 30.3% | 0.6% / 26.2% | 57.1% / 28.3% | 21.8% / 27.4% | 6.2% / 0.0% |
| GRPO-4000 | 33.9% / 34.3% | 0.4% / 25.9% | 48.8% / 29.2% | 13.8% / 29.0% | 3.1% / 0.0% |
| GRPO-5000 | 22.3% / 34.1% | 0.1% / 28.6% | 52.0% / 30.9% | 20.0% / 31.5% | 5.6% / 0.0% |

*Cell format: `% of all samples / extractable accuracy within bucket`.*

### Base — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| DIRECT | 83 | 1.22% | 19.28% |
| COT | 315 | 4.62% | 26.03% |
| LONG_COT | 460 | 6.74% | 27.17% |
| no_valid_tag | 4862 | 71.28% | 25.85% |
| UNSCORABLE | 1101 | 16.14% | 0.00% |

### SFT — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| DIRECT | 1 | 0.01% | 0.00% |
| COT | 4635 | 67.95% | 29.30% |
| LONG_COT | 520 | 7.62% | 27.50% |
| no_valid_tag | 1532 | 22.46% | 31.72% |
| UNSCORABLE | 133 | 1.95% | 0.00% |

### GRPO-2000 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| DIRECT | 970 | 14.22% | 30.31% |
| COT | 42 | 0.62% | 26.19% |
| LONG_COT | 3896 | 57.12% | 28.29% |
| no_valid_tag | 1488 | 21.81% | 27.35% |
| UNSCORABLE | 425 | 6.23% | 0.00% |

### GRPO-4000 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| DIRECT | 2312 | 33.90% | 34.30% |
| COT | 27 | 0.40% | 25.93% |
| LONG_COT | 3329 | 48.81% | 29.20% |
| no_valid_tag | 941 | 13.80% | 29.01% |
| UNSCORABLE | 212 | 3.11% | 0.00% |

### GRPO-5000 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| DIRECT | 1521 | 22.30% | 34.12% |
| COT | 7 | 0.10% | 28.57% |
| LONG_COT | 3547 | 52.00% | 30.93% |
| no_valid_tag | 1364 | 20.00% | 31.52% |
| UNSCORABLE | 382 | 5.60% | 0.00% |

## Problem Type × Strategy Analysis (UVB)

### Base

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 16.91% | no_valid_tag | LONG_COT |
| Landmark Position | 876 | 19.06% | no_valid_tag | LONG_COT |
| Progress Evaluation | 781 | 20.36% | no_valid_tag | COT |
| Trajectory Captioning | 421 | 25.42% | no_valid_tag | COT |
| Goal Detection | 320 | 18.44% | no_valid_tag | DIRECT |
| Cognitive Map | 270 | 24.44% | no_valid_tag | LONG_COT |
| High-level Planning | 261 | 32.18% | no_valid_tag | COT |
| Association Reasoning | 237 | 10.13% | no_valid_tag | LONG_COT |

- **Action Generation** dist `{'DIRECT': 13, 'COT': 84, 'LONG_COT': 87, 'no_valid_tag': 805}` acc `{'DIRECT': 15.384615384615385, 'COT': 20.238095238095237, 'LONG_COT': 21.839080459770116, 'no_valid_tag': 20.124223602484474}`
- **Landmark Position** dist `{'DIRECT': 13, 'COT': 29, 'LONG_COT': 81, 'no_valid_tag': 630}` acc `{'DIRECT': 23.076923076923077, 'COT': 24.137931034482758, 'LONG_COT': 28.395061728395063, 'no_valid_tag': 21.26984126984127}`
- **Progress Evaluation** dist `{'DIRECT': 12, 'COT': 33, 'LONG_COT': 65, 'no_valid_tag': 552}` acc `{'DIRECT': 8.333333333333334, 'COT': 30.303030303030305, 'LONG_COT': 21.53846153846154, 'no_valid_tag': 24.27536231884058}`
- **Trajectory Captioning** dist `{'DIRECT': 6, 'COT': 24, 'LONG_COT': 30, 'no_valid_tag': 279}` acc `{'DIRECT': 0.0, 'COT': 33.333333333333336, 'LONG_COT': 23.333333333333332, 'no_valid_tag': 32.97491039426523}`

### SFT

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 19.86% | COT | COT |
| Landmark Position | 876 | 22.83% | COT | no_valid_tag |
| Progress Evaluation | 781 | 22.02% | COT | LONG_COT |
| Trajectory Captioning | 421 | 29.69% | COT | COT |
| Goal Detection | 320 | 19.38% | no_valid_tag | no_valid_tag |
| Cognitive Map | 270 | 41.11% | COT | COT |
| High-level Planning | 261 | 43.30% | COT | no_valid_tag |
| Association Reasoning | 237 | 13.08% | no_valid_tag | no_valid_tag |

- **Action Generation** dist `{'COT': 946, 'LONG_COT': 30, 'no_valid_tag': 188}` acc `{'COT': 21.67019027484144, 'LONG_COT': 16.666666666666668, 'no_valid_tag': 13.297872340425531}`
- **Landmark Position** dist `{'COT': 576, 'LONG_COT': 98, 'no_valid_tag': 191}` acc `{'COT': 20.3125, 'LONG_COT': 24.489795918367346, 'no_valid_tag': 30.89005235602094}`
- **Progress Evaluation** dist `{'COT': 634, 'LONG_COT': 30, 'no_valid_tag': 114}` acc `{'COT': 22.870662460567825, 'LONG_COT': 23.333333333333332, 'no_valid_tag': 17.54385964912281}`
- **Trajectory Captioning** dist `{'COT': 238, 'LONG_COT': 35, 'no_valid_tag': 126}` acc `{'COT': 31.932773109243698, 'LONG_COT': 28.571428571428573, 'no_valid_tag': 30.952380952380953}`

### GRPO-2000

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 19.70% | LONG_COT | DIRECT |
| Landmark Position | 876 | 18.72% | LONG_COT | COT |
| Progress Evaluation | 781 | 21.00% | LONG_COT | DIRECT |
| Trajectory Captioning | 421 | 31.35% | LONG_COT | LONG_COT |
| Goal Detection | 320 | 21.25% | LONG_COT | DIRECT |
| Cognitive Map | 270 | 35.93% | LONG_COT | COT |
| High-level Planning | 261 | 40.23% | LONG_COT | LONG_COT |
| Association Reasoning | 237 | 15.61% | LONG_COT | COT |

- **Action Generation** dist `{'DIRECT': 73, 'COT': 14, 'LONG_COT': 882, 'no_valid_tag': 177}` acc `{'DIRECT': 26.027397260273972, 'COT': 21.428571428571427, 'LONG_COT': 21.08843537414966, 'no_valid_tag': 14.124293785310735}`
- **Landmark Position** dist `{'DIRECT': 102, 'COT': 9, 'LONG_COT': 530, 'no_valid_tag': 181}` acc `{'DIRECT': 19.607843137254903, 'COT': 33.333333333333336, 'LONG_COT': 19.81132075471698, 'no_valid_tag': 19.88950276243094}`
- **Progress Evaluation** dist `{'DIRECT': 108, 'COT': 3, 'LONG_COT': 479, 'no_valid_tag': 141}` acc `{'DIRECT': 27.77777777777778, 'COT': 0.0, 'LONG_COT': 20.876826722338205, 'no_valid_tag': 24.113475177304963}`
- **Trajectory Captioning** dist `{'DIRECT': 94, 'COT': 2, 'LONG_COT': 223, 'no_valid_tag': 87}` acc `{'DIRECT': 32.97872340425532, 'COT': 0.0, 'LONG_COT': 33.63228699551569, 'no_valid_tag': 29.885057471264368}`

### GRPO-4000

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 18.17% | LONG_COT | COT |
| Landmark Position | 876 | 22.15% | LONG_COT | DIRECT |
| Progress Evaluation | 781 | 25.10% | LONG_COT | COT |
| Trajectory Captioning | 421 | 35.63% | LONG_COT | no_valid_tag |
| Goal Detection | 320 | 17.50% | DIRECT | no_valid_tag |
| Cognitive Map | 270 | 38.52% | LONG_COT | DIRECT |
| High-level Planning | 261 | 47.13% | LONG_COT | LONG_COT |
| Association Reasoning | 237 | 13.50% | LONG_COT | DIRECT |

- **Action Generation** dist `{'DIRECT': 226, 'COT': 10, 'LONG_COT': 816, 'no_valid_tag': 106}` acc `{'DIRECT': 18.141592920353983, 'COT': 40.0, 'LONG_COT': 18.504901960784313, 'no_valid_tag': 17.92452830188679}`
- **Landmark Position** dist `{'DIRECT': 242, 'COT': 2, 'LONG_COT': 397, 'no_valid_tag': 188}` acc `{'DIRECT': 27.68595041322314, 'COT': 0.0, 'LONG_COT': 22.92191435768262, 'no_valid_tag': 19.148936170212767}`
- **Progress Evaluation** dist `{'DIRECT': 141, 'COT': 5, 'LONG_COT': 477, 'no_valid_tag': 132}` acc `{'DIRECT': 30.49645390070922, 'COT': 40.0, 'LONG_COT': 24.31865828092243, 'no_valid_tag': 26.515151515151516}`
- **Trajectory Captioning** dist `{'DIRECT': 164, 'LONG_COT': 234, 'no_valid_tag': 20}` acc `{'DIRECT': 33.53658536585366, 'LONG_COT': 37.17948717948718, 'no_valid_tag': 40.0}`

### GRPO-5000

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 19.86% | LONG_COT | LONG_COT |
| Landmark Position | 876 | 20.78% | LONG_COT | DIRECT |
| Progress Evaluation | 781 | 25.22% | LONG_COT | no_valid_tag |
| Trajectory Captioning | 421 | 36.82% | LONG_COT | no_valid_tag |
| Goal Detection | 320 | 20.62% | DIRECT | DIRECT |
| Cognitive Map | 270 | 45.56% | LONG_COT | no_valid_tag |
| High-level Planning | 261 | 42.53% | LONG_COT | COT |
| Association Reasoning | 237 | 14.77% | LONG_COT | LONG_COT |

- **Action Generation** dist `{'DIRECT': 124, 'COT': 5, 'LONG_COT': 836, 'no_valid_tag': 167}` acc `{'DIRECT': 17.741935483870968, 'COT': 0.0, 'LONG_COT': 21.650717703349283, 'no_valid_tag': 19.161676646706585}`
- **Landmark Position** dist `{'DIRECT': 157, 'LONG_COT': 415, 'no_valid_tag': 236}` acc `{'DIRECT': 25.477707006369428, 'LONG_COT': 22.650602409638555, 'no_valid_tag': 20.338983050847457}`
- **Progress Evaluation** dist `{'DIRECT': 82, 'LONG_COT': 509, 'no_valid_tag': 149}` acc `{'DIRECT': 20.73170731707317, 'LONG_COT': 27.308447937131632, 'no_valid_tag': 27.516778523489933}`
- **Trajectory Captioning** dist `{'DIRECT': 121, 'LONG_COT': 248, 'no_valid_tag': 42}` acc `{'DIRECT': 36.36363636363637, 'LONG_COT': 36.693548387096776, 'no_valid_tag': 47.61904761904762}`

## Checkpoint Trend

| Checkpoint | UVB | VideoMMMU | MMVU | Macro | Format | Unscorable |
|---|---:|---:|---:|---:|---:|---:|
| Base | 22.39% | 17.48% | 21.44% | 21.70% | 12.58% | 16.14% |
| SFT | 28.40% | 28.66% | 36.00% | 29.13% | 75.59% | 1.95% |
| GRPO-2000 | 26.54% | 25.45% | 28.64% | 26.59% | 71.95% | 6.23% |
| GRPO-4000 | 28.40% | 33.17% | 39.20% | 29.98% | 83.10% | 3.11% |
| GRPO-5000 | 28.72% | 31.75% | 38.88% | 30.02% | 74.40% | 5.60% |

# PERSPECTIVE Results

## Overall

### Per benchmark

| Checkpoint | Benchmark | n | Extractable Acc | Format Rate | Unscorable | Scorable |
|---|---|---:|---:|---:|---:|---:|
| Base | UVB | 5355 | 19.01% | 53.69% | 1638 (30.59%) | 3717 |
| Base | VideoMMMU | 841 | 16.88% | 22.59% | 225 (26.75%) | 616 |
| Base | MMVU | 625 | 16.80% | 36.32% | 238 (38.08%) | 387 |
| SFT | UVB | 5355 | 12.16% | 24.07% | 3150 (58.82%) | 2205 |
| SFT | VideoMMMU | 841 | 25.45% | 81.69% | 81 (9.63%) | 760 |
| SFT | MMVU | 625 | 19.68% | 46.40% | 269 (43.04%) | 356 |
| GRPO-2000 | UVB | 5355 | 28.93% | 90.64% | 23 (0.43%) | 5332 |
| GRPO-2000 | VideoMMMU | 841 | 32.46% | 94.53% | 8 (0.95%) | 833 |
| GRPO-2000 | MMVU | 625 | 39.04% | 98.40% | 0 (0.00%) | 625 |
| GRPO-4300 | UVB | 5355 | 29.51% | 96.02% | 16 (0.30%) | 5339 |
| GRPO-4300 | VideoMMMU | 841 | 31.63% | 91.68% | 15 (1.78%) | 826 |
| GRPO-4300 | MMVU | 625 | 42.08% | 97.28% | 0 (0.00%) | 625 |
| GRPO-5000 | UVB | 5355 | 30.21% | 95.87% | 9 (0.17%) | 5346 |
| GRPO-5000 | VideoMMMU | 841 | 31.39% | 91.32% | 19 (2.26%) | 822 |
| GRPO-5000 | MMVU | 625 | 40.96% | 98.24% | 1 (0.16%) | 624 |

*Primary metric: extractable accuracy over all n. Unscorable samples count as incorrect.*

### Macro aggregate (UVB + VideoMMMU + MMVU)

| Checkpoint | n | Extractable Acc | Strict Acc | Format Rate | Unscorable |
|---|---:|---:|---:|---:|---:|
| Base | 6821 | 18.55% | 13.25% | 48.26% | 2101 (30.80%) |
| SFT | 6821 | 14.48% | 9.43% | 33.22% | 3500 (51.31%) |
| GRPO-2000 | 6821 | 30.29% | 28.06% | 91.83% | 31 (0.45%) |
| GRPO-4300 | 6821 | 30.92% | 29.50% | 95.60% | 31 (0.45%) |
| GRPO-5000 | 6821 | 31.34% | 29.98% | 95.53% | 29 (0.43%) |

## Strategy Distribution and Accuracy

| Checkpoint | ABSTRACT | SPATIOTEMPORAL | TEMPORAL | no_valid_tag | UNSCORABLE |
|---|---|---|---|---|---|
| Base | 5.3% / 27.5% | 31.5% / 27.5% | 11.4% / 27.3% | 20.9% / 25.3% | 30.8% / 0.0% |
| SFT | 16.3% / 26.9% | 7.1% / 26.7% | 9.9% / 32.0% | 15.5% / 32.7% | 51.3% / 0.0% |
| GRPO-2000 | 12.6% / 27.9% | 8.4% / 24.5% | 70.9% / 31.7% | 7.7% / 28.9% | 0.5% / 0.0% |
| GRPO-4300 | 27.3% / 28.4% | 23.6% / 27.4% | 44.7% / 34.2% | 3.9% / 36.1% | 0.5% / 0.0% |
| GRPO-5000 | 26.8% / 28.8% | 23.7% / 28.5% | 45.0% / 34.5% | 4.0% / 33.7% | 0.4% / 0.0% |

*Cell format: `% of all samples / extractable accuracy within bucket`.*

### Base — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| ABSTRACT | 360 | 5.28% | 27.50% |
| SPATIOTEMPORAL | 2151 | 31.53% | 27.52% |
| TEMPORAL | 781 | 11.45% | 27.27% |
| no_valid_tag | 1428 | 20.94% | 25.28% |
| UNSCORABLE | 2101 | 30.80% | 0.00% |

### SFT — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| ABSTRACT | 1110 | 16.27% | 26.94% |
| SPATIOTEMPORAL | 484 | 7.10% | 26.65% |
| TEMPORAL | 672 | 9.85% | 31.99% |
| no_valid_tag | 1055 | 15.47% | 32.70% |
| UNSCORABLE | 3500 | 51.31% | 0.00% |

### GRPO-2000 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| ABSTRACT | 859 | 12.59% | 27.94% |
| SPATIOTEMPORAL | 571 | 8.37% | 24.52% |
| TEMPORAL | 4834 | 70.87% | 31.73% |
| no_valid_tag | 526 | 7.71% | 28.90% |
| UNSCORABLE | 31 | 0.45% | 0.00% |

### GRPO-4300 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| ABSTRACT | 1863 | 27.31% | 28.40% |
| SPATIOTEMPORAL | 1610 | 23.60% | 27.39% |
| TEMPORAL | 3048 | 44.69% | 34.19% |
| no_valid_tag | 269 | 3.94% | 36.06% |
| UNSCORABLE | 31 | 0.45% | 0.00% |

### GRPO-5000 — strategy counts

| Strategy | Count | % all | Extractable Acc |
|---|---:|---:|---:|
| ABSTRACT | 1829 | 26.81% | 28.76% |
| SPATIOTEMPORAL | 1619 | 23.74% | 28.47% |
| TEMPORAL | 3068 | 44.98% | 34.49% |
| no_valid_tag | 276 | 4.05% | 33.70% |
| UNSCORABLE | 29 | 0.43% | 0.00% |

## Problem Type × Strategy Analysis (UVB)

### Base

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 15.47% | SPATIOTEMPORAL | ABSTRACT |
| Landmark Position | 876 | 16.10% | SPATIOTEMPORAL | ABSTRACT |
| Progress Evaluation | 781 | 16.90% | SPATIOTEMPORAL | no_valid_tag |
| Trajectory Captioning | 421 | 18.05% | SPATIOTEMPORAL | TEMPORAL |
| Goal Detection | 320 | 15.62% | SPATIOTEMPORAL | TEMPORAL |
| Cognitive Map | 270 | 26.67% | SPATIOTEMPORAL | no_valid_tag |
| High-level Planning | 261 | 20.69% | SPATIOTEMPORAL | SPATIOTEMPORAL |
| Association Reasoning | 237 | 9.70% | SPATIOTEMPORAL | SPATIOTEMPORAL |

- **Action Generation** dist `{'ABSTRACT': 20, 'SPATIOTEMPORAL': 455, 'TEMPORAL': 203, 'no_valid_tag': 179}` acc `{'ABSTRACT': 25.0, 'SPATIOTEMPORAL': 20.0, 'TEMPORAL': 24.137931034482758, 'no_valid_tag': 21.22905027932961}`
- **Landmark Position** dist `{'ABSTRACT': 39, 'SPATIOTEMPORAL': 310, 'TEMPORAL': 107, 'no_valid_tag': 130}` acc `{'ABSTRACT': 35.8974358974359, 'SPATIOTEMPORAL': 22.903225806451612, 'TEMPORAL': 23.364485981308412, 'no_valid_tag': 23.846153846153847}`
- **Progress Evaluation** dist `{'ABSTRACT': 26, 'SPATIOTEMPORAL': 283, 'TEMPORAL': 112, 'no_valid_tag': 106}` acc `{'ABSTRACT': 23.076923076923077, 'SPATIOTEMPORAL': 23.674911660777386, 'TEMPORAL': 24.107142857142858, 'no_valid_tag': 30.18867924528302}`
- **Trajectory Captioning** dist `{'ABSTRACT': 13, 'SPATIOTEMPORAL': 166, 'TEMPORAL': 31, 'no_valid_tag': 69}` acc `{'ABSTRACT': 23.076923076923077, 'SPATIOTEMPORAL': 26.50602409638554, 'TEMPORAL': 29.032258064516128, 'no_valid_tag': 28.985507246376812}`

### SFT

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 4.14% | no_valid_tag | TEMPORAL |
| Landmark Position | 876 | 15.75% | ABSTRACT | no_valid_tag |
| Progress Evaluation | 781 | 4.87% | no_valid_tag | ABSTRACT |
| Trajectory Captioning | 421 | 10.45% | SPATIOTEMPORAL | no_valid_tag |
| Goal Detection | 320 | 11.56% | no_valid_tag | TEMPORAL |
| Cognitive Map | 270 | 19.63% | ABSTRACT | no_valid_tag |
| High-level Planning | 261 | 12.26% | no_valid_tag | ABSTRACT |
| Association Reasoning | 237 | 5.91% | no_valid_tag | SPATIOTEMPORAL |

- **Action Generation** dist `{'ABSTRACT': 14, 'SPATIOTEMPORAL': 56, 'TEMPORAL': 25, 'no_valid_tag': 162}` acc `{'ABSTRACT': 14.285714285714286, 'SPATIOTEMPORAL': 19.642857142857142, 'TEMPORAL': 24.0, 'no_valid_tag': 18.51851851851852}`
- **Landmark Position** dist `{'ABSTRACT': 248, 'SPATIOTEMPORAL': 172, 'TEMPORAL': 75, 'no_valid_tag': 113}` acc `{'ABSTRACT': 19.758064516129032, 'SPATIOTEMPORAL': 20.930232558139537, 'TEMPORAL': 26.666666666666668, 'no_valid_tag': 29.20353982300885}`
- **Progress Evaluation** dist `{'ABSTRACT': 29, 'SPATIOTEMPORAL': 41, 'TEMPORAL': 38, 'no_valid_tag': 87}` acc `{'ABSTRACT': 27.586206896551722, 'SPATIOTEMPORAL': 14.634146341463415, 'TEMPORAL': 21.05263157894737, 'no_valid_tag': 18.39080459770115}`
- **Trajectory Captioning** dist `{'ABSTRACT': 33, 'SPATIOTEMPORAL': 55, 'TEMPORAL': 20, 'no_valid_tag': 51}` acc `{'ABSTRACT': 15.151515151515152, 'SPATIOTEMPORAL': 29.09090909090909, 'TEMPORAL': 25.0, 'no_valid_tag': 35.294117647058826}`

### GRPO-2000

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 20.71% | TEMPORAL | no_valid_tag |
| Landmark Position | 876 | 19.75% | TEMPORAL | TEMPORAL |
| Progress Evaluation | 781 | 22.92% | TEMPORAL | TEMPORAL |
| Trajectory Captioning | 421 | 33.49% | TEMPORAL | no_valid_tag |
| Goal Detection | 320 | 22.19% | TEMPORAL | TEMPORAL |
| Cognitive Map | 270 | 37.41% | ABSTRACT | SPATIOTEMPORAL |
| High-level Planning | 261 | 46.74% | TEMPORAL | TEMPORAL |
| Association Reasoning | 237 | 13.08% | TEMPORAL | TEMPORAL |

- **Action Generation** dist `{'ABSTRACT': 30, 'SPATIOTEMPORAL': 226, 'TEMPORAL': 774, 'no_valid_tag': 148}` acc `{'ABSTRACT': 30.0, 'SPATIOTEMPORAL': 20.79646017699115, 'TEMPORAL': 18.6046511627907, 'no_valid_tag': 30.405405405405407}`
- **Landmark Position** dist `{'ABSTRACT': 137, 'SPATIOTEMPORAL': 95, 'TEMPORAL': 567, 'no_valid_tag': 77}` acc `{'ABSTRACT': 15.328467153284672, 'SPATIOTEMPORAL': 18.94736842105263, 'TEMPORAL': 21.164021164021165, 'no_valid_tag': 18.181818181818183}`
- **Progress Evaluation** dist `{'ABSTRACT': 35, 'SPATIOTEMPORAL': 51, 'TEMPORAL': 591, 'no_valid_tag': 102}` acc `{'ABSTRACT': 20.0, 'SPATIOTEMPORAL': 11.764705882352942, 'TEMPORAL': 24.873096446700508, 'no_valid_tag': 18.627450980392158}`
- **Trajectory Captioning** dist `{'ABSTRACT': 30, 'SPATIOTEMPORAL': 59, 'TEMPORAL': 292, 'no_valid_tag': 36}` acc `{'ABSTRACT': 46.666666666666664, 'SPATIOTEMPORAL': 35.59322033898305, 'TEMPORAL': 30.136986301369863, 'no_valid_tag': 50.0}`

### GRPO-4300

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 19.44% | SPATIOTEMPORAL | TEMPORAL |
| Landmark Position | 876 | 21.23% | ABSTRACT | no_valid_tag |
| Progress Evaluation | 781 | 22.92% | TEMPORAL | ABSTRACT |
| Trajectory Captioning | 421 | 36.82% | SPATIOTEMPORAL | no_valid_tag |
| Goal Detection | 320 | 19.69% | ABSTRACT | no_valid_tag |
| Cognitive Map | 270 | 38.15% | ABSTRACT | no_valid_tag |
| High-level Planning | 261 | 47.89% | SPATIOTEMPORAL | SPATIOTEMPORAL |
| Association Reasoning | 237 | 14.35% | ABSTRACT | TEMPORAL |

- **Action Generation** dist `{'ABSTRACT': 106, 'SPATIOTEMPORAL': 539, 'TEMPORAL': 492, 'no_valid_tag': 43}` acc `{'ABSTRACT': 17.92452830188679, 'SPATIOTEMPORAL': 18.552875695732837, 'TEMPORAL': 20.934959349593495, 'no_valid_tag': 18.6046511627907}`
- **Landmark Position** dist `{'ABSTRACT': 373, 'SPATIOTEMPORAL': 248, 'TEMPORAL': 221, 'no_valid_tag': 32}` acc `{'ABSTRACT': 19.839142091152816, 'SPATIOTEMPORAL': 20.967741935483872, 'TEMPORAL': 22.624434389140273, 'no_valid_tag': 31.25}`
- **Progress Evaluation** dist `{'ABSTRACT': 143, 'SPATIOTEMPORAL': 265, 'TEMPORAL': 339, 'no_valid_tag': 32}` acc `{'ABSTRACT': 27.272727272727273, 'SPATIOTEMPORAL': 24.150943396226417, 'TEMPORAL': 20.94395280235988, 'no_valid_tag': 15.625}`
- **Trajectory Captioning** dist `{'ABSTRACT': 87, 'SPATIOTEMPORAL': 163, 'TEMPORAL': 147, 'no_valid_tag': 22}` acc `{'ABSTRACT': 44.827586206896555, 'SPATIOTEMPORAL': 32.515337423312886, 'TEMPORAL': 35.374149659863946, 'no_valid_tag': 50.0}`

### GRPO-5000

| Category | n | Extractable Acc | Top Strategy | Best-Acc Strategy |
|---|---:|---:|---|---|
| Action Generation | 1183 | 20.46% | SPATIOTEMPORAL | no_valid_tag |
| Landmark Position | 876 | 22.83% | ABSTRACT | no_valid_tag |
| Progress Evaluation | 781 | 25.10% | TEMPORAL | no_valid_tag |
| Trajectory Captioning | 421 | 37.53% | SPATIOTEMPORAL | no_valid_tag |
| Goal Detection | 320 | 17.19% | ABSTRACT | ABSTRACT |
| Cognitive Map | 270 | 35.93% | ABSTRACT | ABSTRACT |
| High-level Planning | 261 | 51.34% | SPATIOTEMPORAL | SPATIOTEMPORAL |
| Association Reasoning | 237 | 12.24% | ABSTRACT | no_valid_tag |

- **Action Generation** dist `{'ABSTRACT': 97, 'SPATIOTEMPORAL': 533, 'TEMPORAL': 497, 'no_valid_tag': 56}` acc `{'ABSTRACT': 13.402061855670103, 'SPATIOTEMPORAL': 22.701688555347094, 'TEMPORAL': 18.91348088531187, 'no_valid_tag': 25.0}`
- **Landmark Position** dist `{'ABSTRACT': 358, 'SPATIOTEMPORAL': 260, 'TEMPORAL': 222, 'no_valid_tag': 33}` acc `{'ABSTRACT': 22.905027932960895, 'SPATIOTEMPORAL': 22.692307692307693, 'TEMPORAL': 21.62162162162162, 'no_valid_tag': 33.333333333333336}`
- **Progress Evaluation** dist `{'ABSTRACT': 120, 'SPATIOTEMPORAL': 257, 'TEMPORAL': 371, 'no_valid_tag': 33}` acc `{'ABSTRACT': 26.666666666666668, 'SPATIOTEMPORAL': 24.513618677042803, 'TEMPORAL': 24.797843665768195, 'no_valid_tag': 27.272727272727273}`
- **Trajectory Captioning** dist `{'ABSTRACT': 82, 'SPATIOTEMPORAL': 163, 'TEMPORAL': 157, 'no_valid_tag': 18}` acc `{'ABSTRACT': 37.80487804878049, 'SPATIOTEMPORAL': 36.809815950920246, 'TEMPORAL': 36.30573248407644, 'no_valid_tag': 55.55555555555556}`

## Checkpoint Trend

| Checkpoint | UVB | VideoMMMU | MMVU | Macro | Format | Unscorable |
|---|---:|---:|---:|---:|---:|---:|
| Base | 19.01% | 16.88% | 16.80% | 18.55% | 48.26% | 30.80% |
| SFT | 12.16% | 25.45% | 19.68% | 14.48% | 33.22% | 51.31% |
| GRPO-2000 | 28.93% | 32.46% | 39.04% | 30.29% | 91.83% | 0.45% |
| GRPO-4300 | 29.51% | 31.63% | 42.08% | 30.92% | 95.60% | 0.45% |
| GRPO-5000 | 30.21% | 31.39% | 40.96% | 31.34% | 95.53% | 0.43% |

# Cross-Experiment Discussion

| Question | LENGTH | PERSPECTIVE |
|---|---|---|
| Best macro extractable | GRPO-5000/4000 tie (~30.0%); **4000 better strict & unscorable** | GRPO-5000 (31.34%), then 4300 (30.92%) |
| SFT vs GRPO | SFT strong (29.13%); GRPO-4000 adds +0.85pp extractable, +4.0pp strict | **SFT broken on UVB answer extraction**; GRPO essential |
| Strategy collapse? | Yes — COT 68% → <1%; DIRECT/LONG_COT dominate | Yes — TEMPORAL 71% at GRPO-2000; later checkpoints more balanced |
| Format vs accuracy | SFT fixes format; GRPO adds strategy diversity + multiple-answer failures | GRPO fixes `<ANSWER>` omission; format 33% (SFT) → 96% (4300) |
| Adaptive reasoning | Weak — top ≠ best strategy per UVB category | Weak — top ≠ best strategy; GRPO-2000 over-uses TEMPORAL everywhere |
| Checkpoint trend | 4000 ≥ 5000 on strict/format; 5000 +0.04pp macro only | 2000→4300→5000 small gains; 4300 best MMVU |
| More stable task | LENGTH — SFT usable baseline | PERSPECTIVE — only after GRPO |

### VideoMMMU discipline trend (LENGTH, extractable accuracy)

| Discipline | SFT | GRPO-4000 | GRPO-5000 | SFT top → GRPO-4000 top |
|---|---:|---:|---:|---|
| Computer | 33.9% | **46.4%** | 26.8% | COT → DIRECT |
| Electronics | 33.3% | **40.7%** | **51.9%** | COT → DIRECT |
| Mechanical | 21.7% | 20.0% | **28.3%** | COT → DIRECT |
| Architecture | 34.1% | 29.3% | 34.1% | COT → LONG_COT |

GRPO shifts VideoMMMU from uniform COT to DIRECT/LONG_COT. Gains are discipline-specific; no single strategy wins all disciplines.

### Does accuracy gain come from reasoning or format recovery?

| Task | Evidence |
|---|---|
| LENGTH | Extractable accuracy within `no_valid_tag` bucket is **~30–32%** (similar to valid-tag buckets), so post-SFT gains are not purely format recovery. GRPO strict gains track format rate more closely. |
| PERSPECTIVE | SFT→GRPO macro jump (+16.9pp) is **primarily answer-extraction recovery** (unscorable 51% → <0.5%). Strategy accuracy within valid tags stays ~28–35%. |

# Takeaways

## Paper-ready claims
1. SFT teaches valid reasoning tags; Base answers are often extractable but untagged.
2. GRPO materially changes strategy mix, not just correctness.
3. Post-SFT accuracy gains are modest; most visible gain is format compliance.
4. Optimal GRPO checkpoint is not the final step.

## Risks
1. 8f vs 16f eval confound for GRPO-2000 vs later checkpoints.
2. Multiple `<ANSWER>` tags depress strict/format metrics while extractable accuracy looks stable.
3. Small per-category counts for best-strategy analysis.

## Sanity checks
- Re-run GRPO-2000 at 16 frames.
- Bootstrap CIs per category × strategy.
- Audit `no_valid_tag` vs `UNSCORABLE` cases.
- Compare eval strategy shares to `strategy_debug.jsonl` training rollouts.



## Inputs Read (raw prediction jsonl)

| Group | Location |
|---|---|
| Base/SFT/GRPO-4000/5000 (0625) | `/scratch/Results_0625/{BASE_LENGTH,SFT_LENGTH,LENGTH_4000,LENGTH_5000,BASE_PERSPECTIVE,SFT_PERSPECTIVE,PERSPECTIVE_4300,PERSPECTIVE_5000}/` |
| GRPO-2000 (0620) | `/scratch/models/models_0620/qwen25vl3b_grpo_{length|perspective}_8f_step2000_merged/` |

Prior reports consulted (not trusted for numbers): `reports/comprehensive_reanalysis_20260621/COMPREHENSIVE_BENCH_REANALYSIS.md`.

# Artifacts

| File | Path |
|---|---|
| Report | `reports/length_perspective_checkpoint_analysis.md` |
| JSON | `reports/length_perspective_checkpoint_analysis.json` |
| Script | `src/scripts/analyze_checkpoint_reasoning.py` |