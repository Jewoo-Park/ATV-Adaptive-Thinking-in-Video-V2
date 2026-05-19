# GRPO Strategy Learning Fix Report

Date: 2026-05-17

## Summary

This change updates the LENGTH and PERSPECTIVE GRPO paths so reward is only available when the generated output matches the forced rollout strategy. It also removes default-strategy tie-break reward, adds strict tagged direct-answer formatting for LENGTH, aligns training/evaluation parsing, and logs the strategy metrics needed to diagnose adaptive strategy learning.

Full 1-epoch training was not started in this turn. The requested 150-step sanity runs completed and passed the target checks; full training should start only after reviewing this report.

## Modified Files

- `.gitignore`: ignores generated benchmark `results/`.
- `src/r1-v/src/open_r1/strict_answer.py`: training strict parser now requires strategy tags for LENGTH and PERSPECTIVE; LENGTH answer-only output is invalid.
- `src/eval/strict_answer.py`: evaluation strict parser mirrors training parser.
- `src/r1-v/src/open_r1/strategy.py`: adds `<DIRECT>` mapping and no-bonus tie behavior with `effective_best_strategy=None`.
- `src/r1-v/src/open_r1/grpo.py`: updates prompts, reward defaults, option-range checks, and strategy-gated accuracy/format rewards.
- `src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py`: passes forced strategy into rewards, masks mismatched rewards, applies strategy bonus only to compliant slots, adds metrics, uses configured temperature, and fixes debug JSONL alignment.
- `src/eval/uvb_eval_only.py`: updates LENGTH/PERSPECTIVE eval prompts to the strict tagged formats.
- `src/eval/mmvu_eval_only.py`: updates eval prompts to the strict tagged formats.
- `src/eval/videommmu_eval_only.py`: updates eval prompts to the strict tagged formats.
- `src/scripts/run_grpo_answer_only_lora.sh`: updates default hyperparameters for sanity/full runs.
- `src/scripts/run_grpo_length_real.sh`: documents no-bonus tie policy.
- `src/scripts/run_grpo_perspective_real.sh`: documents no-bonus tie policy.
- `src/scripts/run_grpo_length_real_full.sh`: sets final/sanity defaults for the new reward schedule.
- `src/scripts/run_grpo_perspective_real_full.sh`: sets final/sanity defaults for the new reward schedule.
- `src/scripts/verify_balanced_strategy_rollout.py`: updates GPU-free verifier for strict direct tag, no-bonus ties, and new defaults.
- `src/scripts/validate_strategy_parser_reward.py`: adds parser/reward validation for LENGTH and PERSPECTIVE.
- `docs/grpo_strategy_learning_fix_report.md`: this report.

## Diff Summary

Parser changes:
- LENGTH valid strategies are now `direct`, `cot`, `long_cot`, or `invalid`.
- PERSPECTIVE valid strategies are now `abstract`, `temporal`, `spatiotemporal`, or `invalid`.
- LENGTH direct output must be:

```text
<DIRECT>None</DIRECT>
<ANSWER>X</ANSWER>
```

- Answer-only `<ANSWER>X</ANSWER>` is invalid in both LENGTH and PERSPECTIVE.
- Multiple strategy tags, malformed answer tags, invalid option letters, and empty reasoning tags are invalid.

Reward changes:
- Reward order is parse -> format -> strategy match -> accuracy -> base reward -> strategy bonus -> final reward.
- `parsed_strategy != forced_strategy` zeros base reward and prevents strategy bonus.
- Accuracy reward requires valid answer tag, valid option letter in range, required strategy tag, strategy match, and correctness.
- Format reward requires both valid answer tag and the forced strategy tag.

Tie changes:
- `STRATEGY_BONUS_THRESHOLD=0.10`.
- `STRATEGY_BONUS_SCALE=0.20`.
- Tie/unclear groups get no strategy bonus.
- `effective_best_strategy=None` for tie/unclear groups.
- Legacy `tie_break_to_direct_rate` and `tie_break_to_abstract_rate` are retained but remain `0.0`.

Prompt changes:
- LENGTH prompt describes Direct, CoT, and Long CoT usage and exact tags.
- PERSPECTIVE prompt describes Abstract, Temporal, and Spatiotemporal usage and exact tags.
- Evaluation prompts were updated to match training.

Logging changes:
- Added tie/no-bonus metrics, per-strategy format/compliance/accuracy metrics, parsed strategy distribution, invalid strategy rate, malformed answer tag rate, invalid letter rate, and effective-best distribution.
- Debug JSONL now logs only the aligned local completion/reward slice in distributed runs.

Script/config changes:
- Default format/accuracy weights are `ANSWER_FORMAT_WEIGHT=0.35` and `ANSWER_ACCURACY_WEIGHT=0.65`.
- Default sampling temperature is `0.8`.
- Default tie-break scale is `0.0`.

## Validation Commands

Python syntax:

```bash
source .venv_grpo_smoke/bin/activate && PYTHONPATH=src/r1-v/src python -m py_compile \
  src/r1-v/src/open_r1/strict_answer.py \
  src/eval/strict_answer.py \
  src/r1-v/src/open_r1/strategy.py \
  src/r1-v/src/open_r1/grpo.py \
  src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py \
  src/scripts/validate_strategy_parser_reward.py \
  src/scripts/verify_balanced_strategy_rollout.py
```

Parser and reward validation:

```bash
source .venv_grpo_smoke/bin/activate && PYTHONPATH=src/r1-v/src python src/scripts/validate_strategy_parser_reward.py
```

Balanced rollout validation:

```bash
source .venv_grpo_smoke/bin/activate && PYTHONPATH=src/r1-v/src python src/scripts/verify_balanced_strategy_rollout.py
```

All three commands passed.

## Parser and Reward Test Results

Representative parser results:

| Mode | Output | Result |
| --- | --- | --- |
| LENGTH | `<DIRECT>None</DIRECT><ANSWER>A</ANSWER>` | valid, `direct` |
| LENGTH | `<COT>brief</COT><ANSWER>B</ANSWER>` | valid, `cot` |
| LENGTH | `<LONG_COT>detailed</LONG_COT><ANSWER>C</ANSWER>` | valid, `long_cot` |
| LENGTH | `<ANSWER>A</ANSWER>` | invalid |
| PERSPECTIVE | `<ABSTRACT>concept</ABSTRACT><ANSWER>A</ANSWER>` | valid, `abstract` |
| PERSPECTIVE | `<TEMPORAL>sequence</TEMPORAL><ANSWER>B</ANSWER>` | valid, `temporal` |
| PERSPECTIVE | `<SPATIOTEMPORAL>motion</SPATIOTEMPORAL><ANSWER>C</ANSWER>` | valid, `spatiotemporal` |
| PERSPECTIVE | `<ANSWER>A</ANSWER>` | invalid |

Reward checks passed:
- Wrong forced LENGTH strategy gives `accuracy_reward=0.0` and `format_reward=0.0`.
- PERSPECTIVE answer-only output gives `accuracy_reward=0.0` and `format_reward=0.0`.
- Tie/unclear groups leave final reward equal to base reward and set `effective_best_strategy=None`.
- Strategy-mismatched slots receive no strategy bonus.
- Existing 150-step debug JSONL local aligned rows have zero mismatch base-reward leaks and zero mismatch strategy-bonus leaks.

## LENGTH 150-Step Sanity Run

Command:

```bash
source .venv_grpo_smoke/bin/activate && \
CUDA_VISIBLE_DEVICES=1,2,3 NUM_GPUS=3 TRAIN_NUM_GPUS=2 MASTER_PORT=12451 \
OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_strategy_fix_sanity150 \
RUN_NAME=grpo_length_strategy_fix_sanity150 \
STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_strategy_fix_sanity150/strategy_debug.jsonl \
MAX_STEPS=150 SAVE_STEPS=150 LOGGING_STEPS=1 \
NUM_GENERATIONS=9 ROLLOUTS_PER_STRATEGY=3 BALANCED_STRATEGY_ROLLOUT=true \
ANSWER_ACCURACY_WEIGHT=0.65 ANSWER_FORMAT_WEIGHT=0.35 \
STRATEGY_BONUS_SCALE=0.20 STRATEGY_BONUS_THRESHOLD=0.10 TIE_BREAK_BONUS_SCALE=0.0 \
TEMPERATURE=0.8 \
bash src/scripts/run_grpo_length_real_full.sh
```

Result: completed 150/150 steps.

| Metric | All 150 mean | Last 50 mean | Target |
| --- | ---: | ---: | --- |
| cot strategy compliance | 0.6789 | 0.9467 | >= 0.25 |
| long_cot strategy compliance | 0.5122 | 0.6767 | >= 0.20 |
| tie_case_rate | 0.3333 | 0.4900 | < 0.70 |
| no_bonus_tie_rate | 0.3333 | 0.4900 | matches tie policy |
| strategy_bonus_applied_rate | 0.6667 | 0.5100 | > 0.15 |
| effective_best_direct_rate | 0.4033 | 0.2600 | < 0.85 |
| effective_best_strategy_none_rate | 0.3333 | 0.4900 | nonzero for ties |
| malformed_answer_tag_rate | 0.0274 | 0.0189 | no significant increase |
| invalid_letter_rate | 0.0174 | 0.0000 | no significant increase |
| KL | 0.0734 | 0.0553 | stable |
| loss | 0.0029 | 0.0022 | stable |
| grad_norm | 0.3997 | 0.3300 | stable |

LENGTH judgment: PASS.

## PERSPECTIVE 150-Step Sanity Run

Command:

```bash
source .venv_grpo_smoke/bin/activate && \
CUDA_VISIBLE_DEVICES=1,2,3 NUM_GPUS=3 TRAIN_NUM_GPUS=2 MASTER_PORT=12452 \
OUTPUT_DIR=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_strategy_fix_sanity150 \
RUN_NAME=grpo_perspective_strategy_fix_sanity150 \
STRATEGY_DEBUG_LOG_PATH=/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_strategy_fix_sanity150/strategy_debug.jsonl \
MAX_STEPS=150 SAVE_STEPS=150 LOGGING_STEPS=1 \
NUM_GENERATIONS=9 ROLLOUTS_PER_STRATEGY=3 BALANCED_STRATEGY_ROLLOUT=true \
ANSWER_ACCURACY_WEIGHT=0.65 ANSWER_FORMAT_WEIGHT=0.35 \
STRATEGY_BONUS_SCALE=0.20 STRATEGY_BONUS_THRESHOLD=0.10 TIE_BREAK_BONUS_SCALE=0.0 \
TEMPERATURE=0.8 \
bash src/scripts/run_grpo_perspective_real_full.sh
```

Result: completed 150/150 steps.

| Metric | All 150 mean | Last 50 mean | Target |
| --- | ---: | ---: | --- |
| forced_strategy == parsed_strategy | 0.9226 | 0.9689 | >= 0.50 |
| format_ok | 0.9226 | 0.9689 | >= 0.50 |
| parsed_strategy_none_rate | 0.0556 | 0.0233 | < 0.40 |
| tie_case_rate | 0.5800 | 0.6500 | < 0.70 |
| no_bonus_tie_rate | 0.5800 | 0.6500 | matches tie policy |
| strategy_bonus_applied_rate | 0.4200 | 0.3500 | > 0.15 |
| effective_best_abstract_rate | 0.1733 | 0.1000 | < 0.85 |
| effective_best_temporal_rate | 0.1433 | 0.1600 | non-trivial |
| effective_best_spatiotemporal_rate | 0.1033 | 0.0900 | non-trivial |
| effective_best_strategy_none_rate | 0.5800 | 0.6500 | nonzero for ties |
| malformed_answer_tag_rate | 0.0096 | 0.0044 | no significant increase |
| invalid_letter_rate | 0.0011 | 0.0011 | no significant increase |
| KL | 0.0330 | 0.0444 | stable |
| loss | 0.0013 | 0.0018 | stable |
| grad_norm | 0.3674 | 0.3281 | stable |

PERSPECTIVE judgment: PASS.

## Output Artifacts

- LENGTH sanity output: `/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_strategy_fix_sanity150`
- LENGTH debug log: `/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_length_strategy_fix_sanity150/strategy_debug.jsonl`
- PERSPECTIVE sanity output: `/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_strategy_fix_sanity150`
- PERSPECTIVE debug log: `/workspace/ATV-Adaptive-Thinking-in-Video-V2/outputs/grpo_perspective_strategy_fix_sanity150/strategy_debug.jsonl`

Both debug files contain 2700 rows from the pre-alignment logging behavior. Interpreting the rank-0 aligned local slice gives:

| Mode | Local rows | forced == parsed | format_ok | mismatch base leaks | mismatch bonus leaks |
| --- | ---: | ---: | ---: | ---: | ---: |
| LENGTH | 1350 | 0.6674 | 0.6963 | 0 | 0 |
| PERSPECTIVE | 1350 | 0.9200 | 0.9385 | 0 | 0 |

Future runs will log only the aligned local slice.

## Pass/Fail Judgment

Both 150-step sanity runs PASS the requested criteria.

Full 1-epoch training was SKIPPED in this turn so the sanity report can be reviewed first. The code path is ready for full training with the same launchers after review.

## Current Git Status Notes

The working tree already contained unrelated/pre-existing modified and untracked files before this fix. No commit was made. Generated outputs remain under ignored artifact directories.
