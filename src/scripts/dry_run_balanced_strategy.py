"""Dry-run sanity for balanced_strategy_rollout.

Exercises everything that does NOT require vLLM/GPU:
  - strategy plan ordering for LENGTH and PERSPECTIVE (test 1-2)
  - strategy directive injection into a chat-template-style prompt (test 3)
  - _apply_strategy_bonus math: shape, ordering, zero-sum, margin gate (test 4-7)
  - balanced=False legacy path: shape stays B*G, no bonus applied (test 8)

What still must be checked on a GPU node with real vLLM (cannot be run here):
  - num_generations=9 with balanced=True actually produces 9 completions in
    prompt-major order [direct,direct,direct, cot,cot,cot, long_cot,long_cot,long_cot]
    (or [abstract]*3 + [temporal]*3 + [spatiotemporal]*3 in PERSPECTIVE).
  - The trainer's strategy_debug_log_path JSONL has one row per slot with
    forced_strategy aligned to slot_idx % G's plan entry.
  - balanced=False reproduces existing free-rollout outputs bit-for-bit.

Run:
  cd src/r1-v && PYTHONPATH=src python ../scripts/dry_run_balanced_strategy.py
"""
from __future__ import annotations

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_R1V = os.path.join(REPO_ROOT, "src", "r1-v", "src")
if SRC_R1V not in sys.path:
    sys.path.insert(0, SRC_R1V)

import torch

from open_r1.strategy import (
    LENGTH_STRATEGIES,
    PERSPECTIVE_STRATEGIES,
    build_balanced_strategy_plan,
    compute_strategy_bonus,
    parsed_tag_to_strategy,
    strategies_for_task,
    strategy_directive,
    strategy_distribution_rates,
)


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise SystemExit(1)


def test_strategy_plan_length():
    header("Test 1: LENGTH plan with G=9, k=3")
    plan = build_balanced_strategy_plan("length", num_generations=9, rollouts_per_strategy=3)
    print(f"  plan = {plan}")
    expected = ["direct"] * 3 + ["cot"] * 3 + ["long_cot"] * 3
    check(plan == expected, f"plan == {expected}")


def test_strategy_plan_perspective():
    header("Test 2: PERSPECTIVE plan with G=9, k=3")
    plan = build_balanced_strategy_plan("perspective", num_generations=9, rollouts_per_strategy=3)
    print(f"  plan = {plan}")
    expected = ["abstract"] * 3 + ["temporal"] * 3 + ["spatiotemporal"] * 3
    check(plan == expected, f"plan == {expected}")


def test_strategy_plan_invalid_G():
    header("Test 2b: plan rejects num_generations not divisible by 3")
    try:
        build_balanced_strategy_plan("length", num_generations=8, rollouts_per_strategy=3)
    except ValueError as e:
        check(True, f"raised ValueError as expected: {e}")
        return
    check(False, "should have raised ValueError")


def test_directive_text():
    header("Test 3: directive text contains the right keywords")
    d = strategy_directive("length", "direct")
    check("<ANSWER>" in d and "Direct Answer" in d, "LENGTH/direct mentions ANSWER and Direct Answer")
    d = strategy_directive("length", "cot")
    check("<COT>" in d and "<ANSWER>" in d, "LENGTH/cot mentions COT and ANSWER")
    d = strategy_directive("length", "long_cot")
    check("<LONG_COT>" in d and "<ANSWER>" in d, "LENGTH/long_cot mentions LONG_COT and ANSWER")
    d = strategy_directive("perspective", "abstract")
    check("<ABSTRACT>" in d, "PERSPECTIVE/abstract mentions ABSTRACT")
    d = strategy_directive("perspective", "temporal")
    check("<TEMPORAL>" in d, "PERSPECTIVE/temporal mentions TEMPORAL")
    d = strategy_directive("perspective", "spatiotemporal")
    check("<SPATIOTEMPORAL>" in d, "PERSPECTIVE/spatiotemporal mentions SPATIOTEMPORAL")


def test_parsed_tag_to_strategy():
    header("Test 3b: parsed_tag_to_strategy mapping")
    check(parsed_tag_to_strategy("length", None) == "direct", "LENGTH None tag -> direct")
    check(parsed_tag_to_strategy("length", "COT") == "cot", "LENGTH COT -> cot")
    check(parsed_tag_to_strategy("length", "LONG_COT") == "long_cot", "LENGTH LONG_COT -> long_cot")
    check(parsed_tag_to_strategy("perspective", "ABSTRACT") == "abstract", "PERSPECTIVE ABSTRACT -> abstract")
    check(parsed_tag_to_strategy("perspective", "TEMPORAL") == "temporal", "PERSPECTIVE TEMPORAL -> temporal")
    check(parsed_tag_to_strategy("perspective", "SPATIOTEMPORAL") == "spatiotemporal", "PERSPECTIVE SPATIOTEMPORAL -> spatiotemporal")


class _StubTrainer:
    """Lightweight test fixture matching the trainer's strategy-related config surface."""

    def __init__(self, task_type: str, alpha: float, threshold: float, G: int = 9, k: int = 3):
        self.reasoning_task_type = task_type
        self.balanced_strategy_rollout = True
        self.num_generations = G
        self.rollouts_per_strategy = k
        self.strategy_bonus_scale = alpha
        self.strategy_bonus_threshold = threshold
        self._strategies = strategies_for_task(task_type)
        plan = build_balanced_strategy_plan(task_type, G, k)
        self._strategy_plan_per_prompt = plan
        self._strategy_index_per_slot = [self._strategies.index(s) for s in plan]

    def _apply_strategy_bonus(self, base_rewards):
        return compute_strategy_bonus(
            base_rewards=base_rewards,
            strategy_index_per_slot=self._strategy_index_per_slot,
            num_strategies=len(self._strategies),
            bonus_scale=self.strategy_bonus_scale,
            bonus_threshold=self.strategy_bonus_threshold,
        )


def test_bonus_math_known_inputs():
    header("Test 4: _apply_strategy_bonus produces expected final reward")
    # LENGTH, G=9, k=3, alpha=0.1, threshold=0.34
    trainer = _StubTrainer("length", alpha=0.1, threshold=0.34)
    # One prompt group, base rewards by strategy:
    #   direct slots [1.0, 1.0, 1.0]      mean = 1.0
    #   cot slots    [0.0, 0.0, 0.0]      mean = 0.0
    #   long_cot     [0.5, 0.5, 0.5]      mean = 0.5
    base = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    final, log = trainer._apply_strategy_bonus(base)
    overall = (1.0 + 0.0 + 0.5) / 3.0   # 0.5
    margin = 1.0 - 0.5                   # 0.5 >= 0.34 -> bonus applied
    bonus_direct = 1.0 - overall         # 0.5
    bonus_cot = 0.0 - overall            # -0.5
    bonus_long = 0.5 - overall           # 0.0
    expected = torch.tensor([
        1.0 + 0.1 * bonus_direct, 1.0 + 0.1 * bonus_direct, 1.0 + 0.1 * bonus_direct,
        0.0 + 0.1 * bonus_cot,    0.0 + 0.1 * bonus_cot,    0.0 + 0.1 * bonus_cot,
        0.5 + 0.1 * bonus_long,   0.5 + 0.1 * bonus_long,   0.5 + 0.1 * bonus_long,
    ])
    print(f"  base    = {base.tolist()}")
    print(f"  final   = {[round(x, 4) for x in final.tolist()]}")
    print(f"  expected= {[round(x, 4) for x in expected.tolist()]}")
    check(torch.allclose(final, expected, atol=1e-6), "final == base + alpha * (strategy_mean - mean(strategy_means))")
    check(log["margin"].item() > 0.34, "margin > threshold (bonus applied)")
    check(log["apply_mask"].item() == 1.0, "apply_mask == 1.0")
    check(log["best_idx"].item() == 0, "best_idx == 0 (direct)")
    # Per-strategy bonus is zero-sum across strategies.
    s_bonus = log["bonus_per_strategy"][0]
    check(abs(float(s_bonus.sum().item())) < 1e-5, "bonus_per_strategy sums to ~0 across strategies")


def test_bonus_math_margin_gate():
    header("Test 5: margin < threshold suppresses bonus")
    trainer = _StubTrainer("length", alpha=0.1, threshold=0.34)
    # Strategy means: direct=0.5, cot=0.4, long_cot=0.4 -> margin = 0.1 < 0.34
    base = torch.tensor([0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    final, log = trainer._apply_strategy_bonus(base)
    print(f"  margin = {float(log['margin'].item()):.4f} (threshold=0.34)")
    print(f"  apply_mask = {float(log['apply_mask'].item())}")
    check(torch.allclose(final, base), "final == base (bonus suppressed)")
    check(log["apply_mask"].item() == 0.0, "apply_mask == 0.0 (gated off)")


def test_bonus_math_two_groups():
    header("Test 6: two prompt groups are shaped independently")
    trainer = _StubTrainer("length", alpha=0.1, threshold=0.34)
    # Group 0: direct=1.0, cot=0.0, long_cot=0.5  -> margin 0.5  -> apply
    # Group 1: direct=0.5, cot=0.5, long_cot=0.5  -> margin 0.0  -> skip
    base = torch.tensor([
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])
    final, log = trainer._apply_strategy_bonus(base)
    print(f"  apply_mask = {log['apply_mask'].tolist()}")
    check(log["apply_mask"][0].item() == 1.0, "group 0: bonus applied")
    check(log["apply_mask"][1].item() == 0.0, "group 1: bonus skipped")
    # Group 1 final == base (no shaping)
    check(torch.allclose(final[9:], base[9:]), "group 1 final == base")


def test_bonus_zero_mean_within_group():
    header("Test 7: bonus is zero-mean within each prompt group")
    trainer = _StubTrainer("perspective", alpha=0.5, threshold=0.0)  # always apply
    # Random rewards across two groups.
    torch.manual_seed(0)
    base = torch.rand(18)
    final, log = trainer._apply_strategy_bonus(base)
    diff = (final - base).view(2, 9)
    print(f"  per-group bonus mean = {diff.mean(dim=1).tolist()}")
    check(torch.allclose(diff.mean(dim=1), torch.zeros(2), atol=1e-6), "per-group bonus sum ~ 0")


def test_balanced_off_no_change():
    header("Test 8: balanced_strategy_rollout=False keeps base path untouched")
    # Read the trainer source as text (no transformers import on this dev machine).
    trainer_path = os.path.join(
        SRC_R1V, "open_r1", "trainer", "vllm_grpo_trainer_modified.py"
    )
    with open(trainer_path, "r", encoding="utf-8") as f:
        src = f.read()
    check("if self.balanced_strategy_rollout:" in src, "_prepare_inputs gates expansion on flag")
    check(
        "rewards, _strategy_log = self._apply_strategy_bonus(rewards)" in src,
        "reward shaping gated on flag (call site present)",
    )
    check(
        "rewards.view(-1, self.num_generations).mean(dim=1)" in src,
        "GRPO group-wise mean still computed on (possibly shaped) rewards",
    )
    check(
        "advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)" in src,
        "advantage formula unchanged; only the reward vector is reshaped",
    )
    # And the loss path itself must not reference strategy state.
    loss_section = src.split("def compute_loss(")[1]
    check(
        "balanced_strategy_rollout" not in loss_section
        and "strategy_bonus" not in loss_section,
        "compute_loss does not depend on strategy fields (KL/clipping/logprob untouched)",
    )


class _MetricsStub(_StubTrainer):
    """_StubTrainer + the metric-emitting wrappers, replicated from the trainer."""

    def __init__(self, task_type: str, alpha: float, threshold: float, G: int = 9, k: int = 3):
        super().__init__(task_type, alpha, threshold, G=G, k=k)
        from collections import defaultdict
        self._metrics = defaultdict(list)
        self.log_strategy_metrics = True

    def _log_strategy_distribution(self, key_prefix, chosen_idx, clear_mask):
        rates = strategy_distribution_rates(
            strategies=self._strategies,
            chosen_idx_per_group=chosen_idx,
            clear_mask_per_group=clear_mask,
            key_prefix=key_prefix,
        )
        for k, v in rates.items():
            self._metrics[f"strategy/{k}"].append(v)

    def _log_strategy_metrics(self, base_rewards, strategy_log):
        base = strategy_log["base"]
        final = strategy_log["final"]
        strategy_mean = strategy_log["strategy_mean"]
        margin = strategy_log["margin"]
        apply_mask = strategy_log["apply_mask"]
        best_idx = strategy_log["best_idx"]
        self._metrics["strategy/base_mean"].append(float(base.mean().item()))
        self._metrics["strategy/final_mean"].append(float(final.mean().item()))
        self._metrics["strategy/strategy_bonus_applied_rate"].append(
            float(apply_mask.mean().item())
        )
        self._metrics["strategy/strategy_margin_mean"].append(float(margin.mean().item()))
        per_strategy_mean = strategy_mean.mean(dim=0)
        for s_idx, s_name in enumerate(self._strategies):
            self._metrics[f"strategy/mean_reward_{s_name}"].append(
                float(per_strategy_mean[s_idx].item())
            )
        self._log_strategy_distribution(
            key_prefix="best_strategy",
            chosen_idx=best_idx,
            clear_mask=apply_mask,
        )


def test_best_strategy_metric_keys_length():
    header("Test 9a: LENGTH best_strategy metric keys are emitted with gating")
    trainer = _MetricsStub("length", alpha=0.1, threshold=0.34)
    # Three groups:
    #   group 0: direct=1.0, cot=0.0, long_cot=0.5  -> margin 0.5  best=direct
    #   group 1: direct=0.0, cot=0.9, long_cot=0.4  -> margin 0.5  best=cot
    #   group 2: direct=0.5, cot=0.5, long_cot=0.5  -> margin 0.0  tie
    base = torch.tensor([
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])
    _, log = trainer._apply_strategy_bonus(base)
    trainer._log_strategy_metrics(base, log)
    metrics = {k: v[-1] for k, v in trainer._metrics.items()}
    print(f"  metrics keys = {sorted(metrics.keys())}")
    expected_keys = {
        "strategy/base_mean",
        "strategy/final_mean",
        "strategy/strategy_bonus_applied_rate",
        "strategy/strategy_margin_mean",
        "strategy/mean_reward_direct",
        "strategy/mean_reward_cot",
        "strategy/mean_reward_long_cot",
        "strategy/best_strategy_direct_rate",
        "strategy/best_strategy_cot_rate",
        "strategy/best_strategy_long_cot_rate",
        "strategy/best_strategy_tie_or_unclear_rate",
    }
    missing = expected_keys - set(metrics.keys())
    check(not missing, f"all required LENGTH keys present (missing={missing})")
    # 1 of 3 groups -> direct clear best, 1 of 3 -> cot clear best, 1 of 3 -> tie.
    check(abs(metrics["strategy/best_strategy_direct_rate"] - 1 / 3) < 1e-6, "direct rate = 1/3")
    check(abs(metrics["strategy/best_strategy_cot_rate"] - 1 / 3) < 1e-6, "cot rate = 1/3")
    check(abs(metrics["strategy/best_strategy_long_cot_rate"] - 0.0) < 1e-6, "long_cot rate = 0")
    check(
        abs(metrics["strategy/best_strategy_tie_or_unclear_rate"] - 1 / 3) < 1e-6,
        "tie_or_unclear rate = 1/3",
    )
    check(
        abs(metrics["strategy/strategy_bonus_applied_rate"] - 2 / 3) < 1e-6,
        "bonus_applied_rate = 2/3 (two clear groups out of three)",
    )


def test_best_strategy_metric_keys_perspective():
    header("Test 9b: PERSPECTIVE best_strategy metric keys are emitted with gating")
    trainer = _MetricsStub("perspective", alpha=0.1, threshold=0.34)
    # group 0: abstract=0.2, temporal=0.9, spatiotemporal=0.4 -> best=temporal (margin 0.5)
    # group 1: 0.3, 0.3, 0.3 -> tie
    base = torch.tensor([
        0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    ])
    _, log = trainer._apply_strategy_bonus(base)
    trainer._log_strategy_metrics(base, log)
    metrics = {k: v[-1] for k, v in trainer._metrics.items()}
    expected_keys = {
        "strategy/best_strategy_abstract_rate",
        "strategy/best_strategy_temporal_rate",
        "strategy/best_strategy_spatiotemporal_rate",
        "strategy/best_strategy_tie_or_unclear_rate",
        "strategy/mean_reward_abstract",
        "strategy/mean_reward_temporal",
        "strategy/mean_reward_spatiotemporal",
    }
    missing = expected_keys - set(metrics.keys())
    check(not missing, f"all required PERSPECTIVE keys present (missing={missing})")
    check(abs(metrics["strategy/best_strategy_temporal_rate"] - 0.5) < 1e-6, "temporal rate = 1/2")
    check(abs(metrics["strategy/best_strategy_abstract_rate"] - 0.0) < 1e-6, "abstract rate = 0")
    check(abs(metrics["strategy/best_strategy_spatiotemporal_rate"] - 0.0) < 1e-6, "spatiotemporal rate = 0")
    check(abs(metrics["strategy/best_strategy_tie_or_unclear_rate"] - 0.5) < 1e-6, "tie_or_unclear = 1/2")


def test_selected_strategy_extensibility():
    header("Test 9c: _log_strategy_distribution reusable for future free-choice mode")
    trainer = _MetricsStub("length", alpha=0.1, threshold=0.34)
    # Imagine 4 groups; the model "selected" direct, cot, long_cot, direct.
    chosen_idx = torch.tensor([0, 1, 2, 0])
    clear_mask = torch.ones(4)  # free-choice has no margin gate
    trainer._log_strategy_distribution("selected_strategy", chosen_idx, clear_mask)
    metrics = {k: v[-1] for k, v in trainer._metrics.items()}
    print(f"  emitted = {sorted(k for k in metrics if 'selected_strategy' in k)}")
    check("strategy/selected_strategy_direct_rate" in metrics, "selected_strategy_direct_rate emitted")
    check("strategy/selected_strategy_cot_rate" in metrics, "selected_strategy_cot_rate emitted")
    check("strategy/selected_strategy_long_cot_rate" in metrics, "selected_strategy_long_cot_rate emitted")
    check(
        "strategy/selected_strategy_tie_or_unclear_rate" in metrics,
        "selected_strategy_tie_or_unclear_rate emitted (always present)",
    )
    check(abs(metrics["strategy/selected_strategy_direct_rate"] - 0.5) < 1e-6, "direct = 2/4")
    check(
        abs(metrics["strategy/selected_strategy_tie_or_unclear_rate"] - 0.0) < 1e-6,
        "tie_or_unclear = 0 when clear_mask is all-ones",
    )


def test_grpo_group_grouping_logic():
    header("Test 9: rewards.view(-1, G) groups same-prompt rollouts together")
    # Even with balanced rollout, the prompt-major flatten order means rewards.view(-1, G)
    # places one prompt's G rollouts on one row. Confirm with synthetic shapes.
    G = 9
    B_total = 2
    # Construct rewards with markers: group 0 -> 0.0...0.8 step 0.1; group 1 -> 1.0...1.8.
    rewards = torch.tensor([
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    ])
    grouped = rewards.view(-1, G)
    check(grouped.shape == (B_total, G), f"grouped shape {tuple(grouped.shape)}")
    check(torch.allclose(grouped[0].mean(), torch.tensor(0.4)), "group 0 mean == 0.4")
    check(torch.allclose(grouped[1].mean(), torch.tensor(1.4)), "group 1 mean == 1.4")


def main():
    print("Balanced strategy rollout: dry-run sanity (no GPU/vLLM required)")
    test_strategy_plan_length()
    test_strategy_plan_perspective()
    test_strategy_plan_invalid_G()
    test_directive_text()
    test_parsed_tag_to_strategy()
    test_bonus_math_known_inputs()
    test_bonus_math_margin_gate()
    test_bonus_math_two_groups()
    test_bonus_zero_mean_within_group()
    test_balanced_off_no_change()
    test_best_strategy_metric_keys_length()
    test_best_strategy_metric_keys_perspective()
    test_selected_strategy_extensibility()
    test_grpo_group_grouping_logic()
    print("\nAll local dry-run checks PASSED.\n")
    print("Reminder: GPU/vLLM-side checks still required on the HPC:")
    print("  1) Run with balanced_strategy_rollout=true on 1 step, batch_size=1, G=9.")
    print("  2) Inspect strategy_debug_log_path JSONL — confirm 9 slots, plan ordering,")
    print("     parsed strategy compliance, base/final/advantage shapes.")
    print("  3) Run again with balanced=false and confirm legacy behavior unchanged.")


if __name__ == "__main__":
    main()
