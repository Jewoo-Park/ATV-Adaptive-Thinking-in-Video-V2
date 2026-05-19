#!/usr/bin/env python3
"""Low-cost validation for strict strategy parsing and reward-shaping gates."""

from __future__ import annotations

import importlib.util
import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R1V_SRC = os.path.join(REPO_ROOT, "src", "r1-v", "src")
EVAL_STRICT = os.path.join(REPO_ROOT, "src", "eval", "strict_answer.py")
if R1V_SRC not in sys.path:
    sys.path.insert(0, R1V_SRC)

from open_r1.grpo import answer_accuracy_reward, answer_format_reward  # noqa: E402
from open_r1.strategy import build_balanced_strategy_plan, compute_strategy_bonus, strategies_for_task  # noqa: E402
from open_r1.strict_answer import parse_strict_output as train_parse  # noqa: E402


def _load_eval_parser():
    spec = importlib.util.spec_from_file_location("eval_strict_answer_for_validation", EVAL_STRICT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.parse_strict_output


eval_parse = _load_eval_parser()


def check(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print(f"OK {message}")


def assert_same_parser(mode: str, text: str, *, ok: bool, strategy: str, letter: str | None = None) -> None:
    t = train_parse(text, task_type=mode)
    e = eval_parse(text, task_type=mode)
    check(t.to_dict() == e.to_dict(), f"{mode} train/eval parser agree for {text!r}")
    check(t.format_ok is ok, f"{mode} format_ok={ok} for {text!r}")
    check(t.parsed_strategy == strategy, f"{mode} parsed_strategy={strategy} for {text!r}")
    if letter is not None:
        check(t.pred_letter == letter, f"{mode} pred_letter={letter} for {text!r}")


def parser_cases() -> None:
    assert_same_parser("length", "<DIRECT>None</DIRECT>\n<ANSWER>A</ANSWER>", ok=True, strategy="direct", letter="A")
    assert_same_parser("length", "<COT>brief</COT>\n<ANSWER>B</ANSWER>", ok=True, strategy="cot", letter="B")
    assert_same_parser("length", "<LONG_COT>detailed</LONG_COT>\n<ANSWER>C</ANSWER>", ok=True, strategy="long_cot", letter="C")
    assert_same_parser("length", "<ANSWER>A</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("length", "<DIRECT>some reasoning</DIRECT>\n<ANSWER>A</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("length", "<DIRECT>None</DIRECT>\n<ANSWER>Z</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("length", "<COT>x</COT><LONG_COT>y</LONG_COT><ANSWER>A</ANSWER>", ok=False, strategy="invalid")

    assert_same_parser("perspective", "<ABSTRACT>concept</ABSTRACT>\n<ANSWER>A</ANSWER>", ok=True, strategy="abstract", letter="A")
    assert_same_parser("perspective", "<TEMPORAL>sequence</TEMPORAL>\n<ANSWER>B</ANSWER>", ok=True, strategy="temporal", letter="B")
    assert_same_parser(
        "perspective",
        "<SPATIOTEMPORAL>motion and position</SPATIOTEMPORAL>\n<ANSWER>C</ANSWER>",
        ok=True,
        strategy="spatiotemporal",
        letter="C",
    )
    assert_same_parser("perspective", "<ANSWER>A</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("perspective", "plain answer A", ok=False, strategy="invalid")
    assert_same_parser("perspective", "<ABSTRACT></ABSTRACT>\n<ANSWER>A</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("perspective", "<TEMPORAL>x</TEMPORAL>\n<ANSWER>Z</ANSWER>", ok=False, strategy="invalid")
    assert_same_parser("perspective", "<ABSTRACT>x</ABSTRACT><TEMPORAL>y</TEMPORAL><ANSWER>A</ANSWER>", ok=False, strategy="invalid")


def reward_cases() -> None:
    os.environ["GRPO_REASONING_TASK_TYPE"] = "length"
    completions = [[{"content": "<COT>brief</COT>\n<ANSWER>A</ANSWER>"}]]
    problem = ["Question?\nA. yes\nB. no"]
    solution = ["<ANSWER>A</ANSWER>"]
    check(
        answer_accuracy_reward(completions, solution, forced_strategy=["direct"], problem=problem)[0] == 0.0,
        "wrong forced LENGTH strategy gets accuracy_reward=0.0",
    )
    check(
        answer_format_reward(completions, forced_strategy=["direct"])[0] == 0.0,
        "wrong forced LENGTH strategy gets format_reward=0.0",
    )
    check(
        answer_accuracy_reward(completions, solution, forced_strategy=["cot"], problem=problem)[0] == 1.0,
        "matching forced LENGTH strategy can get accuracy_reward=1.0",
    )

    os.environ["GRPO_REASONING_TASK_TYPE"] = "perspective"
    completions = [[{"content": "<ANSWER>A</ANSWER>"}]]
    check(
        answer_accuracy_reward(completions, solution, forced_strategy=["abstract"], problem=problem)[0] == 0.0,
        "PERSPECTIVE answer-only gets accuracy_reward=0.0",
    )
    check(
        answer_format_reward(completions, forced_strategy=["abstract"])[0] == 0.0,
        "PERSPECTIVE answer-only gets format_reward=0.0",
    )


def tie_and_bonus_cases() -> None:
    strategies = strategies_for_task("length")
    plan = build_balanced_strategy_plan("length", num_generations=9, rollouts_per_strategy=3)
    sids = [strategies.index(s) for s in plan]

    base = torch.tensor([0.6] * 3 + [0.55] * 3 + [0.5] * 3)
    final, log = compute_strategy_bonus(base, sids, len(strategies), 0.20, 0.10)
    check(torch.allclose(final, base), "tie/unclear case leaves final_reward == base_reward")
    check(log["apply_mask"].item() == 0.0, "tie/unclear case does not apply strategy bonus")
    check(log["effective_best_idx"].item() == -1, "tie/unclear case has effective_best_strategy=None")

    base = torch.tensor([1.0] * 3 + [0.0] * 3 + [0.5] * 3)
    eligible = torch.tensor([0.0] * 3 + [1.0] * 3 + [1.0] * 3)
    final, log = compute_strategy_bonus(base, sids, len(strategies), 0.20, 0.10, slot_eligible_mask=eligible)
    check(torch.allclose(final[:3], base[:3]), "strategy-mismatched slots receive no strategy bonus")


def main() -> None:
    parser_cases()
    reward_cases()
    tie_and_bonus_cases()
    print("strategy parser/reward validation passed")


if __name__ == "__main__":
    main()
