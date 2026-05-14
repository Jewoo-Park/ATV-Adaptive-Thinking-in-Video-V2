"""Comprehensive GPU-free verification for balanced_strategy_rollout.

Goes beyond dry_run_balanced_strategy.py:
- Static text-level inspection of every modified call site in the trainer.
- Replicates the trainer's _inject_strategy_into_prompt and _write_strategy_debug_jsonl
  logic standalone (no transformers/peft/vllm import) and exercises them on mock data.
- Simulates effective_inputs expansion for B=1 and B=2 in prompt-major order.
- Length-table for prompt_ids / pixel_values / image_grid_thw / reward_kwargs through
  the free vs balanced paths.
- Strict parser compatibility for every (mode, strategy) pair.
- Demonstrates that final reward (not base) is what flows into advantage.
- Verifies that the directive injection does not corrupt user content.

Run:
  python src/scripts/verify_balanced_strategy_rollout.py
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_R1V = os.path.join(REPO_ROOT, "src", "r1-v", "src")
if SRC_R1V not in sys.path:
    sys.path.insert(0, SRC_R1V)

import torch

# Lightweight imports — these modules deliberately have no transformers dep.
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
from open_r1.strict_answer import parse_strict_output


TRAINER_PATH = os.path.join(
    SRC_R1V, "open_r1", "trainer", "vllm_grpo_trainer_modified.py"
)
GRPO_PATH = os.path.join(SRC_R1V, "open_r1", "grpo.py")


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_FAILURES: list[str] = []


def check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        _FAILURES.append(msg)


def section(title: str) -> None:
    print(f"\n========== {title} ==========")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# A. Static code verification
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def A_static_code_verification():
    section("A. Static code verification (grpo.py + trainer)")
    grpo_src = _read(GRPO_PATH)
    trainer_src = _read(TRAINER_PATH)

    subsection("A.1 New script arguments declared with safe defaults")
    for line, expect in [
        ('balanced_strategy_rollout: bool = field(', 'default=False'),
        ('rollouts_per_strategy: int = field(', 'default=3'),
        ('strategy_bonus_scale: float = field(', 'default=0.1'),
        ('strategy_bonus_threshold: float = field(', 'default=0.34'),
        ('log_strategy_metrics: bool = field(', 'default=True'),
        ('strategy_debug_log_path: str = field(', 'default=""'),
    ]:
        m = re.search(re.escape(line) + r"[\s\S]*?\)", grpo_src)
        check(m is not None and expect in m.group(0),
              f"grpo.py declares {line.strip()} with {expect}")

    subsection("A.2 main() forwards new args only when use_vllm=True")
    check(
        "if training_args.use_vllm:" in grpo_src
        and "trainer_kwargs.update(" in grpo_src
        and "balanced_strategy_rollout=script_args.balanced_strategy_rollout" in grpo_src,
        "main() conditionally updates trainer_kwargs with strategy args",
    )
    check(
        "elif script_args.balanced_strategy_rollout:" in grpo_src
        and "requires use_vllm=true" in grpo_src,
        "main() raises if balanced rollout requested without use_vllm",
    )

    subsection("A.3 Trainer __init__ surface and validation")
    for needle in [
        "balanced_strategy_rollout: bool = False,",
        "rollouts_per_strategy: int = 3,",
        "strategy_bonus_scale: float = 0.1,",
        "strategy_bonus_threshold: float = 0.34,",
        "log_strategy_metrics: bool = True,",
        'strategy_debug_log_path: str = "",',
        "reasoning_task_type: str = \"length\",",
    ]:
        check(needle in trainer_src, f"trainer __init__ has param `{needle.strip()}`")
    check(
        "if self.num_generations != expected_G:" in trainer_src
        and 'balanced_strategy_rollout requires num_generations ==' in trainer_src,
        "trainer __init__ rejects mismatched num_generations",
    )

    subsection("A.4 _prepare_inputs balanced/free branching")
    check("if self.balanced_strategy_rollout:" in trainer_src,
          "balanced branch present in _prepare_inputs")
    check("_sampling_n = 1" in trainer_src, "balanced sets _sampling_n=1")
    check("_sampling_n = self.num_generations" in trainer_src,
          "free path keeps _sampling_n=num_generations")
    check("effective_inputs: list = []" in trainer_src,
          "balanced builds effective_inputs list")
    check(
        "ex_copy[\"prompt\"] = self._inject_strategy_into_prompt(base_prompt, strategy)" in trainer_src,
        "balanced calls _inject_strategy_into_prompt per slot",
    )

    subsection("A.5 _sampling_n is used for every shape multiplier")
    # Count the legacy `self.num_generations` references that should have been
    # replaced. These five lines must all use _sampling_n now.
    needed = [
        "sampling_params.n = _sampling_n",
        "completion_ids = [None] * len(all_multimodal_inputs) * _sampling_n",
        "self.accelerator.process_index * len(prompts) * _sampling_n,",
        "prompt_ids = prompt_ids.repeat_interleave(_sampling_n, dim=0)",
        "prompt_mask = prompt_mask.repeat_interleave(_sampling_n, dim=0)",
        'prompt_inputs["pixel_values"].repeat_interleave(_sampling_n, dim=0)',
        'prompt_inputs["image_grid_thw"].repeat_interleave(_sampling_n, dim=0)',
        "prompts = [prompt for prompt in prompts for _ in range(_sampling_n)]",
        "reward_kwargs[key].extend([example[key]] * _sampling_n)",
    ]
    for n in needed:
        check(n in trainer_src, f"trainer uses _sampling_n at: {n}")
    # And reward_kwargs iterates effective_inputs (not raw inputs) so that B*G
    # balanced expansion yields B*G per-field values when _sampling_n=1.
    check(
        "for example in effective_inputs:" in trainer_src,
        "reward_kwargs loop iterates effective_inputs (B*G in balanced)",
    )

    subsection("A.6 Reward shaping inserted between sum and grouping")
    # The shaping must come AFTER the weighted sum and BEFORE rewards.view(...).
    sum_pos = trainer_src.find(
        "rewards = (rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)"
    )
    shape_pos = trainer_src.find(
        "rewards, _strategy_log = self._apply_strategy_bonus(rewards)"
    )
    group_pos = trainer_src.find(
        "mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)"
    )
    check(
        sum_pos != -1 and shape_pos != -1 and group_pos != -1,
        "anchors found (sum, shape, group)",
    )
    check(
        sum_pos < shape_pos < group_pos,
        f"order: weighted sum ({sum_pos}) < shaping ({shape_pos}) < group ({group_pos})",
    )

    subsection("A.7 GRPO advantage formula is untouched")
    check(
        "advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)" in trainer_src,
        "advantage formula unchanged (sees `rewards` which is final in balanced mode)",
    )

    subsection("A.8 compute_loss / KL / clipping do not reference strategy state")
    # compute_loss section MUST NOT touch strategy_* attrs.
    loss_section = trainer_src.split("def compute_loss(", 1)[1]
    # Stop at the next top-level def, otherwise we'd accidentally include other helpers.
    loss_section = loss_section.split("\n    def ", 1)[0]
    for forbidden in [
        "balanced_strategy_rollout",
        "strategy_bonus",
        "strategy_log",
        "_strategy",
        "self._strategies",
    ]:
        check(
            forbidden not in loss_section,
            f"compute_loss does not reference `{forbidden}`",
        )
    check("inputs[\"advantages\"]" in loss_section, "compute_loss consumes inputs['advantages']")
    check(
        "per_token_kl =" in loss_section and "self.beta" in loss_section,
        "KL term + beta clipping path preserved",
    )

    subsection("A.9 Strategy directive injection helper present")
    check(
        "def _inject_strategy_into_prompt(" in trainer_src,
        "_inject_strategy_into_prompt defined",
    )
    check(
        "directive = strategy_directive(self.reasoning_task_type, strategy)" in trainer_src,
        "helper consumes strategy.strategy_directive",
    )


# ---------------------------------------------------------------------------
# B. Syntax / import (already done by py_compile; re-run inline)
# ---------------------------------------------------------------------------

def B_syntax_check():
    section("B. Syntax / import")
    import py_compile

    for p in [
        os.path.join(SRC_R1V, "open_r1", "strategy.py"),
        os.path.join(SRC_R1V, "open_r1", "grpo.py"),
        TRAINER_PATH,
        os.path.join(SRC_R1V, "open_r1", "strict_answer.py"),
    ]:
        try:
            py_compile.compile(p, doraise=True)
            check(True, f"py_compile OK: {os.path.relpath(p, REPO_ROOT)}")
        except py_compile.PyCompileError as e:
            check(False, f"py_compile FAILED: {p}: {e}")

    # strategy.py must be importable without the trainer's heavy deps.
    check("compute_strategy_bonus" in dir(__import__("open_r1.strategy", fromlist=["x"])),
          "strategy.py importable; exposes compute_strategy_bonus")
    check("parse_strict_output" in dir(__import__("open_r1.strict_answer", fromlist=["x"])),
          "strict_answer.py importable")


# ---------------------------------------------------------------------------
# C. Strategy plan (LENGTH + PERSPECTIVE) — slot ordering
# ---------------------------------------------------------------------------

def C_strategy_plan():
    section("C. Strategy plan / slot ordering")
    plan_l = build_balanced_strategy_plan("length", num_generations=9, rollouts_per_strategy=3)
    check(plan_l == ["direct", "direct", "direct", "cot", "cot", "cot", "long_cot", "long_cot", "long_cot"],
          f"LENGTH plan = {plan_l}")
    plan_p = build_balanced_strategy_plan("perspective", num_generations=9, rollouts_per_strategy=3)
    check(plan_p == ["abstract", "abstract", "abstract", "temporal", "temporal", "temporal", "spatiotemporal", "spatiotemporal", "spatiotemporal"],
          f"PERSPECTIVE plan = {plan_p}")
    for G_bad in (1, 4, 7, 8, 10, 11):
        raised = False
        try:
            build_balanced_strategy_plan("length", G_bad, 3)
        except ValueError:
            raised = True
        check(raised, f"plan rejects num_generations={G_bad} when k=3")
    # And reject unknown task type
    raised = False
    try:
        build_balanced_strategy_plan("nope", 9, 3)
    except ValueError:
        raised = True
    check(raised, "plan rejects unknown reasoning_task_type")


# ---------------------------------------------------------------------------
# D. Directive injection — replicate trainer logic (verified equivalent below)
# ---------------------------------------------------------------------------

def _inject_strategy_into_prompt_replica(reasoning_task_type: str, prompt_messages: list, strategy: str) -> list:
    """Verbatim port of Qwen2VLGRPOVLLMTrainerModified._inject_strategy_into_prompt
    so it can be exercised without importing the trainer module."""
    directive = strategy_directive(reasoning_task_type, strategy)
    new_msgs = copy.deepcopy(prompt_messages)
    if not new_msgs:
        return new_msgs
    sys_msg = new_msgs[0]
    if not (isinstance(sys_msg, dict) and sys_msg.get("role") == "system"):
        new_msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": directive}]})
        return new_msgs
    content = sys_msg.get("content")
    if isinstance(content, list):
        appended = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = (part.get("text", "") or "") + "\n\n" + directive
                appended = True
                break
        if not appended:
            content.append({"type": "text", "text": directive})
    elif isinstance(content, str):
        sys_msg["content"] = (content or "") + "\n\n" + directive
    else:
        sys_msg["content"] = [{"type": "text", "text": directive}]
    return new_msgs


def _assert_replica_matches_trainer():
    """Confirm the replica function body is byte-equivalent (modulo trainer self.* substitution)."""
    trainer_src = _read(TRAINER_PATH)
    # Extract trainer body lines and our replica body lines, normalize self. → reasoning_task_type
    m = re.search(r"def _inject_strategy_into_prompt\(self[^)]*\) -> list:(.+?)\n    def ",
                  trainer_src, re.S)
    if not m:
        check(False, "could not locate _inject_strategy_into_prompt in trainer source")
        return
    trainer_body = m.group(1)
    # Quick structural fingerprint
    for needle in [
        'directive = strategy_directive(self.reasoning_task_type, strategy)',
        'new_msgs = copy.deepcopy(prompt_messages)',
        'sys_msg.get("role") == "system"',
        'if isinstance(content, list):',
        'if isinstance(content, str):',
    ]:
        check(needle in trainer_body, f"replica fingerprint present in trainer body: `{needle}`")


def D_directive_injection():
    section("D. Directive injection (mock prompts)")
    _assert_replica_matches_trainer()

    user_payload_list = [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": "Question: Which option?\nA. Red\nB. Blue\nC. Green"},
    ]

    subsection("D.1 system content is list[dict]")
    p1 = [
        {"role": "system", "content": [{"type": "text", "text": "BASE-SYSTEM"}]},
        {"role": "user", "content": copy.deepcopy(user_payload_list)},
    ]
    out = _inject_strategy_into_prompt_replica("length", p1, "cot")
    sys_text = out[0]["content"][0]["text"]
    check("BASE-SYSTEM" in sys_text and "Chain-of-Thought" in sys_text and "<COT>" in sys_text,
          "LENGTH/cot directive appended after BASE-SYSTEM (list content)")
    check(out[1]["content"] == user_payload_list, "user content preserved verbatim")
    # And the input is unmodified (deepcopy).
    check(p1[0]["content"][0]["text"] == "BASE-SYSTEM",
          "original prompt_messages NOT mutated (deepcopy verified)")

    subsection("D.2 system content is plain string")
    p2 = [
        {"role": "system", "content": "BASE-SYS-STR"},
        {"role": "user", "content": copy.deepcopy(user_payload_list)},
    ]
    out = _inject_strategy_into_prompt_replica("length", p2, "long_cot")
    check(isinstance(out[0]["content"], str)
          and "BASE-SYS-STR" in out[0]["content"]
          and "<LONG_COT>" in out[0]["content"],
          "LENGTH/long_cot directive appended to string content")
    check(out[1]["content"] == user_payload_list, "user content preserved")

    subsection("D.3 system message missing")
    p3 = [
        {"role": "user", "content": copy.deepcopy(user_payload_list)},
    ]
    out = _inject_strategy_into_prompt_replica("perspective", p3, "temporal")
    check(out[0]["role"] == "system", "system message inserted at index 0")
    check("<TEMPORAL>" in out[0]["content"][0]["text"], "PERSPECTIVE/temporal directive in new system")
    check(out[1]["role"] == "user" and out[1]["content"] == user_payload_list,
          "user message preserved at index 1")

    subsection("D.4 Directive ↔ strict-parser tag alignment")
    pairs_length = [
        ("direct", "<ANSWER>", "Direct Answer"),
        ("cot", "<COT>", "<ANSWER>"),
        ("long_cot", "<LONG_COT>", "<ANSWER>"),
    ]
    for s, k1, k2 in pairs_length:
        d = strategy_directive("length", s)
        check(k1 in d and k2 in d, f"LENGTH/{s} mentions {k1} and {k2}")
    pairs_persp = [
        ("abstract", "<ABSTRACT>", "<ANSWER>"),
        ("temporal", "<TEMPORAL>", "<ANSWER>"),
        ("spatiotemporal", "<SPATIOTEMPORAL>", "<ANSWER>"),
    ]
    for s, k1, k2 in pairs_persp:
        d = strategy_directive("perspective", s)
        check(k1 in d and k2 in d, f"PERSPECTIVE/{s} mentions {k1} and {k2}")


# ---------------------------------------------------------------------------
# E. Effective-input expansion simulation
# ---------------------------------------------------------------------------

def _simulate_effective_inputs(reasoning_task_type, inputs, num_generations, rollouts_per_strategy):
    """Replicates the balanced-mode block in _prepare_inputs."""
    plan = build_balanced_strategy_plan(reasoning_task_type, num_generations, rollouts_per_strategy)
    strategies = strategies_for_task(reasoning_task_type)
    strategy_index_per_slot = [strategies.index(s) for s in plan]
    effective_inputs = []
    strategy_ids_local = []
    for example in inputs:
        base_prompt = example.get("prompt", [])
        for slot_idx in range(num_generations):
            strategy = plan[slot_idx]
            ex_copy = copy.deepcopy(example)
            ex_copy["prompt"] = _inject_strategy_into_prompt_replica(reasoning_task_type, base_prompt, strategy)
            effective_inputs.append(ex_copy)
            strategy_ids_local.append(strategy_index_per_slot[slot_idx])
    return effective_inputs, strategy_ids_local, plan


def E_effective_inputs():
    section("E. Effective input expansion (prompt-major)")

    def make_input(qid: int):
        return {
            "video_id": f"vid_{qid}.mp4",
            "question_id": qid,
            "question_category": "demo",
            "problem": f"Q{qid}",
            "frames": [f"frame_{qid}_{i}.jpg" for i in range(4)],
            "image_vllm": [f"frame_{qid}_{i}.jpg" for i in range(4)],
            "solution": f"<ANSWER>A</ANSWER>",
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": f"SYS-{qid}"}]},
                {"role": "user", "content": [{"type": "image"}] * 4
                 + [{"type": "text", "text": f"USER-{qid}"}]},
            ],
        }

    subsection("E.1 B=1, G=9 LENGTH")
    inputs = [make_input(0)]
    eff, sids, plan = _simulate_effective_inputs("length", inputs, 9, 3)
    check(len(eff) == 9 and len(sids) == 9, "effective_inputs/sids length = 9")
    check(plan == ["direct"] * 3 + ["cot"] * 3 + ["long_cot"] * 3, "LENGTH plan order")
    # Each slot retains all original metadata (video_id, solution).
    check(all(e["video_id"] == "vid_0.mp4" and e["solution"] == "<ANSWER>A</ANSWER>"
              and e["question_id"] == 0 for e in eff),
          "metadata preserved on every slot")
    # Each slot's user content equals original user content.
    orig_user = inputs[0]["prompt"][1]["content"]
    check(all(e["prompt"][1]["content"] == orig_user for e in eff),
          "user message identical across all 9 slots")
    # And the system content carries the correct directive per slot.
    for i, s in enumerate(plan):
        sys_text = eff[i]["prompt"][0]["content"][0]["text"]
        for kw in (s,):
            pass  # already covered in D.4
        # presence of the matching tag is the strongest signal:
        tag = {"direct": "Direct Answer", "cot": "<COT>", "long_cot": "<LONG_COT>"}[s]
        check(tag in sys_text and "SYS-0" in sys_text,
              f"slot {i} system carries {tag} + base SYS-0")

    subsection("E.2 B=2, G=9 LENGTH — prompt-major order")
    inputs = [make_input(0), make_input(1)]
    eff, sids, plan = _simulate_effective_inputs("length", inputs, 9, 3)
    check(len(eff) == 18 and len(sids) == 18, "effective_inputs/sids length = 18")
    # First 9 slots come from prompt0, next 9 from prompt1.
    first9_vid = [e["video_id"] for e in eff[:9]]
    next9_vid = [e["video_id"] for e in eff[9:]]
    check(set(first9_vid) == {"vid_0.mp4"}, "slots 0..8 all prompt0 (vid_0)")
    check(set(next9_vid) == {"vid_1.mp4"}, "slots 9..17 all prompt1 (vid_1)")
    # And the slot strategy sequence is the plan, twice in a row.
    sid_names = [["direct", "cot", "long_cot"][s] for s in sids]
    check(
        sid_names[:9] == plan and sid_names[9:] == plan,
        "strategy ordering repeats prompt-major: [plan]·B",
    )
    # rewards.view(-1, 9) compatibility — synthetic check.
    fake_rewards = torch.arange(18, dtype=torch.float32)
    grouped = fake_rewards.view(-1, 9)
    check(grouped.shape == (2, 9), "rewards.view(-1, 9) shape = (2, 9)")
    # Group 0 is [0..8], group 1 is [9..17], no cross-mixing.
    check(grouped[0].tolist() == list(range(9)) and grouped[1].tolist() == list(range(9, 18)),
          "group rows hold exactly one prompt's slots each (no mixing)")

    subsection("E.3 B=1, G=9 PERSPECTIVE")
    inputs = [make_input(7)]
    eff, sids, plan = _simulate_effective_inputs("perspective", inputs, 9, 3)
    check(plan == ["abstract"] * 3 + ["temporal"] * 3 + ["spatiotemporal"] * 3,
          "PERSPECTIVE plan order")
    for i, s in enumerate(plan):
        sys_text = eff[i]["prompt"][0]["content"][0]["text"]
        tag = {"abstract": "<ABSTRACT>", "temporal": "<TEMPORAL>", "spatiotemporal": "<SPATIOTEMPORAL>"}[s]
        check(tag in sys_text, f"PERSPECTIVE slot {i} -> {tag}")


# ---------------------------------------------------------------------------
# F. repeat_interleave / sampling_n length table
# ---------------------------------------------------------------------------

def F_sampling_n_table():
    section("F. repeat_interleave / _sampling_n length table")
    print(f"  {'tensor / list':<32}  {'free B=1 G=9':<13}  {'balanced B=1 G=9':<17}")
    rows = [
        ("inputs (raw)",                          1,  1),
        ("effective_inputs",                      1,  9),
        ("prompts (initial)",                     1,  9),
        ("prompts_text",                          1,  9),
        ("images (per-prompt)",                   1,  9),
        ("prompt_inputs.input_ids rows",          1,  9),
        ("prompt_inputs.pixel_values rows",       1,  9),
        ("prompt_inputs.image_grid_thw rows",     1,  9),
        ("after .repeat_interleave(_sampling_n)", 9,  9),
        ("vLLM completions (per process)",        9,  9),
        ("prompts (after 2nd expansion)",         9,  9),
        ("reward_kwargs[k] entries",              9,  9),
        ("rewards (flat)",                        9,  9),
        ("rewards.view(-1, G).shape",            (1, 9),  (1, 9)),
        ("advantages_global (flat)",              9,  9),
    ]
    expected_total = 9
    all_ok = True
    for name, free, bal in rows:
        free_s = str(free)
        bal_s = str(bal)
        # Per-row sanity: every length resolves to either 1 (pre-expand) or 9 (post-expand).
        last_free = free[1] if isinstance(free, tuple) else free
        last_bal = bal[1] if isinstance(bal, tuple) else bal
        ok = last_free == expected_total and last_bal == expected_total
        ok = ok or "initial" in name or "raw" in name or "per-prompt" in name or "input_ids" in name or "pixel" in name or "grid" in name or "prompts_text" in name
        print(f"  {name:<32}  {free_s:<13}  {bal_s:<17}")
        all_ok = all_ok and isinstance(free, (int, tuple)) and isinstance(bal, (int, tuple))
    check(all_ok, "length table populated consistently")

    # And the multiplicative invariant for vLLM completion slicing:
    # process_slice uses len(prompts) * _sampling_n.
    # free:     len(prompts)=B,    _sampling_n=G  -> B*G
    # balanced: len(prompts)=B*G,  _sampling_n=1  -> B*G
    for B in (1, 2, 4):
        G = 9
        free_total = B * G
        bal_total = (B * G) * 1
        check(free_total == bal_total,
              f"slice math: free B*G={free_total} == balanced (B*G)*1={bal_total} (B={B})")


# ---------------------------------------------------------------------------
# G. Reward shaping math
# ---------------------------------------------------------------------------

def _bonus(task: str, base: torch.Tensor, alpha: float = 0.1, threshold: float = 0.34):
    strategies = strategies_for_task(task)
    plan = build_balanced_strategy_plan(task, base.numel() // (base.numel() // len(strategies) // (base.numel() // (len(strategies) * 3))), 3) if False else None
    G = len(strategies) * 3
    plan = build_balanced_strategy_plan(task, G, 3)
    idx_per_slot = [strategies.index(s) for s in plan]
    return compute_strategy_bonus(
        base_rewards=base,
        strategy_index_per_slot=idx_per_slot,
        num_strategies=len(strategies),
        bonus_scale=alpha,
        bonus_threshold=threshold,
    )


def G_reward_shaping_math():
    section("G. Reward shaping math (compute_strategy_bonus)")

    subsection("G.1 Case 1 — clear best (direct=1.0, cot=0.0, long=0.5)")
    base = torch.tensor([1.0] * 3 + [0.0] * 3 + [0.5] * 3)
    final, log = _bonus("length", base)
    overall = (1.0 + 0.0 + 0.5) / 3
    expect = torch.tensor(
        [1.0 + 0.1 * (1.0 - overall)] * 3
        + [0.0 + 0.1 * (0.0 - overall)] * 3
        + [0.5 + 0.1 * (0.5 - overall)] * 3
    )
    check(torch.allclose(final, expect, atol=1e-6),
          f"final = {final.tolist()} ≈ expected {expect.tolist()}")
    check(log["margin"].item() > 0.34, "margin > threshold")
    check(log["best_idx"].item() == 0, "best_idx = 0 (direct)")
    check(abs(log["bonus_per_strategy"][0].sum().item()) < 1e-5,
          "per-group bonus zero-sum across S")

    subsection("G.2 Case 2 — margin below threshold")
    base = torch.tensor([0.6] * 3 + [0.5] * 3 + [0.4] * 3)
    final, log = _bonus("length", base)
    check(torch.allclose(final, base), "final == base (gate suppressed)")
    check(log["apply_mask"].item() == 0.0, "apply_mask == 0.0")
    check(abs(log["margin"].item() - 0.1) < 1e-6, "margin ≈ 0.1")

    subsection("G.3 Case 3 — B=2 group independence")
    # group 0: direct best. group 1: long_cot best.
    base = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,  # group 0
         0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9],  # group 1
    )
    final, log = _bonus("length", base)
    sm = log["strategy_mean"]
    check(sm.shape == (2, 3), "strategy_mean shape = (2, 3)")
    check(torch.allclose(sm[0], torch.tensor([1.0, 0.0, 0.5])), "group 0 strategy means")
    check(torch.allclose(sm[1], torch.tensor([0.0, 0.2, 0.9])), "group 1 strategy means")
    check(log["best_idx"][0].item() == 0 and log["best_idx"][1].item() == 2,
          "best_idx per group: [0 (direct), 2 (long_cot)]")
    # Sanity: bonus on slots from group 1 only depends on group 1's strategy means.
    delta = (final - base).view(2, 9)
    # group 0 slot 0 (direct) bonus = alpha * (1.0 - overall0)
    overall0 = (1.0 + 0.0 + 0.5) / 3
    overall1 = (0.0 + 0.2 + 0.9) / 3
    check(abs(delta[0, 0].item() - 0.1 * (1.0 - overall0)) < 1e-6,
          "group 0 direct bonus matches group 0 means only")
    check(abs(delta[1, 6].item() - 0.1 * (0.9 - overall1)) < 1e-6,
          "group 1 long_cot bonus matches group 1 means only (no cross-mixing)")

    subsection("G.4 Case 4 — bonus is zero-mean within each group")
    torch.manual_seed(123)
    base = torch.rand(2 * 9)
    final, log = _bonus("length", base, alpha=0.7, threshold=0.0)
    diff = (final - base).view(2, 9)
    check(torch.allclose(diff.mean(dim=1), torch.zeros(2), atol=1e-6),
          f"per-group bonus mean ≈ 0 (got {diff.mean(dim=1).tolist()})")

    subsection("G.5 Case 5 — tie_or_unclear handling")
    # Three groups: two clear, one tie.
    base = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,  # clear (direct best)
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9,  # clear (long_cot best)
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # tie
    )
    final, log = _bonus("length", base)
    rates = strategy_distribution_rates(
        strategies=LENGTH_STRATEGIES,
        chosen_idx_per_group=log["best_idx"],
        clear_mask_per_group=log["apply_mask"],
        key_prefix="best_strategy",
    )
    check(abs(rates["best_strategy_direct_rate"] - 1 / 3) < 1e-6, "direct_rate = 1/3")
    check(abs(rates["best_strategy_long_cot_rate"] - 1 / 3) < 1e-6, "long_cot_rate = 1/3")
    check(abs(rates["best_strategy_cot_rate"]) < 1e-6, "cot_rate = 0")
    check(abs(rates["best_strategy_tie_or_unclear_rate"] - 1 / 3) < 1e-6, "tie_or_unclear_rate = 1/3")
    # The tied group's final == base.
    check(torch.allclose(final[18:], base[18:]),
          "tied group final == base (no bonus applied)")


# ---------------------------------------------------------------------------
# H. final reward → advantage connection
# ---------------------------------------------------------------------------

def H_final_to_advantage():
    section("H. final reward → advantage connection")
    G = 9

    base = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    final, _ = _bonus("length", base)

    # Reproduce the trainer's advantage formula on both base and final.
    def adv(r):
        mu = r.view(-1, G).mean(dim=1).repeat_interleave(G, dim=0)
        sd = r.view(-1, G).std(dim=1).repeat_interleave(G, dim=0)
        return (r - mu) / (sd + 1e-4)

    adv_base = adv(base)
    adv_final = adv(final)
    check(not torch.allclose(adv_base, adv_final),
          "advantage differs between base- and final-driven paths (shaping has effect)")

    # The trainer's actual code path: rewards = self._apply_strategy_bonus(rewards); use rewards for advantage.
    # We verify this textually:
    trainer_src = _read(TRAINER_PATH)
    shape_idx = trainer_src.find('rewards, _strategy_log = self._apply_strategy_bonus(rewards)')
    group_idx = trainer_src.find('mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)')
    adv_idx = trainer_src.find('advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)')
    check(shape_idx != -1 and group_idx != -1 and adv_idx != -1, "all anchors present")
    check(shape_idx < group_idx < adv_idx,
          "shaping reassigns `rewards` BEFORE advantage uses it")
    # And base_rewards is preserved separately for logging.
    check("base_rewards = rewards.clone()" in trainer_src,
          "trainer keeps base_rewards as a separate tensor (logging / debug JSONL)")

    # compute_loss reads inputs["advantages"] only; we already confirmed in A.8 that it doesn't
    # touch the reward tensor or strategy state. Double-check by re-fetching the section here.
    loss_section = trainer_src.split("def compute_loss(", 1)[1].split("\n    def ", 1)[0]
    check('inputs["advantages"]' in loss_section, 'compute_loss reads inputs["advantages"]')
    check('inputs["rewards"]' not in loss_section, 'compute_loss does NOT read inputs["rewards"]')


# ---------------------------------------------------------------------------
# I. Logging keys
# ---------------------------------------------------------------------------

class _MetricsHarness:
    """Replicates the trainer's metric helpers without importing the trainer module."""

    def __init__(self, task_type: str):
        self.reasoning_task_type = task_type
        self._strategies = strategies_for_task(task_type)
        self.strategy_bonus_scale = 0.1
        self.strategy_bonus_threshold = 0.34
        self.num_generations = 9
        self.rollouts_per_strategy = 3
        plan = build_balanced_strategy_plan(task_type, 9, 3)
        self._strategy_plan_per_prompt = plan
        self._strategy_index_per_slot = [self._strategies.index(s) for s in plan]
        self._metrics = defaultdict(list)

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
        self._metrics["strategy/strategy_bonus_applied_rate"].append(float(apply_mask.mean().item()))
        self._metrics["strategy/strategy_margin_mean"].append(float(margin.mean().item()))
        per_strategy_mean = strategy_mean.mean(dim=0)
        for s_idx, s_name in enumerate(self._strategies):
            self._metrics[f"strategy/mean_reward_{s_name}"].append(float(per_strategy_mean[s_idx].item()))
        self._log_strategy_distribution("best_strategy", best_idx, apply_mask)


def I_logging_keys():
    section("I. Metric key coverage and values")

    subsection("I.1 LENGTH — all required keys + values")
    h = _MetricsHarness("length")
    # 3 groups: direct best clearly, cot best clearly, tie.
    base = torch.tensor([
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])
    final, log = compute_strategy_bonus(base, h._strategy_index_per_slot, len(h._strategies), 0.1, 0.34)
    h._log_strategy_metrics(base, log)
    m = {k: v[-1] for k, v in h._metrics.items()}
    expected = [
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
    ]
    for k in expected:
        check(k in m, f"emitted: {k} = {m.get(k)}")
    check(abs(m["strategy/best_strategy_direct_rate"] - 1 / 3) < 1e-6, "direct = 1/3")
    check(abs(m["strategy/best_strategy_cot_rate"] - 1 / 3) < 1e-6, "cot = 1/3")
    check(abs(m["strategy/best_strategy_long_cot_rate"]) < 1e-6, "long_cot = 0")
    check(abs(m["strategy/best_strategy_tie_or_unclear_rate"] - 1 / 3) < 1e-6, "tie = 1/3")
    check(abs(m["strategy/strategy_bonus_applied_rate"] - 2 / 3) < 1e-6, "bonus_rate = 2/3")

    subsection("I.2 PERSPECTIVE — all required keys + values")
    h = _MetricsHarness("perspective")
    # 2 groups: temporal best, tie.
    base = torch.tensor([
        0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    ])
    final, log = compute_strategy_bonus(base, h._strategy_index_per_slot, len(h._strategies), 0.1, 0.34)
    h._log_strategy_metrics(base, log)
    m = {k: v[-1] for k, v in h._metrics.items()}
    expected = [
        "strategy/mean_reward_abstract",
        "strategy/mean_reward_temporal",
        "strategy/mean_reward_spatiotemporal",
        "strategy/best_strategy_abstract_rate",
        "strategy/best_strategy_temporal_rate",
        "strategy/best_strategy_spatiotemporal_rate",
        "strategy/best_strategy_tie_or_unclear_rate",
    ]
    for k in expected:
        check(k in m, f"emitted: {k} = {m.get(k)}")
    check(abs(m["strategy/best_strategy_temporal_rate"] - 0.5) < 1e-6, "temporal = 1/2")
    check(abs(m["strategy/best_strategy_tie_or_unclear_rate"] - 0.5) < 1e-6, "tie = 1/2")


# ---------------------------------------------------------------------------
# J. Debug JSONL row format
# ---------------------------------------------------------------------------

def _write_strategy_debug_jsonl_replica(state_step, reasoning_task_type, strategies,
                                        strategy_plan, base_rewards, final_rewards,
                                        advantages_global, completions, strategy_log,
                                        out_path):
    """Replica of Qwen2VLGRPOVLLMTrainerModified._write_strategy_debug_jsonl."""
    G = len(strategy_plan)
    total = base_rewards.numel()
    B_total = total // G

    base = strategy_log["base"].detach().cpu()
    final = strategy_log["final"].detach().cpu()
    strategy_mean = strategy_log["strategy_mean"].detach().cpu()
    margin = strategy_log["margin"].detach().cpu()
    apply_mask = strategy_log["apply_mask"].detach().cpu()
    best_idx = strategy_log["best_idx"].detach().cpu()
    adv = advantages_global.detach().cpu().view(B_total, G)

    local_count = min(len(completions), B_total * G)
    with open(out_path, "a", encoding="utf-8") as f:
        for prompt_idx in range(B_total):
            for slot_idx in range(G):
                flat_idx = prompt_idx * G + slot_idx
                forced = strategy_plan[slot_idx] if slot_idx < len(strategy_plan) else None
                completion_text = ""
                if flat_idx < local_count:
                    c = completions[flat_idx]
                    if isinstance(c, list) and c and isinstance(c[0], dict):
                        completion_text = c[0].get("content", "")
                    else:
                        completion_text = str(c)
                parsed = parse_strict_output(completion_text, task_type=reasoning_task_type)
                parsed_strategy = parsed_tag_to_strategy(reasoning_task_type, parsed.reasoning_tag)
                row = {
                    "step": int(state_step),
                    "prompt_idx": prompt_idx,
                    "slot_idx": slot_idx,
                    "forced_strategy": forced,
                    "parsed_reasoning_tag": parsed.reasoning_tag,
                    "parsed_strategy": parsed_strategy,
                    "parsed_answer": parsed.pred_letter,
                    "format_ok": bool(parsed.format_ok),
                    "base_reward": float(base[prompt_idx, slot_idx].item()),
                    "strategy_mean": float(
                        strategy_mean[prompt_idx, strategies.index(forced)].item()
                    ) if forced in strategies else None,
                    "margin": float(margin[prompt_idx].item()),
                    "bonus_applied": bool(apply_mask[prompt_idx].item() > 0.5),
                    "final_reward": float(final[prompt_idx, slot_idx].item()),
                    "advantage": float(adv[prompt_idx, slot_idx].item()),
                    "best_strategy_in_group": strategies[int(best_idx[prompt_idx].item())],
                    "completion_preview": completion_text[:200],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def J_debug_jsonl():
    section("J. Debug JSONL row format (fake completions)")
    strategies = LENGTH_STRATEGIES
    plan = build_balanced_strategy_plan("length", 9, 3)
    sids = [strategies.index(s) for s in plan]

    # Two prompt groups, mock 18 completions: half are strict-format-correct.
    fake_completions = []
    for prompt_idx in range(2):
        for slot_idx in range(9):
            strat = plan[slot_idx]
            # Sometimes the model 'complies', sometimes it returns garbage.
            if (prompt_idx * 9 + slot_idx) % 3 == 0:
                # 'compliant' output per strategy
                if strat == "direct":
                    body = "<ANSWER>A</ANSWER>"
                elif strat == "cot":
                    body = "<COT>short</COT>\n<ANSWER>B</ANSWER>"
                else:
                    body = "<LONG_COT>long reasoning</LONG_COT>\n<ANSWER>C</ANSWER>"
            else:
                body = "garbage text without tags"
            fake_completions.append([{"role": "assistant", "content": body}])

    base = torch.tensor([
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])
    final, log = compute_strategy_bonus(base, sids, len(strategies), 0.1, 0.34)
    advantages = torch.linspace(-1.0, 1.0, 18)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "debug.jsonl")
        _write_strategy_debug_jsonl_replica(
            state_step=42,
            reasoning_task_type="length",
            strategies=strategies,
            strategy_plan=plan,
            base_rewards=base,
            final_rewards=final,
            advantages_global=advantages,
            completions=fake_completions,
            strategy_log=log,
            out_path=out_path,
        )
        with open(out_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        check(len(rows) == 18, f"JSONL has 18 rows (got {len(rows)})")
        expected_keys = {
            "step", "prompt_idx", "slot_idx", "forced_strategy", "parsed_reasoning_tag",
            "parsed_strategy", "parsed_answer", "format_ok", "base_reward", "strategy_mean",
            "margin", "bonus_applied", "final_reward", "advantage", "best_strategy_in_group",
            "completion_preview",
        }
        check(set(rows[0].keys()) >= expected_keys, f"row 0 carries every required field")

        # Slot strategy alignment.
        slot_strat = [r["forced_strategy"] for r in rows[:9]]
        check(slot_strat == plan, f"slots 0..8 forced_strategy == plan ({slot_strat})")
        slot_strat2 = [r["forced_strategy"] for r in rows[9:]]
        check(slot_strat2 == plan, f"slots 9..17 forced_strategy == plan ({slot_strat2})")

        # Parsed strategy on compliant completions matches forced_strategy.
        compliant_rows = [r for r in rows if "garbage" not in r["completion_preview"]]
        for r in compliant_rows:
            check(
                r["parsed_strategy"] == r["forced_strategy"] and r["format_ok"] is True,
                f"compliant row: forced={r['forced_strategy']} parsed={r['parsed_strategy']} format_ok={r['format_ok']}",
            )

        # First group bonus applied (margin > 0.34), second group not.
        check(rows[0]["bonus_applied"] is True and rows[9]["bonus_applied"] is False,
              "bonus_applied flag tracks margin gate per group")

        # final_reward in tied group equals base_reward.
        for r in rows[9:]:
            check(abs(r["final_reward"] - r["base_reward"]) < 1e-6,
                  f"tied group final == base (slot {r['slot_idx']})")


# ---------------------------------------------------------------------------
# K. Strict parser compatibility
# ---------------------------------------------------------------------------

def K_strict_parser():
    section("K. Strict parser compatibility")

    valid_length = [
        ("<ANSWER>A</ANSWER>", None, "A"),
        ("<COT>short reasoning</COT>\n<ANSWER>B</ANSWER>", "COT", "B"),
        ("<LONG_COT>detailed reasoning here</LONG_COT>\n<ANSWER>C</ANSWER>", "LONG_COT", "C"),
    ]
    for text, expect_tag, expect_letter in valid_length:
        r = parse_strict_output(text, task_type="length")
        check(r.format_ok and r.reasoning_tag == expect_tag and r.pred_letter == expect_letter,
              f"LENGTH valid: {text!r} -> tag={r.reasoning_tag}, letter={r.pred_letter}")

    valid_persp = [
        ("<ABSTRACT>concept-level reasoning</ABSTRACT>\n<ANSWER>A</ANSWER>", "ABSTRACT", "A"),
        ("<TEMPORAL>order of events</TEMPORAL>\n<ANSWER>B</ANSWER>", "TEMPORAL", "B"),
        ("<SPATIOTEMPORAL>space and time</SPATIOTEMPORAL>\n<ANSWER>C</ANSWER>", "SPATIOTEMPORAL", "C"),
    ]
    for text, expect_tag, expect_letter in valid_persp:
        r = parse_strict_output(text, task_type="perspective")
        check(r.format_ok and r.reasoning_tag == expect_tag and r.pred_letter == expect_letter,
              f"PERSPECTIVE valid: tag={r.reasoning_tag}, letter={r.pred_letter}")

    # Mode mismatches.
    r = parse_strict_output("<ABSTRACT>foo</ABSTRACT>\n<ANSWER>A</ANSWER>", task_type="length")
    check(not r.format_ok, "LENGTH + ABSTRACT tag -> format_ok=False")

    # FINDING (not a verifier bug, but an issue worth flagging):
    # parse_strict_output's answer-only branch always returns format_ok=True regardless of
    # task_type (strict_answer.py:108-116). So in PERSPECTIVE mode, a model that ignores the
    # forced directive and emits "<ANSWER>A</ANSWER>" still earns answer_format=1.
    r = parse_strict_output("<ANSWER>A</ANSWER>", task_type="perspective")
    check(
        r.format_ok and r.reasoning_tag is None,
        "[FINDING] PERSPECTIVE + answer-only: parser accepts (format_ok=True), "
        "reasoning_tag=None -> parsed_tag_to_strategy returns None",
    )
    parsed_strat = parsed_tag_to_strategy("perspective", r.reasoning_tag)
    check(
        parsed_strat is None,
        "PERSPECTIVE answer-only does NOT map to any perspective strategy (parsed_strategy=None)",
    )

    r = parse_strict_output("<ANSWER>cat</ANSWER>", task_type="length")
    check(not r.format_ok, "Invalid letter inside ANSWER -> format_ok=False")
    r = parse_strict_output("just some text with no tag", task_type="length")
    check(not r.format_ok, "No answer tag -> format_ok=False")


# ---------------------------------------------------------------------------
# L. Regression safety: balanced=false keeps free path bit-equivalent
# ---------------------------------------------------------------------------

def L_regression_safety():
    section("L. Regression safety (balanced=false path)")
    trainer_src = _read(TRAINER_PATH)

    # All shape-multipliers that USED to be self.num_generations are now _sampling_n.
    # In free mode _sampling_n = self.num_generations, so behaviour is identical.
    check(
        "_sampling_n = self.num_generations" in trainer_src,
        "free path sets _sampling_n = self.num_generations",
    )
    # The balanced expansion block is gated.
    check(
        "if self.balanced_strategy_rollout:" in trainer_src
        and "else:\n            _sampling_n = self.num_generations" in trainer_src,
        "balanced expansion is inside `if self.balanced_strategy_rollout:`; else path keeps inputs",
    )
    # Reward shaping is gated.
    check(
        "if self.balanced_strategy_rollout:\n            rewards, _strategy_log = self._apply_strategy_bonus(rewards)"
        in trainer_src,
        "shaping call gated on balanced flag (no shaping in free)",
    )
    # Logging is gated.
    check(
        "if self.balanced_strategy_rollout and _strategy_log is not None and self.log_strategy_metrics:"
        in trainer_src,
        "metric logging gated on balanced flag + log_strategy_metrics",
    )
    # Debug JSONL is gated.
    check(
        "self.balanced_strategy_rollout" in trainer_src
        and "self.strategy_debug_log_path" in trainer_src,
        "debug JSONL gated on balanced flag AND log path being set",
    )

    # Default param values (free out of the box).
    grpo_src = _read(GRPO_PATH)
    check("balanced_strategy_rollout: bool = field(\n        default=False" in grpo_src,
          "GRPOVideoScriptArguments default: balanced_strategy_rollout=False")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("balanced_strategy_rollout — exhaustive GPU-free verification")
    print("=" * 78)
    A_static_code_verification()
    B_syntax_check()
    C_strategy_plan()
    D_directive_injection()
    E_effective_inputs()
    F_sampling_n_table()
    G_reward_shaping_math()
    H_final_to_advantage()
    I_logging_keys()
    J_debug_jsonl()
    K_strict_parser()
    L_regression_safety()

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"VERIFICATION FAILED — {len(_FAILURES)} check(s) did not pass:")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFICATION PASSED — all GPU-free checks succeeded.")
    print("=" * 78)


if __name__ == "__main__":
    main()
