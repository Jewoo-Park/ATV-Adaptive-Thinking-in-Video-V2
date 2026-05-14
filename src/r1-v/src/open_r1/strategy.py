"""Strategy definitions for balanced GRPO rollouts.

Forces each rollout in a prompt group to use a specific reasoning strategy
(LENGTH: direct / cot / long_cot; PERSPECTIVE: abstract / temporal / spatiotemporal),
so that strategy-level reward comparisons inside the same group are well-defined.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


LENGTH_STRATEGIES: tuple[str, ...] = ("direct", "cot", "long_cot")
PERSPECTIVE_STRATEGIES: tuple[str, ...] = ("abstract", "temporal", "spatiotemporal")


def strategies_for_task(task_type: str) -> tuple[str, ...]:
    normalized = str(task_type or "length").strip().lower()
    if normalized == "length":
        return LENGTH_STRATEGIES
    if normalized == "perspective":
        return PERSPECTIVE_STRATEGIES
    raise ValueError(f"Unknown reasoning_task_type: {task_type!r}")


_LENGTH_DIRECTIVES = {
    "direct": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Direct Answer.\n"
        "You MUST output ONLY <ANSWER>X</ANSWER> on a single line with no reasoning, "
        "no <COT> or <LONG_COT> tags, and no text outside the <ANSWER> tag."
    ),
    "cot": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Chain-of-Thought.\n"
        "You MUST output exactly <COT>brief reasoning</COT>\\n<ANSWER>X</ANSWER>. "
        "The <COT>...</COT> block must be non-empty and concise. "
        "Do NOT use <LONG_COT> or any other tag, and do NOT output text outside these two tags."
    ),
    "long_cot": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Long Chain-of-Thought.\n"
        "You MUST output exactly <LONG_COT>extended step-by-step reasoning</LONG_COT>\\n<ANSWER>X</ANSWER>. "
        "The <LONG_COT>...</LONG_COT> block must be non-empty and substantially longer than a short CoT. "
        "Do NOT use <COT> or any other tag, and do NOT output text outside these two tags."
    ),
}


_PERSPECTIVE_DIRECTIVES = {
    "abstract": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Abstract reasoning.\n"
        "You MUST output exactly <ABSTRACT>concept- or category-level reasoning</ABSTRACT>\\n<ANSWER>X</ANSWER>. "
        "Do NOT use <TEMPORAL> or <SPATIOTEMPORAL>, and do NOT output text outside these two tags."
    ),
    "temporal": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Temporal reasoning.\n"
        "You MUST output exactly <TEMPORAL>reasoning about the order or timing of events across frames</TEMPORAL>\\n<ANSWER>X</ANSWER>. "
        "Do NOT use <ABSTRACT> or <SPATIOTEMPORAL>, and do NOT output text outside these two tags."
    ),
    "spatiotemporal": (
        "FORCED STRATEGY FOR THIS ROLLOUT: Spatio-temporal reasoning.\n"
        "You MUST output exactly <SPATIOTEMPORAL>reasoning about spatial layout combined with temporal change</SPATIOTEMPORAL>\\n<ANSWER>X</ANSWER>. "
        "Do NOT use <ABSTRACT> or <TEMPORAL>, and do NOT output text outside these two tags."
    ),
}


def strategy_directive(task_type: str, strategy: str) -> str:
    """Return a one-paragraph directive to append to the system prompt for this strategy."""
    normalized_task = str(task_type or "length").strip().lower()
    normalized_strat = str(strategy or "").strip().lower()
    if normalized_task == "length":
        if normalized_strat not in _LENGTH_DIRECTIVES:
            raise ValueError(f"Unknown LENGTH strategy: {strategy!r}")
        return _LENGTH_DIRECTIVES[normalized_strat]
    if normalized_task == "perspective":
        if normalized_strat not in _PERSPECTIVE_DIRECTIVES:
            raise ValueError(f"Unknown PERSPECTIVE strategy: {strategy!r}")
        return _PERSPECTIVE_DIRECTIVES[normalized_strat]
    raise ValueError(f"Unknown reasoning_task_type: {task_type!r}")


def parsed_tag_to_strategy(task_type: str, reasoning_tag: Optional[str]) -> Optional[str]:
    """Map a strict-parser reasoning_tag to a strategy id (used for sanity logs)."""
    normalized_task = str(task_type or "length").strip().lower()
    if normalized_task == "length":
        if reasoning_tag is None:
            return "direct"
        if reasoning_tag == "COT":
            return "cot"
        if reasoning_tag == "LONG_COT":
            return "long_cot"
        return None
    if normalized_task == "perspective":
        if reasoning_tag == "ABSTRACT":
            return "abstract"
        if reasoning_tag == "TEMPORAL":
            return "temporal"
        if reasoning_tag == "SPATIOTEMPORAL":
            return "spatiotemporal"
        return None
    return None


def strategy_index(task_type: str, strategy: str) -> int:
    strategies = strategies_for_task(task_type)
    s = str(strategy or "").strip().lower()
    if s not in strategies:
        raise ValueError(f"Strategy {strategy!r} not valid for task {task_type!r}")
    return strategies.index(s)


def compute_strategy_bonus(
    base_rewards: torch.Tensor,
    strategy_index_per_slot: Sequence[int],
    num_strategies: int,
    bonus_scale: float,
    bonus_threshold: float,
) -> tuple[torch.Tensor, dict]:
    """Strategy-relative reward shaping (pure-function form, callable without the trainer).

    Args:
        base_rewards: 1D tensor of length B_total * G, in prompt-major slot order.
        strategy_index_per_slot: Length-G list giving the strategy index for each slot.
        num_strategies: S.
        bonus_scale: alpha for `final = base + alpha * (strategy_mean - mean(strategy_means))`.
        bonus_threshold: if (best_strategy_mean - second_best_strategy_mean) < threshold,
            the bonus is suppressed for that prompt group.

    Returns:
        (final_rewards 1D, log_dict). log_dict contains tensors with prompt-group rows:
        base (B_total, G), final (B_total, G), strategy_mean (B_total, S),
        bonus_per_strategy (B_total, S), margin (B_total,), apply_mask (B_total,),
        best_idx (B_total,).
    """
    device = base_rewards.device
    dtype = base_rewards.dtype
    G = len(strategy_index_per_slot)
    if G == 0:
        raise ValueError("strategy_index_per_slot must have length > 0")
    total = base_rewards.numel()
    if total % G != 0:
        raise RuntimeError(
            f"reward length {total} is not divisible by G={G}"
        )
    B_total = total // G
    S = int(num_strategies)

    strat_idx_per_slot = torch.tensor(
        list(strategy_index_per_slot), dtype=torch.long, device=device
    )
    strat_idx_grouped = strat_idx_per_slot.unsqueeze(0).expand(B_total, -1)  # (B_total, G)
    base = base_rewards.view(B_total, G)

    strategy_sum = torch.zeros(B_total, S, device=device, dtype=dtype)
    strategy_cnt = torch.zeros(B_total, S, device=device, dtype=dtype)
    strategy_sum.scatter_add_(1, strat_idx_grouped, base)
    strategy_cnt.scatter_add_(1, strat_idx_grouped, torch.ones_like(base))
    strategy_mean = strategy_sum / strategy_cnt.clamp(min=1.0)  # (B_total, S)

    overall = strategy_mean.mean(dim=1, keepdim=True)
    bonus_per_strategy = strategy_mean - overall  # zero-sum across S per row

    sorted_means, _ = strategy_mean.sort(dim=1, descending=True)
    if S >= 2:
        margin = sorted_means[:, 0] - sorted_means[:, 1]
    else:
        margin = torch.zeros(B_total, device=device, dtype=dtype)
    apply_mask = (margin >= float(bonus_threshold)).to(dtype)
    bonus_per_strategy = bonus_per_strategy * apply_mask.unsqueeze(1)

    bonus_per_slot = bonus_per_strategy.gather(1, strat_idx_grouped)
    final = base + float(bonus_scale) * bonus_per_slot

    best_idx = strategy_mean.argmax(dim=1)

    log_dict = {
        "base": base,
        "final": final,
        "strategy_mean": strategy_mean,
        "bonus_per_strategy": bonus_per_strategy,
        "margin": margin,
        "apply_mask": apply_mask,
        "best_idx": best_idx,
    }
    return final.reshape(-1), log_dict


def strategy_distribution_rates(
    strategies: Sequence[str],
    chosen_idx_per_group: torch.Tensor,
    clear_mask_per_group: torch.Tensor,
    key_prefix: str,
) -> dict[str, float]:
    """Per-strategy "clear choice" rate + tie_or_unclear rate.

    Reusable for:
      - balanced rollout:  key_prefix="best_strategy",     chosen_idx=argmax over strategy means,
                           clear_mask=(margin >= threshold).
      - free-choice rollout (future): key_prefix="selected_strategy", chosen_idx=parsed strategy id,
                           clear_mask=torch.ones (no gate).

    Returns a dict ready to be appended into _metrics (caller passes through `f"strategy/{k}"` if
    namespacing is desired).
    """
    B_total = int(chosen_idx_per_group.numel())
    out: dict[str, float] = {}
    if B_total == 0:
        return out
    clear_f = clear_mask_per_group.to(torch.float32)
    for s_idx, s_name in enumerate(strategies):
        picked = ((chosen_idx_per_group == s_idx).to(torch.float32) * clear_f).sum().item()
        out[f"{key_prefix}_{s_name}_rate"] = float(picked) / B_total
    tie_count = float((1.0 - clear_f).sum().item())
    out[f"{key_prefix}_tie_or_unclear_rate"] = tie_count / B_total
    return out


def build_balanced_strategy_plan(
    task_type: str, num_generations: int, rollouts_per_strategy: int
) -> list[str]:
    """Return the strategy id assigned to each of the num_generations rollouts for one prompt.

    Layout is contiguous per strategy:
      LENGTH, G=9, k=3 -> [direct, direct, direct, cot, cot, cot, long_cot, long_cot, long_cot]
    """
    strategies = strategies_for_task(task_type)
    if num_generations != len(strategies) * rollouts_per_strategy:
        raise ValueError(
            "num_generations must equal len(strategies) * rollouts_per_strategy "
            f"(got num_generations={num_generations}, strategies={len(strategies)}, "
            f"rollouts_per_strategy={rollouts_per_strategy})"
        )
    plan: list[str] = []
    for s in strategies:
        plan.extend([s] * rollouts_per_strategy)
    return plan
