"""Shared one-line GRPO training metrics for stdout / log files."""

from __future__ import annotations


def format_grpo_train_metrics_line(logs: dict[str, float], global_step: int) -> str:
    """Single-line summary: loss, lr, per-reward, KL, etc."""
    parts = [f"[GRPO] step={global_step}"]
    if "loss" in logs and logs["loss"] is not None:
        parts.append(f"loss={float(logs['loss']):.6f}")
    lr = logs.get("learning_rate")
    if lr is not None:
        parts.append(f"lr={float(lr):.2e}")
    acc_key = "rewards/answer_accuracy_reward"
    fmt_key = "rewards/answer_format_reward"
    if acc_key in logs:
        parts.append(f"accuracy_reward={float(logs[acc_key]):.6f}")
    if fmt_key in logs:
        parts.append(f"format_reward={float(logs[fmt_key]):.6f}")
    for key in sorted(logs):
        if (
            key.startswith("rewards/")
            and key not in (acc_key, fmt_key)
            and not key.startswith("rewards_weight/")
        ):
            short = key.split("/", 1)[-1]
            parts.append(f"{short}={float(logs[key]):.6f}")
    if "reward" in logs:
        parts.append(f"reward={float(logs['reward']):.6f}")
    if "reward_std" in logs:
        parts.append(f"reward_std={float(logs['reward_std']):.6f}")
    if "reward/base_mean" in logs:
        parts.append(f"base_reward={float(logs['reward/base_mean']):.6f}")
    if "kl" in logs:
        parts.append(f"kl={float(logs['kl']):.6f}")
    if "kl/max" in logs:
        parts.append(f"kl_max={float(logs['kl/max']):.4f}")
    if "kl/beta" in logs:
        parts.append(f"beta={float(logs['kl/beta']):.4f}")
    if "advantage/abs_max" in logs:
        parts.append(f"adv_abs_max={float(logs['advantage/abs_max']):.4f}")
    if "advantage/low_std_group_rate" in logs:
        parts.append(f"low_std_grp={float(logs['advantage/low_std_group_rate']):.3f}")
    if "strategy/strategy_bonus_applied_rate" in logs:
        parts.append(
            f"strat_bonus={float(logs['strategy/strategy_bonus_applied_rate']):.3f}"
        )
    if "strategy/reward_tie_or_unclear_rate" in logs:
        parts.append(
            f"tie_rate={float(logs['strategy/reward_tie_or_unclear_rate']):.3f}"
        )
    if "completion_length" in logs:
        parts.append(f"completion_length={float(logs['completion_length']):.2f}")
    strat_eval = sorted(
        k
        for k in logs
        if k.startswith("strategy_eval/") and k.endswith("_strict_accuracy")
    )
    if strat_eval:
        parts.append(
            "strategy_eval_strict="
            + ",".join(
                f"{k.split('/')[1].replace('_strict_accuracy', '')}={float(logs[k]):.3f}"
                for k in strat_eval
            )
        )
    if "mm/mm_mismatch_zero_logprob_row_rate" in logs:
        parts.append(
            "mm_zero_logprob_rate="
            f"{float(logs['mm/mm_mismatch_zero_logprob_row_rate']):.4f}"
        )
    if "mm/image_placeholder_fallback_rate" in logs:
        parts.append(
            "frame_placeholder_rate="
            f"{float(logs['mm/image_placeholder_fallback_rate']):.4f}"
        )
    gn = logs.get("grad_norm")
    if gn is not None:
        parts.append(f"grad_norm={float(gn):.6f}")
    return " | ".join(parts)
