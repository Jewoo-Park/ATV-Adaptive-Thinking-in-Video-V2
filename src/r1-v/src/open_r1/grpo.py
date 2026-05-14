import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.trainer.utils import get_kbit_device_map, get_quantization_config

from open_r1.strict_answer import format_answer, normalize_gt_letter, parse_strict_output


@dataclass
## 이 스크립터 전용 인자만 추가됨.
class GRPOVideoScriptArguments(ScriptArguments):
    dataset_name: str = field(default="video_grpo_local", metadata={"help": "Dataset name required by ScriptArguments"})
    reward_funcs: list[str] = field(
        default_factory=lambda: ["answer_accuracy", "answer_format"],
        metadata={"help": "Reward functions for multiple-choice video GRPO. Possible values: answer_accuracy, answer_format"},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum pixels per frame"})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum pixels per frame"})
    train_file: str = field(default="", metadata={"help": "Path to GRPO train JSONL"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Optional path to GRPO eval/test JSONL"})
    reward_weights: str = field(
        default="",
        metadata={"help": "Comma-separated reward weights aligned with --reward_funcs (e.g. '2.0,1.0')"},
    )
    answer_accuracy_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Optional weight override for answer_accuracy reward"},
    )
    answer_format_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Optional weight override for answer_format reward"},
    )
    train_video_only: bool = field(
        default=False,
        metadata={
            "help": "If true, drop train rows whose video_id is a static image path (.png/.jpg/...), "
            "keeping rows that look like real video clips (e.g. .mp4). Test/eval split is unchanged."
        },
    )
    reasoning_task_type: str = field(
        default="length",
        metadata={"help": "Strict output task mode: length or perspective. Do not mix modes in one run."},
    )
    balanced_strategy_rollout: bool = field(
        default=False,
        metadata={
            "help": "If true, force each prompt's num_generations rollouts to be split evenly across the "
            "three strategies of reasoning_task_type (LENGTH: direct/cot/long_cot, "
            "PERSPECTIVE: abstract/temporal/spatiotemporal) and apply strategy-relative reward shaping. "
            "Requires num_generations == 3 * rollouts_per_strategy."
        },
    )
    rollouts_per_strategy: int = field(
        default=3,
        metadata={"help": "Rollouts per strategy when balanced_strategy_rollout=true. num_generations must equal 3 * this."},
    )
    strategy_bonus_scale: float = field(
        default=0.1,
        metadata={"help": "Alpha multiplier for strategy bonus: final = base + alpha * (strategy_mean - mean(strategy_means))."},
    )
    strategy_bonus_threshold: float = field(
        default=0.34,
        metadata={
            "help": "If (best_strategy_mean - second_best_strategy_mean) < threshold for a prompt group, "
            "skip the bonus for that group and use base reward as final reward."
        },
    )
    log_strategy_metrics: bool = field(
        default=True,
        metadata={"help": "Log per-strategy mean reward, bonus-applied rate, and best-strategy distribution."},
    )
    strategy_debug_log_path: str = field(
        default="",
        metadata={
            "help": "If set and balanced rollout is on, write a per-rollout debug JSONL "
            "(forced strategy, parsed tag, base/final reward, advantage) at each step."
        },
    )


GRPOScriptArguments = GRPOVideoScriptArguments


def _reward_task_type() -> str:
    return os.getenv("GRPO_REASONING_TASK_TYPE", "length").strip().lower() or "length"


# Train JSONL mixes video clips (.mp4, …) with image-only QA rows (video_id ending in .png/.jpg, …).
_STATIC_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")


def _train_row_is_video_clip(example: dict) -> bool:
    vid = str(example.get("video_id", "") or "").strip().lower()
    if not vid:
        return True
    return not any(vid.endswith(sfx) for sfx in _STATIC_IMAGE_SUFFIXES)


def answer_accuracy_reward(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completion_contents, solution):
        parsed = parse_strict_output(content, task_type=_reward_task_type())
        gt = normalize_gt_letter(sol)
        reward = (
            1.0
            if parsed.format_ok and parsed.pred_letter is not None and gt is not None and parsed.pred_letter == gt
            else 0.0
        )
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./logs/uvb_grpo_reward.log")
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def _format_ok(content: str) -> int:
    return int(parse_strict_output(content, task_type=_reward_task_type()).format_ok)


def answer_format_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if _format_ok(x) else 0.0 for x in completion_contents]


def write_test_predictions_jsonl(examples_completions: list[tuple[dict, str]], output_path: str) -> None:
    """Write per-sample test predictions to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example, pred_raw in examples_completions:
            parsed = parse_strict_output(pred_raw, task_type=_reward_task_type())
            gt_answer = normalize_gt_letter(example["solution"])
            correct = 1 if parsed.format_ok and parsed.pred_letter is not None and parsed.pred_letter == gt_answer else 0
            row = {
                "video_id": example.get("video_id", ""),
                "question_id": example.get("question_id", 0),
                "question_category": example.get("question_category", ""),
                "problem": example.get("problem", ""),
                "model_output": pred_raw,
                "pred_letter": parsed.pred_letter,
                "gt_letter": gt_answer,
                "gt_answer": format_answer(gt_answer) if gt_answer else example.get("solution", ""),
                "correct": correct,
                "format_ok": parsed.format_ok,
                "reasoning_tag": parsed.reasoning_tag,
                "reasoning_text": parsed.reasoning_text,
                "malformed_type": parsed.malformed_type,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


reward_funcs_registry = {
    "answer_accuracy": answer_accuracy_reward,
    "answer_format": answer_format_reward,
}


LENGTH_SYSTEM_PROMPT = """You are a video multiple-choice question answering assistant.

Choose the appropriate reasoning length.

Allowed outputs are exactly one of:

<ANSWER>X</ANSWER>

<COT>...</COT>
<ANSWER>X</ANSWER>

<LONG_COT>...</LONG_COT>
<ANSWER>X</ANSWER>

X must be a single option letter from A to J.
Do NOT put explanations or option text inside <ANSWER>.
Do NOT output anything outside the allowed tags.

Example:
User: [Video frames] Which option is correct?
Options:
A. Red
B. Blue
C. Green
Assistant: <ANSWER>B</ANSWER>

Now answer the question based on the video frames."""


PERSPECTIVE_SYSTEM_PROMPT = """You are a video multiple-choice question answering assistant.

Choose the appropriate reasoning perspective.

Allowed outputs are exactly one of:

<ABSTRACT>...</ABSTRACT>
<ANSWER>X</ANSWER>

<TEMPORAL>...</TEMPORAL>
<ANSWER>X</ANSWER>

<SPATIOTEMPORAL>...</SPATIOTEMPORAL>
<ANSWER>X</ANSWER>

X must be a single option letter from A to J.
Do NOT put explanations or option text inside <ANSWER>.
Do NOT output anything outside the allowed tags.

Example:
User: [Video frames] Which option is correct?
Options:
A. Before the turn
B. During the turn
C. After the turn
Assistant: <TEMPORAL>The relevant evidence is the order of events across the frames.</TEMPORAL>
<ANSWER>C</ANSWER>

Now answer the question based on the video frames."""


def system_prompt_for_task(task_type: str) -> str:
    normalized = str(task_type or "length").strip().lower()
    if normalized == "length":
        return LENGTH_SYSTEM_PROMPT
    if normalized == "perspective":
        return PERSPECTIVE_SYSTEM_PROMPT
    raise ValueError("reasoning_task_type must be either 'length' or 'perspective'")


def main(script_args, training_args, model_args):
    reasoning_task_type = str(script_args.reasoning_task_type or "length").strip().lower()
    if reasoning_task_type not in {"length", "perspective"}:
        raise ValueError("--reasoning_task_type must be either 'length' or 'perspective'")
    os.environ["GRPO_REASONING_TASK_TYPE"] = reasoning_task_type
    system_prompt = system_prompt_for_task(reasoning_task_type)

    # Bridge TRL ModelConfig quantization flags (e.g. --load_in_8bit/--load_in_4bit)
    # into trainer args so model_init_kwargs reach from_pretrained().
    model_init_kwargs = dict(getattr(training_args, "model_init_kwargs", {}) or {})
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_init_kwargs["quantization_config"] = quantization_config
        if getattr(model_args, "load_in_4bit", False):
            model_init_kwargs["device_map"] = get_kbit_device_map()
        elif getattr(model_args, "load_in_8bit", False):
            # Keep it explicit for 8-bit too; mirrors TRL script behavior.
            model_init_kwargs["device_map"] = get_kbit_device_map()
        if getattr(training_args, "use_vllm", False):
            print(
                "[VIDEO-GRPO] Warning: k-bit quantization with vLLM weight sync is experimental in this repo."
            )
    training_args.model_init_kwargs = model_init_kwargs

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # Default reward weights when the user does not specify any overrides.
    # We bias toward answer accuracy while still enforcing output format.
    default_weight_by_name = {
        "answer_accuracy": 0.8,
        "answer_format": 0.2,
    }
    reward_weights = [
        float(default_weight_by_name.get(name, 1.0)) for name in script_args.reward_funcs
    ]
    if script_args.reward_weights.strip():
        parsed = [x.strip() for x in script_args.reward_weights.split(",") if x.strip()]
        if len(parsed) != len(reward_weights):
            raise ValueError(
                "reward_weights length must match reward_funcs length "
                f"({len(reward_weights)}), got {len(parsed)}."
            )
        reward_weights = [float(x) for x in parsed]
    if script_args.answer_accuracy_weight is not None:
        for i, name in enumerate(script_args.reward_funcs):
            if name == "answer_accuracy":
                reward_weights[i] = float(script_args.answer_accuracy_weight)
    if script_args.answer_format_weight is not None:
        for i, name in enumerate(script_args.reward_funcs):
            if name == "answer_format":
                reward_weights[i] = float(script_args.answer_format_weight)

    data_files = {"train": script_args.train_file}
    if script_args.test_file:
        data_files["test"] = script_args.test_file
    dataset = load_dataset("json", data_files=data_files)

    if script_args.train_video_only:
        n_before = len(dataset["train"])
        dataset["train"] = dataset["train"].filter(_train_row_is_video_clip)
        n_after = len(dataset["train"])
        print(
            f"[VIDEO-GRPO] train_video_only=True: train rows {n_after}/{n_before} "
            f"(dropped {n_before - n_after} static-image rows by video_id suffix)."
        )
        if n_after == 0:
            raise ValueError(
                "train_video_only removed all train rows. Check train JSONL or disable --train_video_only."
            )

    def resolve_frames_for_split(split_name: str, base_jsonl_path: Optional[str]) -> None:
        if split_name not in dataset or not base_jsonl_path:
            return
        base_dir = os.path.dirname(os.path.abspath(base_jsonl_path))

        def _resolve(example):
            frames = example.get("frames", [])
            resolved = []
            for frame_path in frames:
                if os.path.isabs(frame_path):
                    resolved.append(frame_path)
                else:
                    resolved.append(os.path.normpath(os.path.join(base_dir, frame_path)))
            example["frames"] = resolved
            return example

        dataset[split_name] = dataset[split_name].map(_resolve)

    resolve_frames_for_split("train", script_args.train_file)
    resolve_frames_for_split("test", script_args.test_file)

    def make_conversation_video(example):
        frame_tokens = [{"type": "image"} for _ in example["frames"]]
        frame_tokens.append({"type": "text", "text": example["problem"]})
        out = {
            "image_vllm": example["frames"],
            "solution": example["solution"],
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": frame_tokens},
            ],
        }
        if "video_id" in example:
            out["video_id"] = example["video_id"]
        if "question_id" in example:
            out["question_id"] = example["question_id"]
        if "question_category" in example:
            out["question_category"] = example["question_category"]
        out["problem"] = example["problem"]
        return out

    dataset = dataset.map(make_conversation_video)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    trainer_kwargs = dict(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset and training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        reward_weights=reward_weights,
    )
    if training_args.use_vllm:
        trainer_kwargs.update(
            reasoning_task_type=reasoning_task_type,
            balanced_strategy_rollout=script_args.balanced_strategy_rollout,
            rollouts_per_strategy=script_args.rollouts_per_strategy,
            strategy_bonus_scale=script_args.strategy_bonus_scale,
            strategy_bonus_threshold=script_args.strategy_bonus_threshold,
            log_strategy_metrics=script_args.log_strategy_metrics,
            strategy_debug_log_path=script_args.strategy_debug_log_path,
        )
    elif script_args.balanced_strategy_rollout:
        raise ValueError(
            "balanced_strategy_rollout=true requires use_vllm=true (only the vLLM trainer path supports it)."
        )
    trainer = trainer_cls(**trainer_kwargs)

    # GRPOConfig inherits TrainingArguments, which already defines --resume_from_checkpoint (do not duplicate on ScriptArguments).
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        resume_ckpt = os.path.abspath(os.path.expanduser(str(resume_ckpt)))
        if not os.path.isdir(resume_ckpt):
            raise FileNotFoundError(f"resume_from_checkpoint is not a directory: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(training_args.output_dir)

    if script_args.test_file and "test" in dataset:
        examples_completions = trainer.run_test_inference()
        if examples_completions:
            out_path = os.path.join(training_args.output_dir, "test_predictions.jsonl")
            write_test_predictions_jsonl(examples_completions, out_path)
            print(f"[VIDEO-GRPO] Test predictions saved to {out_path}")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOVideoScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
