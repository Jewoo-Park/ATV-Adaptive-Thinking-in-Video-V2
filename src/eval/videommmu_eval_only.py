import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from strict_answer import format_answer, normalize_gt_letter, parse_strict_output

try:
    from vllm import LLM, SamplingParams  # type: ignore

    _HAS_VLLM = True
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _HAS_VLLM = False


def resize_image_to_pixel_bounds(
    image: Image.Image, max_pixels: int | None, min_pixels: int | None
) -> Image.Image:
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    pixels = width * height
    target_pixels = pixels
    if max_pixels is not None and pixels > max_pixels:
        target_pixels = max_pixels
    elif min_pixels is not None and pixels < min_pixels:
        target_pixels = min_pixels
    if target_pixels == pixels:
        return image
    scale = (target_pixels / float(pixels)) ** 0.5
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def strip_image_tags(text: str) -> str:
    """Remove <image N> placeholder tags inserted by VideoMMMU preprocessing."""
    return re.sub(r"<image\s*\d*>\s*", "", text).strip()


def text_stats(text: str, tokenizer) -> dict:
    words = re.findall(r"\S+", text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return {"chars": len(text), "words": len(words), "tokens": len(token_ids)}


LENGTH_SYSTEM_PROMPT = """You are a video multiple-choice question answering assistant. You have learned to choose the best reasoning length for each question.

Choose the appropriate reasoning length.

Reasoning length selection rule:
Use the shortest reasoning length that is sufficient to answer correctly.
Choose Direct only when the answer is immediately clear from the video.
Choose CoT when you need to compare options or connect a few visual clues.
Choose Long CoT when the answer depends on multiple events, temporal order, object tracking, or combining several visual cues.

Allowed outputs are exactly one of:

Direct:
Use when the answer is visually or textually obvious and does not require intermediate reasoning.
<DIRECT>None</DIRECT>
<ANSWER>X</ANSWER>

Chain-of-Thought:
Use when the problem requires moderate reasoning, option comparison, or a small number of evidence-grounded steps.
<COT>...</COT>
<ANSWER>X</ANSWER>

Long Chain-of-Thought:
Use when the problem requires multi-step reasoning, temporal ordering, event tracking, complex visual grounding, or integration of multiple cues.
<LONG_COT>...</LONG_COT>
<ANSWER>X</ANSWER>

You must follow the format strictly.
X must be a single option letter from A to J.
Do NOT put explanations or option text inside <ANSWER>.
Do NOT output anything outside the allowed tags.
Do NOT output answer-only <ANSWER>X</ANSWER>.

Example:
User: [Video frames] Which option is correct?
Options:
A. Red
B. Blue
C. Green
Assistant: <DIRECT>None</DIRECT>
<ANSWER>B</ANSWER>

Now answer the question based on the video frames."""


PERSPECTIVE_SYSTEM_PROMPT = """You are a video multiple-choice question answering assistant. You have learned to choose the best reasoning perspective for each question.

Choose the appropriate reasoning perspective.

Perspective selection rule:
Choose the perspective based on the kind of visual evidence needed to answer correctly.
Choose Abstract when the answer depends on scene gist, object identity, category, or high-level semantic meaning.
Choose Temporal when the answer depends on event order, timing, sequence, duration, or how states change over time.
Choose Spatiotemporal when the answer depends on motion, spatial layout, object location, interactions, or relations that evolve across frames.

Allowed outputs are exactly one of:

Abstract:
Use when the answer can be determined from high-level scene understanding without detailed motion tracking or fine-grained event ordering.
Use for object identity, category, scene type, overall activity, attributes, roles, or semantic comparisons visible from a coarse reading of the video.
Use when the key evidence is what is present or what kind of thing is happening, rather than precisely when it happens or where it moves.
<ABSTRACT>...</ABSTRACT>
<ANSWER>X</ANSWER>

Temporal:
Use when the answer depends on when something happens, the order of events, duration, repetition, or how the situation evolves across time.
Use for before/after relations, sequencing, counting events, onset or offset of actions, phases of a process, or identifying which moment matches a description.
Use when a single frame is insufficient and the critical evidence is how states change from earlier to later frames.
<TEMPORAL>...</TEMPORAL>
<ANSWER>X</ANSWER>

Spatiotemporal:
Use when the answer requires jointly reasoning about where things are and how they move, interact, or change position over time.
Use for trajectories, relative positions, approaching or leaving, directional movement, physical interactions, and fine-grained motion grounded in frame layout.
Use when neither abstract scene gist nor temporal order alone is enough—you must connect spatial relations to temporal change across multiple frames.
<SPATIOTEMPORAL>...</SPATIOTEMPORAL>
<ANSWER>X</ANSWER>

You must follow the format strictly.
X must be a single option letter from A to J.
Do NOT put explanations or option text inside <ANSWER>.
Do NOT output anything outside the allowed tags.
Do NOT output answer-only <ANSWER>X</ANSWER>.

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
    raise ValueError("--reasoning-task-type must be either 'length' or 'perspective'")


def load_rows(path: Path, max_samples: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VideoMMMU GRPO checkpoint with vLLM inference.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.25,
        help="vLLM gpu_memory_utilization (GRPO training default: VLLM_GPU_UTIL=0.25).",
    )
    parser.add_argument("--max-model-len", type=int, default=3136)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (GRPO training default: TEMPERATURE=0.8).",
    )
    parser.add_argument("--frames-per-sample", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-preds", type=str, default="")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument(
        "--processor-path",
        type=str,
        default="",
        help="Optional HF tree for AutoProcessor/tokenizer (merged weights often need a clean Qwen2.5-VL instruct dir).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "vllm", "hf"),
        help="Inference backend. 'auto' tries vLLM then falls back to HF Transformers.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bar during per-sample inference.",
    )
    parser.add_argument(
        "--reasoning-task-type",
        type=str,
        default="length",
        choices=("length", "perspective"),
        help="Strict output mode. Length allows ANSWER/COT/LONG_COT; perspective allows ABSTRACT/TEMPORAL/SPATIOTEMPORAL.",
    )
    args = parser.parse_args()
    run_id = f"eval_{int(time.time())}"

    # Auto-generate default output paths if not specified
    _model_dir = Path(args.model)
    if not args.save_preds:
        args.save_preds = str(_model_dir / f"videommmu_predictions_{run_id}.jsonl")
    if not args.save_json:
        args.save_json = str(_model_dir / f"videommmu_metrics_{run_id}.json")

    test_path = Path(args.test_file)
    rows = load_rows(test_path, args.max_samples)
    if not rows:
        raise SystemExit(f"No rows found in {test_path}")
    print(f"[videommmu_eval] samples={len(rows)} test_file={test_path}", flush=True)
    base_dir = test_path.resolve().parent

    _tok_root = (args.processor_path or "").strip() or args.model
    processor = AutoProcessor.from_pretrained(_tok_root, trust_remote_code=False)
    use_vllm = args.backend in {"auto", "vllm"} and _HAS_VLLM
    if args.backend == "vllm" and not _HAS_VLLM:
        raise SystemExit("[videommmu_eval] --backend=vllm but vLLM is not importable in this environment.")

    llm = None
    sp = None
    hf_model = None
    if use_vllm:
        llm = LLM(
            model=args.model,
            tokenizer=_tok_root,
            dtype="bfloat16",
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": 16},
            enforce_eager=True,
        )
        print("[videommmu_eval] backend=vllm (loaded); running inference …", flush=True)
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_completion_length, n=1)
    else:
        print("[videommmu_eval] backend=hf (vLLM unavailable or disabled); running inference …", flush=True)
        import torch

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map=None
            )
        except Exception:
            from transformers import AutoModelForCausalLM

            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
            )
        hf_model.to(args.device)
        hf_model.eval()

    correct = 0
    format_ok_count = 0
    total = 0
    preds = []
    completion_chars = completion_words = completion_tokens = 0
    reasoning_chars = reasoning_words = reasoning_tokens = 0
    reasoning_types: Counter = Counter()
    has_reasoning = 0
    pred_letter_counts: Counter = Counter()
    gt_letter_counts: Counter = Counter()
    format_class_counts: Counter = Counter()
    malformed_type_counts: Counter = Counter()
    system_prompt = system_prompt_for_task(args.reasoning_task_type)

    _loop = enumerate(rows)
    if not args.disable_progress:
        _loop = tqdm(
            _loop,
            total=len(rows),
            desc="VideoMMMU eval",
            unit="sample",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    for _, row in _loop:
        # Strip <image N> placeholder tags that VideoMMMU preprocessing inserts
        problem = strip_image_tags(row["problem"])

        resolved_frame_paths = []
        for p in row["frames"][: args.frames_per_sample]:
            frame_path = Path(p)
            if frame_path.is_absolute():
                resolved_frame_paths.append(frame_path)
            else:
                resolved_frame_paths.append((base_dir / frame_path).resolve())

        frames = [
            resize_image_to_pixel_bounds(
                Image.open(p).convert("RGB"),
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
            )
            for p in resolved_frame_paths
        ]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in frames] + [{"type": "text", "text": problem}]),
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if use_vllm:
            out = llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": frames}}],
                sampling_params=sp,
                use_tqdm=False,
            )
            text = out[0].outputs[0].text
        else:
            import torch

            inputs = processor(text=prompt, images=frames, return_tensors="pt")
            inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                gen = hf_model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_length,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature,
                )
            text = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        parsed = parse_strict_output(text, task_type=args.reasoning_task_type)
        pred = parsed.pred_letter
        gt = normalize_gt_letter(row["solution"])
        ok = int(pred is not None and gt is not None and pred == gt)
        fmt = int(parsed.format_ok)
        format_class = "valid" if parsed.format_ok else (parsed.malformed_type or "invalid_structure")
        reasoning_type = parsed.reasoning_tag or "none"
        reasoning_text = parsed.reasoning_text or ""
        full_len = text_stats(text, processor.tokenizer)
        reason_len = text_stats(reasoning_text, processor.tokenizer)

        total += 1
        correct += ok
        format_ok_count += fmt
        completion_chars += full_len["chars"]
        completion_words += full_len["words"]
        completion_tokens += full_len["tokens"]
        reasoning_chars += reason_len["chars"]
        reasoning_words += reason_len["words"]
        reasoning_tokens += reason_len["tokens"]
        reasoning_types[reasoning_type] += 1
        if reasoning_text:
            has_reasoning += 1
        pred_letter_counts[pred] += 1
        gt_letter_counts[gt] += 1
        format_class_counts[format_class] += 1
        malformed_type_counts[parsed.malformed_type or "valid"] += 1

        preds.append({
            "video_id": row.get("video_id"),
            "question_id": row.get("question_id"),
            "question_category": row.get("question_category"),
            "problem": row.get("problem"),
            "problem_cleaned": problem,
            "gt_answer": format_answer(gt) if gt else row.get("solution"),
            "gt_letter": gt,
            "model_output": text,
            "reasoning_tag": parsed.reasoning_tag,
            "reasoning_text": reasoning_text,
            "pred_letter": pred,
            "correct": ok,
            "format_ok": fmt,
            "malformed_type": parsed.malformed_type,
            "completion_chars": full_len["chars"],
            "completion_words": full_len["words"],
            "completion_tokens": full_len["tokens"],
            "reasoning_chars": reason_len["chars"],
            "reasoning_words": reason_len["words"],
            "reasoning_tokens": reason_len["tokens"],
        })

    metrics = {
        "dataset": "VideoMMMU",
        "n": total,
        "answer_accuracy": (correct / total) if total else 0.0,
        "answer_format_rate": (format_ok_count / total) if total else 0.0,
        "reasoning_present_rate": (has_reasoning / total) if total else 0.0,
        "avg_completion_chars": (completion_chars / total) if total else 0.0,
        "avg_completion_words": (completion_words / total) if total else 0.0,
        "avg_completion_tokens": (completion_tokens / total) if total else 0.0,
        "avg_reasoning_chars": (reasoning_chars / total) if total else 0.0,
        "avg_reasoning_words": (reasoning_words / total) if total else 0.0,
        "avg_reasoning_tokens": (reasoning_tokens / total) if total else 0.0,
        "reasoning_type_counts": dict(reasoning_types),
        "malformed_type_counts": dict(malformed_type_counts),
        "format_class_counts": dict(format_class_counts),
        "pred_letter_counts": dict(pred_letter_counts),
        "gt_letter_counts": dict(gt_letter_counts),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out_preds = Path(args.save_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    with out_preds.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved predictions: {out_preds}")

    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": preds}, f, ensure_ascii=False, indent=2)
    print(f"saved report json: {out_json}")


if __name__ == "__main__":
    main()
