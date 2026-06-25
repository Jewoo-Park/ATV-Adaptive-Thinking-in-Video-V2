import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FRAME_GLOB_PATTERNS = ("frame_*.jpg", "frame_*.jpeg", "frame_*.png", "*.jpg", "*.jpeg", "*.png")
SUPPORTED_MODES = {"length", "perspective"}
ALLOWED_REASONING_TYPES = {"ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL"}
OPTION_LABEL_RE = re.compile(r"^\s*(?:\(?([A-J])\)?[\).:：])\s*(.+?)\s*$", re.DOTALL)
ANSWER_LETTER_RE = re.compile(r"^(?:answer|option|choice)?\s*[:：]?\s*([A-J])\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw annotation JSONL into SFT-ready JSONL.")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--input", type=str, required=True, help="Raw annotation JSON/JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output SFT JSONL path.")
    parser.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="Root directory containing train/test frame subdirs. Defaults to <input_dir>/frames.",
    )
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--summary", type=str, default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of rows in {path}")
    return data


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_answer_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lower_raw = raw.lower()
    start_tag = "<answer>"
    end_tag = "</answer>"
    if start_tag in lower_raw and end_tag in lower_raw:
        start_idx = lower_raw.index(start_tag) + len(start_tag)
        end_idx = lower_raw.index(end_tag, start_idx)
        return raw[start_idx:end_idx].strip()
    return raw


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def parse_labeled_options(options: List[str]) -> Tuple[List[str], Dict[str, str], str]:
    normalized_options: List[str] = []
    option_text_by_letter: Dict[str, str] = {}
    for raw_option in options:
        raw = str(raw_option or "").strip()
        if not raw:
            continue
        match = OPTION_LABEL_RE.match(raw)
        if not match:
            return [], {}, "skip_non_mcq_bad_options"
        letter = match.group(1).upper()
        text = re.sub(r"\s+", " ", match.group(2).strip())
        if letter in option_text_by_letter:
            return [], {}, "skip_non_mcq_duplicate_option"
        option_text_by_letter[letter] = text
        normalized_options.append(f"{letter}. {text}")
    if len(normalized_options) < 2:
        return [], {}, "skip_non_mcq_no_options"
    return normalized_options, option_text_by_letter, ""


def map_answer_to_letter(answer_text: str, option_text_by_letter: Dict[str, str]) -> Tuple[str, str]:
    raw = normalize_answer_text(answer_text)
    if not raw:
        return "", "skip_missing_answer"
    direct = ANSWER_LETTER_RE.fullmatch(raw)
    if direct:
        letter = direct.group(1).upper()
        if letter in option_text_by_letter:
            return letter, ""
        return "", "skip_answer_letter_not_in_options"

    labeled = OPTION_LABEL_RE.match(raw)
    if labeled:
        letter = labeled.group(1).upper()
        if letter in option_text_by_letter:
            return letter, ""
        return "", "skip_answer_letter_not_in_options"

    normalized_answer = normalize_for_match(raw)
    matches = []
    for letter, option_text in option_text_by_letter.items():
        if normalized_answer == normalize_for_match(option_text):
            matches.append(letter)
        elif normalized_answer == normalize_for_match(f"{letter}. {option_text}"):
            matches.append(letter)
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return "", "skip_ambiguous_answer_mapping"
    return "", "skip_unmappable_answer"


def format_answer(letter: str) -> str:
    if not re.fullmatch(r"[A-J]", str(letter or "")):
        raise ValueError(f"Invalid answer letter: {letter!r}")
    return f"<ANSWER>{letter}</ANSWER>"


def compose_output(reasoning: str, answer: str) -> str:
    reasoning_text = str(reasoning or "").strip()
    return f"{reasoning_text}\n{answer}" if reasoning_text else answer


def build_question_with_options(question: str, options: List[str]) -> str:
    question_text = str(question or "").strip()
    option_block = "\n".join(str(opt).strip() for opt in options if str(opt).strip())
    if not option_block:
        return question_text
    return f"{question_text}\n\nOptions:\n{option_block}"


def collect_frame_paths_from_subdir(frame_subdir: str, frames_root: Path, frames_per_sample: int) -> List[Path]:
    normalized_subdir = str(frame_subdir or "").strip()
    if not normalized_subdir:
        return []

    def gather_from_dir(frame_dir: Path) -> List[Path]:
        if not frame_dir.exists():
            return []
        frames: List[Path] = []
        for pattern in FRAME_GLOB_PATTERNS:
            frames.extend(frame_dir.glob(pattern))
        return sorted({path.resolve() for path in frames if path.is_file()})[:frames_per_sample]

    # Layout A: frames_root/<frame_subdir> (e.g. sft/data/frames/OCR/OCR__... — symlinks into .../train/...)
    direct = gather_from_dir(frames_root / normalized_subdir)
    if direct:
        return direct

    # Layout B: frames_root/{train,test}/<frame_subdir>
    for split_name in ("train", "test"):
        nested = gather_from_dir(frames_root / split_name / normalized_subdir)
        if nested:
            return nested
    return []


def relativize_paths(paths: List[Path], output_dir: Path) -> List[str]:
    serialized: List[str] = []
    for path in paths:
        try:
            serialized.append(os.path.relpath(path, output_dir))
        except ValueError:
            serialized.append(str(path))
    return serialized


def resolve_media(
    row: Dict[str, Any],
    input_path: Path,
    output_dir: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> Dict[str, Any]:
    explicit_frames = row.get("frames")
    if isinstance(explicit_frames, list):
        resolved_frames: List[Path] = []
        for item in explicit_frames:
            text = str(item or "").strip()
            if not text:
                continue
            path = Path(text)
            if not path.is_absolute():
                path = (input_path.parent / path).resolve()
            if path.exists():
                resolved_frames.append(path)
        if resolved_frames:
            return {"frames": relativize_paths(resolved_frames[:frames_per_sample], output_dir)}

    frame_subdir = str(row.get("frame_subdir") or "").strip()
    if frame_subdir:
        frames = collect_frame_paths_from_subdir(frame_subdir, frames_root, frames_per_sample)
        if frames:
            return {"frames": relativize_paths(frames, output_dir)}

    for key in ("image", "image_path", "video_path"):
        text = str(row.get(key) or "").strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = (input_path.parent / text).resolve()
        if path.exists() and path.is_file():
            return {"image": relativize_paths([path], output_dir)[0]}

    return {}


def export_length_rows(
    rows: List[Dict[str, Any]],
    input_path: Path,
    output_path: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    exported: List[Dict[str, Any]] = []
    stats = Counter()

    for row in rows:
        question = str(row.get("question") or "")
        options = [str(opt) for opt in (row.get("options") or [])]
        normalized_options, option_text_by_letter, option_error = parse_labeled_options(options)
        if option_error:
            stats[option_error] += 1
            continue
        instruction = build_question_with_options(question=question, options=normalized_options)
        answer_raw = str(row.get("answer_raw") or row.get("gold_answer") or row.get("answer") or "").strip()
        answer_letter, answer_error = map_answer_to_letter(answer_raw, option_text_by_letter)
        cot_raw = str(row.get("cot_raw") or "").strip()
        long_cot_raw = str(row.get("long_cot_raw") or "").strip()
        if not instruction:
            stats["skip_missing_instruction"] += 1
            continue
        if answer_error:
            stats[answer_error] += 1
            continue

        media_fields = resolve_media(row, input_path, output_path.parent, frames_root, frames_per_sample)
        if not media_fields:
            stats["skip_missing_media"] += 1
            continue

        answer = format_answer(answer_letter)
        base = {
            "question_id": row.get("question_id"),
            "instruction": instruction,
            "input": "",
            "video_path": row.get("video_path"),
            "source_subset": row.get("source_subset"),
            "frame_subdir": row.get("frame_subdir"),
            "gold_answer": answer_letter,
            "answer": answer,
            "answer_letter": answer_letter,
            "sft_mode": "length",
            "task_type": "length",
            **media_fields,
        }

        direct_reasoning = "<DIRECT>None</DIRECT>"
        exported.append(
            {
                **base,
                "reasoning": direct_reasoning,
                "output": compose_output(direct_reasoning, answer),
                "reasoning_depth": "ANSWER",
            }
        )
        stats["answer_rows"] += 1

        if cot_raw:
            reasoning = f"<COT>{cot_raw}</COT>"
            exported.append(
                {
                    **base,
                    "reasoning": reasoning,
                    "output": compose_output(reasoning, answer),
                    "reasoning_depth": "COT",
                }
            )
            stats["cot_rows"] += 1
        else:
            stats["skip_missing_cot"] += 1

        if long_cot_raw:
            reasoning = f"<LONG_COT>{long_cot_raw}</LONG_COT>"
            exported.append(
                {
                    **base,
                    "reasoning": reasoning,
                    "output": compose_output(reasoning, answer),
                    "reasoning_depth": "LONG_COT",
                }
            )
            stats["long_cot_rows"] += 1
        else:
            stats["skip_missing_long_cot"] += 1

    return exported, dict(stats)


def export_perspective_rows(
    rows: List[Dict[str, Any]],
    input_path: Path,
    output_path: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    exported: List[Dict[str, Any]] = []
    stats = Counter()

    for row in rows:
        question = str(row.get("question") or "")
        options = [str(opt) for opt in (row.get("options") or [])]
        normalized_options, option_text_by_letter, option_error = parse_labeled_options(options)
        if option_error:
            stats[option_error] += 1
            continue
        instruction = build_question_with_options(question=question, options=normalized_options)
        reasoning_type = str(row.get("granularity_type") or row.get("reasoning_type") or "").strip().upper()
        reasoning_trace = str(
            row.get("granularity_thinking_raw")
            or row.get("reasoning_raw")
            or row.get("thinking")
            or ""
        ).strip()
        answer_letter, answer_error = map_answer_to_letter(
            str(row.get("gold_answer") or row.get("answer") or row.get("answer_raw") or ""),
            option_text_by_letter,
        )

        if not instruction:
            stats["skip_missing_instruction"] += 1
            continue
        if not reasoning_type:
            stats["skip_missing_reasoning_type"] += 1
            continue
        if reasoning_type not in ALLOWED_REASONING_TYPES:
            stats["skip_invalid_reasoning_type"] += 1
            continue
        if not reasoning_trace:
            stats["skip_missing_reasoning_trace"] += 1
            continue
        if answer_error:
            stats[answer_error] += 1
            continue

        media_fields = resolve_media(row, input_path, output_path.parent, frames_root, frames_per_sample)
        if not media_fields:
            stats["skip_missing_media"] += 1
            continue

        answer = format_answer(answer_letter)
        reasoning = f"<{reasoning_type}>{reasoning_trace}</{reasoning_type}>"
        exported.append(
            {
                "question_id": row.get("question_id"),
                "instruction": instruction,
                "input": "",
                "answer": answer,
                "answer_letter": answer_letter,
                "reasoning": reasoning,
                "output": compose_output(reasoning, answer),
                "reasoning_type": reasoning_type,
                "video_path": row.get("video_path"),
                "source_subset": row.get("source_subset"),
                "frame_subdir": row.get("frame_subdir"),
                "gold_answer": answer_letter,
                "sft_mode": "perspective",
                "task_type": "perspective",
                **media_fields,
            }
        )
        stats["perspective_rows"] += 1
        stats[f"reasoning_type:{reasoning_type}"] += 1

    return exported, dict(stats)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    frames_root = Path(args.frames_root).resolve() if args.frames_root else (input_path.parent / "frames").resolve()

    rows = load_rows(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if args.mode == "length":
        exported_rows, stats = export_length_rows(rows, input_path, output_path, frames_root, args.frames_per_sample)
    else:
        exported_rows, stats = export_perspective_rows(rows, input_path, output_path, frames_root, args.frames_per_sample)

    write_jsonl(output_path, exported_rows)

    summary = {
        "mode": args.mode,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "frames_root": str(frames_root),
        "input_rows": len(rows),
        "exported_rows": len(exported_rows),
        "stats": stats,
    }
    if args.summary:
        write_json(Path(args.summary).resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
