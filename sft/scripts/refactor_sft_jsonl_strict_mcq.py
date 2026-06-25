#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OPTION_LABEL_RE = re.compile(r"^\s*(?:\(?([A-J])\)?[\).:：])\s*(.+?)\s*$", re.DOTALL)
ANSWER_BLOCK_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
OLD_REASONING_TYPE_RE = re.compile(r"<REASONING_TYPE>\s*(.*?)\s*</REASONING_TYPE>", re.DOTALL)
OLD_REASONING_RE = re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.DOTALL)
PERSPECTIVE_TAGS = ("ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL")


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    parse_error = None
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                parse_error = f"line {line_no}: {type(exc).__name__}: {exc}"
                break
            row["_line"] = line_no
            rows.append(row)
    return rows, parse_error


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def parse_options_from_instruction(instruction: str) -> tuple[dict[str, str], str]:
    options: dict[str, str] = {}
    text = str(instruction or "")
    for line in text.splitlines():
        match = OPTION_LABEL_RE.match(line)
        if not match:
            continue
        letter = match.group(1).upper()
        if letter in options:
            return {}, "skip_non_mcq_duplicate_option"
        options[letter] = re.sub(r"\s+", " ", match.group(2).strip())
    if len(options) < 2:
        return {}, "skip_non_mcq_no_options"
    return options, ""


def extract_answer_text(row: dict[str, Any]) -> str:
    for key in ("answer_letter", "gold_answer", "answer", "solution"):
        value = str(row.get(key) or "").strip()
        if value:
            match = ANSWER_BLOCK_RE.fullmatch(value)
            return match.group(1).strip() if match else value
    output = str(row.get("output") or "").strip()
    match = ANSWER_BLOCK_RE.search(output)
    return match.group(1).strip() if match else ""


def map_answer_to_letter(answer_text: str, options: dict[str, str]) -> tuple[str, str]:
    raw = str(answer_text or "").strip()
    if not raw:
        return "", "skip_missing_answer"
    direct = re.fullmatch(r"(?:answer|option|choice)?\s*[:：]?\s*([A-J])", raw, re.IGNORECASE)
    if direct:
        letter = direct.group(1).upper()
        if letter in options:
            return letter, ""
        return "", "skip_answer_letter_not_in_options"
    labeled = OPTION_LABEL_RE.match(raw)
    if labeled:
        letter = labeled.group(1).upper()
        if letter in options:
            return letter, ""
        return "", "skip_answer_letter_not_in_options"

    raw_key = normalize_match(raw)
    matches = [
        letter
        for letter, option_text in options.items()
        if raw_key in {normalize_match(option_text), normalize_match(f"{letter}. {option_text}")}
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return "", "skip_ambiguous_answer_mapping"
    return "", "skip_unmappable_answer"


def answer_block(letter: str) -> str:
    return f"<ANSWER>{letter}</ANSWER>"


def extract_tag(output: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", output, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def convert_length_row(row: dict[str, Any], answer: str) -> tuple[dict[str, Any] | None, str]:
    output = str(row.get("output") or "").strip()
    reasoning_depth = str(row.get("reasoning_depth") or "").strip().upper()
    if not reasoning_depth:
        if extract_tag(output, "LONG_COT") is not None:
            reasoning_depth = "LONG_COT"
        elif extract_tag(output, "COT") is not None:
            reasoning_depth = "COT"
        else:
            reasoning_depth = "ANSWER"
    if reasoning_depth == "ANSWER":
        reasoning = "<DIRECT>None</DIRECT>"
    elif reasoning_depth == "COT":
        text = extract_tag(output, "COT")
        if text is None:
            return None, "skip_missing_cot"
        reasoning = f"<COT>{text}</COT>"
    elif reasoning_depth == "LONG_COT":
        text = extract_tag(output, "LONG_COT")
        if text is None:
            return None, "skip_missing_long_cot"
        reasoning = f"<LONG_COT>{text}</LONG_COT>"
    else:
        return None, "skip_invalid_reasoning_depth"

    new_row = dict(row)
    new_row["answer"] = answer
    new_row["answer_letter"] = answer[8]
    new_row["gold_answer"] = answer[8]
    new_row["reasoning"] = reasoning
    new_row["reasoning_depth"] = reasoning_depth
    new_row["task_type"] = "length"
    new_row["sft_mode"] = "length"
    new_row["output"] = f"{reasoning}\n{answer}" if reasoning else answer
    return new_row, ""


def convert_perspective_row(row: dict[str, Any], answer: str) -> tuple[dict[str, Any] | None, str]:
    output = str(row.get("output") or "").strip()
    reasoning_type = str(row.get("reasoning_type") or row.get("granularity_type") or "").strip().upper()
    old_type = OLD_REASONING_TYPE_RE.search(output)
    if old_type:
        reasoning_type = old_type.group(1).strip().upper()
    if reasoning_type not in PERSPECTIVE_TAGS:
        detected = [tag for tag in PERSPECTIVE_TAGS if extract_tag(output, tag) is not None]
        if len(detected) == 1:
            reasoning_type = detected[0]
        else:
            return None, "skip_invalid_reasoning_type"

    reasoning_text = extract_tag(output, reasoning_type)
    if reasoning_text is None:
        old_reasoning = OLD_REASONING_RE.search(output)
        reasoning_text = old_reasoning.group(1).strip() if old_reasoning else None
    if not reasoning_text:
        return None, "skip_missing_reasoning_trace"

    reasoning = f"<{reasoning_type}>{reasoning_text}</{reasoning_type}>"
    new_row = dict(row)
    new_row["answer"] = answer
    new_row["answer_letter"] = answer[8]
    new_row["gold_answer"] = answer[8]
    new_row["reasoning"] = reasoning
    new_row["reasoning_type"] = reasoning_type
    new_row["task_type"] = "perspective"
    new_row["sft_mode"] = "perspective"
    new_row["output"] = f"{reasoning}\n{answer}"
    return new_row, ""


def convert_rows(rows: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], Counter]:
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for row in rows:
        instruction = str(row.get("instruction") or row.get("question") or "").strip()
        if not instruction:
            stats["skip_missing_instruction"] += 1
            continue
        options, option_error = parse_options_from_instruction(instruction)
        if option_error:
            stats[option_error] += 1
            continue
        letter, answer_error = map_answer_to_letter(extract_answer_text(row), options)
        if answer_error:
            stats[answer_error] += 1
            continue
        answer = answer_block(letter)
        if mode == "length":
            new_row, error = convert_length_row(row, answer)
        else:
            new_row, error = convert_perspective_row(row, answer)
        if error:
            stats[error] += 1
            continue
        out.append(new_row)
        stats["kept_rows"] += 1
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactor prepared SFT JSONL into strict MCQ-only format.")
    parser.add_argument("--mode", required=True, choices=("length", "perspective"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    rows, parse_error = load_jsonl(input_path)
    converted, stats = convert_rows(rows, args.mode)
    dump_jsonl(output_path, converted)
    summary = {
        "mode": args.mode,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "input_rows_parsed": len(rows),
        "output_rows": len(converted),
        "removed_rows": len(rows) - len(converted),
        "parse_error": parse_error,
        "stats": dict(stats),
    }
    dump_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
