#!/usr/bin/env python3
"""Lightweight strict validator for real GRPO JSONL inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


LETTER_RE = re.compile(r"^[A-J]$")
ANSWER_TAG_RE = re.compile(r"^\s*<ANSWER>\s*([A-J])\s*</ANSWER>\s*$", re.DOTALL)


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def _extract_answer_letter(row: dict) -> str | None:
    candidates = [
        row.get("solution"),
        row.get("answer"),
        row.get("gold_answer"),
        row.get("gt_answer"),
    ]
    for raw in candidates:
        if raw is None:
            continue
        text = str(raw).strip()
        m = ANSWER_TAG_RE.fullmatch(text)
        if m:
            return m.group(1)
        if LETTER_RE.fullmatch(text):
            return text
    return None


def _check_media_paths(row: dict, base_dir: Path, errors: list[str], line_no: int) -> None:
    def _exists(p: str) -> bool:
        path = Path(p)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path.exists()

    frames = row.get("frames")
    if frames is not None:
        if not isinstance(frames, list) or len(frames) == 0:
            errors.append(f"L{line_no}: `frames` must be a non-empty list when present")
        else:
            for idx, fp in enumerate(frames):
                if _is_empty(fp):
                    errors.append(f"L{line_no}: `frames[{idx}]` is empty")
                elif not _exists(str(fp)):
                    errors.append(f"L{line_no}: frame path does not exist: {fp}")

    for key in ("image", "image_path", "video_path"):
        val = row.get(key)
        if val is not None and not _is_empty(val) and not _exists(str(val)):
            errors.append(f"L{line_no}: `{key}` path does not exist: {val}")


def validate_jsonl(path: Path, mode: str, *, skip_media_check: bool = False) -> tuple[int, list[str]]:
    errors: list[str] = []
    total = 0
    base_dir = path.resolve().parent

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"L{i}: invalid JSON ({e})")
                continue

            if not isinstance(row, dict):
                errors.append(f"L{i}: each row must be a JSON object")
                continue

            for k in ("video_id", "question_id", "question_category"):
                if _is_empty(row.get(k)):
                    errors.append(f"L{i}: missing/empty required key `{k}`")

            # Repository canonical schema: problem + solution (+ frames)
            problem = row.get("problem")
            if _is_empty(problem):
                # fallback schema guard: question/options/answer
                q = row.get("question")
                opts = row.get("options")
                ans = row.get("answer") or row.get("gold_answer") or row.get("solution")
                if _is_empty(q) or not isinstance(opts, list) or len(opts) < 2 or _is_empty(ans):
                    errors.append(
                        f"L{i}: requires `problem`/`solution` or (`question`,`options`,`answer`)"
                    )

            answer_letter = _extract_answer_letter(row)
            if answer_letter is None:
                errors.append(
                    f"L{i}: answer must be A-J or strict `<ANSWER>X</ANSWER>` in solution/answer/gold_answer"
                )

            if mode == "perspective" and answer_letter is not None:
                # Perspective training rows should not be empty or malformed; strict strategy reasoning
                # is enforced at generation/parser time, but we still require task marker consistency.
                task_type = str(row.get("task_type") or row.get("reasoning_task_type") or "").strip().lower()
                if task_type and task_type != "perspective":
                    errors.append(
                        f"L{i}: perspective mode but task marker is `{task_type}` (expected perspective)"
                    )
            if mode == "length":
                task_type = str(row.get("task_type") or row.get("reasoning_task_type") or "").strip().lower()
                if task_type and task_type != "length":
                    errors.append(
                        f"L{i}: length mode but task marker is `{task_type}` (expected length)"
                    )

            if not skip_media_check:
                _check_media_paths(row, base_dir, errors, i)

    if total == 0:
        errors.append("no non-empty JSONL rows found")
    return total, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate GRPO real-training JSONL schema strictly.")
    ap.add_argument("--input", required=True, help="Path to JSONL file")
    ap.add_argument(
        "--mode",
        required=True,
        choices=("length", "perspective"),
        help="Target reasoning mode for this dataset",
    )
    ap.add_argument(
        "--skip-media-check",
        action="store_true",
        help="Skip per-frame path.exists() checks (avoids millions of NFS stats on large JSONL).",
    )
    args = ap.parse_args()

    path = Path(args.input).expanduser().resolve()
    if not path.exists():
        print(f"[schema] ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(2)

    total, errors = validate_jsonl(path, args.mode, skip_media_check=args.skip_media_check)
    print(f"[schema] file={path}")
    print(f"[schema] mode={args.mode}")
    print(f"[schema] rows_checked={total}")
    print(f"[schema] skip_media_check={args.skip_media_check}")
    if errors:
        print(f"[schema] INVALID: {len(errors)} issue(s)", file=sys.stderr)
        for msg in errors[:200]:
            print(f"[schema] {msg}", file=sys.stderr)
        if len(errors) > 200:
            print(f"[schema] ... and {len(errors) - 200} more issue(s)", file=sys.stderr)
        sys.exit(1)
    print("[schema] VALID")


if __name__ == "__main__":
    main()
