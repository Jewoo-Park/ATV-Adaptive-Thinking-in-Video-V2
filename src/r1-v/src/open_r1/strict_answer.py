import re
from dataclasses import asdict, dataclass
from typing import Optional


LETTERS = "ABCDEFGHIJ"
LETTER_PATTERN = re.compile(r"^[A-J]$")
ANSWER_BLOCK_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
ANSWER_TAG_ANY_RE = re.compile(r"</?ANSWER>", re.DOTALL)
ANSWER_TAG_CI_RE = re.compile(r"</?\s*answer\s*>", re.IGNORECASE)
FINAL_ANSWER_PATTERN = re.compile(r"<ANSWER>\s*([A-J])\s*</ANSWER>\s*$", re.DOTALL)
STRICT_GT_PATTERN = re.compile(r"^\s*<ANSWER>\s*([A-J])\s*</ANSWER>\s*$", re.DOTALL)

LENGTH_REASONING_TAGS = ("DIRECT", "COT", "LONG_COT")
PERSPECTIVE_REASONING_TAGS = ("ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL")


@dataclass(frozen=True)
class StrictAnswerResult:
    pred_letter: Optional[str]
    format_ok: bool
    reasoning_tag: Optional[str]
    reasoning_text: Optional[str]
    malformed_type: Optional[str]
    parsed_strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


def format_answer(letter: str) -> str:
    normalized = str(letter or "").strip()
    if not LETTER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid MCQ answer letter: {letter!r}")
    return f"<ANSWER>{normalized}</ANSWER>"


def normalize_gt_letter(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    match = STRICT_GT_PATTERN.fullmatch(raw)
    if match:
        return match.group(1)
    if LETTER_PATTERN.fullmatch(raw):
        return raw
    prefixed = re.fullmatch(r"(?:answer|option|choice)\s*[:：]?\s*([A-J])", raw, re.IGNORECASE)
    if prefixed:
        return prefixed.group(1).upper()
    return None


def extract_strict_final_answer(text: str, task_type: str) -> Optional[str]:
    result = parse_strict_output(text, task_type=task_type)
    return result.pred_letter if result.format_ok else None


def _allowed_reasoning_tags(task_type: str) -> tuple[str, ...]:
    normalized = str(task_type or "length").strip().lower()
    if normalized == "length":
        return LENGTH_REASONING_TAGS
    if normalized == "perspective":
        return PERSPECTIVE_REASONING_TAGS
    raise ValueError("task_type must be one of: length, perspective")


def _strategy_for_tag(task_type: str, tag: Optional[str]) -> str:
    normalized = str(task_type or "length").strip().lower()
    if normalized == "length":
        return {
            "DIRECT": "direct",
            "COT": "cot",
            "LONG_COT": "long_cot",
        }.get(str(tag or ""), "invalid")
    if normalized == "perspective":
        return {
            "ABSTRACT": "abstract",
            "TEMPORAL": "temporal",
            "SPATIOTEMPORAL": "spatiotemporal",
        }.get(str(tag or ""), "invalid")
    raise ValueError("task_type must be one of: length, perspective")


def _classify_answer_malformed(text: str) -> Optional[str]:
    exact_blocks = list(ANSWER_BLOCK_RE.finditer(text))
    exact_tag_count = len(ANSWER_TAG_ANY_RE.findall(text))
    ci_tag_count = len(ANSWER_TAG_CI_RE.findall(text))
    if not exact_blocks:
        return "malformed_answer_tag" if ci_tag_count else "no_answer_tag"
    if len(exact_blocks) > 1 or exact_tag_count > 2:
        return "multiple_answers"

    block = exact_blocks[0]
    answer_text = block.group(1).strip()
    if not LETTER_PATTERN.fullmatch(answer_text):
        if len(answer_text) == 1:
            return "invalid_letter"
        return "malformed_answer_tag"

    trailing = text[block.end() :].strip()
    if trailing:
        return "extra_text_after_answer"
    return None


def parse_strict_output(text: str, task_type: str = "length") -> StrictAnswerResult:
    raw = str(text or "")
    normalized_task_type = str(task_type or "length").strip().lower()
    malformed = _classify_answer_malformed(raw)
    final_match = FINAL_ANSWER_PATTERN.search(raw)
    pred_letter = final_match.group(1) if final_match else None
    if malformed is not None:
        return StrictAnswerResult(
            pred_letter=None,
            format_ok=False,
            reasoning_tag=None,
            reasoning_text=None,
            malformed_type=malformed,
            parsed_strategy="invalid",
        )
    if pred_letter is None:
        return StrictAnswerResult(
            pred_letter=None,
            format_ok=False,
            reasoning_tag=None,
            reasoning_text=None,
            malformed_type="invalid_structure",
            parsed_strategy="invalid",
        )

    answer_only = re.fullmatch(r"\s*<ANSWER>\s*([A-J])\s*</ANSWER>\s*", raw, re.DOTALL)
    if answer_only:
        return StrictAnswerResult(
            pred_letter=None,
            format_ok=False,
            reasoning_tag=None,
            reasoning_text=None,
            malformed_type=(
                "missing_perspective_reasoning_tag"
                if normalized_task_type == "perspective"
                else "missing_length_reasoning_tag"
            ),
            parsed_strategy="invalid",
        )

    for tag in _allowed_reasoning_tags(task_type):
        pattern = rf"\s*<{tag}>(.*?)</{tag}>\s*<ANSWER>\s*([A-J])\s*</ANSWER>\s*"
        match = re.fullmatch(pattern, raw, re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()
            if tag == "DIRECT" and reasoning_text != "None":
                break
            if not reasoning_text:
                break
            return StrictAnswerResult(
                pred_letter=pred_letter,
                format_ok=True,
                reasoning_tag=tag,
                reasoning_text=reasoning_text,
                malformed_type=None,
                parsed_strategy=_strategy_for_tag(task_type, tag),
            )

    return StrictAnswerResult(
        pred_letter=None,
        format_ok=False,
        reasoning_tag=None,
        reasoning_text=None,
        malformed_type="invalid_structure",
        parsed_strategy="invalid",
    )
