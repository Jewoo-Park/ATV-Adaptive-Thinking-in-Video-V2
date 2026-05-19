#!/usr/bin/env python3
"""Create a tiny generated PERSPECTIVE GRPO video-QA JSONL with valid PNG frames."""

import argparse
import json
import struct
import zlib
from pathlib import Path


ANSWERS = "ABCD"
PERSPECTIVES = ("ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: Path, rgb: tuple[int, int, int], size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    data += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    data += _png_chunk(b"IEND", b"")
    path.write_bytes(data)


def build_problem(idx: int, answer: str, perspective: str) -> str:
    return (
        f"Question: In dummy PERSPECTIVE clip {idx}, which option matches the {perspective.lower()} cue?\n"
        "Options:\n"
        f"A. Object category cue{' (correct)' if answer == 'A' else ''}\n"
        f"B. Before/after timing cue{' (correct)' if answer == 'B' else ''}\n"
        f"C. Moving object location cue{' (correct)' if answer == 'C' else ''}\n"
        f"D. Distractor cue{' (correct)' if answer == 'D' else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/generated_dummy_grpo_smoke/perspective")
    parser.add_argument("--num-examples", type=int, default=180)
    parser.add_argument("--num-frames", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    frames_root = out_dir / "frames"
    jsonl_path = out_dir / "perspective_grpo_dummy.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    palette = {
        "ABSTRACT": (155, 75, 190),
        "TEMPORAL": (40, 150, 190),
        "SPATIOTEMPORAL": (215, 115, 55),
    }
    for idx in range(args.num_examples):
        answer = ANSWERS[idx % len(ANSWERS)]
        perspective = PERSPECTIVES[idx % len(PERSPECTIVES)]
        sample_dir = frames_root / f"sample_{idx:04d}"
        frame_paths = []
        for frame_idx in range(args.num_frames):
            color = palette[perspective]
            if frame_idx == 1:
                color = tuple(min(255, c + 35) for c in color)
            frame_path = sample_dir / f"frame_{frame_idx:03d}.png"
            write_png(frame_path, color)
            frame_paths.append(str(frame_path.relative_to(out_dir)))
        rows.append(
            {
                "video_id": f"dummy_perspective_{idx:04d}.mp4",
                "question_id": idx,
                "question_category": "dummy_perspective",
                "problem": build_problem(idx, answer, perspective),
                "frames": frame_paths,
                "solution": f"<ANSWER>{answer}</ANSWER>",
                "task_type": "perspective",
                "reasoning_task_type": "perspective",
                "reasoning_type": perspective,
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(jsonl_path)


if __name__ == "__main__":
    main()
