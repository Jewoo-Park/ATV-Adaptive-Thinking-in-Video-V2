#!/usr/bin/env python3
"""Normalize HF config.json for vLLM (Transformers 5.x merge exports).

Transformers 5 save_pretrained may emit configs where nested text_config lacks
``architectures``, causing vLLM to raise:
  TypeError: 'NoneType' object is not iterable

This script copies the v4-compatible config layout from a reference model dir
(typically the base instruct checkpoint or a prior merged export) while keeping
weights unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _needs_fix(cfg: dict[str, Any]) -> bool:
    if not cfg.get("architectures"):
        return True
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and not tc.get("architectures"):
        return True
    if str(cfg.get("transformers_version", "")).startswith("5."):
        return True
    return False


def fix_one(model_dir: Path, reference_dir: Path) -> bool:
    cfg_path = model_dir / "config.json"
    ref_path = reference_dir / "config.json"
    if not cfg_path.is_file():
        print(f"[skip] no config.json: {model_dir}", file=sys.stderr)
        return False
    if not ref_path.is_file():
        print(f"[skip] no reference config: {ref_path}", file=sys.stderr)
        return False

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return False
    if not _needs_fix(cfg):
        print(f"[ok] already vLLM-compatible: {cfg_path}")
        return False

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    if not isinstance(ref, dict):
        return False

    bak = cfg_path.with_suffix(".json.bak_tf5")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[ok] backup: {bak}")

    fixed = dict(ref)
    fixed["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    tc = fixed.get("text_config")
    if isinstance(tc, dict):
        tc = dict(tc)
        tc["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        fixed["text_config"] = tc

    cfg_path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] patched config for vLLM: {cfg_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Patch HF config.json for vLLM compatibility.")
    p.add_argument("--reference", type=str, required=True, help="Reference model dir (e.g. base instruct)")
    p.add_argument("model_dirs", nargs="+", help="Model dirs to patch")
    args = p.parse_args()
    ref = Path(args.reference).resolve()
    n = 0
    for d in args.model_dirs:
        if fix_one(Path(d).resolve(), ref):
            n += 1
    print(f"[done] patched={n} reference={ref}")


if __name__ == "__main__":
    main()
