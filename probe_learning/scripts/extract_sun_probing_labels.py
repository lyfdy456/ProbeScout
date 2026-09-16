"""Extract clean probing labels from raw SUN VQA results JSONL."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import (
    TASKS_DIR, SNOW_ROAD_TASK, SUN_TASK_REGISTRY, SUN_PROCESSED_DIR, get_sun_task,
)


def clean_block(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return ""
    block = raw[start : end + 1]
    block = re.sub(r"^\d+$", "", block, flags=re.MULTILINE)
    block = re.sub(r"^编辑$", "", block, flags=re.MULTILINE)
    block = re.sub(r"\n\s*\n", "\n", block)
    return block


def parse_full(block: str, attrs: list[str]) -> dict | None:
    try:
        obj = json.loads(block)
        return obj.get("attributes")
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_regex(block: str, attrs: list[str]) -> dict:
    result = {}
    for attr in attrs:
        pattern = re.escape(attr) + r'["\s:{\n]*"present":\s*([\d"]+|"unknown")'
        m = re.search(pattern, block, re.DOTALL)
        if m:
            val = m.group(1).strip('"')
            if val == "unknown":
                result[attr] = None
            else:
                try:
                    result[attr] = int(val)
                except ValueError:
                    result[attr] = None
    return result


def extract_labels(raw_answer: str, attrs: list[str]) -> dict:
    block = clean_block(raw_answer)
    if not block:
        return {attr: None for attr in attrs}

    parsed = parse_full(block, attrs)
    if parsed is not None:
        result = {}
        for attr in attrs:
            info = parsed.get(attr)
            if info is None:
                result[attr] = None
                continue
            val = info.get("present")
            if val == "unknown" or val is None:
                result[attr] = None
            else:
                result[attr] = int(val)
        return result

    regex_result = parse_regex(block, attrs)
    return {attr: regex_result.get(attr) for attr in attrs}


def extract_label_scores(raw_answer: str, attrs: list[str]) -> dict:
    """Continuous P(present) per attribute using the VQA confidence:
    present=1 -> conf ; present=0 -> 1-conf ; unknown/None -> None.
    Falls back to the binary 0/1 when confidence is missing or unparsable.
    Used as a ranking score so the full-VQA ceiling gets a meaningful AP.
    """
    binary = extract_labels(raw_answer, attrs)
    block = clean_block(raw_answer)
    parsed = parse_full(block, attrs) if block else None

    out = {}
    for attr in attrs:
        present = binary.get(attr)
        if present is None:
            out[attr] = None
            continue
        conf = None
        if isinstance(parsed, dict):
            info = parsed.get(attr)
            if isinstance(info, dict):
                c = info.get("confidence")
                try:
                    conf = float(c)
                except (TypeError, ValueError):
                    conf = None
        if conf is None:
            out[attr] = float(present)          # no confidence -> fall back to 0/1
        else:
            conf = min(max(conf, 0.0), 1.0)
            out[attr] = conf if present == 1 else (1.0 - conf)
    return out


def build_bare_to_full_map() -> dict[str, str]:
    """Map bare SUN filename (sun_xxx.jpg) → double-underscore path (a__abbey__sun_xxx.jpg)."""
    records = pd.read_csv(SUN_PROCESSED_DIR / "records.csv")
    mapping = {}
    for rel_path in records["relative_path"]:
        bare = os.path.basename(rel_path)
        full = rel_path.replace("/", "__")
        mapping[bare] = full
    return mapping


def normalize_image_name(name: str, bare_map: dict[str, str] | None) -> str:
    """If *name* is a bare filename and we have a mapping, convert it."""
    if bare_map and "__" not in name and name in bare_map:
        return bare_map[name]
    return name


def main():
    parser = argparse.ArgumentParser(description="Extract VQA probing labels for a SUN task")
    parser.add_argument(
        "--task", default=SNOW_ROAD_TASK,
        choices=list(SUN_TASK_REGISTRY.keys()),
        help="Task name (registered in paths.py)",
    )
    args = parser.parse_args()

    cfg = get_sun_task(args.task)
    attrs = cfg["attributes"]
    raw_jsonl = cfg["raw_jsonl"]

    raw_path = TASKS_DIR / args.task / "qa" / raw_jsonl
    out_path = TASKS_DIR / args.task / "vqa_probing_labels.jsonl"

    bare_map = build_bare_to_full_map()
    remapped = 0

    results = []
    total = ok = partial = failed = 0
    with open(raw_path) as f:
        for line in f:
            total += 1
            entry = json.loads(line)
            labels = extract_labels(entry["answer"], attrs)

            non_null = sum(1 for v in labels.values() if v is not None)
            if non_null == len(attrs):
                ok += 1
            elif non_null > 0:
                partial += 1
            else:
                failed += 1

            original = entry["image"]
            normalised = normalize_image_name(original, bare_map)
            if normalised != original:
                remapped += 1
            row = {"image": normalised}
            row.update(labels)
            results.append(row)

    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Task: {args.task}")
    print(f"Total: {total}")
    print(f"  Full parse: {ok}")
    print(f"  Partial (regex fallback): {partial}")
    print(f"  Failed (all null): {failed}")
    if remapped:
        print(f"  Remapped bare filenames: {remapped}")
    print(f"Saved to {out_path}")

    for attr in attrs:
        pos = sum(1 for r in results if r[attr] == 1)
        neg = sum(1 for r in results if r[attr] == 0)
        null = sum(1 for r in results if r[attr] is None)
        print(f"  {attr}: pos={pos} neg={neg} null={null}")


if __name__ == "__main__":
    main()
