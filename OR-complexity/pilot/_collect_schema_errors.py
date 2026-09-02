"""Collect ALL jsonschema validation errors across all annotations (not just first)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/zhaoguojiao/Downloads/WPS/AAA重要文件资料/RuihaoZhu_Cornell")
ANNO = ROOT / "OR-complexity/pilot/ai_annotations"
SCHEMA = ROOT / "OR-complexity/operational_complexity/schemas/semantic_hypergraph.schema.json"

import jsonschema


def extract_json(text):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
validator = jsonschema.Draft202012Validator(schema)

patterns = Counter()
per_file = {}
for model_dir in sorted(ANNO.iterdir()):
    if not model_dir.is_dir():
        continue
    for f in sorted(model_dir.glob("*.txt")):
        doc = extract_json(f.read_text(encoding="utf-8"))
        if doc is None:
            print(f"{model_dir.name}/{f.stem}: EXTRACT_FAIL")
            continue
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        per_file[f"{model_dir.name}/{f.stem}"] = len(errors)
        seen = set()
        for e in errors:
            key = " / ".join(str(p) for p in e.path)
            msg = e.message[:120]
            if key in seen:
                continue
            seen.add(key)
            patterns[(key, msg)] += 1
            print(f"  {model_dir.name}/{f.stem}: path=[{key}] {msg}")

print("\n=== summary: distinct error patterns ===")
for (key, msg), cnt in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  x{cnt:2d}  [{key}]  {msg}")

print("\n=== error count per file ===")
for k, v in sorted(per_file.items()):
    print(f"  {v:2d}  {k}")
