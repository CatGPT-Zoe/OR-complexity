"""Compute operational-complexity metrics for the gpt-5.6-sol annotations."""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path("/Users/zhaoguojiao/Downloads/WPS/AAA重要文件资料/RuihaoZhu_Cornell")
ANNO = ROOT / "OR-complexity/pilot/ai_annotations/gpt-5.6-sol"
sys.path.insert(0, str(ROOT / "OR-complexity/operational_complexity"))

from metrics.hypergraph_builder import validate_graph
from metrics.coupling_metrics import compute_metrics

# The new SOH-1.1 metric engine already computes H_all vs H_c internally.
# This helper now only normalizes annotations that omit coupling_active; it does
# not filter out unary obligations, because unary rules are part of H_all.
def ensure_flags(doc):
    for o in doc.get("obligations", []):
        o.setdefault("coupling_active", len(o.get("support_entity_ids", [])) >= 2)
    return doc


def extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None, "no opening brace found"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate), None
                except json.JSONDecodeError as exc:
                    return None, f"JSON decode failed: {exc}"
    return None, "unbalanced braces"


def flatten(prefix, d, out):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flatten(key, v, out)
        else:
            out[key] = v


rows = []
for f in sorted(ANNO.glob("*.txt")):
    name = f.stem
    doc, err = extract_json(f.read_text(encoding="utf-8"))
    if doc is None:
        print(f"== {name}: EXTRACT_FAIL {err}")
        continue
    doc = ensure_flags(doc)
    try:
        validate_graph(doc)
    except Exception as e:
        print(f"== {name}: GRAPH_INVALID {type(e).__name__}: {e}")
        continue
    met = compute_metrics(doc)
    flat = {}
    flatten("", met, flat)
    flat["instance_id"] = name
    flat["n_obligations_total"] = len(doc.get("obligations", []))
    flat["n_obligations_coupling"] = len([o for o in doc.get("obligations", []) if o.get("coupling_active", True) and len(o.get("support_entity_ids", [])) >= 2])
    rows.append(flat)
    print(f"== {name}: ok  vars={met['scale']['n_var_families']} "
          f"constraints={met['scale']['n_constr_families']} (coupling {flat['n_obligations_coupling']}/{flat['n_obligations_total']})")

if rows:
    keys = list(rows[0].keys())
    out_path = ROOT / "OR-complexity/pilot/gpt_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_path}")
else:
    print("\nNo valid annotations to compute.")