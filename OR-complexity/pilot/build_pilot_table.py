"""Pilot batch runner: compute metrics for many SOH-1.0 annotations, output a table.

Usage:
    PYTHONPATH=.. .venv/bin/python build_pilot_table.py INPUT_DIR [--out table.csv]

INPUT_DIR is scanned recursively for *.json files that look like SOH-1.0
semantic annotations. Each file's metrics are flattened into one CSV row,
matching the extraction-sheet goal:
    dataset -> instance -> semantic entities -> obligations -> metric vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "operational_complexity")
    ),
)

from metrics.coupling_metrics import compute_metrics  # noqa: E402
from metrics.hypergraph_builder import validate_graph  # noqa: E402


def is_soh(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        isinstance(doc, dict)
        and doc.get("schema_version") in ("SOH-1.0", "SOH-1.1")
        and doc.get("layer") == "semantic"
    )


def find_annotations(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            if is_soh(path):
                yield path


def flatten(row: dict) -> dict:
    """Flatten nested metric dict into a flat row dict for CSV."""
    out = {
        "dataset": row.get("dataset", ""),
        "instance_id": row.get("instance_id", ""),
    }
    scale = row.get("scale", {})
    local = row.get("local", {})
    entity = row.get("entity", {})
    obligation = row.get("obligation", {})
    out.update({f"scale.{k}": v for k, v in scale.items()})
    out.update({f"local.{k}": v for k, v in local.items()})
    for k, v in entity.items():
        if k == "treewidth":
            out["entity.treewidth_value"] = v.get("value")
            out["entity.treewidth_status"] = v.get("status")
        else:
            out[f"entity.{k}"] = v
    for k, v in obligation.items():
        if k == "treewidth":
            out["obligation.treewidth_value"] = v.get("value")
            out["obligation.treewidth_status"] = v.get("status")
        else:
            out[f"obligation.{k}"] = v
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="directory to scan for SOH-1.0/SOH-1.1 JSONs")
    parser.add_argument("--out", default=None, help="output CSV path (default: stdout)")
    args = parser.parse_args(argv)

    paths = list(find_annotations(args.input_dir))
    if not paths:
        print(f"No SOH-1.0/SOH-1.1 annotations found under {args.input_dir}", file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        try:
            validate_graph(doc)
            result = compute_metrics(doc)
            row = flatten(result)
            row["source_file"] = os.path.relpath(path, args.input_dir)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: ERROR {exc}", file=sys.stderr)
            continue

    if not rows:
        print("No annotations computed successfully.", file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
