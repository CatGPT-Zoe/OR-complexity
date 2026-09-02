"""Command-line interface: compute metrics from a semantic hypergraph JSON."""

from __future__ import annotations

import argparse
import json
import sys

from .coupling_metrics import compute_metrics
from .hypergraph_builder import validate_graph


def _load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute operational complexity metrics from a SOH-1.0 hypergraph JSON."
    )
    parser.add_argument("input", nargs="+", help="Path(s) to SOH-1.0 JSON file(s)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the graph, do not compute metrics.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args(argv)

    for path in args.input:
        try:
            graph = _load_graph(path)
            validate_graph(graph)
            if args.validate_only:
                print(f"{path}: VALID")
                continue
            result = compute_metrics(graph)
            if len(args.input) == 1:
                print(json.dumps(result, indent=2, ensure_ascii=False) if args.pretty else json.dumps(result, ensure_ascii=False))
            else:
                print(f"{path}: {json.dumps(result, ensure_ascii=False)}")
        except Exception as exc:  # noqa: BLE001 - CLI reports any error
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
