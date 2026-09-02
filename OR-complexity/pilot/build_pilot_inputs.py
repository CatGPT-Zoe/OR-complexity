#!/usr/bin/env python3
"""Build pilot annotator inputs + anchor manifest from the 5 source repos.

Supports both the curated 12-anchor pilot protocol and arbitrary problem
selection by dataset / problem id. Each source database registers a loader
with two phases: discover() (cheap enumeration, no data rendering) and
render(key) (lazy: reads external CSV/JSON data into inline tables). Adding a
new database = adding one loader to the SOURCES registry.

OPTEngine is split into two sources:
- `optengine` for canonical test_data/canonical rows
- `optengine_augmented` for perturbations/constraint_augmentation.jsonl

Examples (run from the workspace root):
    # build EVERY problem from all 5 datasets (default)
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py"

    # curated 12-anchor pilot contrast set (hand-picked, human-verified)
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --curated

    # all problems from selected datasets (LEAN and MIPLIB-NL)
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --datasets lean,miplib

    # specific anchors (substring match on anchor id; auto-resolves dataset)
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --ids IndustryOR_50
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --ids LEAN_TP8,MIPLIB_NL_flugpl

    # list discoverable problems without writing outputs
    .venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --list

Outputs:
    pilot/inputs/<anchor_id>.json        (USER INPUT SCHEMA from semantic_annotator_v1.txt)
    pilot/anchor_manifest.csv            (selection: dataset + difficulty + contrast role)
    pilot/anchor12_manifest.csv          (same, for the --curated pilot subset)

OPTEngine problems include canonical.jsonl plus the constraint_augmentation
perturbations: rows with constraint_type == "augmented" are written to
inputs/OPTEngine-augmented/ with a "_aug" suffix in the anchor id (e.g.
OPTEngine_Inventory_5_aug); the original rows of that file are skipped because
they duplicate canonical instances.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # RuihaoZhu_Cornell
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "inputs"
AUG_DIR = OUT_DIR / "OPTEngine-augmented"
MANIFEST_PATH = SCRIPT_DIR / "anchor12_manifest.csv"   # curated 12-anchor pilot subset
MANIFEST_ALL_PATH = SCRIPT_DIR / "anchor_manifest.csv"  # full discovery set

INDUSTRYOR_JSON = ROOT / "IndustryOR.json"
NL4OPT_JSON = ROOT / "NL4OPT" / "NL4OPT_with_optimal_solution.json"
OPTENGINE_CANON = ROOT / "OR-complexity" / "OPTEngine-main" / "test_data" / "canonical" / "canonical.jsonl"
OPTENGINE_AUG = ROOT / "OR-complexity" / "OPTEngine-main" / "test_data" / "perturbations" / "constraint_augmentation.jsonl"
MIPLIB_NL_DIR = ROOT / "OR-complexity" / "MIPLIB-NL-main" / "dataset"
LEAN_ROOT = ROOT / "OR-complexity" / "lean-llm-opt-main"
LEAN_DIR = LEAN_ROOT / "Test_Dataset" / "Large-scale-or"
LEAN_101 = LEAN_DIR / "Large-scale-or-101.csv"


# ---------------------------------------------------------------------------
# shared rendering helpers
# ---------------------------------------------------------------------------

def read_csv_table(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def md_table(headers: list, rows: list) -> str:
    """Render a markdown table for inline presentation."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = list(r.values()) if isinstance(r, dict) else list(r)
        out.append("| " + " | ".join(str(v) for v in cells) + " |")
    return "\n".join(out)


def clean_headers(headers: list) -> list:
    """Replace empty column headers (row-label columns) with a readable name."""
    return [h if (h is not None and h.strip()) else "row_label" for h in headers]


def csv_to_table(path: Path, name: str | None = None) -> dict | None:
    """Read a CSV into a parameter-table dict; None if the file is empty."""
    rows = read_csv_table(path)
    if not rows:
        return None
    headers = clean_headers(list(rows[0].keys()))
    # normalize ragged rows: ensure every row has the same cell count as headers
    norm = []
    for r in rows:
        cells = list(r.values()) if isinstance(r, dict) else list(r)
        norm.append([cells[i] if i < len(cells) else "" for i in range(len(headers))])
    indices = [h for h in headers if h not in ("parameter", "parameter_name")]
    if len(norm) == 1 and len(headers) > 2:
        indices = []  # transposed single-row parameter table
    return {
        "name": name or path.stem,
        "indices": indices,
        "unit": "",
        "values": md_table(headers, norm),
    }


def ext_entry(path: Path, table_name: str | None = None) -> dict:
    """External-file manifest entry (path relative to the workspace root)."""
    return {
        "path": str(path.relative_to(ROOT)),
        "role": "parameter_table",
        "table": table_name or path.stem,
    }


# ---------------------------------------------------------------------------
# dataset sources (one loader per database; extend SOURCES for new databases)
#
# Each loader exposes:
#   discover() -> list of {"key", "anchor_id", "dataset_label"}  (cheap)
#   render(key) -> {"problem_text", "tables", "external_files", "sets", "notes"}
# ---------------------------------------------------------------------------

SOURCES = {}


@lru_cache(maxsize=1)
def load_industryor_docs() -> list[dict]:
    return json.loads(INDUSTRYOR_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_nl4opt_docs() -> list[dict]:
    return [
        json.loads(line)
        for line in NL4OPT_JSON.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@lru_cache(maxsize=1)
def load_optengine_docs() -> list[dict]:
    return [
        json.loads(line)
        for line in OPTENGINE_CANON.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@lru_cache(maxsize=1)
def load_lean_rows() -> list[dict]:
    return list(csv.DictReader(open(LEAN_101, newline="", encoding="utf-8")))


def resolve_lean_path(rel: str) -> Path | None:
    """Resolve a Dataset_address line to an existing file path.

    Addresses in Large-scale-or-101.csv may or may not include the
    Test_Dataset/Large-scale-or/ prefix; try both bases.
    """
    rel = rel.strip().lstrip("/")
    for base in (LEAN_DIR, LEAN_ROOT):
        cand = (base / rel).resolve()
        if cand.exists():
            return cand
    return None


def lean_instance_dir(row: dict) -> Path | None:
    """Directory holding the CSV data for one LEAN problem row."""
    addrs = [l.strip() for l in row["Dataset_address"].splitlines() if l.strip()]
    if not addrs:
        return None
    p = resolve_lean_path(addrs[0])
    return p.parent if p is not None else None


# --- industryor ---

def discover_industryor() -> list[dict]:
    out = []
    for r in load_industryor_docs():
        iid = str(r["id"])
        out.append({"key": iid, "anchor_id": f"IndustryOR_{iid}", "dataset_label": "IndustryOR"})
    return out


def render_industryor(key: str) -> dict:
    docs = {str(r["id"]): r for r in load_industryor_docs()}
    r = docs[key]
    return {"problem_text": r["en_question"], "tables": [], "external_files": [], "sets": [], "notes": ""}


# --- nl4opt ---

def discover_nl4opt() -> list[dict]:
    out = []
    for idx, r in enumerate(load_nl4opt_docs()):
        out.append({"key": str(idx), "anchor_id": f"NL4OPT_{idx:03d}", "dataset_label": "NL4OPT"})
    return out


def render_nl4opt(key: str) -> dict:
    rows = load_nl4opt_docs()
    r = rows[int(key)]
    return {"problem_text": r["en_question"], "tables": [], "external_files": [], "sets": [], "notes": ""}


# --- optengine (canonical.jsonl has 1810 rows; class+size is not unique) ---

@lru_cache(maxsize=1)
def _optengine_index() -> tuple[dict, dict]:
    """Return (discovery list, {key: row}) for OPTEngine canonical problems."""
    seen = {}
    discovery = []
    index = {}
    for r in load_optengine_docs():
        cs = f"{r['problem_class']}_{r['size']}"
        idx = seen.get(cs, 0)
        seen[cs] = idx + 1
        key = cs if idx == 0 else f"{cs}_{idx}"
        discovery.append({"key": key, "anchor_id": f"OPTEngine_{key}", "dataset_label": "OPTEngine"})
        index[key] = r
    return discovery, index


@lru_cache(maxsize=1)
def _optengine_aug_list() -> list[dict]:
    """Augmented constraint-perturbation rows from constraint_augmentation.jsonl.

    Only rows with constraint_type == 'augmented' are kept; canonical.jsonl
    already supplies the original rows, so they are not duplicated here.
    """
    out = []
    seen = Counter()
    for line in OPTENGINE_AUG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("constraint_type") != "augmented":
            continue
        cs = f"{r['problem_class']}_{r['size']}"
        seen[cs] += 1
        suffix = "" if seen[cs] == 1 else f"_{seen[cs]-1}"
        key = f"{cs}{suffix}_aug"
        out.append({
            "key": key,
            "question": r["question"],
            "problem_class": r["problem_class"],
            "size": r["size"],
        })
    return out


def discover_optengine() -> list[dict]:
    discovery, _ = _optengine_index()
    for r in _optengine_aug_list():
        discovery.append({
            "key": r["key"],
            "anchor_id": f"OPTEngine_{r['key']}",
            "dataset_label": "OPTEngine",
            "augmented": True,
        })
    return discovery


def render_optengine(key: str) -> dict:
    canon = _optengine_index()[1]
    if key in canon:
        return {"problem_text": canon[key]["question"], "tables": [], "external_files": [], "sets": [], "notes": ""}
    for r in _optengine_aug_list():
        if r["key"] == key:
            return {"problem_text": r["question"], "tables": [], "external_files": [], "sets": [], "notes": ""}
    raise KeyError(key)


# --- miplib (external data: instance.json + data/*.csv) ---

def discover_miplib() -> list[dict]:
    out = []
    for inst_dir in sorted(MIPLIB_NL_DIR.iterdir()):
        if not (inst_dir / "instance.json").exists():
            continue
        out.append({"key": inst_dir.name, "anchor_id": f"MIPLIB_NL_{inst_dir.name}", "dataset_label": "MIPLIB-NL"})
    return out


def render_miplib_problem(inst_dir: Path) -> dict:
    """Assemble problem text + inline parameter tables for a MIPLIB-NL instance."""
    inst = json.loads((inst_dir / "instance.json").read_text(encoding="utf-8"))
    params = inst.get("parameters", {})
    text = inst["abstract_problem"]
    for k, v in params.items():
        text = text.replace("{" + k + "}", str(v))

    tables = []
    external = []
    seen = set()
    for name, spec in inst.get("files", {}).items():
        path = (inst_dir / spec["path"]) if os.path.isabs(spec["path"]) else (inst_dir / spec["path"])
        if not path.exists():
            continue
        t = csv_to_table(path, name=name)
        if t is None:
            continue
        tables.append(t)
        external.append(ext_entry(path, name))
        seen.add(path)
    data_dir = inst_dir / "data"
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            if path in seen:
                continue
            t = csv_to_table(path)
            if t is None:
                continue
            tables.append(t)
            external.append(ext_entry(path))
    return {"text": text, "tables": tables, "external_files": external}


def render_miplib(key: str) -> dict:
    r = render_miplib_problem(MIPLIB_NL_DIR / key)
    return {"problem_text": r["text"], "tables": r["tables"], "external_files": r["external_files"], "sets": [], "notes": ""}


# --- lean (external data: all CSVs in the instance directory) ---

def discover_lean() -> list[dict]:
    out = []
    for r in load_lean_rows():
        inst_dir = lean_instance_dir(r)
        if inst_dir is None:
            continue
        out.append({"key": inst_dir.name, "anchor_id": f"LEAN_{inst_dir.name}", "dataset_label": "LEAN-LLM-OPT"})
    return out


def render_lean(key: str) -> dict:
    for r in load_lean_rows():
        inst_dir = lean_instance_dir(r)
        if inst_dir is None or inst_dir.name != key:
            continue
        tables = []
        external = []
        for path in sorted(inst_dir.glob("*.csv")):
            t = csv_to_table(path)
            if t is None:
                continue
            tables.append(t)
            external.append(ext_entry(path))
        return {"problem_text": r["Query"].strip(), "tables": tables, "external_files": external, "sets": [], "notes": ""}
    raise KeyError(key)


for _key, _label, _prefix, _disc, _rend in [
    ("industryor", "IndustryOR", "IndustryOR", discover_industryor, render_industryor),
    ("nl4opt", "NL4OPT", "NL4OPT", discover_nl4opt, render_nl4opt),
    ("optengine", "OPTEngine", "OPTEngine", discover_optengine, render_optengine),
    ("miplib", "MIPLIB-NL", "MIPLIB_NL", discover_miplib, render_miplib),
    ("lean", "LEAN-LLM-OPT", "LEAN", discover_lean, render_lean),
]:
    SOURCES[_key] = {
        "label": _label,
        "prefix": _prefix,
        "discover": _disc,
        "render": _rend,
    }


# ---------------------------------------------------------------------------
# curated 12-anchor pilot protocol
# ---------------------------------------------------------------------------

CURATED = [
    {
        "dataset": "industryor", "key": "92",
        "anchor_id": "IndustryOR_Easy_092_bus_rental",
        "difficulty": "Easy", "contrast_role": "simple surface structure",
        "sets": [{"name": "bus_types", "description": "bus and minibus types"}],
        "tables": [
            {
                "name": "bus_params",
                "indices": ["bus_type"],
                "unit": "",
                "values": md_table(
                    ["bus_type", "seats", "rental_cost", "drivers_needed"],
                    [
                        ["bus", 50, 800, 1],
                        ["minibus", 40, 600, 1],
                    ],
                ),
            },
            {
                "name": "context",
                "indices": [],
                "unit": "",
                "values": md_table(
                    ["students", "available_drivers"],
                    [["400", "9"]],
                ),
            },
        ],
        "notes": "Two decision families (count per bus type); seat coverage, driver limit, and demand obligations.",
    },
    {
        "dataset": "industryor", "key": "77",
        "anchor_id": "IndustryOR_Easy_077_machine_assignment",
        "difficulty": "Easy", "contrast_role": "multi-rule / multi-entity",
        "sets": [
            {"name": "parts", "description": "the 10 parts"},
            {"name": "machines", "description": "machines A, B, C"},
        ],
        "tables": [],
        "notes": "Part-machine assignment + machine activation with setup costs + conditional (if-then) processing rules.",
    },
    {
        "dataset": "industryor", "key": "26",
        "anchor_id": "IndustryOR_Hard_026_shift_scheduling",
        "difficulty": "Hard", "contrast_role": "simple surface structure (cyclic coverage)",
        "sets": [
            {"name": "time_periods", "description": "six 4-hour windows over the 24h cycle"},
            {"name": "shift_starts", "description": "six 8-hour shift start times"},
        ],
        "tables": [
            {
                "name": "staff_requirements",
                "indices": ["time_period"],
                "unit": "people",
                "values": md_table(
                    ["period", "required_staff"],
                    [
                        ["2-6", 10], ["6-10", 15], ["10-14", 25],
                        ["14-18", 20], ["18-22", 18], ["22-2", 12],
                    ],
                ),
            }
        ],
        "notes": "Single decision family (staff per shift start); cyclic 24h coverage obligations.",
    },
    {
        "dataset": "industryor", "key": "36",
        "anchor_id": "IndustryOR_Hard_036_vrp",
        "difficulty": "Hard", "contrast_role": "multi-entity / high coupling",
        "sets": [], "tables": [],
        "notes": "Vehicle routing: route choice + vehicle usage + capacity + time constraints.",
    },
    {
        "dataset": "nl4opt", "key": "0",
        "anchor_id": "NL4OPT_000_oil_spill_transport",
        "difficulty": "N/A", "contrast_role": "low coupling LP",
        "sets": [{"name": "transport_methods", "description": "boat and canoe"}],
        "tables": [
            {
                "name": "method_params",
                "indices": ["method"],
                "unit": "",
                "values": md_table(
                    ["method", "ducks_per_trip", "minutes_per_trip"],
                    [["boat", 10, 20], ["canoe", 8, 40]],
                ),
            },
            {
                "name": "context",
                "indices": [],
                "unit": "",
                "values": md_table(
                    ["min_ducks", "max_boat_trips", "min_canoe_fraction"],
                    [["300", "12", "0.60"]],
                ),
            },
        ],
        "notes": "Two decision families; trip-count ratio + capacity + demand obligations.",
    },
    {
        "dataset": "nl4opt", "key": "13",
        "anchor_id": "NL4OPT_013_radiation_beams",
        "difficulty": "N/A", "contrast_role": "higher coupling LP",
        "sets": [
            {"name": "beams", "description": "Beam 1 and Beam 2"},
            {"name": "body_regions", "description": "pancreas, skin, tumor"},
        ],
        "tables": [
            {
                "name": "dose_rates",
                "indices": ["beam", "region"],
                "unit": "units per minute",
                "values": md_table(
                    ["beam", "pancreas", "skin", "tumor"],
                    [
                        ["Beam 1", 0.3, 0.2, 0.6],
                        ["Beam 2", 0.2, 0.1, 0.4],
                    ],
                ),
            },
            {
                "name": "limits",
                "indices": ["region"],
                "unit": "units",
                "values": md_table(
                    ["region", "limit"],
                    [["skin", "at most 4"], ["tumor", "at least 3"]],
                ),
            },
        ],
        "notes": "Two decision families; multiple dose obligations across regions.",
    },
    {
        "dataset": "miplib", "key": "flugpl",
        "anchor_id": "MIPLIB_NL_flugpl_fleet",
        "difficulty": "N/A", "contrast_role": "smaller family-level structure",
        "sets": [{"name": "periods", "description": "six monthly planning periods"}],
        "notes": "Fleet planning: standard aircraft + rentals + overtime; balance and capacity obligations.",
    },
    {
        "dataset": "miplib", "key": "graph20-20-1rand",
        "anchor_id": "MIPLIB_NL_graph20_20_cuts",
        "difficulty": "N/A", "contrast_role": "clearly more complex (multi-index, logical)",
        "sets": [
            {"name": "stations", "description": "20 observation outposts (nodes)"},
            {"name": "links", "description": "37 candidate trails (edges)"},
        ],
        "notes": "Maximum independent cuts: partition variables per cut, edge-disjointness, logical activation.",
    },
    {
        "dataset": "lean", "key": "TP8",
        "anchor_id": "LEAN_TP8_transportation",
        "difficulty": "N/A", "contrast_role": "relatively sparse (transportation)",
        "sets": [
            {"name": "suppliers", "description": "5 warehouses"},
            {"name": "stores", "description": "6 retail stores"},
        ],
        "notes": "Transportation: single flow family + supply/demand capacity obligations.",
    },
    {
        "dataset": "lean", "key": "Mixture6",
        "anchor_id": "LEAN_Mixture6_truck_scheduling",
        "difficulty": "N/A", "contrast_role": "multi-index / highly interacting",
        "sets": [
            {"name": "trucks", "description": "10 candidate trucks"},
            {"name": "periods", "description": "4 consecutive time periods"},
        ],
        "notes": "Truck activation + startup + transported weight; min-up/min-down, ramp, demand, buffer obligations.",
    },
    {
        "dataset": "optengine", "key": "BinPacking_10",
        "anchor_id": "OPTEngine_BinPacking",
        "difficulty": "N/A", "contrast_role": "controlled low coupling",
        "sets": [
            {"name": "packages", "description": "packages to load"},
            {"name": "trucks", "description": "identical delivery trucks (count decided by model)"},
        ],
        "tables": [],
        "notes": "Bin packing: package-truck assignment family + coverage/capacity obligations.",
    },
    {
        "dataset": "optengine", "key": "JobShop_3",
        "anchor_id": "OPTEngine_JobShop",
        "difficulty": "N/A", "contrast_role": "controlled high coupling",
        "sets": [
            {"name": "routes", "description": "3 delivery routes"},
            {"name": "hubs", "description": "3 distribution hubs"},
        ],
        "tables": [],
        "notes": "Job shop: per-procedure start times + precedence + non-overlap (resource exclusion) obligations.",
    },
]

CURATED_INDEX = {(c["dataset"], c["key"]): c for c in CURATED}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def find_problem(dataset_key: str, problem_key: str) -> dict:
    """Find a discovered problem entry by key (uses the first match)."""
    for p in SOURCES[dataset_key]["discover"]():
        if p["key"] == problem_key:
            return p
    raise KeyError(f"{dataset_key}:{problem_key} not found")


def make_anchor(dataset_key: str, prob: dict) -> dict:
    """Build one anchor dict, layering curated metadata on a discovered problem."""
    cur = CURATED_INDEX.get((dataset_key, prob["key"]))
    rendered = SOURCES[dataset_key]["render"](prob["key"])
    if cur is not None:
        return {
            "anchor_id": cur["anchor_id"],
            "dataset": SOURCES[dataset_key]["label"],
            "difficulty": cur["difficulty"],
            "contrast_role": cur["contrast_role"],
            "problem_text": rendered["problem_text"],
            "sets": cur.get("sets", rendered["sets"]),
            "tables": cur.get("tables", rendered["tables"]),
            "notes": cur["notes"],
            "external_files": rendered["external_files"],
        }
    return {
        "anchor_id": prob["anchor_id"],
        "dataset": SOURCES[dataset_key]["label"],
        "difficulty": "N/A",
        "contrast_role": "",
        "problem_text": rendered["problem_text"],
        "sets": rendered["sets"],
        "tables": rendered["tables"],
        "notes": rendered["notes"],
        "external_files": rendered["external_files"],
        "augmented": bool(prob.get("augmented", False)),
    }


def resolve_datasets_for_ids(ids: list[str]) -> list[str]:
    """Map anchor-id substrings to candidate datasets via their id prefix.

    A dataset is a candidate when its id prefix is a prefix of the given id
    (e.g. "MIPLIB_NL_flugpl" -> miplib). Non-prefixed ids (e.g. "TP8") fall
    back to scanning all datasets.
    """
    candidates = set()
    for w in ids:
        wl = w.lower()
        matched = [k for k, src in SOURCES.items() if wl.startswith(src["prefix"].lower())]
        if matched:
            candidates.update(matched)
        else:
            candidates.update(SOURCES)
    return sorted(candidates)


def build_anchor(anchor: dict) -> dict:
    """Build one annotator input dict in the USER INPUT SCHEMA format."""
    return {
        "instance_id": anchor["anchor_id"],
        "annotation_mode": "semantic",
        "problem_text": anchor["problem_text"],
        "sets": anchor.get("sets", []),
        "parameter_tables_inline": anchor.get("tables", []),
        "external_file_manifest": anchor.get("external_files", []),
        "notes": anchor.get("notes", ""),
    }


def write_outputs(anchors: list, manifest_path: Path = MANIFEST_PATH) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["anchor_id", "dataset", "difficulty", "contrast_role"],
        )
        writer.writeheader()
        for a in anchors:
            writer.writerow(
                {
                    "anchor_id": a["anchor_id"],
                    "dataset": a["dataset"],
                    "difficulty": a["difficulty"],
                    "contrast_role": a["contrast_role"],
                }
            )
    for a in anchors:
        inp = build_anchor(a)
        if a.get("augmented", False) or a["anchor_id"].endswith("_aug"):
            out_path = AUG_DIR / f"{a['anchor_id']}.json"
        else:
            out_path = OUT_DIR / f"{a['anchor_id']}.json"
        out_path.write_text(json.dumps(inp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out_path.name}  (text {len(inp['problem_text'])} chars, "
              f"{len(inp['parameter_tables_inline'])} tables, "
              f"{len(inp['external_file_manifest'])} external files)")
    n_aug = len([a for a in anchors if a.get("augmented", False) or a["anchor_id"].endswith("_aug")])
    print(f"\nManifest: {manifest_path}")
    print(f"Inputs  : {OUT_DIR} ({len(anchors) - n_aug} files)")
    print(f"Augmented OPTEngine inputs: {AUG_DIR} ({n_aug} files)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pilot annotator inputs from the 5 source repos.")
    parser.add_argument(
        "--curated", action="store_true",
        help="build only the 12 hand-curated pilot anchors (default: every problem from all datasets)",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="comma-separated dataset keys to build ALL problems from "
             "(industryor, nl4opt, optengine, miplib, lean). Default: all datasets.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="comma-separated anchor-id substrings to keep. Without --datasets, the "
             "matching dataset is auto-resolved from the id prefix.",
    )
    parser.add_argument("--list", action="store_true", help="list discoverable problems and exit.")
    args = parser.parse_args()

    if args.list:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or list(SOURCES)
        for key in datasets:
            src = SOURCES[key]
            print(f"[{key}] {src['label']}")
            for prob in src["discover"]():
                cur = CURATED_INDEX.get((key, prob["key"]))
                aid = cur["anchor_id"] if cur else prob["anchor_id"]
                print(f"    {aid}")
        return 0

    if args.curated:
        anchors = [make_anchor(c["dataset"], find_problem(c["dataset"], c["key"])) for c in CURATED]
        manifest_path = MANIFEST_PATH
    elif args.ids:
        wanted = [w.strip() for w in args.ids.split(",") if w.strip()]
        if args.datasets:
            datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        else:
            datasets = resolve_datasets_for_ids(wanted)
        anchors = []
        for key in datasets:
            if key not in SOURCES:
                print(f"warning: unknown dataset '{key}' (available: {', '.join(SOURCES)})")
                continue
            for prob in SOURCES[key]["discover"]():
                anchors.append(make_anchor(key, prob))
        anchors = [a for a in anchors if any(w in a["anchor_id"] for w in wanted)]
        manifest_path = MANIFEST_ALL_PATH
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        anchors = []
        for key in datasets:
            if key not in SOURCES:
                print(f"warning: unknown dataset '{key}' (available: {', '.join(SOURCES)})")
                continue
            for prob in SOURCES[key]["discover"]():
                anchors.append(make_anchor(key, prob))
        manifest_path = MANIFEST_ALL_PATH
    else:
        # default: build every discoverable problem from every dataset
        anchors = []
        for key in SOURCES:
            for prob in SOURCES[key]["discover"]():
                anchors.append(make_anchor(key, prob))
        manifest_path = MANIFEST_ALL_PATH

    if not anchors:
        print("No anchors selected.")
        return 1

    write_outputs(anchors, manifest_path=manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
