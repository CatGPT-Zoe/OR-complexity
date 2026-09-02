#!/usr/bin/env python3
"""A/B dual-model semantic annotation harness for the 12-anchor pilot.

Each anchor input (pilot/inputs/*.json) is annotated independently by two
strong models (A and B) using prompts/semantic_annotator_v1.txt as the system
prompt. Raw responses are saved, extracted JSON is schema-validated, and valid
hypergraphs are fed to the metric engine.

Usage (from workspace root):
    # default: annotate all curated 12 anchors (whatever *.json are in inputs/)
    .venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" \
        --models gpt-5.5-2026-04-23 deepseek-v4-pro \
        --out "OR-complexity/pilot/ai_annotations"

    # select anchors by dataset and/or anchor-id (reuses build_pilot_inputs)
    .venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" --datasets lean,miplib
    .venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" --ids LEAN_TP8,MIPLIB_NL_flugpl

Provider selection: model name prefix -> provider (deepseek->DeepSeek, qwen->
DashScope, else OpenAI). A single OPENROUTER_API_KEY in .env can also be used
(--provider openrouter) with model IDs like "openai/gpt-5.5-2026-04-23".

Dry-run mode (no network) is available for pipeline testing:
    ... run_ab_annotation.py --dry-run --out /tmp/ab_dry
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
PROMPT_PATH = SCRIPT_DIR.parent / "operational_complexity" / "prompts" / "semantic_annotator_v1.txt"
SCHEMA_PATH = SCRIPT_DIR.parent / "operational_complexity" / "schemas" / "semantic_hypergraph.schema.json"
INPUT_DIR = SCRIPT_DIR / "inputs"


def iter_inputs(input_dir: Path | None = None) -> list[Path]:
    """All annotator input JSONs, including nested group dirs like OPTEngine-augmented.

    Honors the --inputs override; defaults to the canonical pilot/inputs dir.
    """
    base = input_dir or INPUT_DIR
    return sorted(base.rglob("*.json"))


def find_input(anchor_id: str, input_dir: Path | None = None) -> Path:
    """Locate an anchor input JSON (searches group subdirs as well)."""
    base = input_dir or INPUT_DIR
    hit = next((p for p in base.rglob(f"{anchor_id}.json")), None)
    if hit is None:
        raise FileNotFoundError(f"{anchor_id}.json not found under {base}")
    return hit

sys.path.insert(0, str(SCRIPT_DIR.parent / "operational_complexity"))
from metrics.coupling_metrics import compute_metrics  # noqa: E402
from metrics.hypergraph_builder import validate_graph  # noqa: E402


def load_env() -> None:
    """Load API credentials from the project root .env only.

    The root .env is the authoritative config for this workspace (it holds the
    OpenRouter key used by the A/B models). Fallback to the old OptiMUS .env was
    removed: it carried a stale OPENAI_API_KEY that caused runtime 401s, and its
    direct-OpenAI/DashScope/DeepSeek endpoints are not what this pilot uses.
    """
    env_path = WORKSPACE / ".env"
    if _load_dotenv is not None:
        _load_dotenv(env_path, override=False)
        return
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_client(provider: str, *, timeout) -> OpenAI:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=api_key, timeout=timeout)
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
    if provider == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY or QWEN_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=timeout,
        )
    raise ValueError(f"Unknown provider: {provider}")


def _has_provider_key(provider: str) -> bool:
    keys = {
        "openai": ("OPENAI_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    }
    return any(os.getenv(k) for k in keys.get(provider, ()))


def infer_provider(model: str) -> str:
    """Route a model to a provider.

    Provider-specific prefixes (deepseek-, qwen-) use their own provider only if
    the matching API key is present. All other models fall back to OpenRouter,
    since the workspace root .env carries the OpenRouter key (models such as
    gpt-5.5-2026-04-23 and deepseek-v4-pro are OpenRouter model IDs).
    """
    lowered = model.lower()
    if lowered.startswith("deepseek") and _has_provider_key("deepseek"):
        return "deepseek"
    if lowered.startswith("qwen") and _has_provider_key("qwen"):
        return "qwen"
    return "openrouter"


def model_dir_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def resolve_timeout(provider: str, model: str, base_timeout: float):
    read_timeout = base_timeout
    lowered = model.lower()
    if provider == "qwen":
        read_timeout = max(base_timeout, 1200.0)
    elif re.search(r"gpt-5|o[134]-|claude|gemini", lowered):
        read_timeout = max(base_timeout, 900.0)
    if read_timeout > base_timeout:
        return httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)
    return base_timeout


def call_llm(client: OpenAI, model: str, system: str, user: str, *, temperature: float = 0.0) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 8000,
    }
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "temperature" in str(exc).lower():
            kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""


def extract_json(text: str):
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # find first { ... } block spanning the text
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
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate), None
                except json.JSONDecodeError as exc:
                    return None, f"JSON decode failed: {exc}"
    return None, "unbalanced braces"


def validate_with_schema(doc: dict) -> list[str]:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = []
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError as exc:
        errors.append(str(exc))
    return errors


def select_anchors(input_dir: Path, datasets: str, ids: str) -> list[str]:
    """Resolve which input anchors to annotate.

    Mirrors build_pilot_inputs selection: datasets expand via the SOURCES
    registry's discover(), and ids are substring matches on anchor id. Pass
    the chosen anchor ids that exist as *.json in input_dir.
    """
    # import the sibling module without executing its main()
    sys.path.insert(0, str(SCRIPT_DIR))
    import build_pilot_inputs as bpi

    dataset_keys = [d.strip() for d in datasets.split(",") if d.strip()]
    id_subs = [s.strip() for s in ids.split(",") if s.strip()]

    if not dataset_keys and not id_subs:
        # default: every *.json present in input_dir (including nested group dirs)
        return sorted(p.stem for p in iter_inputs(input_dir))

    if not dataset_keys:
        # ids only: infer datasets from id prefix, like build_pilot_inputs
        dataset_keys = bpi.resolve_datasets_for_ids(id_subs)

    discovered = {}
    for key in dataset_keys:
        if key not in bpi.SOURCES:
            print(f"warning: unknown dataset '{key}' (available: {', '.join(bpi.SOURCES)})")
            continue
        for prob in bpi.SOURCES[key]["discover"]():
            discovered[prob["anchor_id"]] = key

    # keep curated ids too (they may differ from the generic prefix id)
    for cur in bpi.CURATED:
        discovered[cur["anchor_id"]] = cur["dataset"]

    candidates = set(p.stem for p in iter_inputs(input_dir))
    if id_subs:
        chosen = {aid for aid in discovered if any(s in aid for s in id_subs)}
    elif dataset_keys:
        chosen = {aid for aid, k in discovered.items() if k in dataset_keys}
    else:
        chosen = candidates

    return sorted(candidates & chosen)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gpt-5.6-sol", "deepseek-v4-pro"])
    parser.add_argument("--provider", default=None, help="force provider (openai|deepseek|qwen|openrouter)")
    parser.add_argument("--inputs", default=str(INPUT_DIR))
    parser.add_argument("--prompt", default=str(PROMPT_PATH))
    parser.add_argument("--out", default=str(SCRIPT_DIR / "ai_annotations"))
    parser.add_argument(
        "--datasets", default="",
        help="comma-separated dataset keys (industryor,nl4opt,optengine,miplib,lean); "
             "annotate all that dataset's inputs present in --inputs (default: all present)",
    )
    parser.add_argument(
        "--ids", default="",
        help="comma-separated anchor-id substrings to keep (applied after --datasets, "
             "or alone to auto-resolve the dataset from the id prefix)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="do not call APIs; dump prompts for inspection")
    args = parser.parse_args(argv)

    load_env()
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    input_dir = Path(args.inputs)
    out_root = Path(args.out)

    anchors = select_anchors(input_dir, args.datasets, args.ids)

    if args.dry_run:
        dry_dir = out_root / "_dry_run_prompts"
        dry_dir.mkdir(parents=True, exist_ok=True)
        for a in anchors:
            user = json.dumps(json.loads(find_input(a, input_dir).read_text(encoding="utf-8")), indent=2)
            (dry_dir / f"{a}_SYSTEM.txt").write_text(system_prompt, encoding="utf-8")
            (dry_dir / f"{a}_USER.txt").write_text(user, encoding="utf-8")
        print(f"Dry-run: wrote {len(anchors)} system/user prompt pairs to {dry_dir}")
        return 0

    clients = {}
    for model in args.models:
        provider = args.provider or infer_provider(model)
        clients[model] = get_client(provider, timeout=resolve_timeout(provider, model, args.timeout))
        print(f"model={model} provider={provider}")

    rows = []
    summary = []
    for anchor in anchors:
        input_doc = json.loads(find_input(anchor, input_dir).read_text(encoding="utf-8"))
        user_prompt = json.dumps(input_doc, indent=2, ensure_ascii=False)
        for model in args.models:
            model_dir = out_root / model_dir_name(model)
            model_dir.mkdir(parents=True, exist_ok=True)
            raw_path = model_dir / f"{anchor}.txt"
            if raw_path.exists():
                raw = raw_path.read_text(encoding="utf-8")
            else:
                print(f"[{model}] {anchor} ...", flush=True)
                raw = call_llm(clients[model], model, system_prompt, user_prompt, temperature=args.temperature)
                raw_path.write_text(raw, encoding="utf-8")

            doc, err = extract_json(raw)
            status = "ok"
            schema_errors = []
            if doc is None:
                status = f"json_extract_failed: {err}"
            else:
                schema_errors = validate_with_schema(doc)
                if schema_errors:
                    status = f"schema_invalid ({len(schema_errors)} errors)"
                else:
                    try:
                        validate_graph(doc)
                    except Exception as exc:
                        status = f"graph_invalid: {exc}"

            if status == "ok":
                metrics = compute_metrics(doc)
                metrics["dataset"] = input_doc.get("instance_id", anchor).split("_")[0]
                metrics["instance_id"] = anchor
                metrics["model"] = model
                rows.append(metrics)
                print(f"  -> ok  n={metrics['scale']['n_var_families']} m={metrics['scale']['n_constr_families']}")
            else:
                print(f"  -> {status}")
            summary.append({"anchor": anchor, "model": model, "status": status})

    if rows:
        table_path = out_root / "pilot_metric_table.csv"
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            import csv

            fieldnames = ["dataset", "instance_id", "model"]
            for key in ["scale", "local", "entity", "obligation", "controls"]:
                for k in rows[0].get(key, {}):
                    fieldnames.append(f"{key}.{k}")
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            def flat(m):
                out = {"dataset": m["dataset"], "instance_id": m["instance_id"], "model": m["model"]}
                for key in ["scale", "local", "entity", "obligation", "controls"]:
                    for k, v in m.get(key, {}).items():
                        out[f"{key}.{k}"] = v if not isinstance(v, dict) else json.dumps(v)
                return out

            writer.writeheader()
            for m in rows:
                writer.writerow(flat(m))
        print(f"\nMetric table: {table_path} ({len(rows)} rows)")
    else:
        print("\nNo valid annotations produced.")

    (out_root / "annotation_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
