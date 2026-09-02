# 12-Anchor Pilot: A/B Dual-Model Semantic Annotation

Goal: build an "intuition test" of the metric design — annotate 12 carefully
chosen instances with two strong models independently (A/B), compute the v1.0
metric vector for each, and inspect whether the metrics distinguish instances
the way our theory predicts (coupling-heavy vs. sparse, multi-entity vs.
single-entity, high-arity vs. low-arity).

## Anchor selection (purposive contrast sampling)

| anchor | dataset | role |
|---|---|---|
| IndustryOR_Easy_092_bus_rental | IndustryOR | simple surface, 1 entity family + 2 obligations |
| IndustryOR_Easy_077_machine_assignment | IndustryOR | multi-rule/multi-entity (parts×machines, if-then rules) |
| IndustryOR_Hard_026_shift_scheduling | IndustryOR | cyclic 24h coverage, 1 decision family |
| IndustryOR_Hard_036_vrp | IndustryOR | multi-entity VRP (route + vehicle + capacity) |
| NL4OPT_000_oil_spill_transport | NL4OPT | low coupling LP (2 vars, ratio constraint) |
| NL4OPT_013_radiation_beams | NL4OPT | higher coupling LP (multi-region dose obligations) |
| MIPLIB_NL_flugpl_fleet | MIPLIB-NL | small family-level (fleet planning) |
| MIPLIB_NL_graph20_20_cuts | MIPLIB-NL | multi-index / logical cuts |
| LEAN_TP8_transportation | lean-llm-opt | sparse transportation structure |
| LEAN_Mixture6_truck_scheduling | lean-llm-opt | multi-index, min-up/down + ramp + buffer |
| OPTEngine_BinPacking | OPTEngine | controlled low coupling (canonical) |
| OPTEngine_JobShop | OPTEngine | controlled high coupling (precedence + exclusion) |

## Files

- `anchor_manifest.csv` — full-set selection table (dataset + difficulty + contrast role)
- `anchor12_manifest.csv` — curated 12-anchor pilot selection table
- `inputs/*.json` — annotator inputs (problem text + inline parameter tables);
  OPTEngine constraint-augmentation variants live in
  `inputs/OPTEngine-augmented/*_aug.json`
- `build_pilot_inputs.py` — regenerates inputs from the 5 source repos
- `run_ab_annotation.py` — A/B harness (system prompt + 2 models, schema check,
  metric table)
- `ai_annotations/` — created at runtime; per-model raw `.txt` + metric CSV +
  `annotation_status.json`

## How to run (needs network; run from the workspace root)

```bash
# 1. Build inputs (default: EVERY problem from all 5 datasets; re-run to refresh)
.venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py"

# curated 12-anchor pilot contrast set (hand-picked, human-verified)
.venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --curated

# build ALL problems from selected datasets instead of the full set
.venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --datasets lean,miplib

# build specific anchors (substring match on anchor id)
.venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --ids LEAN_TP8,MIPLIB_NL_flugpl

# list discoverable problems without writing outputs
.venv/bin/python "OR-complexity/pilot/build_pilot_inputs.py" --list

# 2. Annotate with two strong models, A and B, independently
#    (default: every *.json currently in inputs/)
.venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" \
    --models gpt-5.5-2026-04-23 deepseek-v4-pro \
    --out "OR-complexity/pilot/ai_annotations"

# restrict annotation by dataset and/or anchor id (same syntax as above)
.venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" --datasets lean,miplib
.venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" --ids LEAN_TP8,MIPLIB_NL_flugpl

# provider auto-inference: deepseek-* -> DeepSeek, qwen-* -> DashScope,
# openai/... -> OpenRouter (if OPENROUTER_API_KEY set), else OpenAI.
# Force a provider with --provider openrouter|deepseek|qwen|openai.

# 3. Inspect the metric table
"OR-complexity/pilot/ai_annotations/pilot_metric_table.csv"
```

Dry-run (no network; writes system/user prompt pairs for inspection):

```bash
.venv/bin/python "OR-complexity/pilot/run_ab_annotation.py" --dry-run \
    --out /tmp/ab_dry
```

## Building inputs from a new dataset

`build_pilot_inputs.py` keeps one loader per source database in a `SOURCES`
registry. Each loader returns a list of problem dicts with `problem_text`,
inline `tables`, and `external_files` (workspace-relative paths to any
CSV/JSON backing data). To add a database, implement a loader with the same
shape (including any per-dataset external-file parsing), register it, and add
curated anchors if desired.

## QC notes

- Each raw response is saved verbatim before any parsing.
- `extract_json` strips markdown fences, then balances braces.
- Outputs are validated against
  `operational_complexity/schemas/semantic_hypergraph.schema.json` and the
  graph validator (unknown entity refs are rejected).
- coupling_active=false obligations are valid annotations (fixed,
  non-coupling assignments); the metric engine includes ONLY coupling-active
  obligations, so their presence never changes the metric values.
- Only schema-valid hypergraphs enter `pilot_metric_table.csv`.
