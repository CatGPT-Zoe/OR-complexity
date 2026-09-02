"""Verification tests: PDF hand-computed values + synthetic topology controls.

SOH-1.1 semantics:
  H_all = (V_S, O_S)  -- full obligation inventory (includes unary)
  H_c   = (V_S, E_c)  -- coupling projection, E_c = {o : |supp(o)| >= 2}

All structural coupling metrics (incidence density, overlap, spectral,
treewidth, hubness, cycle rank, LCC) are computed on H_c.
Unary obligations contribute only to the burden sidecar, never to H_c.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics.coupling_metrics import compute_metrics  # noqa: E402
from metrics.hypergraph_builder import (  # noqa: E402
    validate_graph,
    build_entity_primal_graph,
    build_obligation_graph,
)
from metrics.treewidth import treewidth_or_bounds  # noqa: E402

EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))


def load_example(name: str) -> dict:
    with open(os.path.join(EXAMPLES_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def approx(actual, expected, tol=1e-3):
    if actual is None:
        return False
    return abs(float(actual) - expected) <= tol


# ---------------------------------------------------------------------------
# SOH-1.1 A/A+ worked example (Inventory problem, n=3)
# PDF hand-computed gold values (2026-09-01 spec)
# ---------------------------------------------------------------------------

A_DOC = {
    "schema_version": "SOH-1.1",
    "instance_id": "Inventory_A",
    "layer": "semantic",
    "sets": [],
    "parameters": [],
    "entities": [
        {
            "id": "replenishment_order_quantity",
            "canonical_name": "replenishment_order_quantity",
            "semantic_type": "action",
            "entity_role": "decision",
            "indices": [],
            "description": "R",
            "value_domain": "continuous_nonnegative",
            "endogenous": True,
            "source_status": "explicit",
            "source_spans": ["order quantity"],
        },
        {
            "id": "inventory_state",
            "canonical_name": "inventory_state",
            "semantic_type": "state",
            "entity_role": "state",
            "indices": [],
            "description": "I",
            "value_domain": "continuous_nonnegative",
            "endogenous": True,
            "source_status": "explicit",
            "source_spans": ["stock"],
        },
        {
            "id": "unmet_demand",
            "canonical_name": "unmet_demand",
            "semantic_type": "service_outcome",
            "entity_role": "service_outcome",
            "indices": [],
            "description": "U",
            "value_domain": "continuous_nonnegative",
            "endogenous": True,
            "source_status": "explicit",
            "source_spans": ["shortage"],
        },
    ],
    "objective": {
        "sense": "minimize",
        "normalized_semantics": "min cost",
        "excluded_from_feasibility_hypergraph": True,
    },
    "obligations": [
        {
            "id": "o1",
            "canonical_family": "replenishment_lower_bound",
            "structural_form": "LOWER_BOUND",
            "business_role": "action_limit",
            "normalized_semantics": "each period order >= lower bound",
            "support_entity_ids": ["replenishment_order_quantity"],
            "incidence_slots": [{"entity_id": "replenishment_order_quantity", "role": "order", "aggregation_axes": []}],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": False,
            "source_grounding": "explicit",
            "source_spans": ["between 6 and 35 gallons"],
        },
        {
            "id": "o2",
            "canonical_family": "replenishment_upper_bound",
            "structural_form": "UPPER_BOUND",
            "business_role": "action_limit",
            "normalized_semantics": "each period order <= upper bound",
            "support_entity_ids": ["replenishment_order_quantity"],
            "incidence_slots": [{"entity_id": "replenishment_order_quantity", "role": "order", "aggregation_axes": []}],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": False,
            "source_grounding": "explicit",
            "source_spans": ["between 6 and 35 gallons"],
        },
        {
            "id": "o3",
            "canonical_family": "inventory_flow_balance",
            "structural_form": "BALANCE",
            "business_role": "conservation",
            "normalized_semantics": "previous stock + receipts reconcile demand, stock and unmet",
            "support_entity_ids": ["inventory_state", "replenishment_order_quantity", "unmet_demand"],
            "incidence_slots": [
                {"entity_id": "inventory_state", "role": "prev_stock", "semantic_role": "previous_state",
                 "aggregation_axes": [], "temporal_offset": 0},
                {"entity_id": "replenishment_order_quantity", "role": "receipt", "semantic_role": "usable_replenishment",
                 "aggregation_axes": [], "temporal_offset": 0},
                {"entity_id": "inventory_state", "role": "end_stock", "semantic_role": "resulting_state",
                 "aggregation_axes": [], "temporal_offset": 0},
                {"entity_id": "unmet_demand", "role": "unmet", "semantic_role": "unmet_demand",
                 "aggregation_axes": [], "temporal_offset": 0},
            ],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": True,
            "source_grounding": "explicit",
            "source_spans": ["balance"],
        },
        {
            "id": "o4",
            "canonical_family": "inventory_capacity",
            "structural_form": "UPPER_BOUND",
            "business_role": "state_capacity",
            "normalized_semantics": "physical stock within capacity",
            "support_entity_ids": ["inventory_state"],
            "incidence_slots": [{"entity_id": "inventory_state", "role": "stock", "aggregation_axes": []}],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": False,
            "source_grounding": "explicit",
            "source_spans": ["capacity"],
        },
        {
            "id": "o5",
            "canonical_family": "shortage_requires_stock_exhaustion",
            "structural_form": "IMPLICATION",
            "business_role": "service_semantics",
            "normalized_semantics": "unmet > 0 implies stock = 0",
            "support_entity_ids": ["unmet_demand", "inventory_state"],
            "incidence_slots": [
                {"entity_id": "unmet_demand", "role": "unmet", "aggregation_axes": [], "temporal_offset": 0},
                {"entity_id": "inventory_state", "role": "stock", "aggregation_axes": [], "temporal_offset": 0},
            ],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": True,
            "source_grounding": "explicit",
            "source_spans": ["stock-out semantics"],
        },
        {
            "id": "o6",
            "canonical_family": "lost_sales_no_backlog",
            "structural_form": "NO_CARRYOVER",
            "business_role": "service_semantics",
            "normalized_semantics": "unmet demand is lost, not deferred",
            "support_entity_ids": ["unmet_demand"],
            "incidence_slots": [{"entity_id": "unmet_demand", "role": "unmet", "aggregation_axes": []}],
            "quantifier": "for all t",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": False,
            "source_grounding": "explicit",
            "source_spans": ["lost and cannot be recovered"],
        },
    ],
    "derived_checks": [{"predicate": "unmet_demand[t] <= demand[t]", "derived_from": ["entity_semantics:unmet_demand"]}],
    "unresolved_items": [],
}

A_PLUS_DOC = dict(A_DOC)
A_PLUS_DOC["instance_id"] = "Inventory_A_plus"
# A+ adds day-5 upper bound override and terminal inventory minimum
A_PLUS_DOC["obligations"][1]["parameter_overrides"] = [
    {"scope": "t=5", "parameter_id": "upper_bound", "value": 39, "modifier": "override"}
]
A_PLUS_DOC["obligations"].append({
    "id": "o7",
    "canonical_family": "terminal_inventory_minimum",
    "structural_form": "LOWER_BOUND",
    "business_role": "terminal_condition",
    "normalized_semantics": "final-day stock >= floor",
    "support_entity_ids": ["inventory_state"],
    "incidence_slots": [{"entity_id": "inventory_state", "role": "stock", "aggregation_axes": []}],
    "quantifier": "t = T",
    "scope_kind": "terminal",
    "parameter_refs": [],
    "coupling_active": False,
    "source_grounding": "explicit",
    "source_spans": ["final-day minimum"],
})


class TestSOH11APlusGold:
    """SOH-1.1 canonical A/A+ example from the 2026-09-01 PDF.

    A  : n=3, m_all=6, I_all=9,  a_all=1.5,  m_c=2, I_c=5, a_c=2.5
        D_I=0.8333, D_cap=1.0, J+=0.6667, C_spec=0.6830, C_tw=1.0, hub=0, mu=1, LCC=1
        burden=(6,4,2,0,0)
    A+ : n=3, m_all=7, I_all=10, a_all=1.4286, coupling identical to A
        burden=(7,5,2,1,1)
    """

    def test_a_schema_valid(self):
        validate_graph(A_DOC)

    def test_a_plus_schema_valid(self):
        validate_graph(A_PLUS_DOC)

    def test_a_scale(self):
        r = compute_metrics(A_DOC)
        s = r["scale"]
        assert s["n_var_families"] == 3
        assert s["m_all"] == 6
        assert s["m_unary"] == 4
        assert s["m_rel"] == 2
        assert s["total_incidence_all"] == 9
        assert approx(s["mean_arity_all"], 1.5)
        assert s["total_incidence"] == 5
        assert s["mean_arity"] == 2.5

    def test_a_local(self):
        r = compute_metrics(A_DOC)["local"]
        assert approx(r["incidence_density"], 0.8333, tol=1e-4)
        assert approx(r["intersection_density"], 1.0, tol=1e-4)
        assert approx(r["conditional_jaccard"], 0.6667, tol=1e-4)
        assert approx(r["all_pair_jaccard"], 0.6667, tol=1e-4)

    def test_a_entity_coupling(self):
        r = compute_metrics(A_DOC)["entity"]
        assert approx(r["spectral_coupling_index"], 0.6830, tol=1e-4)
        assert approx(r["spectral_radius_raw"], 1.3660, tol=1e-4)
        assert r["treewidth"]["value"] == 2
        assert approx(r["treewidth_norm"], 1.0, tol=1e-4)
        assert approx(r["hubness"], 0.0, tol=1e-4)
        assert r["cycle_rank"] == 1
        assert approx(r["cycle_density"], 1.0 / 3, tol=1e-4)
        assert approx(r["largest_component_ratio"], 1.0, tol=1e-4)

    def test_a_burden(self):
        r = compute_metrics(A_DOC)["burden"]
        assert r["m_all"] == 6
        assert r["m_unary"] == 4
        assert r["m_rel"] == 2
        assert r["n_exception"] == 0
        assert r["l_decision"] == 0
        assert r["n_conditional"] == 0

    def test_aplus_scale(self):
        r = compute_metrics(A_PLUS_DOC)
        s = r["scale"]
        assert s["n_var_families"] == 3
        assert s["m_all"] == 7
        assert s["m_unary"] == 5
        assert s["m_rel"] == 2
        assert s["total_incidence_all"] == 10
        assert approx(s["mean_arity_all"], 10 / 7, tol=1e-4)
        assert s["total_incidence"] == 5
        assert s["mean_arity"] == 2.5

    def test_aplus_coupling_identical_to_a(self):
        """A+ adds unary obligations only -> coupling projection unchanged."""
        ra = compute_metrics(A_DOC)
        rap = compute_metrics(A_PLUS_DOC)
        for k in ["local", "entity", "obligation"]:
            for f in ra[k]:
                if isinstance(ra[k][f], dict):
                    continue
                assert ra[k][f] == rap[k][f] or approx(ra[k][f], rap[k][f]), \
                    f"{k}.{f} differs: A={ra[k][f]} A+={rap[k][f]}"

    def test_aplus_burden(self):
        r = compute_metrics(A_PLUS_DOC)["burden"]
        assert r["m_all"] == 7
        assert r["m_unary"] == 5
        assert r["m_rel"] == 2
        assert r["n_exception"] == 1  # day-5 override
        assert r["l_decision"] == 0  # no temporal offset in this fixture
        assert r["n_conditional"] == 0


# ---------------------------------------------------------------------------
# PDF worked example (routing + facility + inventory, n=6)
# Updated for SOH-1.1: o1 (single-entity assignment) and o3 (single-entity
# flow conservation) are unary -> excluded from H_c. Thus H_c has 8
# obligations, not 10.
# ---------------------------------------------------------------------------

class TestPDFIntegratedExample:
    def test_schema_valid(self):
        g = load_example("integrated_example.json")
        validate_graph(g)

    def test_scale_vector(self):
        r = compute_metrics(load_example("integrated_example.json"))
        assert r["scale"]["n_var_families"] == 6
        # H_c has 8 obligations (o1 and o3 are unary)
        assert r["scale"]["n_constr_families"] == 8
        assert r["scale"]["m_all"] == 10
        assert r["scale"]["m_unary"] == 2
        assert r["scale"]["total_incidence"] == 17
        assert r["scale"]["total_incidence_all"] == 19
        assert approx(r["scale"]["mean_arity"], 17 / 8)
        assert approx(r["scale"]["mean_arity_all"], 19 / 10)

    def test_local_coupling(self):
        r = compute_metrics(load_example("integrated_example.json"))["local"]
        # H_c: n=6, m=8, I=17 -> D_I = 17/(6*8) = 0.35417
        assert approx(r["incidence_density"], 17 / 48)
        # 15 of 28 pairs intersect -> 0.5357
        assert approx(r["intersection_density"], 15 / 28)
        # sum of positive intersections = 16 -> 16/15
        assert approx(r["mean_positive_overlap"], 16 / 15)
        # sum of positive Jaccards = 5.0 -> 5/15 = 1/3
        assert approx(r["conditional_jaccard"], 1 / 3, tol=1e-4)
        # all-pair Jaccard = D_\cap * J_+ = (15/28)*(1/3) = 5/28
        assert approx(r["all_pair_jaccard"], 5 / 28, tol=1e-4)
        # identity J_all = D_\cap * J_+
        assert approx(r["all_pair_jaccard"], r["intersection_density"] * r["conditional_jaccard"], tol=1e-9)

    def test_entity_spectral(self):
        r = compute_metrics(load_example("integrated_example.json"))["entity"]
        # rho(P) ~ 0.441888 ; C_spec^V = rho/(n-1) = 0.441888/5
        assert approx(r["spectral_radius_raw"], 0.441888, tol=1e-4)
        assert approx(r["spectral_coupling_index"], 0.441888 / 5, tol=1e-4)

    def test_entity_treewidth(self):
        g = load_example("integrated_example.json")
        adj = build_entity_primal_graph(g)
        tw = treewidth_or_bounds(adj)
        assert tw["value"] == 3
        r = compute_metrics(g)["entity"]
        assert r["treewidth"]["value"] == 3
        # C_tw^V = 3/(6-1) = 0.6
        assert tw["value"] / 5 == 0.6

    def test_entity_hubness(self):
        r = compute_metrics(load_example("integrated_example.json"))["entity"]
        # Freeman centralization = 6/20 = 0.3 (d_max=4)
        assert approx(r["hubness"], 0.3)

    def test_entity_cycle_and_lcc(self):
        r = compute_metrics(load_example("integrated_example.json"))["entity"]
        assert r["cycle_rank"] == 4
        assert approx(r["cycle_density"], 4 / 10)
        assert approx(r["largest_component_ratio"], 1.0)

    def test_obligation_spectral(self):
        r = compute_metrics(load_example("integrated_example.json"))["obligation"]
        # rho(W) ~ 1.293805 ; C_spec^O = rho/(m-1) = 1.293805/7
        assert approx(r["weighted_spectral_radius_raw"], 1.293805, tol=1e-4)
        assert approx(r["spectral_coupling_index"], 1.293805 / 7, tol=1e-4)

    def test_obligation_treewidth(self):
        g = load_example("integrated_example.json")
        adj, _, _, _ = build_obligation_graph(g)
        tw = treewidth_or_bounds(adj)
        assert tw["value"] == 4
        r = compute_metrics(g)["obligation"]
        assert r["treewidth"]["value"] == 4

    def test_obligation_hubness(self):
        r = compute_metrics(load_example("integrated_example.json"))["obligation"]
        # H_c has 8 obligations; obligation graph: 15 edges
        # d_max=5, sum(d_max-d)=12, denom=(7*6)=42 -> 12/42
        assert approx(r["hubness"], 12 / 42, tol=1e-4)

    def test_obligation_cycle_and_lcc(self):
        r = compute_metrics(load_example("integrated_example.json"))["obligation"]
        # |E_O|=15 -> mu=15-8+1=8 ; C_cyc=8/21
        assert r["cycle_rank"] == 8
        assert approx(r["cycle_density"], 8 / 21, tol=1e-3)
        assert approx(r["largest_component_ratio"], 1.0)


# ---------------------------------------------------------------------------
# Product-mix example (single entity family, two obligations)
# SOH-1.1: both obligations are unary -> H_c is empty
# ---------------------------------------------------------------------------

class TestProductMix:
    def test_schema_valid(self):
        validate_graph(load_example("product_mix.json"))

    def test_metrics(self):
        r = compute_metrics(load_example("product_mix.json"))
        # one entity family, two obligations, both unary
        assert r["scale"]["n_var_families"] == 1
        assert r["scale"]["m_all"] == 2
        assert r["scale"]["m_unary"] == 2
        assert r["scale"]["m_rel"] == 0
        assert r["scale"]["total_incidence"] == 0
        assert r["scale"]["total_incidence_all"] == 2
        assert approx(r["scale"]["mean_arity_all"], 1.0)
        # H_c is empty -> incidence density 0
        assert r["local"]["incidence_density"] == 0.0
        # no pairs -> intersection_density 0
        assert r["local"]["intersection_density"] == 0.0
        assert r["local"]["mean_positive_overlap"] is None
        assert r["local"]["conditional_jaccard"] is None
        # entity graph: single node, no coupling edges
        assert r["entity"]["treewidth"]["value"] == 0
        assert r["entity"]["hubness"] == 0.0
        assert r["entity"]["cycle_rank"] == 0
        # entity adj is empty -> LCC 0
        assert r["entity"]["largest_component_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Synthetic topology controls
# ---------------------------------------------------------------------------


def make_graph(entity_ids, obligation_supports):
    """Build a minimal SOH-like graph dict from supports."""
    entities = [
        {
            "id": eid,
            "canonical_name": eid,
            "semantic_type": "other",
            "entity_role": "decision",
            "indices": [],
            "description": eid,
            "endogenous": True,
            "source_status": "explicit",
        }
        for eid in entity_ids
    ]
    obligations = [
        {
            "id": f"o{i}",
            "canonical_family": f"family_{i}",
            "structural_form": "BALANCE",
            "business_role": "test",
            "normalized_semantics": f"obligation {i}",
            "support_entity_ids": list(support),
            "incidence_slots": [],
            "quantifier": "for all",
            "scope_kind": "universal",
            "parameter_refs": [],
            "coupling_active": True,
            "source_grounding": "explicit",
            "source_spans": ["synthetic"],
        }
        for i, support in enumerate(obligation_supports)
    ]
    return {
        "schema_version": "SOH-1.1",
        "instance_id": "synthetic",
        "layer": "semantic",
        "sets": [],
        "parameters": [],
        "entities": entities,
        "objective": {
            "sense": "satisfy",
            "normalized_semantics": "none",
            "excluded_from_feasibility_hypergraph": True,
        },
        "obligations": obligations,
        "derived_checks": [],
        "unresolved_items": [],
    }


class TestTopologyControls:
    def test_star_is_hub(self):
        # center entity a connected to b,c,d via three obligations
        g = make_graph(
            ["a", "b", "c", "d"],
            [["a", "b"], ["a", "c"], ["a", "d"]],
        )
        r = compute_metrics(g)
        # star: hubness high, treewidth low, cycle rank 0
        assert r["entity"]["hubness"] == 1.0
        assert r["entity"]["treewidth"]["value"] == 1
        assert r["entity"]["cycle_rank"] == 0

    def test_cycle_has_cycle(self):
        g = make_graph(
            ["a", "b", "c", "d"],
            [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]],
        )
        r = compute_metrics(g)
        # 4-cycle: cycle rank 1, hubness 0, treewidth 2
        assert r["entity"]["cycle_rank"] == 1
        assert r["entity"]["hubness"] == 0.0
        assert r["entity"]["treewidth"]["value"] == 2

    def test_clique_high_treewidth_high_cycle(self):
        g = make_graph(
            ["a", "b", "c", "d"],
            [
                ["a", "b"], ["a", "c"], ["a", "d"],
                ["b", "c"], ["b", "d"], ["c", "d"],
            ],
        )
        r = compute_metrics(g)
        # K4: treewidth 3, hubness 0, cycle rank = 6-4+1 = 3
        assert r["entity"]["treewidth"]["value"] == 3
        assert r["entity"]["hubness"] == 0.0
        assert r["entity"]["cycle_rank"] == 3

    def test_two_components_lcc(self):
        # two disconnected modules
        g = make_graph(
            ["a", "b", "c", "d", "e", "f"],
            [["a", "b", "c"], ["d", "e", "f"]],
        )
        r = compute_metrics(g)
        assert approx(r["entity"]["largest_component_ratio"], 0.5)

    def test_disconnected_chain(self):
        g = make_graph(
            ["a", "b", "c", "d"],
            [["a", "b"], ["c", "d"]],
        )
        r = compute_metrics(g)
        # two components, each a single edge
        assert r["entity"]["cycle_rank"] == 0
        assert approx(r["entity"]["largest_component_ratio"], 0.5)
        assert r["entity"]["treewidth"]["value"] == 1

    def test_single_entity_multi_rule(self):
        # one entity family, many obligations with identical support
        g = make_graph(
            ["production"],
            [["production"], ["production"], ["production"]],
        )
        r = compute_metrics(g)
        # All obligations are unary -> H_c is empty
        assert r["entity"]["treewidth"]["value"] == 0
        assert r["entity"]["hubness"] == 0.0
        assert r["entity"]["cycle_rank"] == 0
        # obligation graph on H_c is empty -> spectral 0
        assert approx(r["obligation"]["spectral_coupling_index"], 0.0)
        assert r["obligation"]["treewidth"]["value"] == 0


# ---------------------------------------------------------------------------
# Degenerate / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_graph(self):
        g = make_graph([], [])
        r = compute_metrics(g)
        assert r["scale"]["n_var_families"] == 0
        assert r["scale"]["m_all"] == 0
        assert r["scale"]["m_rel"] == 0
        assert r["scale"]["total_incidence"] == 0

    def test_single_obligation_no_pairs(self):
        g = make_graph(["a", "b"], [["a", "b"]])
        r = compute_metrics(g)
        # no pairs -> intersection_density 0, conditional undefined (None)
        assert r["local"]["intersection_density"] == 0.0
        assert r["local"]["mean_positive_overlap"] is None
        assert r["local"]["conditional_jaccard"] is None
