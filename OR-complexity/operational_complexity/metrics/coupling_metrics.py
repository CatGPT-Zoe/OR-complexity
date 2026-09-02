"""SOH-1.1 coupling metrics for family-level semantic-obligation hypergraphs.

Computes the two-layer decomposition described in the SOH-1.1 spec:
  H_all = (V_S, O_S)   full obligation inventory (unary + relational)
  H_c   = (V_S, E_c)   coupling projection, E_c = {o : |supp(o)| >= 2}

All structural coupling metrics (incidence density, overlap, spectral, treewidth,
hubness, cycle rank, LCC) are computed on H_c. Unary rules contribute only to the
burden sidecar, never to the coupling projection.

The metric vector is:
  C_S = (D_I, D_∩, J_+, C_spec, C_tw, C_hub, μ, C_LCC)
  B_S = (m_all, m_unary, m_rel, N_exception, L_decision, N_conditional)
"""

from __future__ import annotations

from collections import deque
from math import comb

import numpy as np

from .hypergraph_builder import (
    all_obligations,
    build_entity_primal_graph,
    build_obligation_graph,
    build_incidence_matrix,
    burden_sidecar,
    obligation_pair_stats,
    obligation_support,
    unary_obligations,
)
from .treewidth import treewidth_or_bounds

EPS = 1e-12


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return (mat + mat.T) / 2.0


def _spectral_radius(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    vals = np.linalg.eigvalsh(_symmetrize(mat))
    return float(np.max(np.abs(vals))) if len(vals) else 0.0


def _largest_component_ratio(adj: dict) -> float:
    if not adj:
        return 0.0
    seen = set()
    largest = 0
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        size = 0
        while q:
            v = q.popleft()
            size += 1
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        largest = max(largest, size)
    return largest / len(adj)


def _connected_components(adj: dict):
    seen = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def _edge_count(adj: dict) -> int:
    return sum(len(v) for v in adj.values()) // 2


def _freeman_centralization(adj: dict) -> float:
    n = len(adj)
    if n < 3:
        return 0.0
    degrees = {v: len(nbs) for v, nbs in adj.items()}
    dmax = max(degrees.values()) if degrees else 0
    numerator = sum(dmax - d for d in degrees.values())
    denom = (n - 1) * (n - 2)
    return float(numerator / denom) if denom else 0.0


def _cycle_rank(adj: dict) -> int:
    if not adj:
        return 0
    e = _edge_count(adj)
    v = len(adj)
    c = len(_connected_components(adj))
    return int(e - v + c)


def _normalized_cycle_density(adj: dict) -> float:
    v = len(adj)
    if v < 3:
        return 0.0
    mu = _cycle_rank(adj)
    max_mu = (v - 1) * (v - 2) / 2
    return float(mu / max_mu) if max_mu > 0 else 0.0


def _obligation_weighted_matrix(weights: dict) -> np.ndarray:
    ids = list(weights.keys())
    n = len(ids)
    W = np.zeros((n, n), dtype=float)
    if n == 0:
        return W
    idx = {oid: i for i, oid in enumerate(ids)}
    for i, oid_i in enumerate(ids):
        for oid_j, w in weights[oid_i].items():
            W[i, idx[oid_j]] = w
    return W


def _obligation_spectral(weights: dict) -> tuple:
    """Return (raw spectral radius, normalized coupling index) on H_c."""
    W = _obligation_weighted_matrix(weights)
    rho = _spectral_radius(W)
    n = W.shape[0]
    norm = float(rho / max(n - 1, 1)) if n else 0.0
    return float(rho), norm


def compute_metrics(graph: dict) -> dict:
    """Compute the SOH-1.1 metric vector from a semantic hypergraph."""
    # ---- scale / burden over H_all ----
    m_all = len(all_obligations(graph))
    m_unary = len(unary_obligations(graph))
    I_all = sum(len(obligation_support(o)) for o in all_obligations(graph))

    # ---- coupling projection H_c ----
    B, entity_ids, obligation_ids = build_incidence_matrix(graph)
    n = len(entity_ids)
    m = len(obligation_ids)
    arities = [sum(row) for row in B]
    I = int(sum(arities))
    mean_arity = float(I / m) if m else 0.0
    mean_arity_all = float(I_all / m_all) if m_all else 0.0

    # incidence density and overlap statistics on H_c
    pair_stats = obligation_pair_stats(graph)
    num_pairs = comb(m, 2) if m >= 2 else 0
    positive_pairs = [p for p in pair_stats if p[2] > 0]
    positive_count = len(positive_pairs)
    inter_sum = sum(p[2] for p in positive_pairs)
    jaccard_sum = sum(p[4] for p in positive_pairs)

    incidence_density = float(I / (n * m)) if n and m else 0.0
    intersection_density = float(positive_count / num_pairs) if num_pairs else 0.0
    mean_positive_overlap = float(inter_sum / positive_count) if positive_count else None
    conditional_jaccard = float(jaccard_sum / positive_count) if positive_count else None
    all_pair_jaccard = float(jaccard_sum / num_pairs) if num_pairs else 0.0

    # entity layer (primal graph from H_c)
    entity_adj = build_entity_primal_graph(graph)
    primal_graph_edges = _edge_count(entity_adj)
    entity_degrees = [len(nbs) for nbs in entity_adj.values()]
    entity_max_deg = max(entity_degrees) if entity_degrees else 0
    entity_hubness = _freeman_centralization(entity_adj)
    entity_cycle_rank = _cycle_rank(entity_adj)
    entity_cycle_density = _normalized_cycle_density(entity_adj)
    entity_lcc = _largest_component_ratio(entity_adj)

    # weighted primal matrix P_uv = average co-incidence frequency over H_c
    P = np.zeros((n, n), dtype=float)
    for row in B:
        ones = [i for i, x in enumerate(row) if x]
        for i in range(len(ones)):
            for j in range(i + 1, len(ones)):
                P[ones[i], ones[j]] += 1.0 / m if m else 0.0
                P[ones[j], ones[i]] += 1.0 / m if m else 0.0
    entity_spectral = _spectral_radius(P) if n else 0.0
    entity_spectral_norm = float(entity_spectral / max(n - 1, 1)) if n else 0.0

    # obligation graph metrics on H_c
    obl_adj, obl_weights, _, _ = build_obligation_graph(graph)
    obl_treewidth = treewidth_or_bounds(obl_adj)
    obl_hubness = _freeman_centralization(obl_adj)
    obl_cycle_rank = _cycle_rank(obl_adj)
    obl_cycle_density = _normalized_cycle_density(obl_adj)
    obl_lcc = _largest_component_ratio(obl_adj)
    obl_spectral_raw, obl_spectral_norm = _obligation_spectral(obl_weights)

    # treewidth of entity primal graph
    ent_treewidth = treewidth_or_bounds(entity_adj)
    tw_value = ent_treewidth.get("value") if ent_treewidth else None
    ent_treewidth_norm = float(tw_value / (n - 1)) if n > 1 and tw_value is not None else 0.0

    burden = burden_sidecar(graph)

    result = {
        "instance_id": graph.get("instance_id"),
        "schema_version": graph.get("schema_version"),
        "dataset": graph.get("dataset"),
        "scale": {
            "n_var_families": n,
            "n_constr_families": m,
            "m_all": m_all,
            "m_unary": m_unary,
            "m_rel": m,
            "total_incidence": I,
            "total_incidence_all": I_all,
            "mean_arity": mean_arity,
            "mean_arity_all": mean_arity_all,
        },
        "local": {
            "incidence_density": incidence_density,
            "intersection_density": intersection_density,
            "mean_positive_overlap": mean_positive_overlap,
            "conditional_jaccard": conditional_jaccard,
            "all_pair_jaccard": all_pair_jaccard,
        },
        "entity": {
            "spectral_coupling_index": entity_spectral_norm,
            "spectral_radius_raw": float(entity_spectral),
            "treewidth": ent_treewidth,
            "treewidth_norm": ent_treewidth_norm,
            "hubness": float(entity_hubness),
            "cycle_rank": int(entity_cycle_rank),
            "cycle_density": float(entity_cycle_density),
            "largest_component_ratio": float(entity_lcc),
            "primal_edge_count": int(primal_graph_edges),
            "max_degree": int(entity_max_deg),
        },
        "obligation": {
            "spectral_coupling_index": float(obl_spectral_norm),
            "weighted_spectral_radius_raw": float(obl_spectral_raw),
            "treewidth": obl_treewidth,
            "hubness": float(obl_hubness),
            "cycle_rank": int(obl_cycle_rank),
            "cycle_density": float(obl_cycle_density),
            "largest_component_ratio": float(obl_lcc),
        },
        "burden": burden,
        "controls": {
            "scale_vars": n,
            "scale_constr": m,
            "schema_version": graph.get("schema_version"),
        },
    }
    return result
