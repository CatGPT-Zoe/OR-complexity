"""Two-layer semantic obligation structures (SOH-1.1).

H_all = (V_S, O_S): the complete semantic obligation inventory, including
unary obligations (|supp(o)| == 1) such as terminal inventory floors and
explicit business bounds.

H_c   = (V_S, E_c), E_c = {o in O_S : |supp(o)| >= 2}:
the coupling projection that drives cross-entity structural metrics.

The engine accepts both SOH-1.0 (coupling_active flag) and SOH-1.1
(structural_form / canonical_family / temporal incidence slots) annotations.
An obligation participates in H_c when it is coupling-active AND its support
involves at least two distinct entity families.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def schema_version(graph: dict) -> str:
    return graph.get("schema_version", "SOH-1.0")


def is_coupling_obligation(o: dict) -> bool:
    """An obligation enters H_c iff coupling-active and multi-entity."""
    if not o.get("coupling_active", True):
        return False
    support = set(o.get("support_entity_ids", []))
    return len(support) >= 2


def all_obligations(graph: dict) -> list:
    """O_S: every obligation in the semantic inventory."""
    return list(graph.get("obligations", []))


def coupling_obligations(graph: dict) -> list:
    """E_c: obligations that participate in the coupling projection."""
    return [o for o in graph.get("obligations", []) if is_coupling_obligation(o)]


def all_entities(graph: dict) -> list:
    return list(graph.get("entities", []))


def coupling_entities(graph: dict) -> list:
    """Entities referenced by at least one coupling projection obligation."""
    used = {eid for o in coupling_obligations(graph) for eid in o.get("support_entity_ids", [])}
    return [e for e in graph.get("entities", []) if e.get("id") in used]


def obligation_support(o: dict) -> set:
    return set(o.get("support_entity_ids", []))


def unary_obligations(graph: dict) -> list:
    """Unary obligations in the full inventory (|supp| == 1)."""
    return [o for o in all_obligations(graph) if len(obligation_support(o)) == 1]


def validate_graph(graph: dict) -> None:
    """Minimal structural validation for SOH-like inputs (SOH-1.0 / SOH-1.1)."""
    version = graph.get("schema_version")
    if version not in ("SOH-1.0", "SOH-1.1"):
        raise ValueError(f"schema_version must be SOH-1.0 or SOH-1.1, got {version!r}")
    if graph.get("layer") != "semantic":
        raise ValueError("layer must be semantic")
    entities = graph.get("entities", [])
    obligations = graph.get("obligations", [])
    entity_ids = {e["id"] for e in entities}
    if len(entity_ids) != len(entities):
        raise ValueError("duplicate entity id")
    for o in obligations:
        for eid in o.get("support_entity_ids", []):
            if eid not in entity_ids:
                raise ValueError(f"obligation {o.get('id')} references unknown entity {eid}")


def build_incidence_matrix(graph: dict):
    """Return (B, entity_ids, obligation_ids) over the COUPLING projection H_c.

    B[o_idx][v_idx] = 1 iff entity v appears in coupling obligation o.
    """
    entities = coupling_entities(graph)
    obligations = coupling_obligations(graph)
    entity_ids = [e["id"] for e in entities]
    obligation_ids = [o["id"] for o in obligations]
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}
    B = [[0 for _ in entity_ids] for _ in obligations]
    for oi, o in enumerate(obligations):
        for eid in obligation_support(o):
            B[oi][entity_index[eid]] = 1
    return B, entity_ids, obligation_ids


def build_entity_primal_graph(graph: dict):
    """Primal graph over coupling entities (projection of H_c).

    Two entities are adjacent if they co-occur in at least one coupling
    obligation. Returns adjacency dict {entity_id: set(neighbor_entity_ids)}.
    """
    entities = [e["id"] for e in coupling_entities(graph)]
    adj = {eid: set() for eid in entities}
    for o in coupling_obligations(graph):
        support = list(dict.fromkeys(o.get("support_entity_ids", [])))
        if len(support) < 2:
            continue
        for u, v in combinations(support, 2):
            adj[u].add(v)
            adj[v].add(u)
    return adj


def build_obligation_graph(graph: dict):
    """Obligation intersection graph over H_c.

    Two obligations are adjacent if they share at least one entity.
    Edge weight is Jaccard overlap of supports.
    Returns (adjacency, weight_matrix, support_sizes, pairwise_intersections).
    """
    obligations = coupling_obligations(graph)
    ids = [o["id"] for o in obligations]
    supports = {o["id"]: obligation_support(o) for o in obligations}
    adj = {oid: set() for oid in ids}
    weights = {oid: {} for oid in ids}
    support_sizes = {oid: len(supports[oid]) for oid in ids}
    intersections = {}
    for i, oid_i in enumerate(ids):
        si = supports[oid_i]
        for oid_j in ids[i + 1 :]:
            sj = supports[oid_j]
            inter = len(si & sj)
            intersections[(oid_i, oid_j)] = inter
            if inter > 0:
                union = len(si | sj)
                w = inter / union if union else 0.0
                adj[oid_i].add(oid_j)
                adj[oid_j].add(oid_i)
                weights[oid_i][oid_j] = w
                weights[oid_j][oid_i] = w
    return adj, weights, support_sizes, intersections


def entity_family_burdens(graph: dict):
    """Support counts per entity over H_c and total incidence burden."""
    counts = defaultdict(int)
    for o in coupling_obligations(graph):
        for eid in obligation_support(o):
            counts[eid] += 1
    total = sum(counts.values())
    return dict(counts), total


def obligation_pair_stats(graph: dict):
    """Pairwise overlap counts and Jaccard values over H_c."""
    obligations = coupling_obligations(graph)
    supports = {o["id"]: obligation_support(o) for o in obligations}
    ids = [o["id"] for o in obligations]
    stats = []
    for i, oid_i in enumerate(ids):
        si = supports[oid_i]
        for oid_j in ids[i + 1 :]:
            sj = supports[oid_j]
            inter = len(si & sj)
            union = len(si | sj)
            j = inter / union if union else 0.0
            stats.append((oid_i, oid_j, inter, union, j))
    return stats


# ---------------------------------------------------------------------------
# SOH-1.1 burden / temporal / exception extraction
# ---------------------------------------------------------------------------

def _temporal_offsets(o: dict) -> list[int]:
    """Collect numeric temporal offsets from an obligation's incidence slots.

    Reads temporal_offset (SOH-1.1) or lag_or_offset (SOH-1.0) per slot.
    """
    offsets = []
    for slot in o.get("incidence_slots", []):
        if not isinstance(slot, dict):
            continue
        toff = slot.get("temporal_offset")
        if toff is None:
            lag = slot.get("lag_or_offset")
            if isinstance(lag, str):
                # tolerate strings like "-1", "t-1", "prev"
                lag = lag.strip().lower()
                for token in lag.replace("t", " ").replace("t+", " ").split():
                    if token.lstrip("-").isdigit():
                        toff = int(token)
                        break
            elif isinstance(lag, (int, float)):
                toff = int(lag)
        if isinstance(toff, (int, float)) and not isinstance(toff, bool):
            offsets.append(int(toff))
    return offsets


def decision_lag(graph: dict) -> int:
    """L_decision: max absolute temporal offset across all obligations."""
    best = 0
    for o in all_obligations(graph):
        for off in _temporal_offsets(o):
            best = max(best, abs(off))
    return best


def exception_count(graph: dict) -> int:
    """N_exception: number of parameter_overrides across all obligations."""
    n = 0
    for o in all_obligations(graph):
        n += len(o.get("parameter_overrides", []) or [])
    return n


def conditional_count(graph: dict) -> int:
    """N_conditional: obligations whose scope_kind is conditional or subset."""
    n = 0
    for o in all_obligations(graph):
        kind = o.get("scope_kind")
        if kind in ("conditional", "subset"):
            n += 1
    return n


def burden_sidecar(graph: dict) -> dict:
    """B_S: (m_all, m_unary, m_rel, N_exception, L_decision, N_conditional).

    A   : (6, 4, 2, 0, 0, 0)
    A+  : (7, 5, 2, 1, 1, 0)
    """
    m_all = len(all_obligations(graph))
    m_unary = len(unary_obligations(graph))
    m_rel = len(coupling_obligations(graph))
    return {
        "m_all": m_all,
        "m_unary": m_unary,
        "m_rel": m_rel,
        "n_exception": exception_count(graph),
        "l_decision": decision_lag(graph),
        "n_conditional": conditional_count(graph),
    }
