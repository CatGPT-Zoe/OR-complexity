"""SOH-1.2 instance-layer metrics: the instance-level semantic obligation hypergraph.

Family-layer SOH treats each CSDE as one vertex; obligations connecting distinct
families are coupling edges (H_c^F). SOH-1.2 adds a second layer:

  H_I = (V_I, E_I)
    V_I : entity instances   (entity_id, {role: value} index binding)
    E_I : instance obligation edges (one instantiated rule per edge)

Dual of the family hypergraph:
  - vertices are INSTANCES of entity families (e.g. P(1,1), P(1,2), ..., P(3,3))
  - hyperedges are instance obligations (e.g. precedence edge P(1,1)--P(1,2),
    hub-usage edge P(1,1)--P(2,1)--P(3,1))

Information sources, in priority order:
  1. top-level instance_hyperedges (SOH-1.2 annotations)
  2. per-obligation instance_edges / incidence_instances (SOH-1.2 annotations)
  3. synthesis from incidence_slots.arguments index templates + sets domains
     (upgrade path for SOH-1.1 annotations)

Instance metrics complement C_S/B_S with an instance-layer summary:
  I_S = (nV_I, nE_I, arity_I, D_I^I, D_int^I, rho_I, LCC_I, ...)
and an intra-family coupling count (edges whose incident instances belong to a
single family) vs cross-family coupling.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import comb

import numpy as np

EPS = 1e-12


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return (mat + mat.T) / 2.0


def _spectral_radius(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    vals = np.linalg.eigvalsh(_symmetrize(mat))
    return float(np.max(np.abs(vals))) if len(vals) else 0.0


def _edge_count(adj: dict) -> int:
    return sum(len(v) for v in adj.values()) // 2


def _connected_components(adj: dict):
    seen = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        q = [start]
        seen.add(start)
        comp = []
        while q:
            v = q.pop()
            comp.append(v)
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def _largest_component_ratio(adj: dict) -> float:
    if not adj:
        return 0.0
    largest = max((len(c) for c in _connected_components(adj)), default=0)
    return float(largest / len(adj)) if adj else 0.0


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


# ---------------------------------------------------------------------------
# Instance-domain resolution
# ---------------------------------------------------------------------------

def _set_members(graph: dict, set_id: str, role_map: dict):
    """Return concrete member values for a set, honoring compound members."""
    for s in graph.get("sets", []):
        sid = s.get("id") or s.get("name")
        if sid is None or sid != set_id:
            continue
        mems = s.get("members")
        # dict shorthand: {"index_range": [1, 22]} -> expand to a range list
        if isinstance(mems, dict):
            rng = mems.get("index_range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = rng
                if all(isinstance(v, (int, float)) for v in rng):
                    return [i for i in range(int(lo), int(hi) + 1)]
            return []
        if mems is not None:
            if mems and isinstance(mems[0], dict):
                out = {}
                for obj in mems:
                    for role, val in obj.items():
                        out.setdefault(role, []).append(val)
                return out
            return [m for m in mems if not isinstance(m, dict)]
        if s.get("values") is not None:
            return list(s["values"])
        if s.get("elements") is not None:
            return list(s["elements"])
        if s.get("cardinality") is not None:
            try:
                return [i + 1 for i in range(int(s["cardinality"]))]
            except (TypeError, ValueError):
                return []
    return []


def _set_size(graph: dict, set_id: str) -> int | None:
    """Return domain cardinality for a set if determinable."""
    members = _set_members(graph, set_id, {})
    if members:
        return len(members)
    for s in graph.get("sets", []):
        if (s.get("id") or s.get("name")) == set_id and s.get("cardinality") is not None:
            try:
                return int(s["cardinality"])
            except (TypeError, ValueError):
                return None
    return None


def entity_roles(graph: dict, entity_id: str) -> list[str]:
    """Index role names of an entity family, in canonical order."""
    for e in graph.get("entities", []):
        if e.get("id") != entity_id:
            continue
        sig = e.get("index_signature") or e.get("indices") or []
        roles = []
        for item in sig:
            if isinstance(item, str):
                roles.append(item)
            elif isinstance(item, dict):
                if item.get("role") is not None:
                    roles.append(item["role"])
        if not roles and e.get("arguments"):
            roles = list(e["arguments"])
        return roles
    return []


def entity_role_sets(graph: dict, entity_id: str) -> dict:
    """Map role -> [concrete values] for an entity family's index domain."""
    out = {}
    for e in graph.get("entities", []):
        if e.get("id") != entity_id:
            continue
        sig = e.get("index_signature") or e.get("indices") or []
        for item in sig:
            if isinstance(item, dict) and item.get("set_id"):
                vals = _set_members(graph, item["set_id"], {})
                if vals:
                    out[item["role"]] = vals
        if not out and e.get("arguments"):
            # fall back: role names matching set ids/names contained in arguments
            from_str = e.get("index_signature") or e.get("indices") or []
            _ = from_str
        break
    return out


def _match_role_to_set(graph: dict, role: str):
    """Return concrete values for a role name by matching set id/name/canonical_name."""
    for s in graph.get("sets", []):
        sid = s.get("id") or s.get("name")
        if sid is None:
            continue
        names = [str(sid)]
        if s.get("canonical_name"):
            names.append(str(s["canonical_name"]))
        if s.get("name"):
            names.append(str(s["name"]))
        if any(role.lower() in n.lower() for n in names):
            vals = _set_members(graph, sid, {})
            if vals:
                return vals
    return []


# ---------------------------------------------------------------------------
# Edge extraction (priority: top-level edges > per-obligation edges > synthesis)
# ---------------------------------------------------------------------------

def _binding_from_slot(slot: dict) -> dict:
    """Normalize index binding from slot fields."""
    if isinstance(slot, dict):
        b = slot.get("binding")
        if isinstance(b, dict):
            return b
        b = slot.get("index_binding")
        if isinstance(b, dict):
            return b
        b = slot.get("index_bindings")
        if isinstance(b, dict):
            return b
        args = slot.get("arguments")
        if isinstance(args, list):
            return dict(enumerate(args))  # positional
    return {}


def _extract_top_level_edges(graph: dict) -> list | None:
    """Consume top-level instance_hyperedges if present."""
    hedges = graph.get("instance_hyperedges")
    if not isinstance(hedges, list) or not hedges:
        return None
    edges = []
    for he in hedges:
        vs = he.get("instance_vertices", [])
        if not isinstance(vs, list) or not vs:
            continue
        incidence = []
        for item in vs:
            if isinstance(item, str):
                incidence.append((item, {}, None))
                continue
            eid = item.get("entity_id")
            if not eid:
                continue
            b = item.get("index_binding") or item.get("binding") or {}
            incidence.append((eid, dict(b), item.get("temporal_role")))
        if incidence:
            edges.append({
                "id": he.get("id") or f"edge_{len(edges)}",
                "family": he.get("obligation_family") or he.get("obligation_id") or "unknown",
                "incidence": incidence,
                "boundary_kind": he.get("boundary_kind", "generic"),
                "source": "top_level",
            })
    return edges or None


def _extract_obligation_edges(graph: dict, obligation: dict) -> list:
    """Explicit edge list from instance_edges, else one canonical edge from
    incidence_instances."""
    edges = []
    iedges = obligation.get("instance_edges")
    if isinstance(iedges, list) and iedges:
        for i, ie in enumerate(iedges):
            slots = ie.get("slots") if isinstance(ie, dict) else None
            if not isinstance(slots, list):
                continue
            inc = []
            for slot in slots:
                if isinstance(slot, str):
                    inc.append((slot, {}, None))
                    continue
                eid = slot.get("entity_id")
                if not eid:
                    continue
                b = slot.get("index_binding") or slot.get("binding") or {}
                inc.append((eid, dict(b), slot.get("temporal_role")))
            if inc:
                edges.append({
                    "id": ie.get("id") if isinstance(ie, dict) else f"e{i}",
                    "family": obligation.get("canonical_family") or obligation.get("id"),
                    "incidence": inc,
                    "boundary_kind": (ie.get("boundary_kind") if isinstance(ie, dict) else None) or "generic",
                    "source": "instance_edges",
                })
        return edges
    inc_inst = obligation.get("incidence_instances")
    if isinstance(inc_inst, list) and inc_inst:
        inc = []
        for item in inc_inst:
            if isinstance(item, str):
                inc.append((item, {}, None))
                continue
            eid = item.get("entity_id")
            if not eid:
                continue
            b = item.get("binding") or item.get("index_binding") or {}
            inc.append((eid, dict(b), item.get("temporal_role")))
        if inc:
            return [{
                "id": obligation.get("id"),
                "family": obligation.get("canonical_family") or obligation.get("id"),
                "incidence": inc,
                "boundary_kind": obligation.get("scope_kind", "generic"),
                "source": "incidence_instances",
            }]
    return []


# ---------------------------------------------------------------------------
# Synthesis from SOH-1.1 index-templated incidence slots
# ---------------------------------------------------------------------------

def _parse_arg(expr) -> tuple:
    """Parse an index expression into (basename, offset) or (literal, None).

    'stage'       -> ('stage', 0)
    'stage+1'     -> ('stage', 1)
    'route_1'     -> ('route_1', 0)
    3             -> (3, None)  literal
    'Route 1'     -> ('Route 1', None) literal
    """
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return (expr, None)
    s = str(expr).strip()
    m = __import__("re").fullmatch(r"([A-Za-z_]\w*)\s*([+-]\s*\d+)?", s)
    if m:
        base = m.group(1)
        off = int(m.group(2).replace(" ", "")) if m.group(2) else 0
        return (base, off)
    return (s, None)


def _resolve_binding(binding: dict, domain: dict, graph: dict) -> list:
    """Resolve a symbolic binding to concrete {role: value} bindings.

    Returns a list of candidate concrete bindings (cross product over each
    role whose current value is symbolic and resolvable from sets).
    """
    concrete = {}
    symbolic = {}
    for role, val in binding.items():
        if isinstance(val, (int, float)):
            concrete[str(role)] = val
            continue
        base, off = _parse_arg(val)
        if off is not None and not isinstance(base, (int, float)):
            vals = domain.get(base) or _match_role_to_set(graph, base)
            if vals:
                symbolic[str(role)] = ([_offset_by(v, off) for v in vals], base)
                continue
        # literal
        concrete[str(role)] = val
    if not symbolic:
        return [concrete]
    keys = list(symbolic.keys())
    result = []
    for combo in product(*(symbolic[k][0] for k in keys)):
        b = dict(concrete)
        for k, v in zip(keys, combo):
            b[k] = v
        result.append(b)
    return result


def _offset_by(v, off: int):
    if isinstance(v, int):
        return v + off
    return v


def _synthesize_edges(graph: dict, obligation: dict, max_edges: int = 512) -> list:
    """Synthesize concrete instance edges from incidence_slots index templates."""
    slots = obligation.get("incidence_slots")
    if not isinstance(slots, list) or len(slots) < 2:
        return []
    parsed = []
    domains = {}
    for slot in slots:
        if not isinstance(slot, dict):
            return []
        eid = slot.get("entity_id")
        if not eid:
            return []
        args = slot.get("arguments") or []
        binding = _binding_from_slot(slot)
        if not args and binding:
            args = [f"{r}@{v}" if not isinstance(v, (int, float)) else f"{r}@{v}" for r, v in binding.items()]
        if not args:
            return []
        parsed.append((eid, args, binding))
        for arg in args:
            base, off = _parse_arg(arg)
            if off is not None and not isinstance(base, (int, float)):
                vals = entity_role_sets(graph, eid).get(base) or _match_role_to_set(graph, base)
                if vals:
                    domains[f"{eid}:{base}"] = vals
    if not domains:
        return []
    # resolve each slot against domains: produce list of candidate bindings per slot
    slot_candidates = []
    for eid, args, binding in parsed:
        if binding:
            cands = _resolve_binding(binding, {k.split(":")[1]: v for k, v in domains.items() if k.startswith(eid + ":") or True}, graph)
            slot_candidates.append([(eid, c, None) for c in cands])
        else:
            roles = [a for a in args if _parse_arg(a)[1] is not None]
            per_slot = []
            for arg in args:
                base, off = _parse_arg(arg)
                vals = domains.get(f"{eid}:{base}")
                if vals:
                    per_slot.append([(v, off) for v in vals])
            # cross product of arg positions
            if not per_slot:
                continue
            for combo in product(*per_slot):
                b = {}
                for role_item, (v, _off) in zip(args, combo):
                    b[role_item] = v
                slot_candidates.append([(eid, b, None)])
    if not slot_candidates:
        return []
    edges = []
    seen = set()
    count = 0
    for combo in product(*slot_candidates):
        count += 1
        if count > max_edges:
            break
        # dedupe edge by (entity_id, frozen binding) incidence set
        key = tuple(sorted((eid, tuple(sorted((str(k), repr(v)) for k, v in b.items()))) for eid, b, _ in combo))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "id": f"{obligation.get('id')}_syn_{count}",
            "family": obligation.get("canonical_family") or obligation.get("id"),
            "incidence": [(eid, b, None) for eid, b, _ in combo],
            "boundary_kind": "synthesized",
            "source": "synthesized",
        })
    return edges


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _norm_val(v):
    """Normalize binding values for vertex identity: digit-only strings == ints."""
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except ValueError:
                return v
    return v


def _add_instance_sets(incidence, max_size: int = 64):
    """Normalize an edge's incidence into a set of (entity_id, frozen_binding)."""
    out = set()
    for eid, b, _role in incidence:
        frozen = tuple(sorted((str(k), repr(_norm_val(v))) for k, v in b.items()))
        out.add((eid, frozen))
    return out


def compute_instance_metrics(graph: dict) -> dict:
    """Compute the SOH-1.2 instance-layer metric vector.

    Returns a dict (empty-safe) with keys prefixed for flattening:
      scale, arity, coupling, primal (spectral/LCC/cycle/hub), vertex stats.
    """
    edges = _extract_top_level_edges(graph) or []
    if not edges:
        for o in graph.get("obligations", []):
            edges.extend(_extract_obligation_edges(graph, o))
        if not edges:
            for o in graph.get("obligations", []):
                edges.extend(_synthesize_edges(graph, o))

    # total instance-domain size across indexed families
    domain_total = 0
    indexed_families = 0
    for e in graph.get("entities", []):
        roles = entity_roles(graph, e.get("id"))
        dom = entity_role_sets(graph, e.get("id"))
        if e.get("instance_level") is True and roles:
            size = 1
            for role in roles:
                vals = dom.get(role) or _match_role_to_set(graph, role)
                if not vals:
                    size = None
                    break
                size *= len(vals)
            if size is not None:
                domain_total += size
                indexed_families += 1

    # vertex universe: union of instances over all edges
    vertex_universe = {}
    edge_sets = []
    for e in edges:
        eset = _add_instance_sets(e["incidence"])
        edge_sets.append(eset)
        for eid, frozen in eset:
            vertex_universe[(eid, frozen)] = 1

    n_vertices = len(vertex_universe)
    n_edges = len(edge_sets)
    incidence_total = sum(len(s) for s in edge_sets)
    arities = [len(s) for s in edge_sets]

    # family-level attribution
    edge_families = [len({eid for eid, _ in s}) for s in edge_sets]
    intra_family = sum(1 for f in edge_families if f == 1)
    cross_family = sum(1 for f in edge_families if f >= 2)

    mean_arity = float(sum(arities) / len(arities)) if arities else 0.0
    max_arity = max(arities) if arities else 0
    incidence_density = float(incidence_total / (n_vertices * n_edges)) if n_vertices and n_edges else 0.0
    coupling_edges = sum(1 for a in arities if a >= 2)
    unary_instance_edges = sum(1 for a in arities if a < 2)

    # overlap stats on instance incidence
    num_pairs = comb(n_edges, 2) if n_edges >= 2 else 0
    positive_pairs = 0
    inter_sum = 0
    jac_sum = 0
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            inter = len(edge_sets[i] & edge_sets[j])
            if inter > 0:
                positive_pairs += 1
                inter_sum += inter
                union = len(edge_sets[i] | edge_sets[j])
                jac_sum += inter / union if union else 0.0
    intersection_density = float(positive_pairs / num_pairs) if num_pairs else 0.0
    mean_positive_overlap = float(inter_sum / positive_pairs) if positive_pairs else None
    mean_instance_jaccard = float(jac_sum / num_pairs) if num_pairs else 0.0

    # instance primal graph: instances adjacent if co-occur in an edge
    adj = {v: set() for v in vertex_universe}
    for s in edge_sets:
        vs = list(s)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                adj[vs[i]].add(vs[j])
                adj[vs[j]].add(vs[i])
    n = len(adj)
    degrees = [len(nbs) for nbs in adj.values()]
    P = np.zeros((n, n), dtype=float)
    idx = {v: i for i, v in enumerate(adj)}
    for s in edge_sets:
        vs = list(s)
        if not n_edges:
            continue
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                P[idx[vs[i]], idx[vs[j]]] += 1.0 / n_edges if n_edges else 0.0
                P[idx[vs[j]], idx[vs[i]]] += 1.0 / n_edges if n_edges else 0.0
    spectral = _spectral_radius(P) if n else 0.0
    spectral_norm = float(spectral / max(n - 1, 1)) if n else 0.0

    family_involved = len({eid for s in edge_sets for eid, _ in s})

    return {
        "instance": {
            "n_families_indexed": int(indexed_families),
            "domain_total": int(domain_total),
            "n_vertices": int(n_vertices),
            "n_edges": int(n_edges),
            "edges_top_level": int(sum(1 for e in edges if e["source"] == "top_level")),
            "edges_explicit": int(sum(1 for e in edges if e["source"] != "synthesized")),
            "edges_synthesized": int(sum(1 for e in edges if e["source"] == "synthesized")),
            "incidence": int(incidence_total),
            "mean_arity": float(mean_arity),
            "max_arity": int(max_arity),
            "incidence_density": float(incidence_density),
            "coupling_edges": int(coupling_edges),
            "unary_instance_edges": int(unary_instance_edges),
            "intra_family_edges": int(intra_family),
            "cross_family_edges": int(cross_family),
            "family_involved": int(family_involved),
            "vertices_per_family": float(n_vertices / family_involved) if family_involved else 0.0,
            "coupling_ratio": float(cross_family / n_edges) if n_edges else 0.0,
            "intersection_density": float(intersection_density),
            "mean_positive_overlap": (float(mean_positive_overlap) if mean_positive_overlap is not None else None),
            "mean_instance_jaccard": float(mean_instance_jaccard),
            "spectral_coupling_index": float(spectral_norm),
            "spectral_radius_raw": float(spectral),
            "largest_component_ratio": _largest_component_ratio(adj),
            "cycle_rank": int(_cycle_rank(adj)),
            "cycle_density": float(_normalized_cycle_density(adj)),
            "hubness": float(_freeman_centralization(adj)),
            "max_degree": int(max(degrees)) if degrees else 0,
            "primal_edges": int(_edge_count(adj)),
        }
    }