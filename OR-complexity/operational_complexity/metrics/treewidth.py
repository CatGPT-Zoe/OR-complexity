"""Treewidth computation: exact branch-and-bound for small graphs, bounds for large.

Implements the v1.0 spec's recommendation:
- small graphs (N <= cutoff): exact treewidth via branch-and-bound over
  elimination orders (QuickBB-style), with min-fill upper bound and
  degeneracy lower bound;
- larger graphs: report (lower_bound, upper_bound) and status="bounded".

The treewidth of a graph equals the minimum, over all vertex elimination
orders, of the maximum degree of a vertex at the moment it is eliminated
(fill edges added). This module never claims a tractability theorem; it only
computes the combinatorial parameter.
"""

from __future__ import annotations

import time
from typing import Optional

#: graphs larger than this are not searched exactly
DEFAULT_CUTOFF = 25
#: per-call wall-clock budget for the exact search
DEFAULT_TIME_LIMIT = 30.0


def _min_fill_upper_bound(adj: dict) -> int:
    """Greedy min-fill elimination: an upper bound on treewidth.

    adj: {vertex: set(neighbors)}. Returns the width of the greedy order.
    """
    cur = {v: set(nb) for v, nb in adj.items()}
    width = 0
    while cur:
        best_v, best_score = None, None
        for v in cur:
            nb = list(cur[v])
            fill = 0
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] not in cur[nb[i]]:
                        fill += 1
            score = (fill, len(nb), v)
            if best_score is None or score < best_score:
                best_score, best_v = score, v
        v = best_v
        nb = list(cur[v])
        width = max(width, len(nb))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                cur[nb[i]].add(nb[j])
                cur[nb[j]].add(nb[i])
        for u in nb:
            cur[u].discard(v)
        del cur[v]
    return width


def _degeneracy_lower_bound(adj: dict) -> int:
    """Degeneracy: min-degree removal lower bound on treewidth."""
    cur = {v: set(nb) for v, nb in adj.items()}
    lb = 0
    while cur:
        v = min(cur, key=lambda u: len(cur[u]))
        lb = max(lb, len(cur[v]))
        for u in cur[v]:
            cur[u].discard(v)
        del cur[v]
    return lb


def _exact_search(
    adj: dict,
    time_limit: float,
) -> Optional[int]:
    """Branch-and-bound over elimination orders.

    Returns the exact treewidth if the search completes within the time
    budget, otherwise None (caller falls back to bounds).
    """
    start = time.time()
    best = _min_fill_upper_bound(adj)

    def rec(cur: dict, current_width: int) -> bool:
        # returns False on timeout
        nonlocal best
        if time.time() - start > time_limit:
            return False
        if current_width >= best:
            return True
        if not cur:
            best = current_width
            return True
        items = []
        for v in cur:
            nb = list(cur[v])
            fill = 0
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] not in cur[nb[i]]:
                        fill += 1
            items.append((fill, len(nb), v))
        items.sort()
        for _, deg, v in items:
            new_width = max(current_width, deg)
            if new_width >= best:
                continue
            nb = list(cur[v])
            new_cur = {u: set(nb_u) for u, nb_u in cur.items() if u != v}
            for u in nb:
                new_cur[u].discard(v)
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    new_cur[nb[i]].add(nb[j])
                    new_cur[nb[j]].add(nb[i])
            if not rec(new_cur, new_width):
                return False
        return True

    if not rec(adj, 0):
        return None
    return best


def treewidth_or_bounds(
    adj: dict,
    cutoff: int = DEFAULT_CUTOFF,
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> dict:
    """Compute exact treewidth or certified bounds.

    Args:
        adj: adjacency as {vertex: set(neighbors)} (undirected).
        cutoff: max vertex count for the exact search.
        time_limit: wall-clock budget for the exact search (seconds).

    Returns:
        {"status": "exact"|"bounded", "lower_bound": int,
         "upper_bound": int, "value": int|None}
    """
    n = len(adj)
    if n <= 1:
        return {"status": "exact", "lower_bound": 0, "upper_bound": 0, "value": 0}

    lb = _degeneracy_lower_bound(adj)
    ub = _min_fill_upper_bound(adj)

    if lb == ub:
        return {"status": "exact", "lower_bound": lb, "upper_bound": ub, "value": lb}

    if n > cutoff:
        return {"status": "bounded", "lower_bound": lb, "upper_bound": ub, "value": None}

    exact = _exact_search(adj, time_limit)
    if exact is None:
        return {"status": "bounded", "lower_bound": lb, "upper_bound": ub, "value": None}
    return {
        "status": "exact",
        "lower_bound": exact,
        "upper_bound": exact,
        "value": exact,
    }
