"""Small DAG utilities over ``node -> predecessors`` mappings."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import TypeVar

N = TypeVar("N", bound=Hashable)


class CycleError(ValueError):
    """Raised when a precedence graph contains a cycle."""


def topological_order(preds: Mapping[N, Iterable[N]]) -> list[N]:
    """Kahn's algorithm; deterministic for a given mapping order.

    Every predecessor must itself be a key of ``preds``.
    """
    indegree = {node: 0 for node in preds}
    succs: dict[N, list[N]] = {node: [] for node in preds}
    for node, node_preds in preds.items():
        for pred in node_preds:
            succs[pred].append(node)
            indegree[node] += 1

    order = [node for node in preds if indegree[node] == 0]
    for node in order:  # grows while iterating
        for succ in succs[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                order.append(succ)

    if len(order) != len(indegree):
        stuck = [node for node in preds if indegree[node] > 0]
        raise CycleError(f"precedence cycle involving {stuck}")
    return order


def ancestors(preds: Mapping[N, Iterable[N]]) -> dict[N, set[N]]:
    """All transitive predecessors of every node."""
    result: dict[N, set[N]] = {}
    for node in topological_order(preds):
        acc: set[N] = set()
        for pred in preds[node]:
            acc.add(pred)
            acc |= result[pred]
        result[node] = acc
    return result


def transitive_reduction(preds: Mapping[N, Iterable[N]]) -> dict[N, tuple[N, ...]]:
    """Drop every edge implied by a longer path; the partial order is unchanged."""
    anc = ancestors(preds)
    reduced: dict[N, tuple[N, ...]] = {}
    for node, node_preds in preds.items():
        direct = list(dict.fromkeys(node_preds))
        reduced[node] = tuple(p for p in direct if not any(p in anc[q] for q in direct if q != p))
    return reduced


def contract(preds: Mapping[N, Iterable[N]], keep: Collection[N]) -> dict[N, tuple[N, ...]]:
    """Remove the nodes not in ``keep`` while preserving ordering through them.

    A kept node inherits the (kept) predecessors of each removed predecessor, so
    ``a -> x -> b`` with ``x`` removed becomes ``a -> b``.
    """
    effective: dict[N, tuple[N, ...]] = {}
    for node in topological_order(preds):
        acc: dict[N, None] = {}
        for pred in preds[node]:
            if pred in keep:
                acc[pred] = None
            else:
                acc.update(dict.fromkeys(effective[pred]))
        effective[node] = tuple(acc)
    return transitive_reduction({node: effective[node] for node in preds if node in keep})
