from __future__ import annotations

import pytest

from hospital_sim.domain.dag import (
    CycleError,
    ancestors,
    contract,
    topological_order,
    transitive_reduction,
)


def test_topological_order_respects_precedence() -> None:
    preds = {"d": ["b", "c"], "b": ["a"], "c": ["a"], "a": []}
    order = topological_order(preds)
    assert sorted(order) == ["a", "b", "c", "d"]
    for node, node_preds in preds.items():
        assert all(order.index(p) < order.index(node) for p in node_preds)


def test_cycle_is_detected() -> None:
    with pytest.raises(CycleError):
        topological_order({"a": ["c"], "b": ["a"], "c": ["b"]})


def test_ancestors_are_transitive() -> None:
    assert ancestors({"a": [], "b": ["a"], "c": ["b"]})["c"] == {"a", "b"}


def test_transitive_reduction_drops_implied_edges() -> None:
    reduced = transitive_reduction({"a": [], "b": ["a"], "c": ["a", "b"]})
    assert reduced == {"a": (), "b": ("a",), "c": ("b",)}


def test_contract_bridges_over_removed_nodes() -> None:
    preds = {"a": [], "x": ["a"], "b": ["x"], "c": []}
    assert contract(preds, {"a", "b", "c"}) == {"a": (), "b": ("a",), "c": ()}


def test_contract_bridges_over_chains_of_removed_nodes() -> None:
    preds = {"a": [], "x": ["a"], "y": ["x"], "b": ["y"]}
    assert contract(preds, {"a", "b"}) == {"a": (), "b": ("a",)}


def test_contract_removed_root_leaves_successor_free() -> None:
    assert contract({"x": [], "b": ["x"]}, {"b"}) == {"b": ()}
