"""Search-based reference for team allocation on small instances.

For a fixed dispatching rule, try every way of forming each team and keep the
best episode. This is a hindsight bound: the search sees how each choice
actually turns out, which no online rule can. It tells us how much any
allocation rule could possibly gain under that dispatcher on that instance.

Resources that are indistinguishable (same roles, attributes, roster, no
individual randomness) are interchangeable, so teams are enumerated up to
that symmetry. This keeps the tree small enough for a handful of patients.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from itertools import combinations

from hospital_sim.baselines.dispatching import DISPATCHERS
from hospital_sim.domain.model import ProblemInstance
from hospital_sim.sim.resources import Team
from hospital_sim.sim.simulator import Simulator

Objective = Callable[[Simulator], float]


class SearchTooLarge(RuntimeError):
    """The instance has more allocation combinations than the leaf budget."""


def _resource_class(sim: Simulator, r: int) -> Hashable:
    resource = sim.instance.scenario.resources[r]
    calendar = (
        None
        if resource.calendar is None
        else sim.instance.scenario.calendar_by_name[resource.calendar]
    )
    individual = resource.breakdown is not None or (
        calendar is not None and calendar.absence_probability > 0
    )
    return (
        sim.resources.roles_of[r],
        tuple(sorted(resource.attributes.items())),
        resource.calendar,
        r if individual else None,  # own random downtime: not interchangeable
    )


def distinct_teams(sim: Simulator, op: int) -> list[Team]:
    """Every valid team for ``op`` from idle resources, one per symmetry class."""
    manager = sim.resources
    reqs = manager.requirements[sim.type_of[op]]
    seen: set[Hashable] = set()
    teams: list[Team] = []

    def extend(position: int, used: frozenset[int], partial: tuple[tuple[int, ...], ...]) -> None:
        if position == len(reqs):
            signature = tuple(
                tuple(sorted(str(_resource_class(sim, r)) for r in group)) for group in partial
            )
            if signature not in seen:
                seen.add(signature)
                teams.append(partial)
            return
        role, qty = reqs[position]
        idle = [r for r in manager.idle_members(role) if r not in used]
        tried: set[Hashable] = set()
        for group in combinations(idle, qty):
            key = tuple(sorted(str(_resource_class(sim, r)) for r in group))
            if key in tried:
                continue
            tried.add(key)
            extend(position + 1, used | set(group), (*partial, group))

    extend(0, frozenset(), ())
    return teams


def best_allocation(
    instance: ProblemInstance,
    dispatcher: str,
    objective: Objective,
    *,
    max_leaves: int = 20_000,
) -> tuple[float, int]:
    """Lowest ``objective`` over all allocations under ``dispatcher``.

    Returns ``(best value, number of complete schedules tried)``.
    """
    dispatch = DISPATCHERS[dispatcher]

    def play(prefix: tuple[int, ...]) -> tuple[float, list[int]]:
        """Follow ``prefix`` (team index per start), then always team 0."""
        sim = Simulator(instance)
        option_counts: list[int] = []
        while sim.run_until_decision():
            op = dispatch(sim, sim.startable_ops())
            teams = distinct_teams(sim, op)
            index = prefix[len(option_counts)] if len(option_counts) < len(prefix) else 0
            option_counts.append(len(teams))
            sim.start(op, teams[index])
        return objective(sim), option_counts

    best = float("inf")
    leaves = 0
    stack: list[tuple[int, ...]] = [()]
    while stack:
        prefix = stack.pop()
        value, counts = play(prefix)
        leaves += 1
        if leaves > max_leaves:
            raise SearchTooLarge(f"more than {max_leaves} allocation combinations")
        best = min(best, value)
        path = prefix + (0,) * (len(counts) - len(prefix))
        # Siblings of every choice made beyond the prefix; each leaf is visited once.
        for depth in range(len(prefix), len(counts)):
            for alternative in range(1, counts[depth]):
                stack.append((*path[:depth], alternative))
    return best, leaves
