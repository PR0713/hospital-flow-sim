"""Allocation rules: which idle resources form the team of an operation.

A rule is a cost over (role, resource); the team is the feasible assignment of
least total cost. One implementation serves the environment and the baselines,
so nobody is compared under a different version of a rule.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from hospital_sim.sim.resources import Team
from hospital_sim.sim.simulator import Simulator

# cost(sim, role index, resource index) -> lower is preferred
Cost = Callable[[Simulator, int, int], float]
AllocationRule = Callable[[Simulator, int], Team | None]

_INFEASIBLE = 1e12
# Ties are broken by least cumulative work, then by declaration order.
_TIE_WORK = 1e-6
_TIE_INDEX = 1e-9
# Among equally flexible candidates for a speed-governing role, take the faster.
_TIE_SPEED = 1e-3


def min_cost_team(sim: Simulator, op: int, cost: Cost) -> Team | None:
    """The cheapest valid team for ``op`` among idle resources, or ``None``."""
    manager = sim.resources
    reqs = manager.requirements[sim.type_of[op]]
    if not reqs:
        return ()
    if not manager.can_staff(sim.type_of[op]):
        return None
    work = manager.work_time
    busiest = max(work) or 1.0

    def full_cost(role: int, r: int) -> float:
        return cost(sim, role, r) + _TIE_WORK * work[r] / busiest + _TIE_INDEX * r

    if manager.disjoint[sim.type_of[op]]:
        # No resource can fill two of these roles: choose per role.
        return tuple(
            tuple(
                sorted(sorted(manager.idle_members(role), key=lambda r: full_cost(role, r))[:qty])
            )
            for role, qty in reqs
        )

    slots = [role for role, qty in reqs for _ in range(qty)]
    candidates = sorted({r for role, _ in reqs for r in manager.idle_members(role)})
    eligible = [set(manager.members[role]) for role in slots]
    matrix = np.full((len(slots), len(candidates)), _INFEASIBLE)
    for i, role in enumerate(slots):
        for j, r in enumerate(candidates):
            if r in eligible[i]:
                matrix[i, j] = full_cost(role, r)
    rows, cols = linear_sum_assignment(matrix)
    if matrix[rows, cols].max() >= _INFEASIBLE:
        return None
    chosen: dict[int, list[int]] = {role: [] for role, _ in reqs}
    for i, j in zip(rows, cols, strict=True):
        chosen[slots[i]].append(candidates[j])
    return tuple(tuple(sorted(chosen[role])) for role, _ in reqs)


def _flexibility(sim: Simulator, role: int, r: int) -> float:
    # What assigning r takes away: one idle member from each role it can fill,
    # weighted by how few idle members that role has left.
    idle = sim.resources.idle_count
    return sum(1.0 / idle[other] for other in sim.resources.roles_of[r] if idle[other] > 0)


def flexible(sim: Simulator, op: int) -> Team | None:
    """Keep versatile and scarce resources free: prefer the resource whose
    assignment removes the least future staffing ability. Speed only breaks
    ties, and only in the roles that govern the operation's duration."""
    speed_roles = sim.speed_role_indices[sim.type_of[op]]

    def cost(s: Simulator, role: int, r: int) -> float:
        tie = _TIE_SPEED * s.resource_speed[r] if role in speed_roles else 0.0
        return _flexibility(s, role, r) - tie

    return min_cost_team(sim, op, cost)


def fastest(sim: Simulator, op: int) -> Team | None:
    """Put the fastest resources in the roles that govern speed; elsewhere
    behave like ``flexible``."""
    speed_roles = sim.speed_role_indices[sim.type_of[op]]

    def cost(s: Simulator, role: int, r: int) -> float:
        if role in speed_roles:
            return -100.0 * s.resource_speed[r]
        return _flexibility(s, role, r)

    return min_cost_team(sim, op, cost)


def least_used(sim: Simulator, op: int) -> Team | None:
    """Spread work evenly: prefer resources with the least cumulative work."""
    return min_cost_team(sim, op, lambda s, role, r: s.resources.work_time[r])


def first_idle(sim: Simulator, op: int) -> Team | None:
    """The first idle eligible resources in declaration order (the M2a default)."""
    return sim.resources.first_idle_team(sim.type_of[op])


RULES: dict[str, AllocationRule] = {
    "flexible": flexible,
    "fastest": fastest,
    "least_used": least_used,
    "first_idle": first_idle,
}
