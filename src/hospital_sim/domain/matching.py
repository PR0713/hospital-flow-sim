"""Team feasibility as bipartite matching.

When a resource is eligible for several roles, per-role counts are not enough:
one surgeon qualified as both lead and assistant cannot fill both slots of the
same operation. Each unit of each requirement is a slot, and a team exists iff
every slot can be matched to a distinct resource.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

from hospital_sim.domain.model import RequirementSpec

R = TypeVar("R", bound=Hashable)


def find_team(
    requirements: Sequence[RequirementSpec],
    candidates: Mapping[str, Sequence[R]],
) -> dict[str, tuple[R, ...]] | None:
    """Return one valid ``role -> resources`` assignment, or ``None`` if none exists.

    ``candidates[role]`` lists the usable resources for that role in preference
    order; earlier candidates are tried first, so the result is deterministic.
    """
    slots = [req.role for req in requirements for _ in range(req.quantity)]
    slot_of: dict[R, int] = {}

    def assign(slot: int, visited: set[R]) -> bool:
        for resource in candidates.get(slots[slot], ()):
            if resource in visited:
                continue
            visited.add(resource)
            if resource not in slot_of or assign(slot_of[resource], visited):
                slot_of[resource] = slot
                return True
        return False

    for slot in range(len(slots)):
        if not assign(slot, set()):
            return None

    team: dict[str, list[R]] = {req.role: [] for req in requirements}
    for role in team:
        team[role] = [r for r in candidates[role] if r in slot_of and slots[slot_of[r]] == role]
    return {role: tuple(members) for role, members in team.items()}
