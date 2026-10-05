"""Resource allocation manager.

Tracks which resources are available, decides whether an operation type can
be staffed right now, and acquires or releases whole teams atomically. A
resource is available when no operation holds it and nothing blocks it (off
duty, absent, broken down, or in turnover after its last operation). Resources,
roles and operation types are addressed by integer index (their position in
the scenario) because this sits on the simulator's hot path.

A team is a tuple aligned with the operation type's requirements:
``team[i]`` holds the resource indices filling ``requirements[i]``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from hospital_sim.domain.matching import find_team
from hospital_sim.domain.model import RequirementSpec, Scenario

Team = tuple[tuple[int, ...], ...]

FREE = -1


class AllocationError(ValueError):
    """A team violates eligibility, quantity, distinctness or availability."""


class ResourceManager:
    def __init__(self, scenario: Scenario) -> None:
        self.resource_ids = [r.id for r in scenario.resources]
        self.role_names = [role.name for role in scenario.roles]
        role_index = {name: i for i, name in enumerate(self.role_names)}
        resource_index = {rid: i for i, rid in enumerate(self.resource_ids)}

        self.members: list[tuple[int, ...]] = [
            tuple(resource_index[rid] for rid in scenario.resources_by_role[name])
            for name in self.role_names
        ]
        self._member_sets = [frozenset(m) for m in self.members]
        self.roles_of: list[tuple[int, ...]] = [
            tuple(sorted(role_index[name] for name in r.roles)) for r in scenario.resources
        ]

        # Per operation type: (role index, quantity) in requirement order.
        self.requirements: list[tuple[tuple[int, int], ...]] = [
            tuple((role_index[req.role], req.quantity) for req in t.requirements)
            for t in scenario.operation_types
        ]
        # If no resource is shared between two roles an operation type needs,
        # per-role idle counts decide feasibility; otherwise matching is needed.
        self.disjoint = [self._roles_disjoint(reqs) for reqs in self.requirements]
        self._match_specs = [
            [RequirementSpec(str(role), qty) for role, qty in reqs] for reqs in self.requirements
        ]

        self.holder: list[int] = []
        self.idle_count: list[int] = []
        # Cumulative busy time and finished operations per resource.
        self.work_time: list[float] = []
        self.jobs_done: list[int] = []
        # Number of reasons a resource may not start new work (0 = none).
        self.blocked: list[int] = []
        self.reset()

    def _roles_disjoint(self, reqs: Sequence[tuple[int, int]]) -> bool:
        seen: set[int] = set()
        for role, _ in reqs:
            if seen & self._member_sets[role]:
                return False
            seen |= self._member_sets[role]
        return True

    def reset(self) -> None:
        # holder[r] is the operation id using resource r, or FREE.
        self.holder = [FREE] * len(self.resource_ids)
        self.idle_count = [len(m) for m in self.members]
        self.work_time = [0.0] * len(self.resource_ids)
        self.jobs_done = [0] * len(self.resource_ids)
        self.blocked = [0] * len(self.resource_ids)

    # --- queries -----------------------------------------------------------

    def is_idle(self, resource: int) -> bool:
        """Free and not blocked: may be assigned to a new operation now."""
        return self.holder[resource] == FREE and self.blocked[resource] == 0

    def idle_members(self, role: int) -> list[int]:
        holder, blocked = self.holder, self.blocked
        return [r for r in self.members[role] if holder[r] == FREE and blocked[r] == 0]

    def can_staff(self, op_type: int) -> bool:
        """True iff a full team for this operation type is idle right now."""
        idle = self.idle_count
        for role, qty in self.requirements[op_type]:
            if idle[role] < qty:
                return False
        return self.disjoint[op_type] or self.first_idle_team(op_type) is not None

    def first_idle_team(self, op_type: int) -> Team | None:
        """The team made of the first idle eligible resources, or ``None``."""
        reqs = self.requirements[op_type]
        if self.disjoint[op_type]:
            team: list[tuple[int, ...]] = []
            for role, qty in reqs:
                idle = self.idle_members(role)
                if len(idle) < qty:
                    return None
                team.append(tuple(idle[:qty]))
            return tuple(team)
        matched = find_team(
            self._match_specs[op_type],
            {str(role): self.idle_members(role) for role, _ in reqs},
        )
        if matched is None:
            return None
        return tuple(matched[str(role)] for role, _ in reqs)

    # --- mutations ---------------------------------------------------------

    def validate(self, op_type: int, team: Team) -> None:
        reqs = self.requirements[op_type]
        if len(team) != len(reqs):
            raise AllocationError(f"team has {len(team)} role groups, operation needs {len(reqs)}")
        used: set[int] = set()
        for (role, qty), group in zip(reqs, team, strict=True):
            name = self.role_names[role]
            if len(group) != qty:
                raise AllocationError(
                    f"role '{name}' needs {qty} resources, team gives {len(group)}"
                )
            for r in group:
                rid = self.resource_ids[r]
                if r not in self._member_sets[role]:
                    raise AllocationError(f"resource '{rid}' is not eligible for role '{name}'")
                if self.holder[r] != FREE:
                    raise AllocationError(f"resource '{rid}' is busy")
                if self.blocked[r]:
                    raise AllocationError(f"resource '{rid}' is unavailable")
                if r in used:
                    raise AllocationError(f"resource '{rid}' appears twice in the team")
                used.add(r)

    def acquire(self, op: int, op_type: int, team: Team) -> None:
        """Take every resource of ``team`` for ``op``, or none of them."""
        self.validate(op_type, team)
        for group in team:
            for r in group:
                self.holder[r] = op
                for role in self.roles_of[r]:
                    self.idle_count[role] -= 1

    def release(self, team: Team, worked: float = 0.0, hold: Collection[int] = ()) -> None:
        """Free every resource of ``team`` together, crediting ``worked`` time.

        Resources in ``hold`` are freed but blocked once more (turnover); the
        caller unblocks them when the turnover ends.
        """
        for group in team:
            for r in group:
                self.holder[r] = FREE
                self.work_time[r] += worked
                self.jobs_done[r] += 1
                if r in hold:
                    self.blocked[r] += 1
                if self.blocked[r] == 0:
                    for role in self.roles_of[r]:
                        self.idle_count[role] += 1

    def block(self, resource: int) -> None:
        """Add a reason this resource may not start new work. A running
        operation is not interrupted."""
        self.blocked[resource] += 1
        if self.blocked[resource] == 1 and self.holder[resource] == FREE:
            for role in self.roles_of[resource]:
                self.idle_count[role] -= 1

    def unblock(self, resource: int) -> None:
        self.blocked[resource] -= 1
        if self.blocked[resource] == 0 and self.holder[resource] == FREE:
            for role in self.roles_of[resource]:
                self.idle_count[role] += 1
