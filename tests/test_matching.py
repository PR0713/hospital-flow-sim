from __future__ import annotations

from hospital_sim.domain import RequirementSpec, find_team

SURGERY = [RequirementSpec("Lead"), RequirementSpec("Assistant")]


def test_disjoint_roles_fill_by_count() -> None:
    team = find_team(
        [RequirementSpec("Nurse", 2), RequirementSpec("Room")],
        {"Nurse": ["n1", "n2", "n3"], "Room": ["r1"]},
    )
    assert team == {"Nurse": ("n1", "n2"), "Room": ("r1",)}


def test_insufficient_count_is_infeasible() -> None:
    assert find_team([RequirementSpec("Nurse", 3)], {"Nurse": ["n1", "n2"]}) is None


def test_missing_role_is_infeasible() -> None:
    assert find_team([RequirementSpec("Nurse")], {}) is None


def test_one_dual_qualified_resource_cannot_fill_two_slots() -> None:
    # Per-role counts say 1 lead and 1 assistant are available; it is one person.
    assert find_team(SURGERY, {"Lead": ["s1"], "Assistant": ["s1"]}) is None


def test_matching_reassigns_to_make_room() -> None:
    # Greedy gives s1 to Assistant first, leaving nobody to lead; the matching
    # must move the assistant slot to s2 so that s1 can lead.
    team = find_team(
        [RequirementSpec("Assistant"), RequirementSpec("Lead")],
        {"Assistant": ["s1", "s2"], "Lead": ["s1"]},
    )
    assert team == {"Assistant": ("s2",), "Lead": ("s1",)}


def test_team_never_uses_a_resource_twice() -> None:
    team = find_team(
        [RequirementSpec("Lead"), RequirementSpec("Assistant", 2)],
        {"Lead": ["s1", "s2"], "Assistant": ["s1", "s2", "s3"]},
    )
    assert team is not None
    members = [r for group in team.values() for r in group]
    assert len(members) == len(set(members)) == 3
    assert len(team["Lead"]) == 1 and len(team["Assistant"]) == 2
