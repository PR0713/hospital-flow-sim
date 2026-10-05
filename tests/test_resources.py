from __future__ import annotations

import pytest

from hospital_sim.sim.resources import FREE, AllocationError, ResourceManager

from .conftest import make_scenario


@pytest.fixture
def manager() -> ResourceManager:
    scenario = make_scenario(
        resources={"Nurse": 3, "Room": 1},
        op_types={
            "Check": ({"Nurse": 1}, 5, True),
            "Procedure": ({"Room": 1, "Nurse": 2}, 5, True),
            "Surgery": ({"Lead": 1, "Assistant": 1}, 5, True),
        },
        pathways={"p": [("a", "Check", []), ("b", "Procedure", []), ("c", "Surgery", [])]},
        extra_resources=[
            {"id": "senior", "roles": ["Lead", "Assistant"]},
            {"id": "junior", "roles": ["Assistant"]},
        ],
    )
    return ResourceManager(scenario)


CHECK, PROCEDURE, SURGERY = 0, 1, 2


def ids(manager: ResourceManager, *names: str) -> tuple[int, ...]:
    return tuple(manager.resource_ids.index(n) for n in names)


def test_first_idle_team_follows_the_requirement_vector(manager: ResourceManager) -> None:
    team = manager.first_idle_team(PROCEDURE)
    assert team == (ids(manager, "room_1"), ids(manager, "nurse_1", "nurse_2"))


def test_acquire_marks_every_member_busy_and_release_frees_all(manager: ResourceManager) -> None:
    team = manager.first_idle_team(PROCEDURE)
    assert team is not None
    manager.acquire(7, PROCEDURE, team)
    held = [r for r, h in enumerate(manager.holder) if h == 7]
    assert sorted(held) == sorted(r for g in team for r in g)
    manager.release(team)
    assert all(h == FREE for h in manager.holder)
    assert manager.idle_count == [len(m) for m in manager.members]


def test_cannot_staff_when_one_role_is_short(manager: ResourceManager) -> None:
    manager.acquire(1, CHECK, (ids(manager, "nurse_1"),))
    manager.acquire(2, CHECK, (ids(manager, "nurse_2"),))
    assert manager.can_staff(CHECK)  # one nurse left
    assert not manager.can_staff(PROCEDURE)  # room is free but only one nurse
    assert manager.first_idle_team(PROCEDURE) is None


def test_no_double_booking(manager: ResourceManager) -> None:
    manager.acquire(1, CHECK, (ids(manager, "nurse_1"),))
    with pytest.raises(AllocationError, match="busy"):
        manager.acquire(2, CHECK, (ids(manager, "nurse_1"),))


def test_ineligible_resource_is_rejected(manager: ResourceManager) -> None:
    with pytest.raises(AllocationError, match="not eligible for role 'Nurse'"):
        manager.acquire(1, CHECK, (ids(manager, "room_1"),))


@pytest.mark.parametrize(
    ("team_names", "message"),
    [
        ((("room_1",), ("nurse_1",)), "needs 2 resources"),
        ((("room_1",), ("nurse_1", "nurse_1")), "appears twice"),
        ((("room_1",),), "role groups"),
        ((("room_1",), ("nurse_1", "room_1")), "not eligible"),
    ],
)
def test_failed_acquire_holds_nothing(
    manager: ResourceManager, team_names: tuple[tuple[str, ...], ...], message: str
) -> None:
    """All-or-nothing: an invalid team must not leave any resource held."""
    team = tuple(ids(manager, *group) for group in team_names)
    with pytest.raises(AllocationError, match=message):
        manager.acquire(1, PROCEDURE, team)
    assert all(h == FREE for h in manager.holder)
    assert manager.idle_count == [len(m) for m in manager.members]


def test_partially_busy_team_is_rejected_whole(manager: ResourceManager) -> None:
    manager.acquire(1, CHECK, (ids(manager, "nurse_2"),))
    with pytest.raises(AllocationError, match="busy"):
        manager.acquire(2, PROCEDURE, (ids(manager, "room_1"), ids(manager, "nurse_1", "nurse_2")))
    assert manager.is_idle(ids(manager, "room_1")[0])
    assert manager.is_idle(ids(manager, "nurse_1")[0])


def test_shared_roles_use_matching(manager: ResourceManager) -> None:
    # The senior must lead; first-idle for Assistant alone would grab the senior.
    assert manager.first_idle_team(SURGERY) == (ids(manager, "senior"), ids(manager, "junior"))


def test_shared_roles_counts_are_not_enough(manager: ResourceManager) -> None:
    # Junior busy: one idle Lead and one idle Assistant remain, but it is one person.
    junior = ids(manager, "junior")[0]
    manager.holder[junior] = 99
    for role in manager.roles_of[junior]:
        manager.idle_count[role] -= 1
    assert manager.idle_count[manager.role_names.index("Lead")] == 1
    assert manager.idle_count[manager.role_names.index("Assistant")] == 1
    assert not manager.can_staff(SURGERY)


def test_resource_in_two_roles_updates_both_idle_counts(manager: ResourceManager) -> None:
    lead, assistant = manager.role_names.index("Lead"), manager.role_names.index("Assistant")
    manager.acquire(1, SURGERY, (ids(manager, "senior"), ids(manager, "junior")))
    assert manager.idle_count[lead] == 0 and manager.idle_count[assistant] == 0
