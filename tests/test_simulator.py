from __future__ import annotations

import math
import random
from typing import Any

import pytest

from hospital_sim.domain import Scenario
from hospital_sim.generators import generate_instance
from hospital_sim.sim import (
    AllocationError,
    InvariantViolation,
    OpState,
    SimulationError,
    Simulator,
    check_invariants,
    compute_metrics,
    fifo,
    run,
)

from .conftest import make_scenario


def sim_of(scenario: Scenario, seed: int = 0) -> Simulator:
    return Simulator(generate_instance(scenario, seed), debug=True)


def op(sim: Simulator, patient: int, step: str) -> int:
    return next(o.id for o in sim.instance.patients[patient].operations if o.step_id == step)


def by_step(sim: Simulator, patient: int = 0) -> dict[str, tuple[float, float]]:
    return {
        o.step_id: (sim.start_time[o.id], sim.end_time[o.id])
        for o in sim.instance.patients[patient].operations
    }


# --- hand-checkable deterministic schedules -----------------------------------


def test_dag_runs_independent_operations_in_parallel() -> None:
    # a(5) and b(7) are independent, c(10) needs both: makespan = 7 + 10.
    scenario = make_scenario(
        resources={"Nurse": 3},
        op_types={
            "A": ({"Nurse": 1}, 5, False),
            "B": ({"Nurse": 1}, 7, False),
            "C": ({"Nurse": 1}, 10, False),
        },
        pathways={"p": [("a", "A", []), ("b", "B", []), ("c", "C", ["a", "b"])]},
    )
    sim = run(sim_of(scenario))
    assert by_step(sim) == {"a": (0, 5), "b": (0, 7), "c": (7, 17)}
    assert compute_metrics(sim).makespan == 17


def test_patient_presence_serializes_otherwise_parallel_operations() -> None:
    scenario = make_scenario(
        resources={"Nurse": 3},
        op_types={"A": ({"Nurse": 1}, 5, True), "B": ({"Nurse": 1}, 7, True)},
        pathways={"p": [("a", "A", []), ("b", "B", [])]},
    )
    sim = sim_of(scenario)
    assert sim.run_until_decision()
    sim.start(op(sim, 0, "a"))
    assert not sim.can_start(op(sim, 0, "b"))  # nurses are free, the patient is not
    with pytest.raises(SimulationError, match="patient 0 is busy"):
        sim.start(op(sim, 0, "b"))
    run(sim)
    assert by_step(sim) == {"a": (0, 5), "b": (5, 12)}


CONTENTION_TYPES: dict[str, tuple[dict[str, int], float, bool]] = {
    "Consult": ({"Nurse": 1, "Doctor": 1}, 10, True),
    "Scan": ({"Nurse": 1, "Room": 1}, 6, True),
}
CONTENTION_PATHWAYS = {"consult": [("x", "Consult", [])], "scan": [("y", "Scan", [])]}


def contention(nurses: int) -> Simulator:
    scenario = make_scenario(
        resources={"Nurse": nurses, "Doctor": 1, "Room": 1},
        op_types=CONTENTION_TYPES,
        pathways=CONTENTION_PATHWAYS,
        arrivals={"kind": "static", "n_patients": 2},
    )
    # Find a seed giving one patient of each pathway.
    for seed in range(50):
        sim = sim_of(scenario, seed)
        if {p.pathway for p in sim.instance.patients} == {"consult", "scan"}:
            return sim
    raise AssertionError("no seed with one patient per pathway")


def test_contention_on_a_shared_role_forces_serialization() -> None:
    # Doctor and room are different roles and both free, but the two
    # operations share the single nurse, so they cannot overlap: 10 + 6.
    sim = run(contention(nurses=1))
    assert compute_metrics(sim).makespan == 16
    (s0, e0), (s1, e1) = sorted((sim.start_time[o], sim.end_time[o]) for o in range(2))
    assert e0 == s1


def test_without_contention_the_same_operations_overlap() -> None:
    sim = run(contention(nurses=2))
    assert compute_metrics(sim).makespan == 10
    assert sim.start_time[0] == sim.start_time[1] == 0


def flow_shop() -> Simulator:
    # Two-machine flow shop: P0 = (5, 1), P1 = (1, 5). Johnson's rule puts the
    # job with the short first stage first; the optimum is 7.
    scenario = make_scenario(
        resources={"M1": 1, "M2": 1},
        op_types={
            "LongFirst": ({"M1": 1}, 5, True),
            "ShortSecond": ({"M2": 1}, 1, True),
            "ShortFirst": ({"M1": 1}, 1, True),
            "LongSecond": ({"M2": 1}, 5, True),
        },
        pathways={
            "slow_start": [("m1", "LongFirst", []), ("m2", "ShortSecond", ["m1"])],
            "fast_start": [("m1", "ShortFirst", []), ("m2", "LongSecond", ["m1"])],
        },
        arrivals={"kind": "static", "n_patients": 2},
    )
    for seed in range(50):
        sim = sim_of(scenario, seed)
        if [p.pathway for p in sim.instance.patients] == ["slow_start", "fast_start"]:
            return sim
    raise AssertionError("no suitable seed")


def test_known_optimal_schedule_is_reachable_and_fifo_is_worse() -> None:
    assert compute_metrics(run(flow_shop(), fifo)).makespan == 11

    def fast_start_first(sim: Simulator, startable: list[int]) -> int:
        return max(startable, key=lambda o: sim.patient_of[o])

    sim = run(flow_shop(), fast_start_first)
    assert compute_metrics(sim).makespan == 7
    assert by_step(sim, 1) == {"m1": (0, 1), "m2": (1, 6)}
    assert by_step(sim, 0) == {"m1": (1, 6), "m2": (6, 7)}


def test_metrics_of_the_serialized_case() -> None:
    sim = run(contention(nurses=1))
    metrics = compute_metrics(sim)
    first = min(range(2), key=lambda o: sim.start_time[o])
    first_duration = sim.end_time[first] - sim.start_time[first]
    assert sorted(metrics.flow_times) == sorted([first_duration, 16])
    assert sorted(metrics.op_waits) == [0, first_duration]
    assert metrics.total_wait == first_duration
    assert metrics.utilization == pytest.approx({"Nurse": 1.0, "Doctor": 10 / 16, "Room": 6 / 16})


# --- precedence, eligibility, all-or-nothing ------------------------------------


@pytest.fixture
def chain() -> Simulator:
    scenario = make_scenario(
        resources={"Nurse": 2, "Doctor": 1},
        op_types={
            "Prep": ({"Nurse": 1}, 5, True),
            "Treat": ({"Doctor": 1, "Nurse": 2}, 10, True),
            "Rest": ({"Nurse": 1}, 3, True),
        },
        pathways={"p": [("a", "Prep", []), ("x", "Treat", ["a"]), ("b", "Rest", ["x"])]},
        arrivals={"kind": "static", "n_patients": 2},
    )
    sim = sim_of(scenario)
    assert sim.run_until_decision()
    return sim


def test_operation_cannot_start_before_its_predecessors(chain: Simulator) -> None:
    assert sim_states(chain, 0) == ["READY", "WAITING", "WAITING"]
    with pytest.raises(SimulationError, match="WAITING, not READY"):
        chain.start(op(chain, 0, "x"))
    assert chain.startable_ops() == [op(chain, 0, "a"), op(chain, 1, "a")]


def sim_states(sim: Simulator, patient: int) -> list[str]:
    return [sim.state[o.id].name for o in sim.instance.patients[patient].operations]


def test_precedence_holds_in_the_finished_schedule(chain: Simulator) -> None:
    run(chain)
    for o in chain.instance.operations:
        for pred in o.predecessors:
            assert chain.end_time[pred] <= chain.start_time[o.id]


def test_explicit_team_must_be_eligible(chain: Simulator) -> None:
    doctor = chain.resources.resource_ids.index("doctor_1")
    with pytest.raises(AllocationError, match="not eligible for role 'Nurse'"):
        chain.start(op(chain, 0, "a"), ((doctor,),))
    assert chain.state[op(chain, 0, "a")] == OpState.READY
    assert all(chain.resources.is_idle(r) for r in range(3))


def test_explicit_team_is_used(chain: Simulator) -> None:
    nurse_2 = chain.resources.resource_ids.index("nurse_2")
    chain.start(op(chain, 0, "a"), ((nurse_2,),))
    assert chain.resources.holder[nurse_2] == op(chain, 0, "a")
    assert chain.resources.is_idle(chain.resources.resource_ids.index("nurse_1"))


def test_operation_needing_two_nurses_waits_and_holds_nothing(chain: Simulator) -> None:
    """All-or-nothing: Treat (doctor + 2 nurses) must not grab the doctor and
    one nurse while the other nurse is busy."""
    chain.start(op(chain, 0, "a"))
    chain.start(op(chain, 1, "a"))
    assert chain.advance()  # both preps finish at t=5
    treat0, treat1 = op(chain, 0, "x"), op(chain, 1, "x")
    assert chain.startable_ops() == [treat0, treat1]
    chain.start(treat0)
    # Doctor and both nurses are taken together; the second Treat holds nothing.
    assert not chain.can_start(treat1)
    assert set(chain.resources.holder) == {treat0}
    with pytest.raises(SimulationError, match="no idle team"):
        chain.start(treat1)
    assert set(chain.resources.holder) == {treat0}


def test_team_is_released_together_on_completion(chain: Simulator) -> None:
    chain.start(op(chain, 0, "a"))
    chain.start(op(chain, 1, "a"))
    chain.advance()
    treat0 = op(chain, 0, "x")
    chain.start(treat0)
    assert sum(not chain.resources.is_idle(r) for r in range(3)) == 3
    assert math.isnan(chain.end_time[treat0])  # realized duration stays hidden
    chain.advance()
    assert chain.now == 15 and chain.end_time[treat0] == 15
    assert all(chain.resources.is_idle(r) for r in range(3))


def test_no_resource_is_ever_double_booked(example_scenario: Scenario) -> None:
    sim = run(Simulator(generate_instance(example_scenario, 5), debug=True))
    intervals: dict[int, list[tuple[float, float]]] = {}
    for o in range(sim.n_ops):
        team = sim.team[o]
        assert team is not None
        for group in team:
            for r in group:
                intervals.setdefault(r, []).append((sim.start_time[o], sim.end_time[o]))
    for spans in intervals.values():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


# --- arrivals, waiting, delays --------------------------------------------------


def test_operations_are_hidden_until_the_patient_arrives() -> None:
    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types={"A": ({"Nurse": 1}, 1, True)},
        pathways={"p": [("a", "A", [])]},
        arrivals={"kind": "poisson", "n_patients": 3, "rate": 0.01},
    )
    sim = sim_of(scenario)
    assert sim.state == [OpState.HIDDEN] * 3 and sim.startable_ops() == []
    with pytest.raises(SimulationError, match="HIDDEN"):
        sim.start(0)
    assert sim.run_until_decision()
    assert sim.now == sim.instance.patients[0].arrival_time > 0
    assert sim.state == [OpState.READY, OpState.HIDDEN, OpState.HIDDEN]
    run(sim)
    for patient in sim.instance.patients:
        assert sim.start_time[patient.id] >= patient.arrival_time


def test_waiting_is_allowed_while_something_could_start() -> None:
    sim = contention(nurses=2)
    assert sim.run_until_decision()
    first, second = sim.startable_ops()
    sim.start(first)
    assert sim.startable_ops() == [second]
    sim.advance()  # deliberately idle the second operation until the first ends
    sim.start(second)
    run(sim)
    assert sim.start_time[second] == sim.end_time[first]


DELAY_TYPES: dict[str, tuple[dict[str, int], float, bool]] = {
    "Draw": ({"Nurse": 1}, 5, True),
    "Result": ({}, 30, False),
    "Transfer": ({}, 4, True),
    "Review": ({"Nurse": 1}, 10, True),
}


def test_pure_delay_starts_by_itself_and_blocks_no_resource() -> None:
    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types=DELAY_TYPES,
        pathways={
            "p": [("draw", "Draw", []), ("res", "Result", ["draw"]), ("rev", "Review", ["res"])]
        },
    )
    sim = sim_of(scenario)
    decisions = []
    while sim.run_until_decision():
        decisions.append(sim.startable_ops())
        sim.start(decisions[-1][0])
    assert decisions == [[op(sim, 0, "draw")], [op(sim, 0, "rev")]]  # never the delay
    assert by_step(sim) == {"draw": (0, 5), "res": (5, 35), "rev": (35, 45)}
    assert sim.team[op(sim, 0, "res")] == ()


def test_delay_that_occupies_the_patient_is_a_decision() -> None:
    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types=DELAY_TYPES,
        pathways={"p": [("move", "Transfer", []), ("draw", "Draw", [])]},
    )
    sim = sim_of(scenario)
    assert sim.run_until_decision()
    assert sim.startable_ops() == [op(sim, 0, "move"), op(sim, 0, "draw")]
    sim.start(op(sim, 0, "draw"))
    assert not sim.can_start(op(sim, 0, "move"))
    run(sim)
    assert by_step(sim) == {"draw": (0, 5), "move": (5, 9)}


# --- cancellation ---------------------------------------------------------------


def test_cancelled_operation_does_not_block_or_lose_precedence(chain: Simulator) -> None:
    a, x, b = (op(chain, 0, s) for s in "axb")
    chain.cancel(x)
    assert chain.state[x] == OpState.CANCELLED
    assert chain.state[b] == OpState.WAITING  # still waits for a, through x
    chain.start(a)
    chain.advance()
    assert chain.state[b] == OpState.READY and chain.ready_time[b] == 5


def test_cancelling_a_ready_operation_releases_its_successors(chain: Simulator) -> None:
    a, x = op(chain, 0, "a"), op(chain, 0, "x")
    chain.cancel(a)
    assert chain.state[x] == OpState.READY
    assert a not in chain.ready_ops


def test_running_or_finished_operations_cannot_be_cancelled(chain: Simulator) -> None:
    a = op(chain, 0, "a")
    chain.start(a)
    with pytest.raises(SimulationError, match="RUNNING and cannot be cancelled"):
        chain.cancel(a)
    chain.advance()
    with pytest.raises(SimulationError, match="DONE and cannot be cancelled"):
        chain.cancel(a)


def test_patient_completes_when_remaining_operations_are_cancelled(chain: Simulator) -> None:
    a, x, b = (op(chain, 0, s) for s in "axb")
    chain.start(a)
    chain.advance()
    chain.cancel(b)
    chain.cancel(x)
    assert chain.completion_time[0] == 5
    run(chain)
    assert chain.done and chain.state[x] == chain.state[b] == OpState.CANCELLED


def test_cancelling_a_hidden_operation_before_arrival() -> None:
    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types={"A": ({"Nurse": 1}, 2, True)},
        pathways={"p": [("a", "A", []), ("b", "A", ["a"])]},
        arrivals={"kind": "poisson", "n_patients": 1, "rate": 0.1},
    )
    sim = sim_of(scenario)
    sim.cancel(op(sim, 0, "a"))
    run(sim)
    arrival = sim.instance.patients[0].arrival_time
    assert by_step(sim)["b"] == (arrival, arrival + 2)


# --- reproducibility and invariants ---------------------------------------------


def schedule(sim: Simulator) -> list[Any]:
    return [sim.start_time, sim.end_time, sim.team, sim.completion_time]


def test_same_seed_gives_the_same_schedule(example_scenario: Scenario) -> None:
    first = run(Simulator(generate_instance(example_scenario, 21)))
    second = run(Simulator(generate_instance(example_scenario, 21)))
    assert schedule(first) == schedule(second)
    assert schedule(first) != schedule(run(Simulator(generate_instance(example_scenario, 22))))


def test_reset_replays_the_same_episode(example_scenario: Scenario) -> None:
    sim = run(Simulator(generate_instance(example_scenario, 4), debug=True))
    before = [list(x) for x in schedule(sim)]
    sim.reset()
    assert sim.now == 0 and not sim.done
    assert [list(x) for x in schedule(run(sim))] == before


def test_durations_do_not_depend_on_dispatch_order(example_scenario: Scenario) -> None:
    """Common random numbers: an operation takes equally long under any policy
    (the default duration model ignores the team)."""

    def lifo(sim: Simulator, startable: list[int]) -> int:
        return startable[-1]

    a = run(Simulator(generate_instance(example_scenario, 8)), fifo)
    b = run(Simulator(generate_instance(example_scenario, 8)), lifo)
    assert a.start_time != b.start_time
    for o in range(a.n_ops):
        assert a.end_time[o] - a.start_time[o] == pytest.approx(b.end_time[o] - b.start_time[o])


@pytest.mark.parametrize("seed", range(15))
def test_invariants_hold_under_random_dispatch_with_waits(
    example_scenario: Scenario, seed: int
) -> None:
    rng = random.Random(seed)

    def chaotic(sim: Simulator, startable: list[int]) -> int | None:
        if sim.has_pending_events() and rng.random() < 0.3:
            return None
        return rng.choice(startable)

    sim = run(Simulator(generate_instance(example_scenario, seed), debug=True), chaotic)
    assert sim.done
    assert all(s == OpState.DONE for s in sim.state)


def test_invariant_checker_catches_a_double_booking(chain: Simulator) -> None:
    chain.start(op(chain, 0, "a"))
    idle = next(r for r in range(3) if chain.resources.is_idle(r))
    chain.resources.holder[idle] = op(chain, 0, "a")  # corrupt the state
    with pytest.raises(InvariantViolation):
        check_invariants(chain)


def test_invariant_checker_catches_a_precedence_violation(chain: Simulator) -> None:
    chain.state[op(chain, 0, "x")] = OpState.READY
    with pytest.raises(InvariantViolation, match="unfinished predecessors"):
        check_invariants(chain)


def test_default_example_is_unchanged_since_m2a(example_scenario: Scenario) -> None:
    """Golden value recorded at M2a. The realism layer is off by default
    (rho = 0, one arrival stream, no speed roles) and must not move it."""
    sim = run(Simulator(generate_instance(example_scenario, 3)))
    assert compute_metrics(sim).makespan == 726.8204833997011
