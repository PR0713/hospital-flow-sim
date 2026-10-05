"""M2c-1: shifts, breaks, absences, breakdowns, turnover."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from hospital_sim.domain import BreakdownSpec, Calendar, Scenario, ScenarioError
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import (
    AllocationError,
    DeadlockError,
    Simulator,
    compute_metrics,
    run,
)
from hospital_sim.sim import availability as av

from .conftest import make_scenario

# The overloaded variant exercises every kind of downtime.
STRESS = Path(__file__).parent.parent / "configs" / "example_hospital_stress.yaml"


# --- interval streams -----------------------------------------------------------


def take(stream: Any, n: int) -> list[tuple[float, float]]:
    return [next(stream) for _ in range(n)]


def test_off_duty_is_the_complement_of_shifts_plus_breaks() -> None:
    cal = Calendar("c", 1440, shifts=((480, 720), (780, 1020)), breaks=((600, 15),))
    assert av.off_duty_within_period(cal) == [(0, 480), (600, 615), (720, 780), (1020, 1440)]
    assert take(av.planned_off_duty(cal), 5)[-1] == (1440, 1920)  # repeats each period


def test_round_the_clock_calendar_has_no_planned_downtime() -> None:
    assert list(av.planned_off_duty(Calendar("c", 100, shifts=((0, 100),)))) == []
    assert list(av.absences(Calendar("c", 100, shifts=((0, 100),)), np.random.default_rng(0))) == []


def test_absences_miss_whole_shifts_at_the_configured_rate() -> None:
    cal = Calendar("c", 100, shifts=((10, 40), (60, 90)), absence_probability=0.3)
    missed = take(av.absences(cal, np.random.default_rng(1)), 3000)
    assert all((s % 100, e - s) in {(10, 30), (60, 30)} for s, e in missed)
    shifts_elapsed = missed[-1][0] / 100 * 2
    assert len(missed) / shifts_elapsed == pytest.approx(0.3, rel=0.05)


def test_breakdown_stream_is_an_alternating_exponential_process() -> None:
    downs = take(av.breakdowns(BreakdownSpec(mtbf=200, mttr=20), np.random.default_rng(2)), 5000)
    repairs = np.array([e - s for s, e in downs])
    ups = np.array([downs[0][0]] + [b[0] - a[1] for a, b in zip(downs, downs[1:], strict=False)])
    assert (ups > 0).all() and (repairs > 0).all()
    assert stats.kstest(ups, "expon", args=(0, 200)).pvalue > 0.01
    assert stats.kstest(repairs, "expon", args=(0, 20)).pvalue > 0.01
    assert repairs.sum() / downs[-1][1] == pytest.approx(20 / 220, rel=0.05)  # unavailability


# --- shifts and breaks in the engine --------------------------------------------


def one_nurse(calendar: dict[str, Any], n_patients: int, duration: float) -> Simulator:
    scenario = make_scenario(
        resources={},
        op_types={"Task": ({"Nurse": 1}, duration, True)},
        pathways={"p": [("t", "Task", [])]},
        arrivals={"kind": "static", "n_patients": n_patients},
        extra_resources=[{"id": "nurse", "roles": ["Nurse"], "calendar": "cal"}],
        calendars=[{"name": "cal", **calendar}],
    )
    return Simulator(generate_instance(scenario, 0), debug=True)


def spans(sim: Simulator) -> list[tuple[float, float]]:
    return sorted((sim.start_time[o], sim.end_time[o]) for o in range(sim.n_ops))


def test_shift_end_is_lazy_and_work_resumes_next_shift() -> None:
    # On duty 0-100 of every 200. Three 60-minute tasks.
    sim = run(one_nurse({"period": 200, "shifts": [{"start": 0, "end": 100}]}, 3, 60))
    # The second task starts at 60, runs past the end of the shift and is not
    # interrupted; the third waits for the next shift.
    assert spans(sim) == [(0, 60), (60, 120), (200, 260)]
    assert sim.overtime == [20]
    assert [x for x in sim.downtime_log if x[1] < 260] == [(0, 100.0, 200.0, "off duty")]

    metrics = compute_metrics(sim)
    assert metrics.overtime == {"Nurse": 20}
    # Busy 180 of (260 - 100 rostered off + 20 overtime) on-duty minutes.
    assert metrics.utilization == {"Nurse": 1.0}


def test_nothing_starts_while_off_duty() -> None:
    sim = one_nurse({"period": 200, "shifts": [{"start": 50, "end": 150}]}, 1, 10)
    assert sim.run_until_decision()
    assert sim.now == 50  # the patient arrived at 0 and waited for the shift
    sim.reset()
    sim.advance()  # t=0: arrival, and the nurse is rostered off
    assert sim.now == 0 and sim.ready_ops == [0] and sim.startable_ops() == []
    assert not sim.resources.is_idle(0)
    with pytest.raises(AllocationError, match="unavailable"):
        sim.start(0, ((0,),))


def test_break_blocks_new_starts_but_not_running_work() -> None:
    calendar = {
        "period": 1000,
        "shifts": [{"start": 0, "end": 1000}],
        "breaks": [{"start": 30, "duration": 10}],
    }
    # Free exactly when the break begins: the next task waits for it to end.
    assert spans(run(one_nurse(calendar, 2, 30))) == [(0, 30), (40, 70)]
    # Busy across the break: it works through, and the overlap is overtime.
    sim = run(one_nurse(calendar, 2, 25))
    assert spans(sim) == [(0, 25), (25, 50)]
    assert sim.overtime == [10]


def test_durations_are_not_changed_by_the_roster(example_scenario: Scenario) -> None:
    calendar = Calendar(
        "day", 600, shifts=((0, 300),), breaks=((120, 20),), absence_probability=0.2
    )
    rostered = dataclasses.replace(
        example_scenario,
        calendars=(calendar,),
        resources=tuple(
            dataclasses.replace(r, calendar="day") if "Nurse" in r.roles else r
            for r in example_scenario.resources
        ),
    )
    plain = run(Simulator(generate_instance(example_scenario, 4)))
    with_roster = run(Simulator(generate_instance(rostered, 4), debug=True))
    assert with_roster.start_time != plain.start_time
    for o in range(plain.n_ops):
        assert with_roster.end_time[o] - with_roster.start_time[o] == pytest.approx(
            plain.end_time[o] - plain.start_time[o]
        )


# --- absences and breakdowns in the engine --------------------------------------


def test_absent_resource_misses_exactly_whole_shifts() -> None:
    sim = one_nurse(
        {"period": 100, "shifts": [{"start": 0, "end": 60}], "absence_probability": 0.4}, 60, 20
    )
    run(sim)
    absent = [(s, e) for _, s, e, reason in sim.downtime_log if reason == "absent"]
    assert absent and all(s % 100 == 0 and e - s == 60 for s, e in absent)
    for start, _ in spans(sim):
        assert start % 100 < 60, "started while rostered off"
        assert not any(s <= start < e for s, e in absent), "started while absent"


def breakdown_sim(seed: int = 0) -> Simulator:
    scenario = make_scenario(
        resources={},
        op_types={"Scan": ({"Scanner": 1}, 15, True)},
        pathways={"p": [("s", "Scan", [])]},
        arrivals={"kind": "poisson", "n_patients": 300, "rate": 0.03},
        extra_resources=[
            {"id": "scanner", "roles": ["Scanner"], "breakdown": {"mtbf": 120, "mttr": 30}}
        ],
    )
    return Simulator(generate_instance(scenario, seed), debug=True)


def test_breakdowns_only_take_effect_between_operations() -> None:
    sim = run(breakdown_sim())
    downs = [(s, e) for _, s, e, reason in sim.downtime_log if reason == "breakdown"]
    assert len(downs) > 20
    hit_while_busy = 0
    for start, end in spans(sim):
        assert end - start == 15  # never interrupted or stretched
        assert not any(s <= start < e for s, e in downs), "started on a broken scanner"
        hit_while_busy += any(start < s < end for s, e in downs)
    assert hit_while_busy > 0  # the lazy-removal path was exercised
    assert sim.overtime[0] > 0


def test_availability_does_not_depend_on_the_policy_and_replays_on_reset() -> None:
    def lifo(sim: Simulator, startable: list[int]) -> int:
        return startable[-1]

    def truth(sim: Simulator, until: float) -> list[Any]:
        return [x for x in sim.downtime_log if x[3] != "turnover" and x[1] < until]

    scenario = load_scenario(STRESS)
    a = run(Simulator(generate_instance(scenario, 13)))
    b = run(Simulator(generate_instance(scenario, 13)), lifo)
    assert a.start_time != b.start_time
    until = min(compute_metrics(a).makespan, compute_metrics(b).makespan)
    assert truth(a, until) == truth(b, until) and len(truth(a, until)) > 5
    assert {"absent", "breakdown", "off duty"} <= {x[3] for x in a.downtime_log}

    first = (list(a.start_time), list(a.downtime_log))
    a.reset()
    run(a)
    assert (a.start_time, a.downtime_log) == first


# --- turnover -------------------------------------------------------------------


def test_turnover_holds_the_room_but_frees_staff_and_patient() -> None:
    scenario = make_scenario(
        resources={"Room": 1, "Nurse": 1},
        op_types={
            "Procedure": ({"Room": 1, "Nurse": 1}, 10, True),
            "Checkout": ({"Nurse": 1}, 3, True),
        },
        pathways={"p": [("proc", "Procedure", []), ("out", "Checkout", ["proc"])]},
        arrivals={"kind": "static", "n_patients": 2},
        turnover={"Procedure": {"Room": 5}},
    )
    sim = Simulator(generate_instance(scenario, 0), debug=True)
    assert sim.run_until_decision()
    sim.start(0)
    sim.advance()  # t=10: procedure done
    room, nurse = (sim.resources.resource_ids.index(x) for x in ("room_1", "nurse_1"))
    assert sim.resources.is_idle(nurse) and not sim.resources.is_idle(room)
    assert sim.resources.holder[room] == -1 and sim.in_turnover[room]
    # The patient and the nurse move on at once; the next procedure cannot start.
    assert sim.startable_ops() == [1]
    run(sim)
    times = {
        (o.patient_id, o.step_id): (sim.start_time[o.id], sim.end_time[o.id])
        for o in sim.instance.operations
    }
    assert times[(0, "out")] == (10, 13)
    assert times[(1, "proc")] == (15, 25)  # 10 + 5 minutes of cleaning
    metrics = compute_metrics(sim)
    assert metrics.turnover["Room"] == 5 + (metrics.makespan - 25)
    assert metrics.utilization["Room"] == pytest.approx(20 / metrics.makespan)


# --- deadlock, validation -------------------------------------------------------


def two_nurse_scenario(second_shift: tuple[float, float]) -> Scenario:
    return make_scenario(
        resources={},
        op_types={"Lift": ({"Nurse": 2}, 5, True)},
        pathways={"p": [("l", "Lift", [])]},
        extra_resources=[
            {"id": "n1", "roles": ["Nurse"], "calendar": "early"},
            {"id": "n2", "roles": ["Nurse"], "calendar": "late"},
        ],
        calendars=[
            {"name": "early", "period": 100, "shifts": [{"start": 0, "end": 50}]},
            {
                "name": "late",
                "period": 100,
                "shifts": [{"start": second_shift[0], "end": second_shift[1]}],
            },
        ],
    )


def test_rosters_that_never_overlap_are_rejected() -> None:
    with pytest.raises(ScenarioError, match="never have a full team on duty"):
        two_nurse_scenario((50, 100))
    sim = run(Simulator(generate_instance(two_nurse_scenario((40, 100)), 0), debug=True))
    assert spans(sim) == [(40, 45)]  # the only overlap of the two shifts


def test_engine_reports_a_deadlock_instead_of_spinning() -> None:
    ok = two_nurse_scenario((40, 100))
    never = dataclasses.replace(
        ok, calendars=(ok.calendars[0], dataclasses.replace(ok.calendars[1], shifts=((50, 100),)))
    )  # bypasses validation
    sim = Simulator(generate_instance(never, 0), stall_limit=50)
    with pytest.raises(DeadlockError, match="no pending work"):
        run(sim)


def test_calendar_and_turnover_validation() -> None:
    def build(**overrides: Any) -> None:
        make_scenario(
            **{
                "resources": {},
                "op_types": {"T": ({"Nurse": 1}, 5, True)},
                "pathways": {"p": [("t", "T", [])]},
                "extra_resources": [{"id": "n", "roles": ["Nurse"], "calendar": "c"}],
                "calendars": [{"name": "c", "period": 100, "shifts": [{"start": 0, "end": 50}]}],
                **overrides,
            }
        )

    build()
    cases = {
        "unknown calendar 'c'": {"calendars": []},
        "0 <= start < end <= period": {
            "calendars": [{"name": "c", "period": 100, "shifts": [{"start": 60, "end": 120}]}]
        },
        "time order and not overlap": {
            "calendars": [
                {
                    "name": "c",
                    "period": 100,
                    "shifts": [{"start": 0, "end": 50}, {"start": 40, "end": 80}],
                }
            ]
        },
        "must lie inside one shift": {
            "calendars": [
                {
                    "name": "c",
                    "period": 100,
                    "shifts": [{"start": 0, "end": 50}],
                    "breaks": [{"start": 45, "duration": 10}],
                }
            ]
        },
        "turnover role 'Room' is not one of its required roles": {"turnover": {"T": {"Room": 5}}},
    }
    for message, overrides in cases.items():
        with pytest.raises(ScenarioError) as exc:
            build(**overrides)
        assert message in "\n".join(exc.value.errors), message


# --- everything together --------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_invariants_hold_on_the_stress_example_under_random_dispatch(seed: int) -> None:
    rng = random.Random(seed)

    def chaotic(sim: Simulator, startable: list[int]) -> int | None:
        return None if rng.random() < 0.3 else rng.choice(startable)

    sim = run(Simulator(generate_instance(load_scenario(STRESS), seed), debug=True), chaotic)
    assert sim.done
    # No operation ever started on a resource that was down or in turnover.
    for o in range(sim.n_ops):
        team = sim.team[o]
        assert team is not None
        for group in team:
            for r in group:
                assert not any(
                    res == r and s <= sim.start_time[o] < e for res, s, e, _ in sim.downtime_log
                )


def test_gantt_shows_downtime_and_turnover() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from hospital_sim.eval import plot_gantt

    sim = run(Simulator(generate_instance(load_scenario(STRESS), 11)))
    kinds = {p.get_gid() for p in plot_gantt(sim).axes[0].patches}
    assert {"operation", "turnover", "off duty", "absent"} <= kinds
