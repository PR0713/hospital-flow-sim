"""M2c-2: overrides by patient class, appointments and no-shows, patients
leaving, and pathway reveal with its four leak tests."""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError
from scipy import stats

from hospital_sim.domain import ProblemInstance, Scenario, ScenarioError
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.env.observation import OP_FEATURES, PATIENT_FEATURES
from hospital_sim.generators import (
    ScenarioConfig,
    generate_instance,
    load_config,
    load_scenario,
    without_operations,
)
from hospital_sim.sim import OpState, PatientStatus, Simulator, compute_metrics, run

from .conftest import build, make_scenario
from .test_env_integrity import assert_identical_until, same, trace, with_hidden

FULL = Path(__file__).parent.parent / "configs" / "example_hospital_full.yaml"


def full() -> Scenario:
    return load_scenario(FULL)


# --- requirement and duration overrides -----------------------------------------


def override_config() -> dict[str, Any]:
    return {
        "name": "overrides",
        "roles": [{"name": "Nurse"}],
        "resources": [{"roles": ["Nurse"], "count": 4}],
        "operation_types": [
            {
                "name": "Care",
                "requirements": [{"role": "Nurse"}],
                "duration": {"distribution": "deterministic", "mean": 10},
                "overrides": [
                    {
                        "when": {"patient_class": "urgent"},
                        "requirements": [{"role": "Nurse", "quantity": 3}],
                    },
                    {"when": {"patient_class": "slow"}, "duration_scale": 2.0},
                ],
            }
        ],
        "pathways": [{"name": "p", "steps": [{"id": "c", "op_type": "Care"}]}],
        "arrivals": [
            {"name": "routine", "kind": "static", "n_patients": 1},
            {"name": "urgent", "kind": "static", "n_patients": 1, "urgent": True},
            {"name": "slow", "kind": "static", "n_patients": 1},
        ],
    }


def test_overrides_become_derived_types_chosen_by_patient_class() -> None:
    sc = build(override_config())
    assert [t.name for t in sc.operation_types] == ["Care", "Care[urgent]", "Care[slow]"]
    assert sc.resolve_op_type("Care", "urgent", None) == "Care[urgent]"
    assert sc.resolve_op_type("Care", "routine", None) == "Care"

    sim = run(Simulator(generate_instance(sc, 0), debug=True))
    by_class = {p.patient_class: p.operations[0] for p in sim.instance.patients}
    assert {c: op.op_type for c, op in by_class.items()} == {
        "routine": "Care",
        "urgent": "Care[urgent]",
        "slow": "Care[slow]",
    }
    team_size = {c: len((sim.team[op.id] or ((),))[0]) for c, op in by_class.items()}
    assert team_size == {"routine": 1, "urgent": 3, "slow": 1}
    duration = {c: sim.end_time[op.id] - sim.start_time[op.id] for c, op in by_class.items()}
    assert duration == {"routine": 10, "urgent": 10, "slow": 20}


def test_override_by_acuity_and_first_match_wins() -> None:
    config = override_config()
    config["complexity"] = {"acuity_classes": 3}
    config["operation_types"][0]["overrides"] = [
        {"when": {"patient_class": "urgent", "acuity_class": 2}, "duration_scale": 5.0},
        {"when": {"acuity_class": 2}, "duration_scale": 3.0},
    ]
    sc = build(config)
    assert sc.resolve_op_type("Care", "urgent", 2) == "Care[urgent,acuity=2]"
    assert sc.resolve_op_type("Care", "routine", 2) == "Care[acuity=2]"
    assert sc.resolve_op_type("Care", "urgent", 1) == "Care"
    for seed in range(5):
        instance = generate_instance(sc, seed)
        for patient in instance.patients:
            expected = sc.resolve_op_type("Care", patient.patient_class, patient.acuity_class)
            assert patient.operations[0].op_type == expected


def test_override_validation() -> None:
    def errors(mutate: Any) -> str:
        config = override_config()
        mutate(config)
        with pytest.raises((ScenarioError, ValidationError)) as exc:
            build(config)
        return str(exc.value)

    def infeasible(c: dict[str, Any]) -> None:
        c["operation_types"][0]["overrides"][0]["requirements"][0]["quantity"] = 5

    def unknown_class(c: dict[str, Any]) -> None:
        c["operation_types"][0]["overrides"][0]["when"] = {"patient_class": "vip"}

    def no_acuity(c: dict[str, Any]) -> None:
        c["operation_types"][0]["overrides"][0]["when"] = {"acuity_class": 1}

    def derived_in_pathway(c: dict[str, Any]) -> None:
        c["pathways"][0]["steps"][0]["op_type"] = "Care[urgent]"

    def both(c: dict[str, Any]) -> None:
        c["operation_types"][0]["overrides"][1]["duration"] = {"mean": 5, "cv": 0.2}

    def empty_when(c: dict[str, Any]) -> None:
        c["operation_types"][0]["overrides"][0]["when"] = {}

    assert "needs 5 x 'Nurse' but only 4 exist" in errors(infeasible)
    assert "unknown patient class 'vip'" in errors(unknown_class)
    assert "acuity class 1 does not exist" in errors(no_acuity)
    assert "is a derived type; name its base type" in errors(derived_in_pathway)
    assert "not both" in errors(both)
    assert "needs patient_class, acuity_class, or both" in errors(empty_when)


@pytest.mark.slow
def test_urgent_surgery_in_the_full_example_takes_three_nurses() -> None:
    sc = full()
    seen = 0
    for seed in range(40):
        sim = run(Simulator(generate_instance(sc, seed), debug=True))
        nurses = set(sim.resources.members[sim.resources.role_names.index("Nurse")])
        for op in sim.instance.operations:
            if op.op_type.startswith("Surgery") and sim.state[op.id] == OpState.DONE:
                used = sum(r in nurses for g in sim.team[op.id] or () for r in g)
                urgent = sim.instance.patients[op.patient_id].urgent
                assert used == (3 if urgent else 2)
                seen += urgent
    assert seen > 0


# --- appointments and no-shows --------------------------------------------------


def clinic(no_show: float = 0.0, sd: float = 0.0, n: int = 4, **extra: Any) -> Scenario:
    return make_scenario(
        resources={"Nurse": 1},
        op_types={"Visit": ({"Nurse": 1}, 10, True)},
        pathways={"p": [("v", "Visit", [])]},
        arrivals={
            "kind": "scheduled",
            "n_patients": n,
            "appointments": {
                "start": 20,
                "interval": 30,
                "punctuality_sd": sd,
                "no_show_probability": no_show,
                "no_show_timeout": 15,
            },
            **extra,
        },
    )


def test_punctual_appointments_arrive_on_the_booked_times() -> None:
    instance = generate_instance(clinic(), 0)
    assert [p.arrival_time for p in instance.patients] == [20, 50, 80, 110]
    assert [p.scheduled_time for p in instance.patients] == [20, 50, 80, 110]
    assert [p.appointment for p in instance.patients] == [0, 1, 2, 3]


def test_lateness_and_no_show_rates() -> None:
    instance = generate_instance(clinic(no_show=0.2, sd=6.0, n=4000), 1)
    came = [p for p in instance.patients if math.isfinite(p.arrival_time)]
    assert 1 - len(came) / 4000 == pytest.approx(0.2, abs=0.02)
    late = np.array([p.arrival_time - (p.scheduled_time or 0) for p in came])
    assert late.mean() == pytest.approx(0.0, abs=0.4) and late.std() == pytest.approx(6.0, rel=0.08)
    assert late.max() < 15  # nobody turns up after being declared a no-show


def find_no_show_instance(appointment: int = 1) -> ProblemInstance:
    """Four bookings (at 20, 50, 80, 110) of which exactly the given one is missed."""
    for seed in range(400):
        instance = generate_instance(clinic(no_show=0.3, sd=4.0), seed)
        absent = [math.isinf(p.arrival_time) for p in instance.patients]
        booked = next(p for p in instance.patients if p.appointment == appointment)
        if sum(absent) == 1 and math.isinf(booked.arrival_time):
            return instance
    raise AssertionError("no suitable instance")


def test_a_no_show_is_declared_after_the_timeout_and_never_counted_as_flow() -> None:
    instance = find_no_show_instance()
    sim = Simulator(instance, debug=True)
    absent = next(p.id for p in instance.patients if math.isinf(p.arrival_time))
    assert sim.patient_status[absent] == PatientStatus.EXPECTED
    run(sim)
    assert sim.patient_status[absent] == PatientStatus.NO_SHOW
    assert sim.completion_time[absent] == 50 + 15  # booked at 50, 15-minute timeout
    assert sim.state[instance.patients[absent].operations[0].id] == OpState.CANCELLED
    metrics = compute_metrics(sim)
    assert metrics.n_no_show == 1 and len(metrics.flow_times) == 3
    assert all(math.isfinite(f) for f in metrics.flow_times)


def test_the_booking_list_is_visible_and_does_not_betray_a_no_show() -> None:
    instance = find_no_show_instance(appointment=3)  # the last booking, at 110
    sc = instance.scenario
    env = HospitalEnv(EnvConfig(sc, debug=True))
    obs, _ = env.reset(seed=0, options={"instance": instance})
    f = PATIENT_FEATURES.index
    rows = obs["patients"][obs["patient_valid"] > 0]
    # All four appointments are listed from the start, in booking order, and
    # the first to arrive is now present.
    assert len(rows) == 4
    assert rows[:, f("expected")].sum() + rows[:, f("in_system")].sum() == 4
    upcoming = rows[rows[:, f("expected")] == 1][:, f("scheduled_in")] * env.time_scale
    assert env.sim is not None
    assert list(np.round(upcoming + env.sim.now)) == [50, 80, 110]

    # Twin: the same patient merely arrives 12 minutes late instead.
    absent = next(p for p in instance.patients if math.isinf(p.arrival_time))
    late = dataclasses.replace(absent, arrival_time=122.0)
    others = [p for p in instance.patients if p.id != absent.id]
    twin = dataclasses.replace(
        instance, patients=tuple(sorted([*others, late], key=lambda p: p.id))
    )
    trace_a, _ = trace(sc, instance)
    trace_b, _ = trace(sc, twin)
    assert_identical_until(trace_a, trace_b, 122.0, min_steps=3)

    final = trace_a[-1][1]["patients"]
    assert final[:, f("no_show")].sum() == 1 and final[3, f("no_show")] == 1


# --- patients leaving -----------------------------------------------------------


def queue_instance(patience: tuple[float, ...]) -> ProblemInstance:
    sc = make_scenario(
        resources={"Nurse": 1},
        op_types={"Visit": ({"Nurse": 1}, 30, True)},
        pathways={"p": [("v", "Visit", [])]},
        arrivals={"kind": "static", "n_patients": 3, "patience": {"mean": 50}},
    )
    return with_hidden(generate_instance(sc, 0), patience=patience)


def test_a_patient_leaves_when_patience_runs_out_before_being_seen() -> None:
    sim = run(Simulator(queue_instance((math.inf, 40.0, math.inf)), debug=True))
    # FIFO: patient 0 is seen 0-30, patient 1 from 30 (before 40), patient 2 from 60.
    assert sim.n_left == 0 and compute_metrics(sim).makespan == 90

    sim = run(Simulator(queue_instance((math.inf, math.inf, 45.0)), debug=True))
    # Patient 2 would be seen at 60 but gives up at 45.
    assert sim.n_left == 1 and sim.patient_status[2] == PatientStatus.LEFT
    assert sim.completion_time[2] == 45 and sim.state[2] == OpState.CANCELLED
    metrics = compute_metrics(sim)
    assert metrics.makespan == 60 and metrics.n_left == 1
    assert sorted(metrics.flow_times) == [30, 45, 60]  # the wait of the leaver counts


def test_a_patient_already_being_seen_does_not_leave() -> None:
    sim = run(Simulator(queue_instance((5.0, math.inf, math.inf)), debug=True))
    assert sim.n_left == 0 and sim.end_time[0] == 30


def test_leaving_is_penalized_and_rewards_still_sum_exactly() -> None:
    instance = queue_instance((math.inf, math.inf, 45.0))
    env = HospitalEnv(EnvConfig(instance.scenario, reward_scale=1.0, allow_wait=False))
    assert env.leave_penalty == 2 * 30  # twice the longest pathway's expected work
    env.reset(seed=0, options={"instance": instance})
    rewards = []
    while True:
        obs, reward, terminated, truncated, info = env.step(
            int(np.flatnonzero(env.action_masks())[0])
        )
        rewards.append(reward)
        if terminated or truncated:
            break
    episode = info["episode"]
    assert episode["patients_left"] == 1 and episode["leave_penalty"] == 60
    assert -sum(rewards) == pytest.approx(episode["total_weighted_flow_time"] + 60)
    assert obs["patients"][2, PATIENT_FEATURES.index("left")] == 1


def test_patience_does_not_leak() -> None:
    sc = queue_instance(()).scenario
    # Whoever is served third (the policy is random) gives up at 45 in one
    # world and at 55 in the other; the first two are seen at 0 and 30.
    a = queue_instance((45.0, 45.0, 45.0))
    b = queue_instance((55.0, 55.0, 55.0))
    trace_a, _ = trace(sc, a)
    trace_b, _ = trace(sc, b)
    assert_identical_until(trace_a, trace_b, 45.0, min_steps=2)


def test_patience_is_lognormal_with_the_configured_mean() -> None:
    sc = make_scenario(
        resources={"Nurse": 1},
        op_types={"Visit": ({"Nurse": 1}, 1, True)},
        pathways={"p": [("v", "Visit", [])]},
        arrivals={
            "kind": "poisson",
            "n_patients": 5000,
            "rate": 1.0,
            "patience": {"mean": 90, "cv": 0.6},
        },
    )
    patience = np.array(generate_instance(sc, 2).hidden.patience)
    assert patience.mean() == pytest.approx(90, rel=0.04)
    assert patience.std() / patience.mean() == pytest.approx(0.6, rel=0.08)
    sigma = math.sqrt(math.log1p(0.36))
    target = stats.norm(math.log(90) - sigma**2 / 2, sigma)
    assert stats.kstest(np.log(patience), target.cdf).pvalue > 0.01


# --- pathway reveal: generation and engine --------------------------------------


def reveal_scenario(
    probability: float = 0.5,
    correlation: float = 0.0,
    gated: bool = True,
    n: int = 1,
    repeat: dict[str, Any] | None = None,
) -> Scenario:
    extra = {"revealed_by": "surgery"} if gated else {}
    return build(
        {
            "name": "reveal",
            "roles": [{"name": "Nurse"}],
            "resources": [{"roles": ["Nurse"], "count": 3}],
            "operation_types": [
                {
                    "name": name,
                    "requirements": [{"role": "Nurse"}],
                    "duration": {"distribution": "deterministic", "mean": mean},
                }
                for name, mean in (("Surgery", 60), ("ICU", 100), ("Recovery", 30), ("Scan", 10))
            ],
            "pathways": [
                {
                    "name": "p",
                    "steps": [
                        {"id": "scan", "op_type": "Scan", **({"repeat": repeat} if repeat else {})},
                        {"id": "surgery", "op_type": "Surgery", "after": ["scan"]},
                        {
                            "id": "icu",
                            "op_type": "ICU",
                            "after": ["surgery"],
                            "probability": probability,
                            **extra,
                        },
                        {"id": "recovery", "op_type": "Recovery", "after": ["surgery", "icu"]},
                    ],
                }
            ],
            "arrivals": {"kind": "static", "n_patients": n},
            "complexity": {"rho": 0.3, "reveal_correlation": correlation},
        }
    )


def with_icu(sc: Scenario, wanted: bool = True) -> ProblemInstance:
    for seed in range(100):
        instance = generate_instance(sc, seed)
        if any(op.step_id == "icu" for op in instance.operations) == wanted:
            return instance
    raise AssertionError("no such instance")


def ids(instance: ProblemInstance) -> dict[str, int]:
    return {op.step_id: op.id for op in instance.operations}


def test_gated_operation_is_hidden_until_its_trigger_completes() -> None:
    instance = with_icu(reveal_scenario())
    op = ids(instance)
    assert instance.hidden.reveal_trigger[op["icu"]] == op["surgery"]
    sim = Simulator(instance, debug=True)
    assert sim.run_until_decision()
    assert sim.state[op["icu"]] == OpState.HIDDEN and op["icu"] not in sim.visible_ops
    sim.start(op["scan"])
    sim.advance()
    sim.start(op["surgery"])
    assert sim.state[op["icu"]] == OpState.HIDDEN
    assert sim.state[op["recovery"]] == OpState.WAITING
    sim.advance()  # surgery done at 70: the stay in intensive care appears
    assert sim.now == 70
    assert sim.state[op["icu"]] == OpState.READY and sim.visible_ops[-1] == op["icu"]
    assert sim.state[op["recovery"]] == OpState.WAITING  # never ready without it
    run(sim)
    assert sim.start_time[op["recovery"]] == 170


def test_cancelling_a_trigger_cancels_what_it_would_have_revealed() -> None:
    instance = with_icu(reveal_scenario())
    op = ids(instance)
    sim = Simulator(instance, debug=True)
    assert sim.run_until_decision()
    sim.cancel(op["surgery"])
    assert sim.state[op["icu"]] == OpState.CANCELLED and op["icu"] not in sim.visible_ops
    run(sim)
    assert sim.done and sim.end_time[op["recovery"]] == 40  # after the scan only


def test_nothing_is_gated_without_reveal_config(example_scenario: Scenario) -> None:
    instance = generate_instance(example_scenario, 0)
    assert instance.hidden.reveal_trigger == () and instance.hidden.patience == ()


def test_gating_changes_visibility_only() -> None:
    """With the complexity link off, gating a step changes nothing about who
    the patients are or how long anything takes."""
    for seed in range(10):
        open_ = generate_instance(reveal_scenario(gated=False, n=5), seed)
        gated = generate_instance(reveal_scenario(gated=True, n=5), seed)
        assert [p.operations for p in open_.patients] == [p.operations for p in gated.patients]
        assert open_.hidden.duration_quantiles == gated.hidden.duration_quantiles
        assert open_.hidden.reveal_trigger == ()


@pytest.mark.parametrize("correlation", [0.0, 0.8])
def test_revealed_steps_follow_the_hidden_complexity(correlation: float) -> None:
    instance = generate_instance(reveal_scenario(0.3, correlation, n=4000), 3)
    happened = np.array(
        [any(op.step_id == "icu" for op in p.operations) for p in instance.patients]
    )
    factor = np.array(instance.hidden.patient_factors)
    assert happened.mean() == pytest.approx(0.3, abs=0.025)  # marginal rate unchanged
    gap = factor[happened].mean() - factor[~happened].mean()
    if correlation == 0:
        assert abs(gap) < 0.1
    else:
        assert gap > 1.0  # complex patients are the ones who need intensive care


def test_a_step_cannot_be_revealed_by_a_step_that_did_not_happen() -> None:
    config = {
        "name": "chain",
        "roles": [{"name": "Nurse"}],
        "resources": [{"roles": ["Nurse"]}],
        "operation_types": [
            {
                "name": "T",
                "requirements": [{"role": "Nurse"}],
                "duration": {"distribution": "deterministic", "mean": 1},
            }
        ],
        "pathways": [
            {
                "name": "p",
                "steps": [
                    {"id": "a", "op_type": "T"},
                    {"id": "b", "op_type": "T", "after": ["a"], "probability": 0.5},
                    {"id": "c", "op_type": "T", "after": ["b"], "revealed_by": "b"},
                ],
            }
        ],
        "arrivals": {"kind": "static", "n_patients": 300},
    }
    instance = generate_instance(build(config), 0)
    shapes = {tuple(op.step_id for op in p.operations) for p in instance.patients}
    assert shapes == {("a",), ("a", "b", "c")}


@pytest.mark.slow
def test_repeats_are_a_hidden_chain_of_copies() -> None:
    sc = reveal_scenario(repeat={"probability": 0.5, "max": 3}, n=3000, gated=False)
    instance = generate_instance(sc, 5)
    counts = np.array(
        [sum(op.step_id.startswith("scan") for op in p.operations) for p in instance.patients]
    )
    share = np.bincount(counts, minlength=5)[1:] / len(counts)
    assert share == pytest.approx([0.5, 0.25, 0.125, 0.125], abs=0.025)  # truncated geometric

    patient = next(
        p for p in instance.patients if sum(o.step_id.startswith("scan") for o in p.operations) == 3
    )
    op = {o.step_id: o for o in patient.operations}
    assert op["scan#2"].predecessors == (op["scan"].id,)
    assert op["scan#3"].predecessors == (op["scan#2"].id,)
    assert op["surgery"].predecessors == (op["scan#3"].id,)  # waits for the last repeat
    trigger = instance.hidden.reveal_trigger
    assert trigger[op["scan#2"].id] == op["scan"].id and trigger[op["scan#3"].id] == op["scan#2"].id

    sim = run(Simulator(dataclasses.replace(instance, patients=instance.patients), debug=False))
    assert sim.start_time[op["surgery"].id] >= sim.end_time[op["scan#3"].id]


def test_without_operations_builds_a_consistent_twin() -> None:
    instance = with_icu(reveal_scenario())
    op = ids(instance)
    twin = without_operations(instance, {op["icu"]})
    assert [o.step_id for o in twin.operations] == ["scan", "surgery", "recovery"]
    assert [o.id for o in twin.operations] == [0, 1, 2]
    assert twin.operations[2].predecessors == (1,) and twin.operations[1].successors == (2,)
    assert twin.hidden.reveal_trigger == ()
    kept = [instance.hidden.duration_quantiles[op[s]] for s in ("scan", "surgery", "recovery")]
    assert list(twin.hidden.duration_quantiles) == kept
    assert compute_metrics(run(Simulator(twin, debug=True))).makespan == 100


# --- pathway reveal: the four leak tests ----------------------------------------


def first_reveal_time(env: HospitalEnv) -> float:
    """When the first hidden operation was shown. Test-only use of ground truth."""
    sim = env.sim
    assert sim is not None
    triggers = sim.instance.hidden.reveal_trigger
    times = [sim.end_time[t] for t in set(triggers) if t >= 0 and not math.isnan(sim.end_time[t])]
    return min(times, default=math.inf)


def test_leak_1_twin_with_and_without_the_gated_step() -> None:
    sc = reveal_scenario(n=3)
    instance = generate_instance(
        sc,
        next(
            s
            for s in range(50)
            if sum(op.step_id == "icu" for op in generate_instance(sc, s).operations) == 1
        ),
    )
    icu = next(op for op in instance.operations if op.step_id == "icu")
    twin = without_operations(instance, {icu.id})
    trace_a, env_a = trace(sc, instance)
    trace_b, _ = trace(sc, twin)
    assert env_a.sim is not None
    reveal = env_a.sim.end_time[instance.hidden.reveal_trigger[icu.id]]
    assert_identical_until(trace_a, trace_b, reveal, min_steps=4)


@pytest.mark.parametrize("prior", [True, False])
def test_leak_2_truncation_twin_over_many_seeds(prior: bool) -> None:
    """The real instance against a copy in which nothing hidden exists."""
    sc = full()
    checked = 0
    for seed in range(60):
        instance = generate_instance(sc, seed)
        gated = {i for i, t in enumerate(instance.hidden.reveal_trigger) if t >= 0}
        if not gated:
            continue
        twin = without_operations(instance, gated)

        def run_trace(inst: ProblemInstance) -> Any:
            env = HospitalEnv(EnvConfig(sc, observation="graph", gated_work_prior=prior))
            rng = np.random.default_rng(0)
            obs, info = env.reset(seed=0, options={"instance": inst})
            records = [(info["time"], obs, env.action_masks(), 0.0, info)]
            while True:
                obs, reward, terminated, truncated, info = env.step(
                    int(rng.choice(np.flatnonzero(env.action_masks())))
                )
                records.append((info["time"], obs, env.action_masks(), reward, info))
                if terminated or truncated:
                    return records, env

        trace_a, env_a = run_trace(instance)
        trace_b, _ = run_trace(twin)
        reveal = first_reveal_time(env_a)
        if sum(r[0] < reveal for r in trace_a) < 6:
            continue  # the first reveal comes too early to compare much
        assert_identical_until(trace_a, trace_b, reveal, min_steps=5)
        checked += 1
        if checked == 6:
            return
    raise AssertionError(f"only {checked} usable seeds")


def test_leak_3_poisoned_hidden_operations_change_nothing_before_the_reveal() -> None:
    sc = full()
    type_names = [t.name for t in sc.operation_types]
    checked = 0
    for seed in range(60):
        instance = generate_instance(sc, seed)
        triggers = instance.hidden.reveal_trigger
        gated = {i for i, t in enumerate(triggers) if t >= 0}
        if not gated:
            continue
        # Give every hidden operation a different type and a different duration.
        quantiles = list(instance.hidden.duration_quantiles)
        patients = []
        for patient in instance.patients:
            operations = []
            for op in patient.operations:
                if op.id in gated:
                    quantiles[op.id] = 1.0 - quantiles[op.id]
                    other = "Surgery" if not op.op_type.startswith("Surgery") else "Recovery"
                    assert other in type_names
                    op = dataclasses.replace(op, op_type=other)
                operations.append(op)
            patients.append(dataclasses.replace(patient, operations=tuple(operations)))
        poisoned = dataclasses.replace(
            with_hidden(instance, duration_quantiles=tuple(quantiles)), patients=tuple(patients)
        )
        trace_a, env_a = trace(sc, instance)
        trace_b, _ = trace(sc, poisoned)
        reveal = first_reveal_time(env_a)
        if sum(r[0] < reveal for r in trace_a) < 6:
            continue  # the first reveal comes too early to compare much
        assert_identical_until(trace_a, trace_b, reveal, min_steps=5)
        checked += 1
        if checked == 5:
            return
    raise AssertionError(f"only {checked} usable seeds")


def test_leak_4_shapes_do_not_depend_on_what_is_hidden() -> None:
    sc = reveal_scenario(n=4)
    seen_counts = set()
    shapes = set()
    for seed in range(30):
        instance = generate_instance(sc, seed)
        seen_counts.add(sum(t >= 0 for t in instance.hidden.reveal_trigger))
        env = HospitalEnv(EnvConfig(sc, observation="graph"))
        obs, _ = env.reset(seed=0, options={"instance": instance})
        graph = env.graph()
        shapes.add(
            (
                tuple((k, v.shape) for k, v in sorted(obs.items())),
                graph["nodes"]["op"].shape,
                graph["edges"]["op__precedes__op"].shape,
                int(env.action_masks().sum()),
            )
        )
    assert len(seen_counts) >= 3  # instances with 0, 1, 2... hidden operations
    assert len(shapes) == 1  # ...all look the same at the start


# --- observation of a growing graph ---------------------------------------------


def test_graph_grows_at_the_reveal_and_edges_are_rewired() -> None:
    sc = reveal_scenario(n=1)
    instance = with_icu(sc)
    env = HospitalEnv(EnvConfig(sc, observation="graph", allow_wait=False, debug=True))
    obs, _ = env.reset(seed=0, options={"instance": instance})
    prior = OP_FEATURES.index("expected_gated_work")

    def edges() -> set[tuple[int, int]]:
        return {(int(a), int(b)) for a, b in env.graph()["edges"]["op__precedes__op"].T}

    # Rows: 0 scan, 1 surgery, 2 recovery. Intensive care is unknown.
    assert int(obs["op_valid"].sum()) == 3
    assert edges() == {(0, 1), (1, 2)}
    # The template says: after surgery, intensive care (100 min) with probability 0.5.
    assert obs["ops"][1, prior] * env.time_scale == pytest.approx(50.0)
    assert obs["ops"][0, prior] == 0 and obs["ops"][2, prior] == 0

    obs, *_ = env.step(0)  # scan
    obs, *_ = env.step(1)  # surgery; when it completes the stay appears
    assert int(obs["op_valid"].sum()) == 4
    assert edges() == {(0, 1), (1, 3), (3, 2)}  # recovery now follows row 3
    assert obs["ops"][1, prior] == 0  # nothing left to reveal
    assert obs["ops"][3, OP_FEATURES.index("is_ready")] == 1
    assert obs["ops"][2, OP_FEATURES.index("open_predecessors")] == 1
    assert env.action_masks()[:4].tolist() == [False, False, False, True]

    quiet = HospitalEnv(EnvConfig(sc, gated_work_prior=False))
    obs, _ = quiet.reset(seed=0, options={"instance": instance})
    assert (obs["ops"][:, prior] == 0).all()


def test_prior_is_identical_whether_or_not_the_step_was_drawn() -> None:
    sc = reveal_scenario(n=1)
    prior = OP_FEATURES.index("expected_gated_work")
    values = []
    for wanted in (True, False):
        env = HospitalEnv(EnvConfig(sc))
        obs, _ = env.reset(seed=0, options={"instance": with_icu(sc, wanted)})
        values.append(obs["ops"][:3, prior].tolist())
    assert values[0] == values[1]


# --- the full example -----------------------------------------------------------


@pytest.mark.slow
def test_full_example_passes_check_env_and_runs() -> None:
    sc, config_warnings = load_config(FULL).to_scenario()
    assert config_warnings == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(
            HospitalEnv(EnvConfig(sc, observation="graph", on_invalid_action="fallback")),
            skip_render_check=True,
        )
    env = HospitalEnv(EnvConfig(sc, observation="graph", debug=True))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=3)
    episodes = grew = 0
    visible = int(obs["op_valid"].sum())
    arrived_rows = 0
    while episodes < 40:
        obs, _, terminated, truncated, _ = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        now_visible = int(obs["op_valid"].sum())
        assert now_visible >= visible
        grew += now_visible > visible
        visible = now_visible
        arrived_rows = max(arrived_rows, int(obs["patient_valid"].sum()))
        if terminated or truncated:
            assert env.observation_space.contains(obs)
            episodes += 1
            obs, _ = env.reset()
            visible = int(obs["op_valid"].sum())
    assert grew > 100 and arrived_rows > 6


def test_reveal_config_validation() -> None:
    def errors(step: dict[str, Any]) -> str:
        config = ScenarioConfig.model_validate(
            {
                "name": "x",
                "roles": [{"name": "N"}],
                "resources": [{"roles": ["N"]}],
                "operation_types": [
                    {
                        "name": "T",
                        "requirements": [{"role": "N"}],
                        "duration": {"distribution": "deterministic", "mean": 1},
                    }
                ],
                "pathways": [{"name": "p", "steps": [{"id": "a", "op_type": "T"}, step]}],
                "arrivals": {"kind": "static", "n_patients": 1},
            }
        )
        with pytest.raises(ScenarioError) as exc:
            config.to_scenario()
        return "\n".join(exc.value.errors)

    assert "revealed_by must name another step" in errors(
        {"id": "b", "op_type": "T", "revealed_by": "zz"}
    )
    assert "revealed_by must name another step" in errors(
        {"id": "b", "op_type": "T", "revealed_by": "b"}
    )
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(
            {"name": "x", "arrivals": {"kind": "scheduled", "n_patients": 2}}
        )
    with pytest.raises(ValidationError):
        clinic(rate=0.1)


def test_same_helper_is_exported() -> None:
    assert callable(same)
