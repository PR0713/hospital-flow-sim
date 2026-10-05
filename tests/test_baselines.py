"""M4: dispatching rules, evaluation harness, allocation search."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hospital_sim.baselines as baselines_package
from hospital_sim.baselines import (
    DISPATCHER_FACTORIES,
    DISPATCHERS,
    Policy,
    all_policies,
    best_allocation,
    distinct_teams,
    remaining_work,
)
from hospital_sim.domain import Scenario
from hospital_sim.env import EnvConfig
from hospital_sim.eval import EnvAgent, evaluate, sample_instances
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import OpState, Simulator, compute_metrics
from hospital_sim.sim.allocation import RULES

from .conftest import make_scenario
from .test_env_integrity import FORBIDDEN

CONFIGS = Path(__file__).parent.parent / "configs"


def scenario(name: str) -> Scenario:
    return load_scenario(CONFIGS / f"example_hospital{name}.yaml")


# --- dispatching rules on a hand-checkable case ---------------------------------


def choice_case() -> Simulator:
    """One nurse, three patients ready at t=0, in this ready order:

    patient 0 (routine): Long (30) then Short (5)   -> 35 of work left
    patient 1 (routine): Short (5)                   ->  5
    patient 2 (urgent, weight 4): Medium (10) with 2 helpers -> 10
    """
    sc = make_scenario(
        resources={"Nurse": 1, "Helper": 2},
        op_types={
            "Long": ({"Nurse": 1}, 30, True),
            "Short": ({"Nurse": 1}, 5, True),
            "Medium": ({"Nurse": 1, "Helper": 2}, 10, True),
        },
        pathways={
            "two_step": [("a", "Long", []), ("b", "Short", ["a"])],
            "quick": [("a", "Short", [])],
            "team": [("a", "Medium", [])],
        },
        arrivals=[
            {"name": "first", "kind": "static", "n_patients": 1, "pathway_mix": {"two_step": 1}},
            {"name": "second", "kind": "static", "n_patients": 1, "pathway_mix": {"quick": 1}},
            {
                "name": "urgent",
                "kind": "static",
                "n_patients": 1,
                "pathway_mix": {"team": 1},
                "weight": 4.0,
                "urgent": True,
            },
        ],
    )
    sim = Simulator(generate_instance(sc, 0), debug=True)
    assert sim.run_until_decision()
    return sim


@pytest.mark.parametrize(
    ("rule", "patient"),
    [
        ("fifo", 0),  # first in ready order
        ("spt", 1),  # 5 minutes
        ("longest_remaining_work", 0),  # 35 minutes of work left
        ("most_constrained_first", 2),  # team of three
        ("urgent_first", 2),
        ("weight_aware", 2),  # 4 / 10 beats 1 / 5 and 1 / 35
        ("acuity_aware", 2),  # no acuity classes here: same as weight_aware
    ],
)
def test_each_rule_picks_the_expected_operation(rule: str, patient: int) -> None:
    sim = choice_case()
    startable = sim.startable_ops()
    assert [sim.patient_of[o] for o in startable] == [0, 1, 2]
    dispatch = (
        DISPATCHER_FACTORIES[rule](sim.instance.scenario)
        if rule in DISPATCHER_FACTORIES
        else DISPATCHERS[rule]
    )
    assert sim.patient_of[dispatch(sim, startable)] == patient


def test_remaining_work_counts_only_visible_unstarted_operations() -> None:
    sim = choice_case()
    assert [remaining_work(sim, p) for p in range(3)] == [35, 5, 10]
    sim.start(0)
    assert remaining_work(sim, 0) == 5  # the running operation no longer counts


def test_weight_aware_without_weights_prefers_the_shortest_remaining_work() -> None:
    sim = choice_case()
    sim.weight[2] = 1.0
    assert sim.patient_of[DISPATCHERS["weight_aware"](sim, sim.startable_ops())] == 1


def test_random_policy_is_reproducible_and_varies() -> None:
    sc = scenario("")
    policy = Policy("random")
    a = policy.run(generate_instance(sc, 3))
    b = policy.run(generate_instance(sc, 3))
    assert a.start_time == b.start_time
    assert a.start_time != Policy("fifo").run(generate_instance(sc, 3)).start_time


@pytest.mark.parametrize("name", ["", "_dynamic", "_stress"])
def test_all_policies_finish_every_patient_without_idling(name: str) -> None:
    sc = scenario(name)
    for policy in all_policies():
        for seed in range(3):
            sim = policy.run(generate_instance(sc, seed), debug=True)
            assert sim.done and all(s == OpState.DONE for s in sim.state), policy.name


def test_baselines_never_touch_hidden_state() -> None:
    offenders = []
    for path in sorted(Path(baselines_package.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(node, "attr", None) or getattr(node, "id", None)
            if name in FORBIDDEN:
                offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert offenders == []


# --- evaluation harness ---------------------------------------------------------


@pytest.fixture(scope="module")
def stress_eval() -> Any:
    return evaluate(
        scenario("_stress"), [Policy("fifo"), Policy("urgent_first"), Policy("spt")], 60
    )


def test_instances_are_common_reproducible_and_never_from_train() -> None:
    sc = scenario("")
    first = sample_instances(sc, 5, base_seed=1)
    assert first == sample_instances(sc, 5, base_seed=1)
    assert first != sample_instances(sc, 5, base_seed=2)
    assert first != sample_instances(sc, 5, base_seed=1, split="test")
    with pytest.raises(ValueError, match="must not use the train split"):
        sample_instances(sc, 5, split="train")
    with pytest.raises(ValueError, match="must not use the train split"):
        evaluate(sc, [Policy("fifo")], 2, split="train")


def test_evaluation_is_reproducible_and_paired(stress_eval: Any) -> None:
    again = evaluate(scenario("_stress"), [Policy("fifo")], 60)
    assert np.array_equal(
        again.series("fifo+flexible", "makespan"), stress_eval.series("fifo+flexible", "makespan")
    )
    assert all(len(r) == 60 for r in stress_eval.records.values())


def test_confidence_interval_arithmetic() -> None:
    result = evaluate(scenario(""), [Policy("fifo")], 30)
    x = result.series("fifo+flexible", "makespan")
    mean, sd, half = result.summary("makespan")["fifo+flexible"]
    assert mean == pytest.approx(x.mean()) and sd == pytest.approx(x.std(ddof=1))
    assert half == pytest.approx(2.045229642 * sd / np.sqrt(30), rel=1e-6)  # t(0.975, 29)


def test_common_random_numbers_tighten_comparisons(stress_eval: Any) -> None:
    a, b = "urgent_first+flexible", "fifo+flexible"
    _, paired = stress_eval.paired_difference("mean_flow_time", a, b)
    summary = stress_eval.summary("mean_flow_time")
    unpaired = float(np.hypot(summary[a][2], summary[b][2]))
    assert paired < 0.5 * unpaired


def test_urgent_first_helps_the_urgent_class(stress_eval: Any) -> None:
    fifo = stress_eval.by_class("fifo+flexible", "mean_flow_time")
    urgent = stress_eval.by_class("urgent_first+flexible", "mean_flow_time")
    assert set(fifo) == {"elective", "urgent"}
    assert urgent["urgent"][0] < fifo["urgent"][0]
    assert fifo["urgent"][3] < 60  # not every day has an urgent patient
    # ...and the weighted objective sees it.
    diff, half = stress_eval.paired_difference(
        "weighted_mean_flow_time", "urgent_first+flexible", "fifo+flexible"
    )
    assert diff < 0


def test_class_means_are_consistent_with_the_overall_mean(stress_eval: Any) -> None:
    for record in stress_eval.records["fifo+flexible"]:
        total = sum(c["patients"] for c in record.by_class.values())
        pooled = sum(c["patients"] * c["mean_flow_time"] for c in record.by_class.values()) / total
        assert pooled == pytest.approx(record.values["mean_flow_time"])


def test_table_and_ranking(stress_eval: Any) -> None:
    table = stress_eval.table()
    assert table.splitlines()[0] == "| Policy | mean_flow_time | makespan | mean_op_wait | left |"
    assert len(table.splitlines()) == 2 + 3
    assert stress_eval.ranking("mean_flow_time")[0] in table.splitlines()[2]
    assert set(stress_eval.utilization("fifo+flexible")) == {
        r.name for r in scenario("_stress").roles
    }


def test_env_agents_are_evaluated_on_the_same_instances_and_train_is_refused() -> None:
    sc = scenario("")

    def lowest(observation: dict[str, Any], mask: np.ndarray) -> int:
        return int(np.flatnonzero(mask)[0])

    agent = EnvAgent("lowest_slot", lowest, EnvConfig(sc, split="eval", allow_wait=False))
    result = evaluate(sc, [Policy("fifo"), agent], 5)
    assert set(result.records) == {"fifo+flexible", "lowest_slot"}
    assert (result.series("lowest_slot", "makespan") > 0).all()

    trained = EnvAgent("cheater", lowest, EnvConfig(sc, split="train"))
    with pytest.raises(ValueError, match="train-split env"):
        evaluate(sc, [trained], 2)


# --- allocation search ----------------------------------------------------------


def surgeons(juniors: int, patients: int = 2) -> Scenario:
    return make_scenario(
        resources={},
        op_types={"Surgery": ({"Lead": 1, "Assistant": 1}, 60, True)},
        pathways={"p": [("s", "Surgery", [])]},
        arrivals={"kind": "static", "n_patients": patients},
        extra_resources=[
            {"id": "senior", "roles": ["Lead", "Assistant"], "count": 2},
            {"id": "junior", "roles": ["Assistant"], "count": juniors},
        ],
    )


def makespan(sim: Simulator) -> float:
    return compute_metrics(sim).makespan


def test_symmetric_resources_are_enumerated_once() -> None:
    sim = Simulator(generate_instance(surgeons(2), 0))
    assert sim.run_until_decision()
    teams = distinct_teams(sim, sim.startable_ops()[0])
    ids = sim.resources.resource_ids
    named = sorted(tuple(ids[r] for group in team for r in group) for team in teams)
    # Lead is a senior; the assistant is either the other senior or a junior.
    assert named == [("senior_1", "junior_1"), ("senior_1", "senior_2")]


def test_search_finds_the_parallel_schedule() -> None:
    instance = generate_instance(surgeons(2), 0)
    best, leaves = best_allocation(instance, "fifo", makespan)
    assert best == 60 and leaves <= 4
    assert makespan(Policy("fifo", "first_idle").run(instance)) == 120
    assert makespan(Policy("fifo", "flexible").run(instance)) == best


@pytest.mark.parametrize("seed", range(6))
def test_search_is_never_worse_than_any_rule(seed: int) -> None:
    base = scenario("")
    sc = make_small(base, patients=4)
    instance = generate_instance(sc, seed)

    def flow(sim: Simulator) -> float:
        return compute_metrics(sim).mean_flow_time

    best, _ = best_allocation(instance, "fifo", flow)
    for rule in RULES:
        assert best <= flow(Policy("fifo", rule).run(instance)) + 1e-9


def make_small(base: Scenario, patients: int) -> Scenario:
    import dataclasses

    return dataclasses.replace(
        base, arrivals=(dataclasses.replace(base.arrivals[0], n_patients=patients),)
    )
