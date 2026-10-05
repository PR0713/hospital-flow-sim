"""Allocation rules shared by the environment and the baselines."""

from __future__ import annotations

from typing import Any

import pytest

from hospital_sim.generators import generate_instance
from hospital_sim.sim import Simulator, compute_metrics
from hospital_sim.sim.allocation import RULES

from .conftest import make_scenario


def surgeons(juniors: int) -> Any:
    return make_scenario(
        resources={},
        op_types={"Surgery": ({"Lead": 1, "Assistant": 1}, 60, True)},
        pathways={"p": [("s", "Surgery", [])]},
        arrivals={"kind": "static", "n_patients": 2},
        extra_resources=[
            {"id": "senior", "roles": ["Lead", "Assistant"], "count": 2},
            {"id": "junior", "roles": ["Assistant"], "count": juniors},
        ],
    )


def makespan(scenario: Any, rule: str) -> float:
    sim = Simulator(generate_instance(scenario, 0), debug=True)
    while sim.run_until_decision():
        op = sim.startable_ops()[0]
        sim.start(op, RULES[rule](sim, op))
    return compute_metrics(sim).makespan


def test_flexible_rule_keeps_the_versatile_surgeon_free() -> None:
    # Two seniors (lead or assist) and two juniors (assist only). Pairing each
    # senior with a junior runs both surgeries at once; pairing the seniors
    # with each other leaves nobody to lead the second.
    assert makespan(surgeons(2), "flexible") == 60
    assert makespan(surgeons(2), "first_idle") == 120


def test_with_three_surgeons_no_rule_can_run_two_surgeries_at_once() -> None:
    assert {makespan(surgeons(1), rule) for rule in RULES} == {120}


@pytest.mark.parametrize("rule", list(RULES))
def test_every_rule_returns_a_valid_team_or_none(rule: str) -> None:
    sim = Simulator(generate_instance(surgeons(2), 0), debug=True)
    assert sim.run_until_decision()
    first, second = sim.startable_ops()
    sim.start(first, RULES[rule](sim, first))  # debug mode validates the team
    team = RULES[rule](sim, second)
    if team is None:
        assert not sim.can_start(second)
    else:
        sim.start(second, team)
