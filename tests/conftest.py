from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from hospital_sim.domain import Scenario
from hospital_sim.generators import ScenarioConfig, load_scenario

EXAMPLE_CONFIG = Path(__file__).parent.parent / "configs" / "example_hospital.yaml"

_BASE: dict[str, Any] = {
    "name": "tiny",
    "roles": [{"name": "Nurse"}, {"name": "Doctor"}, {"name": "Room", "kind": "room"}],
    "resources": [
        {"roles": ["Nurse"], "count": 2},
        {"roles": ["Doctor"], "count": 1},
        {"roles": ["Room"], "count": 1},
    ],
    "operation_types": [
        {
            "name": "Triage",
            "requirements": [{"role": "Nurse"}],
            "duration": {"distribution": "deterministic", "mean": 5},
        },
        {
            "name": "Consult",
            "requirements": [
                {"role": "Doctor"},
                {"role": "Room"},
                {"role": "Nurse", "quantity": 2},
            ],
            "duration": {"mean": 20, "cv": 0.5},
        },
    ],
    "pathways": [
        {
            "name": "visit",
            "steps": [
                {"id": "triage", "op_type": "Triage"},
                {"id": "consult", "op_type": "Consult", "after": ["triage"]},
            ],
        }
    ],
    "arrivals": {"kind": "static", "n_patients": 3},
}


@pytest.fixture
def base_config() -> dict[str, Any]:
    """A minimal valid config as a plain dict; tests mutate their own copy."""
    return copy.deepcopy(_BASE)


@pytest.fixture
def example_scenario() -> Scenario:
    return load_scenario(EXAMPLE_CONFIG)


def build(config: dict[str, Any]) -> Scenario:
    scenario, _ = ScenarioConfig.model_validate(config).to_scenario()
    return scenario


def make_scenario(
    resources: dict[str, int],
    op_types: dict[str, tuple[dict[str, int], float | dict[str, Any], bool]],
    pathways: dict[str, list[tuple[str, str, list[str]]]],
    arrivals: dict[str, Any] | list[dict[str, Any]] | None = None,
    extra_resources: list[dict[str, Any]] | None = None,
    complexity: dict[str, Any] | None = None,
    speed_roles: dict[str, list[str]] | None = None,
    calendars: list[dict[str, Any]] | None = None,
    turnover: dict[str, dict[str, float]] | None = None,
) -> Scenario:
    """Compact scenario builder for simulator tests.

    ``op_types[name] = (requirements, duration, requires_patient)`` where a
    numeric duration means deterministic. ``pathways[name]`` lists
    ``(step id, op type, predecessors)``.
    """
    roles = set(resources) | {r for e in extra_resources or [] for r in e["roles"]}
    return build(
        {
            "name": "test",
            "roles": [{"name": role} for role in sorted(roles)],
            "resources": [{"roles": [role], "count": n} for role, n in resources.items()]
            + (extra_resources or []),
            "operation_types": [
                {
                    "name": name,
                    "requirements": [{"role": r, "quantity": q} for r, q in reqs.items()],
                    "duration": duration
                    if isinstance(duration, dict)
                    else {"distribution": "deterministic", "mean": duration},
                    "requires_patient": requires_patient,
                    "speed_roles": (speed_roles or {}).get(name, []),
                    "turnover": (turnover or {}).get(name, {}),
                }
                for name, (reqs, duration, requires_patient) in op_types.items()
            ],
            "pathways": [
                {
                    "name": name,
                    "steps": [{"id": s, "op_type": t, "after": after} for s, t, after in steps],
                }
                for name, steps in pathways.items()
            ],
            "arrivals": arrivals or {"kind": "static", "n_patients": 1},
            "complexity": complexity or {},
            "calendars": calendars or [],
        }
    )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    from .test_performance import HISTORY, RESULTS

    if RESULTS:
        terminalreporter.section("core engine throughput (logged, not enforced)")
        for n_patients, rate in RESULTS:
            if n_patients < 0:
                terminalreporter.write_line(
                    f"env, {-n_patients} patients: {rate:>9,.0f} steps/s (with observations)"
                )
            else:
                terminalreporter.write_line(f"{n_patients:>4} patients: {rate:>9,.0f} decisions/s")
        terminalreporter.write_line(f"appended to {HISTORY}")
