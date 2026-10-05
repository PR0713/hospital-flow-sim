from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from hospital_sim.domain import DurationKind, RequirementSpec, RoleKind, Scenario
from hospital_sim.generators import ScenarioConfig, load_config, load_scenario

from .conftest import EXAMPLE_CONFIG, build


def test_example_hospital_loads_without_warnings() -> None:
    scenario, warnings = load_config(EXAMPLE_CONFIG).to_scenario()
    assert warnings == []
    assert len(scenario.resources) == 22
    assert {p.name for p in scenario.pathways} == {"surgical", "diagnostic"}


def test_example_surgery_team(example_scenario: Scenario) -> None:
    surgery = example_scenario.operation_type_by_name["Surgery"]
    assert set(surgery.requirements) == {
        RequirementSpec("OperatingRoom", 1),
        RequirementSpec("LeadSurgeon", 1),
        RequirementSpec("AssistantSurgeon", 1),
        RequirementSpec("Anaesthetist", 1),
        RequirementSpec("Nurse", 2),
    }
    assert surgery.duration.kind is DurationKind.LOGNORMAL


def test_example_role_membership(example_scenario: Scenario) -> None:
    pool = example_scenario.resources_by_role
    assert pool["LeadSurgeon"] == ("surgeon_senior_1", "surgeon_senior_2")
    assert pool["AssistantSurgeon"] == ("surgeon_senior_1", "surgeon_senior_2", "surgeon_junior")
    assert {r.name: r.kind for r in example_scenario.roles}["LabAnalyzer"] is RoleKind.EQUIPMENT
    assert not example_scenario.operation_type_by_name["LabAnalysis"].requires_patient


def test_json_and_yaml_give_the_same_scenario(tmp_path: Path) -> None:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    json_path = tmp_path / "hospital.json"
    json_path.write_text(json.dumps(data))
    assert load_scenario(json_path) == load_scenario(EXAMPLE_CONFIG)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "hospital.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="unsupported config format"):
        load_config(path)


def test_unknown_keys_are_rejected(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][0]["requirments"] = []
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(base_config)


@pytest.mark.parametrize(
    "duration",
    [
        {"mean": 10},  # lognormal without cv
        {"distribution": "deterministic", "mean": 10, "cv": 0.2},
        {"distribution": "empirical"},
        {"distribution": "empirical", "samples": [1, 2], "mean": 3},
        {"distribution": "gamma", "cv": 0.3},  # no mean
        {"mean": -5, "cv": 0.3},
    ],
)
def test_bad_duration_parameters_are_rejected(
    base_config: dict[str, Any], duration: dict[str, Any]
) -> None:
    base_config["operation_types"][0]["duration"] = duration
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(base_config)


def test_empirical_duration_mean_is_derived(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][0]["duration"] = {
        "distribution": "empirical",
        "samples": [4, 6, 8],
    }
    spec = build(base_config).operation_type_by_name["Triage"].duration
    assert spec.mean == 6 and spec.samples == (4, 6, 8)


def test_poisson_arrivals_need_a_rate(base_config: dict[str, Any]) -> None:
    base_config["arrivals"] = {"kind": "poisson", "n_patients": 5}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(base_config)


def test_pathway_needs_exactly_one_source(base_config: dict[str, Any]) -> None:
    base_config["pathways"][0]["random"] = {"n_ops": 3, "density": 0.5}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(base_config)


def test_resource_id_expansion(base_config: dict[str, Any]) -> None:
    base_config["resources"] = [
        {"roles": ["Nurse"], "count": 2},
        {"id": "dr_house", "roles": ["Doctor"]},
        {"id": "bay", "roles": ["Room"], "count": 2},
    ]
    assert [r.id for r in build(base_config).resources] == [
        "nurse_1",
        "nurse_2",
        "dr_house",
        "bay_1",
        "bay_2",
    ]


def test_random_pathway_from_config(base_config: dict[str, Any]) -> None:
    base_config["pathways"].append({"name": "rnd", "random": {"n_ops": 6, "density": 0.4}})
    first, second = build(base_config), build(base_config)
    assert len(first.pathway_by_name["rnd"].steps) == 6
    assert first == second
