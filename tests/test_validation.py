from __future__ import annotations

from typing import Any

import pytest

from hospital_sim.domain import ScenarioError
from hospital_sim.generators import ScenarioConfig

from .conftest import build


def errors_of(config: dict[str, Any]) -> str:
    with pytest.raises(ScenarioError) as exc:
        build(config)
    return "\n".join(exc.value.errors)


def test_base_config_is_valid(base_config: dict[str, Any]) -> None:
    scenario, warnings = ScenarioConfig.model_validate(base_config).to_scenario()
    assert warnings == []
    assert scenario.resources_by_role["Nurse"] == ("nurse_1", "nurse_2")


def test_requirement_exceeding_pool_is_flagged(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][1]["requirements"][2]["quantity"] = 3
    assert "needs 3 x 'Nurse' but only 2 exist" in errors_of(base_config)


def test_shared_resource_across_roles_is_flagged(base_config: dict[str, Any]) -> None:
    # One person holds both roles, and Consult needs one of each at once.
    base_config["roles"] += [{"name": "Lead"}, {"name": "Assistant"}]
    base_config["resources"].append({"id": "s1", "roles": ["Lead", "Assistant"]})
    base_config["operation_types"][1]["requirements"] = [{"role": "Lead"}, {"role": "Assistant"}]
    assert "no full team exists" in errors_of(base_config)


def test_shared_resource_is_fine_with_enough_people(base_config: dict[str, Any]) -> None:
    base_config["roles"] += [{"name": "Lead"}, {"name": "Assistant"}]
    base_config["resources"].append({"id": "s", "roles": ["Lead", "Assistant"], "count": 2})
    base_config["operation_types"][1]["requirements"] = [{"role": "Lead"}, {"role": "Assistant"}]
    build(base_config)


def test_unknown_role_in_requirement(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][0]["requirements"] = [{"role": "Urologist"}]
    assert "unknown role 'Urologist'" in errors_of(base_config)


def test_unknown_role_on_resource(base_config: dict[str, Any]) -> None:
    base_config["resources"].append({"id": "x", "roles": ["Porter"]})
    assert "resource 'x': unknown role 'Porter'" in errors_of(base_config)


def test_duplicate_role_in_requirements(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][0]["requirements"] = [{"role": "Nurse"}, {"role": "Nurse"}]
    assert "listed more than once" in errors_of(base_config)


def test_duplicate_names_are_flagged(base_config: dict[str, Any]) -> None:
    base_config["resources"].append({"id": "nurse_1", "roles": ["Nurse"]})
    base_config["operation_types"].append(base_config["operation_types"][0])
    message = errors_of(base_config)
    assert "duplicate resource 'nurse_1'" in message
    assert "duplicate operation type 'Triage'" in message


def test_pathway_cycle_is_flagged(base_config: dict[str, Any]) -> None:
    base_config["pathways"][0]["steps"][0]["after"] = ["consult"]
    assert "precedence cycle" in errors_of(base_config)


def test_unknown_predecessor_and_op_type(base_config: dict[str, Any]) -> None:
    steps = base_config["pathways"][0]["steps"]
    steps[1]["after"] = ["nope"]
    steps[0]["op_type"] = "Dance"
    message = errors_of(base_config)
    assert "unknown predecessor 'nope'" in message
    assert "unknown operation type 'Dance'" in message


def test_pathway_that_can_be_empty_is_flagged(base_config: dict[str, Any]) -> None:
    base_config["pathways"][0]["steps"] = [
        {"id": "triage", "op_type": "Triage", "probability": 0.5}
    ]
    assert "could have no operations" in errors_of(base_config)


def test_optional_step_in_branch_group_is_flagged(base_config: dict[str, Any]) -> None:
    base_config["pathways"][0]["steps"][0].update(branch_group="g", probability=0.5)
    assert "both optional and in a branch group" in errors_of(base_config)


def test_unknown_pathway_in_mix(base_config: dict[str, Any]) -> None:
    base_config["arrivals"]["pathway_mix"] = {"ghost": 1.0}
    assert "unknown pathway 'ghost'" in errors_of(base_config)


def test_all_errors_are_reported_together(base_config: dict[str, Any]) -> None:
    base_config["operation_types"][0]["requirements"] = [{"role": "Urologist"}]
    base_config["pathways"][0]["steps"][0]["after"] = ["consult"]
    with pytest.raises(ScenarioError) as exc:
        build(base_config)
    assert len(exc.value.errors) == 2


def test_unused_definitions_produce_warnings(base_config: dict[str, Any]) -> None:
    base_config["roles"].append({"name": "Porter"})
    base_config["pathways"][0]["steps"].pop()
    _, warnings = ScenarioConfig.model_validate(base_config).to_scenario()
    assert "operation type 'Consult' is not used by any pathway" in warnings
    assert "role 'Porter' is not required by any operation type" in warnings
