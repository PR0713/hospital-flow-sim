"""Requirement vectors and resource counts are data, not code.

Every scenario here is written as YAML and loaded through the normal loader;
nothing in the package is special-cased for any role, quantity or count.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from hospital_sim.domain import RequirementSpec, ScenarioError
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import generate_instance, load_config, load_scenario
from hospital_sim.sim import Simulator, compute_metrics, run

EXAMPLE = Path(__file__).parent.parent / "configs" / "example_hospital.yaml"


def edited_example(tmp_path: Path, surgery_nurses: int, nurses: int) -> Path:
    """The shipped example with two numbers changed in the YAML."""
    data = yaml.safe_load(EXAMPLE.read_text())
    for resource in data["resources"]:
        if resource["roles"] == ["Nurse"]:
            resource["count"] = nurses
    surgery = next(t for t in data["operation_types"] if t["name"] == "Surgery")
    for requirement in surgery["requirements"]:
        if requirement["role"] == "Nurse":
            requirement["quantity"] = surgery_nurses
    path = tmp_path / "edited.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_surgery_with_three_of_four_nurses(tmp_path: Path) -> None:
    scenario, warnings = load_config(edited_example(tmp_path, 3, 4)).to_scenario()
    assert warnings == []
    surgery = scenario.operation_type_by_name["Surgery"]
    assert RequirementSpec("Nurse", 3) in surgery.requirements
    assert len(scenario.resources_by_role["Nurse"]) == 4

    for seed in range(8):
        sim = run(Simulator(generate_instance(scenario, seed), debug=True))
        nurse_role = sim.resources.role_names.index("Nurse")
        nurses = set(sim.resources.members[nurse_role])
        surgeries = [o for o in range(sim.n_ops) if sim.op_types[sim.type_of[o]].name == "Surgery"]
        for op in surgeries:
            team = sim.team[op]
            assert team is not None
            assert sum(r in nurses for group in team for r in group) == 3
        # While a surgery runs only one nurse is left, so at most one other
        # nurse-using operation can overlap it; never more than four in use.
        events = []
        for o in range(sim.n_ops):
            used = sum(r in nurses for group in sim.team[o] or () for r in group)
            events += [(sim.start_time[o], used), (sim.end_time[o], -used)]
        in_use = peak = 0
        for _, delta in sorted(events, key=lambda e: (e[0], e[1])):
            in_use += delta
            peak = max(peak, in_use)
        assert peak <= 4
        if surgeries:
            assert peak >= 3


def test_surgery_needing_more_nurses_than_exist_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="needs 3 x 'Nurse' but only 2 exist"):
        load_scenario(edited_example(tmp_path, 3, 2))


def test_environment_follows_the_edited_requirements(tmp_path: Path) -> None:
    scenario = load_scenario(edited_example(tmp_path, 3, 4))
    env = HospitalEnv(EnvConfig(scenario, debug=True))
    obs, _ = env.reset(seed=1)
    nurse = [r.name for r in scenario.roles].index("Nurse")
    surgery = [t.name for t in scenario.operation_types].index("Surgery")
    rng = np.random.default_rng(0)
    seen = False
    while True:
        rows = np.flatnonzero(obs["op_type"] == surgery)
        if len(rows):
            assert (obs["op_requirements"][rows, nurse] == 3).all()
            seen = True
        obs, _, terminated, truncated, _ = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            break
    assert seen and terminated


TEMPLATE = """
name: quantities
roles: [{{name: Nurse}}, {{name: Room, kind: room}}]
resources:
  - {{roles: [Nurse], count: {nurses}}}
  - {{roles: [Room], count: {rooms}}}
operation_types:
  - name: Procedure
    requirements: [{{role: Nurse, quantity: {per_op}}}, {{role: Room}}]
    duration: {{distribution: deterministic, mean: 10}}
pathways:
  - name: p
    steps: [{{id: proc, op_type: Procedure}}]
arrivals: {{kind: static, n_patients: {patients}}}
"""


@pytest.mark.parametrize(
    ("nurses", "per_op", "rooms"),
    [(4, 1, 4), (4, 2, 4), (4, 3, 4), (5, 2, 4), (4, 4, 4), (6, 2, 2), (6, 1, 1), (9, 3, 2)],
)
def test_schedule_follows_quantities_and_counts(
    tmp_path: Path, nurses: int, per_op: int, rooms: int
) -> None:
    """Six identical 10-minute procedures: the number that can run at once is
    min(nurses // per_op, rooms), which fixes the makespan."""
    path = tmp_path / "q.yaml"
    path.write_text(TEMPLATE.format(nurses=nurses, per_op=per_op, rooms=rooms, patients=6))
    sim = run(Simulator(generate_instance(load_scenario(path), 0), debug=True))
    parallel = min(nurses // per_op, rooms)
    assert compute_metrics(sim).makespan == 10 * math.ceil(6 / parallel)
    starts = sorted(sim.start_time)
    assert starts == [10 * (i // parallel) for i in range(6)]


SUBSTITUTION = """
name: substitution
roles:
  - {name: ScanOperator}        # anyone allowed to run the scanner
  - {name: Radiologist}
  - {name: Scanner, kind: equipment}
resources:
  - {id: radiologist, roles: [Radiologist, ScanOperator]}
  - {id: senior_tech, roles: [ScanOperator]}
  - {roles: [Scanner], count: 2}
operation_types:
  - name: Scan
    requirements: [{role: ScanOperator}, {role: Scanner}]
    duration: {distribution: deterministic, mean: 20}
  - name: Report
    requirements: [{role: Radiologist}]
    duration: {distribution: deterministic, mean: 5}
pathways:
  - name: p
    steps: [{id: scan, op_type: Scan}, {id: report, op_type: Report, after: [scan]}]
arrivals: {kind: static, n_patients: 2}
"""


def test_either_or_staffing_is_a_shared_role(tmp_path: Path) -> None:
    """A scan run by "a radiologist or a senior technician" needs no schema
    feature: define a role both belong to."""
    path = tmp_path / "s.yaml"
    path.write_text(SUBSTITUTION)
    sim = run(Simulator(generate_instance(load_scenario(path), 0), debug=True))
    ids = sim.resources.resource_ids
    scans = [o for o in range(sim.n_ops) if sim.op_types[sim.type_of[o]].name == "Scan"]
    operators = {ids[(sim.team[o] or ((),))[0][0]] for o in scans}
    assert operators == {"radiologist", "senior_tech"}  # each ran one scan
    assert [sim.start_time[o] for o in scans] == [0, 0]  # in parallel
