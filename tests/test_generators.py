from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from hospital_sim.domain import ArrivalKind, ArrivalSpec, ProblemInstance, Scenario
from hospital_sim.domain.dag import ancestors, topological_order
from hospital_sim.generators import generate_instance, random_pathway

from .conftest import build

TYPES = ["A", "B", "C"]


# --- random pathways ---------------------------------------------------------


@pytest.mark.parametrize("density", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_random_pathway_is_a_dag(density: float) -> None:
    pathway = random_pathway("p", TYPES, 12, density, np.random.default_rng(1))
    assert len(pathway.steps) == 12
    assert {s.op_type for s in pathway.steps} <= set(TYPES)
    topological_order({s.id: s.predecessors for s in pathway.steps})  # raises on a cycle


def test_density_zero_is_fully_parallel() -> None:
    pathway = random_pathway("p", TYPES, 8, 0.0, np.random.default_rng(0))
    assert all(s.predecessors == () for s in pathway.steps)


def test_density_one_is_a_chain() -> None:
    pathway = random_pathway("p", TYPES, 8, 1.0, np.random.default_rng(0))
    assert [s.predecessors for s in pathway.steps] == [()] + [(f"s{i}",) for i in range(7)]


def test_density_controls_number_of_ordered_pairs() -> None:
    def ordered_pairs(density: float) -> int:
        pathway = random_pathway("p", TYPES, 15, density, np.random.default_rng(7))
        return sum(
            len(a) for a in ancestors({s.id: s.predecessors for s in pathway.steps}).values()
        )

    assert ordered_pairs(0.1) < ordered_pairs(0.4) < ordered_pairs(0.9)


def test_random_pathway_is_reproducible() -> None:
    first = random_pathway("p", TYPES, 10, 0.3, np.random.default_rng(5))
    assert first == random_pathway("p", TYPES, 10, 0.3, np.random.default_rng(5))
    assert first != random_pathway("p", TYPES, 10, 0.3, np.random.default_rng(6))


# --- instances ---------------------------------------------------------------


def check_structure(instance: ProblemInstance) -> None:
    """Invariants every generated instance must satisfy."""
    ops = instance.operations
    assert [op.id for op in ops] == list(range(len(ops)))
    assert len(instance.hidden.duration_quantiles) == len(ops)
    assert all(0 <= u < 1 for u in instance.hidden.duration_quantiles)
    for patient in instance.patients:
        assert patient.operations, "patient without operations"
        own = {op.id for op in patient.operations}
        for op in patient.operations:
            assert op.patient_id == patient.id
            assert set(op.predecessors) <= own, "precedence must stay within a patient"
            for pred in op.predecessors:
                assert op.id in ops[pred].successors
            for succ in op.successors:
                assert op.id in ops[succ].predecessors
        topological_order({op.id: op.predecessors for op in patient.operations})


def test_same_seed_gives_identical_instance(example_scenario: Scenario) -> None:
    assert generate_instance(example_scenario, 42) == generate_instance(example_scenario, 42)


def test_different_seeds_give_different_instances(example_scenario: Scenario) -> None:
    assert generate_instance(example_scenario, 1) != generate_instance(example_scenario, 2)


@pytest.mark.parametrize("seed", range(20))
def test_example_instances_are_well_formed(example_scenario: Scenario, seed: int) -> None:
    instance = generate_instance(example_scenario, seed)
    assert len(instance.patients) == 10
    assert all(p.arrival_time == 0.0 for p in instance.patients)
    check_structure(instance)


def test_branch_group_picks_exactly_one(example_scenario: Scenario) -> None:
    seen: set[str] = set()
    for seed in range(30):
        for patient in generate_instance(example_scenario, seed).patients:
            if patient.pathway != "surgical":
                continue
            imaging = [op.step_id for op in patient.operations if op.step_id in {"xray", "ct"}]
            assert len(imaging) == 1
            seen.update(imaging)
    assert seen == {"xray", "ct"}


def test_skipped_optional_step_passes_precedence_through(example_scenario: Scenario) -> None:
    with_review = without_review = 0
    for seed in range(30):
        instance = generate_instance(example_scenario, seed)
        for patient in instance.patients:
            if patient.pathway != "surgical":
                continue
            by_step = {op.step_id: op for op in patient.operations}
            pre_op_preds = {instance.operations[i].step_id for i in by_step["pre_op"].predecessors}
            imaging = "xray" if "xray" in by_step else "ct"
            if "anaesthetic_review" in by_step:
                with_review += 1
                assert pre_op_preds == {"anaesthetic_review", imaging}
            else:
                without_review += 1
                assert pre_op_preds == {"lab", imaging}
    assert with_review and without_review


def test_pathway_mix_is_respected(base_config: dict[str, Any]) -> None:
    base_config["pathways"].append(
        {"name": "quick", "steps": [{"id": "triage", "op_type": "Triage"}]}
    )
    base_config["arrivals"] = {
        "kind": "static",
        "n_patients": 2000,
        "pathway_mix": {"visit": 0.25, "quick": 0.75},
    }
    instance = generate_instance(build(base_config), 0)
    share = sum(p.pathway == "quick" for p in instance.patients) / 2000
    assert share == pytest.approx(0.75, abs=0.03)


def test_poisson_arrivals(base_config: dict[str, Any]) -> None:
    base_config["arrivals"] = {"kind": "poisson", "n_patients": 4000, "rate": 0.2}
    instance = generate_instance(build(base_config), 3)
    times = [p.arrival_time for p in instance.patients]
    assert times[0] > 0 and times == sorted(times)
    assert np.mean(np.diff(times)) == pytest.approx(5.0, rel=0.05)
    check_structure(instance)


def test_streams_are_independent(example_scenario: Scenario) -> None:
    """Changing the arrival process must not change who the patients are or
    how lucky each operation is (common random numbers)."""
    poisson = ArrivalSpec(ArrivalKind.POISSON, 10, 0.1, example_scenario.arrivals[0].pathway_mix)
    static = generate_instance(example_scenario, 11)
    dynamic = generate_instance(example_scenario, 11, arrivals=poisson)

    assert [p.arrival_time for p in static.patients] != [p.arrival_time for p in dynamic.patients]
    assert static.hidden == dynamic.hidden
    assert [dataclasses.replace(p, arrival_time=0.0) for p in dynamic.patients] == list(
        static.patients
    )


@pytest.mark.parametrize("n_patients", [1, 3, 25])
def test_batch_size_is_not_baked_in(example_scenario: Scenario, n_patients: int) -> None:
    arrivals = dataclasses.replace(example_scenario.arrivals[0], n_patients=n_patients)
    instance = generate_instance(example_scenario, 0, arrivals=arrivals)
    assert len(instance.patients) == n_patients
    check_structure(instance)


def test_random_pathways_generate_valid_instances(base_config: dict[str, Any]) -> None:
    base_config["pathways"] = [
        {"name": f"rnd{i}", "random": {"n_ops": 4 + i, "density": 0.3, "seed": i}} for i in range(4)
    ]
    base_config["arrivals"]["n_patients"] = 12
    instance = generate_instance(build(base_config), 9)
    assert len({len(p.operations) for p in instance.patients}) > 1  # varying DAG sizes
    check_structure(instance)
