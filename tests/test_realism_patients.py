"""M2b: latent complexity, acuity class, arrival processes, patient classes."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import special, stats

from hospital_sim.domain import ProblemInstance, Scenario, ScenarioError
from hospital_sim.generators import ScenarioConfig, generate_instance
from hospital_sim.sim import Simulator, compute_metrics, run

from .conftest import build, make_scenario

OPS_PER_PATIENT = 4


def scenario(
    n: int = 5000,
    complexity: dict[str, Any] | None = None,
    arrivals: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> Scenario:
    return make_scenario(
        resources={"Staff": 10},
        op_types={"T": ({"Staff": 1}, {"mean": 10, "cv": 0.5}, False)},
        pathways={
            "p": [(f"s{i}", "T", []) for i in range(OPS_PER_PATIENT)],
            "q": [("only", "T", [])],
        },
        arrivals=arrivals or {"kind": "static", "n_patients": n, "pathway_mix": {"p": 1.0}},
        complexity=complexity,
    )


def scores(instance: ProblemInstance) -> np.ndarray:
    """Normal scores of the duration quantiles, one row per patient."""
    u = np.array(instance.hidden.duration_quantiles).reshape(-1, OPS_PER_PATIENT)
    return special.ndtri(u)


# --- latent complexity ----------------------------------------------------------


def test_rho_zero_is_exactly_the_independent_behaviour() -> None:
    plain = generate_instance(scenario(200), 5)
    explicit = generate_instance(scenario(200, {"rho": 0.0, "acuity_classes": 3}), 5)
    assert plain.hidden.duration_quantiles == explicit.hidden.duration_quantiles


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7])
def test_within_and_between_patient_correlation(rho: float) -> None:
    z = scores(generate_instance(scenario(complexity={"rho": rho}), 2))
    within = [np.corrcoef(z[:, i], z[:, j])[0, 1] for i in range(4) for j in range(i + 1, 4)]
    between = [np.corrcoef(z[:-1, i], z[1:, i])[0, 1] for i in range(4)]
    assert np.mean(within) == pytest.approx(rho, abs=0.03)
    assert np.mean(between) == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize("rho", [0.3, 0.7])
def test_copula_keeps_quantiles_uniform(rho: float) -> None:
    instance = generate_instance(scenario(complexity={"rho": rho}), 3)
    u = np.array(instance.hidden.duration_quantiles).reshape(-1, OPS_PER_PATIENT)
    # One operation per patient, so the sample is independent.
    assert stats.kstest(u[:, 0], "uniform").pvalue > 0.01


def test_complex_patients_are_slow_at_everything() -> None:
    spread = {"kind": "poisson", "n_patients": 3000, "rate": 0.1, "pathway_mix": {"p": 1.0}}
    instance = generate_instance(scenario(complexity={"rho": 0.6}, arrivals=spread), 4)
    sim = run(Simulator(instance))
    durations = np.array([sim.end_time[o] - sim.start_time[o] for o in range(sim.n_ops)])
    per_patient = durations.reshape(-1, OPS_PER_PATIENT)
    factor = np.array(instance.hidden.patient_factors)
    assert stats.spearmanr(factor, per_patient.sum(axis=1)).statistic > 0.6
    assert stats.spearmanr(per_patient[:, 0], per_patient[:, 1]).statistic > 0.4


def test_complexity_does_not_change_arrivals_or_pathways(example_scenario: Scenario) -> None:
    correlated = dataclasses.replace(
        example_scenario, complexity=dataclasses.replace(example_scenario.complexity, rho=0.5)
    )
    a, b = generate_instance(example_scenario, 6), generate_instance(correlated, 6)
    assert a.patients == b.patients
    assert a.hidden.patient_factors == b.hidden.patient_factors
    assert a.hidden.duration_quantiles != b.hidden.duration_quantiles


# --- visible acuity class -------------------------------------------------------


def test_acuity_is_off_by_default() -> None:
    assert all(p.acuity_class is None for p in generate_instance(scenario(50), 0).patients)


def test_acuity_classes_are_balanced_and_informative() -> None:
    instance = generate_instance(
        scenario(complexity={"rho": 0.5, "acuity_classes": 3, "acuity_correlation": 0.7}), 7
    )
    classes = np.array([p.acuity_class for p in instance.patients])
    factor = np.array(instance.hidden.patient_factors)
    assert sorted(set(classes)) == [0, 1, 2]
    assert np.bincount(classes) / len(classes) == pytest.approx([1 / 3] * 3, abs=0.03)
    means = [factor[classes == k].mean() for k in range(3)]
    assert means[0] < -0.4 and abs(means[1]) < 0.1 and means[2] > 0.4
    # Informative but noisy: the classes overlap.
    assert factor[classes == 0].max() > factor[classes == 2].min()


@pytest.mark.parametrize(("correlation", "expected"), [(0.0, 0.0), (1.0, 1.0)])
def test_acuity_correlation_extremes(correlation: float, expected: float) -> None:
    instance = generate_instance(
        scenario(complexity={"acuity_classes": 4, "acuity_correlation": correlation}), 8
    )
    classes = np.array([p.acuity_class for p in instance.patients])
    factor = np.array(instance.hidden.patient_factors)
    if expected == 0.0:
        assert abs(stats.spearmanr(classes, factor).statistic) < 0.04
    else:  # a perfect reading sorts patients into classes by their true factor
        assert all(factor[classes == k].max() <= factor[classes == k + 1].min() for k in range(3))


def test_true_factor_is_only_in_the_hidden_part() -> None:
    instance = generate_instance(scenario(20, {"rho": 0.5, "acuity_classes": 3}), 0)
    assert len(instance.hidden.patient_factors) == 20
    fields = {f.name for f in dataclasses.fields(instance.patients[0])}
    assert "acuity_class" in fields and not {"factor", "complexity"} & fields


# --- arrival processes ----------------------------------------------------------


def arrival_times(arrivals: dict[str, Any] | list[dict[str, Any]], seed: int = 0) -> np.ndarray:
    instance = generate_instance(scenario(arrivals=arrivals), seed)
    return np.array([p.arrival_time for p in instance.patients])


def dispersion(times: np.ndarray, window: float) -> float:
    """Index of dispersion of counts per window (1 for a Poisson process)."""
    counts = np.bincount((times // window).astype(int))[:-1]  # drop the partial last window
    return float(counts.var() / counts.mean())


def test_homogeneous_poisson_gaps_are_exponential() -> None:
    times = arrival_times({"kind": "poisson", "n_patients": 5000, "rate": 0.2})
    gaps = np.diff(np.concatenate([[0.0], times]))
    assert stats.kstest(gaps, "expon", args=(0, 5.0)).pvalue > 0.01
    assert dispersion(times, 50.0) == pytest.approx(1.0, abs=0.15)


def test_rate_profile_shapes_arrivals() -> None:
    rates = [0.5, 0.0, 0.25, 1.0]
    times = arrival_times(
        {"kind": "poisson", "n_patients": 8000, "rate_profile": {"period": 400, "rates": rates}}
    )
    bins = ((times % 400) // 100).astype(int)
    counts = np.bincount(bins, minlength=4)
    assert counts[1] == 0  # nobody arrives while the rate is zero
    observed = counts[[0, 2, 3]]
    expected = np.array([0.5, 0.25, 1.0]) / 1.75 * observed.sum()
    assert stats.chisquare(observed, expected).pvalue > 0.01
    # Overall rate: mean of the profile.
    assert len(times) / times[-1] == pytest.approx(np.mean(rates), rel=0.05)


def test_time_rescaling_of_a_profiled_process_gives_a_unit_poisson() -> None:
    rates = np.array([0.5, 0.1, 0.25, 1.0])
    times = arrival_times(
        {
            "kind": "poisson",
            "n_patients": 6000,
            "rate_profile": {"period": 400, "rates": list(rates)},
        }
    )

    def cumulative(t: np.ndarray) -> np.ndarray:
        whole, rest = np.divmod(t, 400.0)
        edges = np.concatenate([[0.0], np.cumsum(rates * 100.0)])
        k = np.minimum((rest // 100).astype(int), 3)
        return whole * edges[-1] + edges[k] + rates[k] * (rest - 100.0 * k)

    gaps = np.diff(np.concatenate([[0.0], cumulative(times)]))
    assert stats.kstest(gaps, "expon").pvalue > 0.01


@pytest.mark.parametrize("batch_mean", [2.0, 4.0])
def test_bursts_overdisperse_counts_without_changing_the_rate(batch_mean: float) -> None:
    times = arrival_times(
        {"kind": "poisson", "n_patients": 20000, "rate": 0.2, "batch_mean": batch_mean}
    )
    assert len(times) / times[-1] == pytest.approx(0.2, rel=0.05)
    # Compound Poisson with geometric batches: dispersion = 2 * mean - 1.
    assert dispersion(times, 200.0) == pytest.approx(2 * batch_mean - 1, rel=0.15)
    assert len(np.unique(times)) < 0.7 * len(times)  # patients share arrival instants


def test_horizon_mode_stops_arrivals_and_then_drains() -> None:
    spec = {"kind": "poisson", "horizon": 480, "rate": 0.05, "pathway_mix": {"q": 1.0}}
    counts = []
    for seed in range(200):
        times = arrival_times(spec, seed)
        assert len(times) == 0 or times.max() <= 480
        counts.append(len(times))
    assert np.mean(counts) == pytest.approx(0.05 * 480, rel=0.05)
    assert np.var(counts) / np.mean(counts) == pytest.approx(1.0, abs=0.25)  # Poisson count

    instance = generate_instance(scenario(arrivals=spec), 1)
    sim = run(Simulator(instance, debug=True))
    assert sim.done and compute_metrics(sim).makespan >= max(
        p.arrival_time for p in instance.patients
    )


def test_horizon_and_count_together_apply_whichever_comes_first() -> None:
    capped = arrival_times({"kind": "poisson", "horizon": 10_000, "n_patients": 5, "rate": 0.05})
    assert len(capped) == 5
    cut = arrival_times({"kind": "poisson", "horizon": 100, "n_patients": 10_000, "rate": 0.05})
    assert 0 < len(cut) < 30 and cut.max() <= 100


# --- patient classes ------------------------------------------------------------

ELECTIVE = {
    "name": "elective",
    "kind": "poisson",
    "n_patients": 30,
    "rate": 0.1,
    "pathway_mix": {"p": 1.0},
}
URGENT = {
    "name": "urgent",
    "kind": "poisson",
    "n_patients": 5,
    "rate": 0.02,
    "weight": 5.0,
    "urgent": True,
    "pathway_mix": {"q": 1.0},
}


def test_classes_are_merged_in_arrival_order() -> None:
    instance = generate_instance(scenario(arrivals=[ELECTIVE, URGENT]), 0)
    times = [p.arrival_time for p in instance.patients]
    assert times == sorted(times) and len(times) == 35
    assert [p.id for p in instance.patients] == list(range(35))
    assert [op.id for op in instance.operations] == list(range(instance.n_operations))
    urgent = [p for p in instance.patients if p.urgent]
    assert len(urgent) == 5
    assert all(p.patient_class == "urgent" and p.weight == 5.0 and p.pathway == "q" for p in urgent)
    assert all(p.weight == 1.0 and p.pathway == "p" for p in instance.patients if not p.urgent)


def test_adding_a_class_does_not_change_the_existing_patients() -> None:
    """Common random numbers across scenario variants: the elective patients
    keep their arrival times, pathways and duration luck."""
    alone = generate_instance(scenario(arrivals=[ELECTIVE], complexity={"rho": 0.4}), 9)
    both = generate_instance(scenario(arrivals=[ELECTIVE, URGENT], complexity={"rho": 0.4}), 9)

    def fingerprint(instance: ProblemInstance, cls: str) -> list[Any]:
        q = instance.hidden.duration_quantiles
        return [
            (p.arrival_time, p.pathway, tuple(q[op.id] for op in p.operations))
            for p in instance.patients
            if p.patient_class == cls
        ]

    assert fingerprint(alone, "elective") == fingerprint(both, "elective")


def test_weighted_flow_time() -> None:
    sc = make_scenario(
        resources={"Nurse": 1},
        op_types={"T": ({"Nurse": 1}, 10, True)},
        pathways={"p": [("t", "T", [])]},
        arrivals=[
            {"name": "routine", "kind": "static", "n_patients": 1},
            {"name": "urgent", "kind": "static", "n_patients": 1, "weight": 3.0, "urgent": True},
        ],
    )
    sim = run(Simulator(generate_instance(sc, 0), debug=True))  # FIFO: routine first
    metrics = compute_metrics(sim)
    assert metrics.flow_times == (10, 20) and metrics.weights == (1.0, 3.0)
    assert metrics.mean_flow_time == 15
    assert metrics.weighted_mean_flow_time == pytest.approx((10 + 3 * 20) / 4)
    assert sim.weight == [1.0, 3.0]


# --- config validation of the new fields ----------------------------------------


@pytest.mark.parametrize(
    "arrivals",
    [
        {"kind": "poisson", "rate": 0.1},  # neither count nor horizon
        {"kind": "poisson", "n_patients": 5},  # no rate
        {
            "kind": "poisson",
            "n_patients": 5,
            "rate": 0.1,
            "rate_profile": {"period": 10, "rates": [1]},
        },
        {"kind": "static", "n_patients": 5, "horizon": 100},
        {"kind": "static"},
        {"kind": "poisson", "n_patients": 5, "rate": 0.1, "batch_mean": 0.5},
        {"kind": "static", "n_patients": 5, "weight": 0},
    ],
)
def test_bad_arrival_configs_are_rejected(
    base_config: dict[str, Any], arrivals: dict[str, Any]
) -> None:
    base_config["arrivals"] = arrivals
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(base_config)


def test_semantic_errors_in_new_fields(base_config: dict[str, Any]) -> None:
    base_config["arrivals"] = [
        {"name": "a", "kind": "static", "n_patients": 1},
        {
            "name": "a",
            "kind": "poisson",
            "n_patients": 1,
            "rate_profile": {"period": 10, "rates": [0, 0]},
        },
    ]
    base_config["operation_types"][0]["speed_roles"] = ["Doctor"]
    base_config["operation_types"][0]["duration"] = {
        "distribution": "deterministic",
        "mean": 5,
        "shift": 2,
    }
    base_config["operation_types"][1]["duration"] = {"mean": 20, "cv": 0.5, "shift": 25}
    base_config["resources"][0]["attributes"] = {"speed": 0}
    base_config["complexity"] = {"acuity_classes": 1}
    with pytest.raises(ScenarioError) as exc:
        build(base_config)
    message = "\n".join(exc.value.errors)
    for expected in (
        "duplicate arrival stream 'a'",
        "rate_profile needs non-negative rates, one positive",
        "speed role 'Doctor' is not one of its required roles",
        "shift is only valid for lognormal and gamma",
        "shift must be in (0, mean)",
        "speed must be positive",
        "acuity_classes must be 0 (off) or >= 2",
    ):
        assert expected in message, expected


def test_dynamic_example_loads_and_runs() -> None:
    from pathlib import Path

    from hospital_sim.generators import load_config

    path = Path(__file__).parent.parent / "configs" / "example_hospital_dynamic.yaml"
    sc, warnings = load_config(path).to_scenario()
    assert warnings == []
    for seed in range(5):
        instance = generate_instance(sc, seed)
        assert all(p.acuity_class in (0, 1, 2) for p in instance.patients)
        assert run(Simulator(instance, debug=True)).done
