"""M2b: heavy tails, team speed, and distributional checks on durations."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest
from scipy import stats

from hospital_sim.domain import DurationKind, DurationSpec
from hospital_sim.generators import generate_instance
from hospital_sim.sim import Simulator, StartContext, run
from hospital_sim.sim.durations import (
    Complicated,
    Distribution,
    Scaled,
    StandardDurationModel,
    make_distribution,
)

from .conftest import make_scenario

GRID = (np.arange(200_000) + 0.5) / 200_000

SHIFTED = DurationSpec(DurationKind.LOGNORMAL, mean=80, cv=0.4, shift=30)
COMPLICATED = DurationSpec(
    DurationKind.LOGNORMAL,
    mean=60,
    cv=0.4,
    complication_probability=0.1,
    complication_multiplier=3.0,
)
BOTH = DurationSpec(
    DurationKind.GAMMA,
    mean=50,
    cv=0.5,
    shift=10,
    complication_probability=0.05,
    complication_multiplier=2.5,
)
WRAPPED: dict[str, Distribution] = {
    "shifted": make_distribution(SHIFTED),
    "complicated": make_distribution(COMPLICATED),
    "shifted_complicated": make_distribution(BOTH),
    "scaled": Scaled(make_distribution(COMPLICATED), 0.8),
}


def draws(dist: Distribution) -> np.ndarray:
    return np.array([dist.ppf(float(u)) for u in GRID])


def test_shifted_lognormal_keeps_configured_mean_and_cv() -> None:
    x = draws(WRAPPED["shifted"])
    assert x.min() > 30
    assert x.mean() == pytest.approx(80, rel=2e-3)
    assert x.std() / x.mean() == pytest.approx(0.4, rel=2e-2)


def test_complication_mixture_mean_and_share() -> None:
    dist = WRAPPED["complicated"]
    assert isinstance(dist, Complicated)
    assert dist.mean == pytest.approx(60 * (0.9 + 0.1 * 3.0))
    assert draws(dist).mean() == pytest.approx(dist.mean, rel=2e-3)
    # The top 10% of quantiles are the complicated cases.
    assert dist.ppf(0.95) == pytest.approx(3.0 * dist.base.ppf(0.5))


def test_complication_adds_a_second_mode_of_long_cases() -> None:
    mixture = WRAPPED["complicated"]
    assert isinstance(mixture, Complicated)
    # Against the routine case alone, long cases become far more common.
    assert mixture.sf(4 * 60) > 100 * mixture.base.sf(4 * 60)

    # Against a single lognormal with the same mean and CV, the shape differs:
    # most cases are shorter, and a distinct cluster of long ones remains.
    x = draws(mixture)
    matched = make_distribution(
        DurationSpec(DurationKind.LOGNORMAL, mean=float(x.mean()), cv=float(x.std() / x.mean()))
    )
    assert mixture.ppf(0.5) < matched.ppf(0.5)
    assert mixture.sf(4 * 60) > 1.3 * matched.sf(4 * 60)


@pytest.mark.parametrize("name", list(WRAPPED))
def test_wrapped_distributions_are_self_consistent(name: str) -> None:
    dist = WRAPPED[name]
    x = draws(dist)
    assert x.mean() == pytest.approx(dist.mean, rel=2e-3)
    for t in (0.5 * dist.mean, dist.mean, 2 * dist.mean, 4 * dist.mean):
        assert dist.sf(t) == pytest.approx((x > t).mean(), abs=2e-4)
        longer = x[x > t]
        if len(longer) > 500:
            assert dist.expected_remaining(t) == pytest.approx(longer.mean() - t, rel=2e-2)
    assert dist.expected_remaining(0.0) == pytest.approx(dist.mean)
    assert dist.expected_remaining(1e9) >= 0


def test_overrunning_a_complicated_case_raises_the_estimate() -> None:
    # Once a case has run past what a routine case would take, it is almost
    # certainly a complication, and the expected remaining time jumps.
    dist = WRAPPED["complicated"]
    assert dist.expected_remaining(150) > dist.expected_remaining(60) > 0


# --- distributional checks on durations realized by the simulator ---------------

N = 4000
TYPE_SPECS: dict[str, dict[str, Any]] = {
    "Logn": {"mean": 30, "cv": 0.6},
    "Gam": {"distribution": "gamma", "mean": 20, "cv": 0.4},
    "Heavy": {
        "mean": 40,
        "cv": 0.4,
        "shift": 10,
        "complication": {"probability": 0.1, "multiplier": 3.0},
    },
}


@functools.cache
def realized(rho: float, seed: int = 0) -> tuple[Simulator, dict[str, np.ndarray]]:
    scenario = make_scenario(
        resources={"Staff": 60},
        op_types={name: ({"Staff": 1}, spec, False) for name, spec in TYPE_SPECS.items()},
        pathways={"p": [(name.lower(), name, []) for name in TYPE_SPECS]},
        # Spread arrivals out: a static batch of this size is slow to simulate.
        arrivals={"kind": "poisson", "n_patients": N, "rate": 0.4},
        complexity={"rho": rho},
    )
    sim = run(Simulator(generate_instance(scenario, seed)))
    out = {}
    for i, t in enumerate(sim.op_types):
        ops = [o for o in range(sim.n_ops) if sim.type_of[o] == i]
        out[t.name] = np.array([sim.end_time[o] - sim.start_time[o] for o in ops])
    return sim, out


@pytest.mark.parametrize("rho", [0.0, 0.6])
@pytest.mark.parametrize("name", list(TYPE_SPECS))
def test_realized_durations_follow_the_configured_distribution(rho: float, name: str) -> None:
    """Kolmogorov-Smirnov against the target CDF. With rho > 0 this also
    shows that the copula leaves every marginal distribution unchanged."""
    sim, samples = realized(rho)
    type_index = [t.name for t in sim.op_types].index(name)
    dist = sim.duration_model.base(type_index)
    result = stats.kstest(samples[name], lambda x: 1.0 - np.vectorize(dist.sf)(x))
    assert result.pvalue > 0.01, result
    assert samples[name].mean() == pytest.approx(dist.mean, rel=0.05)


def test_realized_tail_frequencies_match() -> None:
    sim, samples = realized(0.0, seed=1)
    for i, t in enumerate(sim.op_types):
        dist = sim.duration_model.base(i)
        for q in (0.9, 0.99):
            threshold = dist.ppf(q) if t.name != "Heavy" else float(np.quantile(samples[t.name], q))
            expected = dist.sf(threshold)
            observed = float((samples[t.name] > threshold).mean())
            sd = (expected * (1 - expected) / N) ** 0.5
            assert abs(observed - expected) < 4 * sd + 1 / N, (t.name, q)


def test_a_wrong_distribution_is_rejected_by_the_same_check() -> None:
    """The KS check has teeth: gamma durations do not pass as lognormal."""
    _, samples = realized(0.0)
    wrong = make_distribution(DurationSpec(DurationKind.LOGNORMAL, mean=20, cv=0.4))
    result = stats.kstest(samples["Gam"], lambda x: 1.0 - np.vectorize(wrong.sf)(x))
    assert result.pvalue < 1e-4


# --- team speed and the resource-state-aware interface --------------------------


def speed_scenario() -> Any:
    return make_scenario(
        resources={"Nurse": 1},
        op_types={
            "Surgery": ({"Lead": 1, "Assistant": 1, "Nurse": 1}, 60, True),
            "Check": ({"Lead": 1}, 60, True),
        },
        pathways={"s": [("surgery", "Surgery", [])], "c": [("check", "Check", [])]},
        arrivals={"kind": "static", "n_patients": 1, "pathway_mix": {"s": 1.0}},
        extra_resources=[
            {"id": "fast", "roles": ["Lead", "Assistant"], "attributes": {"speed": 2.0}},
            {"id": "slow", "roles": ["Lead", "Assistant"], "attributes": {"speed": 0.5}},
        ],
        speed_roles={"Surgery": ["Lead"]},
    )


@pytest.mark.parametrize(
    ("lead", "assistant", "duration"), [("fast", "slow", 30), ("slow", "fast", 120)]
)
def test_duration_scales_with_the_speed_of_the_governing_role(
    lead: str, assistant: str, duration: float
) -> None:
    sim = Simulator(generate_instance(speed_scenario(), 0), debug=True)
    assert sim.run_until_decision()
    ids = sim.resources.resource_ids
    # Requirement order follows the scenario: Lead, Assistant, Nurse.
    team = ((ids.index(lead),), (ids.index(assistant),), (ids.index("nurse_1"),))
    assert sim.expected_duration[0] == 60  # before a team is known
    sim.start(0, team)
    assert sim.expected_remaining(0) == duration
    run(sim)
    assert sim.end_time[0] == duration


def test_team_speed_is_a_geometric_mean_and_optional() -> None:
    scenario = make_scenario(
        resources={},
        op_types={"Pair": ({"Doc": 2}, 60, True), "Solo": ({"Doc": 1}, 60, True)},
        pathways={"p": [("a", "Pair", []), ("b", "Solo", [])]},
        extra_resources=[
            {"id": "fast", "roles": ["Doc"], "attributes": {"speed": 4.0}},
            {"id": "normal", "roles": ["Doc"]},
        ],
        speed_roles={"Pair": ["Doc"]},
    )
    model = StandardDurationModel(scenario)
    assert model.team_speed(0, ((0, 1),)) == pytest.approx(2.0)
    assert model.team_speed(1, ((0,),)) == 1.0  # Solo has no speed roles
    sim = run(Simulator(generate_instance(scenario, 0), debug=True))
    durations = {
        o.step_id: sim.end_time[o.id] - sim.start_time[o.id] for o in sim.instance.operations
    }
    assert durations == {"a": 30, "b": 60}


def test_custom_model_can_read_resource_state() -> None:
    """The interface a fatigue model will use: slower with accumulated work."""

    class Tiring(StandardDurationModel):
        def at_start(self, ctx: StartContext) -> Distribution:
            worked = sum(ctx.resources.work_time[r] for group in ctx.team for r in group)
            return Scaled(self.base(ctx.op_type), 1.0 + worked / 100.0)

    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types={"Task": ({"Nurse": 1}, 50, True)},
        pathways={"p": [("t", "Task", [])]},
        arrivals={"kind": "static", "n_patients": 3},
    )
    sim = run(
        Simulator(generate_instance(scenario, 0), duration_model=Tiring(scenario), debug=True)
    )
    durations = [sim.end_time[o] - sim.start_time[o] for o in range(3)]
    assert durations == pytest.approx([50, 75, 112.5])
    assert sim.resources.work_time == pytest.approx([237.5])
    assert sim.resources.jobs_done == [3]


# --- log-logistic: the genuinely heavy-tailed option ----------------------------

HEAVY_SPEC = {"distribution": "loglogistic", "mean": 40, "tail_index": 2.5}


@functools.cache
def loglogistic_run() -> tuple[Distribution, np.ndarray]:
    scenario = make_scenario(
        resources={"Staff": 60},
        op_types={"H": ({"Staff": 1}, HEAVY_SPEC, False)},
        pathways={"p": [("h", "H", [])]},
        arrivals={"kind": "poisson", "n_patients": 20_000, "rate": 0.5},
    )
    sim = run(Simulator(generate_instance(scenario, 0)))
    return sim.duration_model.base(0), np.array(sim.end_time) - np.array(sim.start_time)


def test_loglogistic_mean_and_ks() -> None:
    dist, x = loglogistic_run()
    assert dist.mean == pytest.approx(40)
    assert x.mean() == pytest.approx(40, rel=0.05)
    assert stats.kstest(x, lambda t: 1.0 - np.vectorize(dist.sf)(t)).pvalue > 0.01
    # Independent cross-check against scipy's own log-logistic (Fisk).
    assert stats.kstest(x, stats.fisk(c=2.5, scale=dist.ppf(0.5)).cdf).pvalue > 0.01


def test_loglogistic_tail_is_a_power_law() -> None:
    dist, x = loglogistic_run()
    # Observed exceedance frequencies deep in the tail match the model...
    for q in (0.99, 0.999):
        threshold = dist.ppf(q)
        expected = 1 - q
        sd = (expected * q / len(x)) ** 0.5
        assert abs((x > threshold).mean() - expected) < 4 * sd
    # ...doubling t divides the tail probability by about 2 ** tail_index...
    assert dist.sf(4000) / dist.sf(2000) == pytest.approx(2**-2.5, rel=0.01)
    # ...and a Hill estimate on the largest 2% of realized durations recovers the index.
    top = np.sort(x)[-400:]
    hill = 1.0 / np.mean(np.log(top[1:] / top[0]))
    assert hill == pytest.approx(2.5, rel=0.2)


def test_loglogistic_tail_dwarfs_lognormal_and_the_complication_mixture() -> None:
    dist, x = loglogistic_run()
    matched = make_distribution(
        DurationSpec(DurationKind.LOGNORMAL, mean=40, cv=float(np.sqrt(1 / 0.9 - 1)))
    )
    mixture = make_distribution(
        DurationSpec(
            DurationKind.LOGNORMAL,
            mean=30,
            cv=0.4,
            complication_probability=0.1,
            complication_multiplier=3.0,
        )
    )
    far = 20 * 40  # twenty times the mean
    assert dist.sf(far) > 1e-4
    assert dist.sf(far) > 100 * matched.sf(far)
    assert dist.sf(far) > 1e6 * mixture.sf(far)


def test_loglogistic_expected_remaining() -> None:
    dist, _ = loglogistic_run()
    grid = np.array([dist.ppf(float(u)) for u in (np.arange(2_000_000) + 0.5) / 2_000_000])
    for t in (0.0, 40.0, 200.0):
        longer = grid[grid > t]
        assert dist.expected_remaining(t) == pytest.approx(longer.mean() - t, rel=0.03)
    # Heavy tail: the longer a case has run, the longer it is expected to continue.
    assert dist.expected_remaining(400) > dist.expected_remaining(200) > dist.expected_remaining(40)


def test_loglogistic_config_validation() -> None:
    from pydantic import ValidationError

    from hospital_sim.generators.config import DurationConfig

    for bad in (
        {"distribution": "loglogistic", "mean": 40},
        {"distribution": "loglogistic", "mean": 40, "tail_index": 1.0},
        {"distribution": "loglogistic", "mean": 40, "tail_index": 2.5, "cv": 0.5},
        {"mean": 40, "cv": 0.5, "tail_index": 2.5},
    ):
        with pytest.raises(ValidationError):
            DurationConfig.model_validate(bad)
