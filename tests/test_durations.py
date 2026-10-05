from __future__ import annotations

import math

import numpy as np
import pytest

from hospital_sim.domain import DurationKind, DurationSpec
from hospital_sim.sim.durations import Distribution, make_distribution

GRID = (np.arange(200_000) + 0.5) / 200_000  # midpoint quantiles

SPECS = {
    "lognormal": DurationSpec(DurationKind.LOGNORMAL, mean=30, cv=0.5),
    "lognormal_wide": DurationSpec(DurationKind.LOGNORMAL, mean=90, cv=1.2),
    "gamma": DurationSpec(DurationKind.GAMMA, mean=60, cv=0.3),
    "exponential": DurationSpec(DurationKind.GAMMA, mean=10, cv=1.0),
    "truncated_normal": DurationSpec(
        DurationKind.TRUNCATED_NORMAL, mean=20, cv=0.5, low=5, high=40
    ),
    "empirical": DurationSpec(DurationKind.EMPIRICAL, mean=6, samples=(8, 4, 6, 4)),
    "deterministic": DurationSpec(DurationKind.DETERMINISTIC, mean=12),
}


def draws(dist: Distribution) -> np.ndarray:
    return np.array([dist.ppf(float(u)) for u in GRID])


@pytest.mark.parametrize("name", ["lognormal", "lognormal_wide", "gamma", "exponential"])
def test_mean_and_cv_match_the_config(name: str) -> None:
    spec = SPECS[name]
    x = draws(make_distribution(spec))
    assert make_distribution(spec).mean == pytest.approx(spec.mean)
    assert x.mean() == pytest.approx(spec.mean, rel=2e-3)
    assert x.std() / x.mean() == pytest.approx(spec.cv, rel=2e-2)


@pytest.mark.parametrize("name", list(SPECS))
def test_reported_mean_is_the_mean_of_the_draws(name: str) -> None:
    dist = make_distribution(SPECS[name])
    assert draws(dist).mean() == pytest.approx(dist.mean, rel=2e-3)


@pytest.mark.parametrize("name", list(SPECS))
def test_ppf_is_monotone_and_finite(name: str) -> None:
    dist = make_distribution(SPECS[name])
    x = [dist.ppf(u) for u in (0.0, 1e-9, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-9, 1.0)]
    assert all(math.isfinite(v) and v >= 0 for v in x)
    assert x == sorted(x)


def test_truncated_normal_respects_bounds() -> None:
    x = draws(make_distribution(SPECS["truncated_normal"]))
    assert x.min() >= 5 and x.max() <= 40


def test_empirical_resamples_the_given_values() -> None:
    x = draws(make_distribution(SPECS["empirical"]))
    values, counts = np.unique(x, return_counts=True)
    assert list(values) == [4, 6, 8]
    assert list(counts / len(x)) == pytest.approx([0.5, 0.25, 0.25])


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0, 2.0])
def test_expected_remaining_matches_the_conditional_mean(name: str, fraction: float) -> None:
    dist = make_distribution(SPECS[name])
    elapsed = fraction * dist.mean
    x = draws(dist)
    longer = x[x > elapsed]
    expected = longer.mean() - elapsed if len(longer) else 0.0
    assert dist.expected_remaining(elapsed) == pytest.approx(expected, rel=1e-2, abs=1e-2)


def test_exponential_is_memoryless() -> None:
    dist = make_distribution(SPECS["exponential"])
    for elapsed in (0.0, 3.0, 25.0, 80.0):
        assert dist.expected_remaining(elapsed) == pytest.approx(10.0)


def test_lognormal_remaining_time_grows_once_an_operation_overruns() -> None:
    # Heavy tail: the longer it has already taken, the longer it is expected to go on.
    dist = make_distribution(SPECS["lognormal_wide"])
    assert dist.expected_remaining(400) > dist.expected_remaining(200) > dist.expected_remaining(90)


def test_expected_remaining_is_safe_far_in_the_tail() -> None:
    for name in SPECS:
        value = make_distribution(SPECS[name]).expected_remaining(1e9)
        assert math.isfinite(value) and value >= 0
