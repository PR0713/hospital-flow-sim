"""Minimal calibration: fit duration and arrival parameters from an event log.

No real data was available when this was written. The functions are tested by
round trip only: simulate with known parameters, fit, recover them. Treat
them as a starting point, not as a validated fitting pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DurationFit:
    distribution: str  # "lognormal", "gamma" or "deterministic"
    mean: float
    cv: float
    n: int
    aic: float
    # Kolmogorov-Smirnov p-value against the fitted distribution. Optimistic,
    # because the parameters were estimated from the same data.
    ks_pvalue: float

    def to_config(self) -> dict[str, Any]:
        """A ``duration:`` block for a scenario file."""
        if self.distribution == "deterministic":
            return {"distribution": "deterministic", "mean": round(self.mean, 4)}
        return {
            "distribution": self.distribution,
            "mean": round(self.mean, 4),
            "cv": round(self.cv, 4),
        }


def _fit_lognormal(x: np.ndarray) -> DurationFit:
    logs = np.log(x)
    mu, sigma = float(logs.mean()), float(logs.std())
    dist = stats.lognorm(s=sigma, scale=math.exp(mu))
    return DurationFit(
        distribution="lognormal",
        mean=math.exp(mu + 0.5 * sigma**2),
        cv=math.sqrt(math.expm1(sigma**2)),
        n=len(x),
        aic=4.0 - 2.0 * float(dist.logpdf(x).sum()),
        ks_pvalue=float(stats.kstest(x, dist.cdf).pvalue),
    )


def _fit_gamma(x: np.ndarray) -> DurationFit:
    shape, _, scale = stats.gamma.fit(x, floc=0)
    dist = stats.gamma(a=shape, scale=scale)
    return DurationFit(
        distribution="gamma",
        mean=float(shape * scale),
        cv=1.0 / math.sqrt(shape),
        n=len(x),
        aic=4.0 - 2.0 * float(dist.logpdf(x).sum()),
        ks_pvalue=float(stats.kstest(x, dist.cdf).pvalue),
    )


def fit_durations(
    rows: Iterable[dict[str, Any]],
    *,
    families: Sequence[str] = ("lognormal", "gamma"),
    min_samples: int = 30,
) -> dict[str, DurationFit]:
    """Per operation type, the best-fitting family (lowest AIC) by maximum
    likelihood. Types with fewer than ``min_samples`` durations are skipped.

    Limits to keep in mind: the durations of an operation type are pooled, so
    differences between teams (speed) or patients are folded into the fitted
    spread; and only finished operations are in a log.
    """
    durations: dict[str, list[float]] = {}
    for row in rows:
        durations.setdefault(row["op_type"], []).append(row["end_time"] - row["start_time"])

    fits: dict[str, DurationFit] = {}
    for op_type, values in sorted(durations.items()):
        x = np.array(values, dtype=float)
        if len(x) < min_samples:
            continue
        if x.std() <= 1e-9 * max(x.mean(), 1.0):
            fits[op_type] = DurationFit("deterministic", float(x.mean()), 0.0, len(x), 0.0, 1.0)
            continue
        x = x[x > 0]
        candidates = []
        if "lognormal" in families:
            candidates.append(_fit_lognormal(x))
        if "gamma" in families:
            candidates.append(_fit_gamma(x))
        fits[op_type] = min(candidates, key=lambda f: f.aic)
    return fits


def fit_arrival_profile(
    rows: Iterable[dict[str, Any]],
    *,
    period: float,
    n_bins: int,
    patient_class: str | None = None,
) -> dict[str, Any]:
    """Arrival rate per time bin, as a ``rate_profile:`` block.

    Every episode in the log is taken to cover exactly one ``period`` starting
    at time zero (for example one day). The rate of a bin is the number of
    patients who arrived in it, divided by the bin width and the number of
    episodes. Episodes with no finished operation are not in a log, so very
    quiet periods are slightly over-estimated.
    """
    arrivals: dict[tuple[int, int], float] = {}
    episodes: set[int] = set()
    for row in rows:
        episodes.add(row["episode"])
        if patient_class is None or row["patient_class"] == patient_class:
            arrivals[(row["episode"], row["patient"])] = row["arrival_time"]
    if not episodes:
        raise ValueError("the event log is empty")
    times = np.array([t for t in arrivals.values() if 0 <= t < period])
    counts, _ = np.histogram(times, bins=n_bins, range=(0.0, period))
    width = period / n_bins
    return {
        "period": period,
        "rates": [round(float(c) / (width * len(episodes)), 6) for c in counts],
    }
