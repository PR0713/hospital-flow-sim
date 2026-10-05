"""Performance log. These tests never fail on speed.

Each run measures decisions per second for 10, 50 and 200 patients, prints
the figures in the pytest summary and appends them to benchmarks/history.csv,
so a regression shows up as a drop between rows.
"""

from __future__ import annotations

import csv
import datetime as dt
import platform
from pathlib import Path

import pytest

from hospital_sim import __version__
from hospital_sim.domain import Scenario
from hospital_sim.eval.benchmark import decisions_per_second

pytestmark = pytest.mark.slow

HISTORY = Path(__file__).parent.parent / "benchmarks" / "history.csv"
FIELDS = ["timestamp", "version", "python", "machine", "n_patients", "decisions_per_second"]

RESULTS: list[tuple[int, float]] = []


@pytest.mark.parametrize(("n_patients", "episodes"), [(10, 200), (50, 60), (200, 15)])
def test_log_decisions_per_second(
    example_scenario: Scenario, n_patients: int, episodes: int
) -> None:
    rate = decisions_per_second(example_scenario, n_patients, episodes)
    RESULTS.append((n_patients, rate))

    new_file = not HISTORY.exists()
    with HISTORY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "version": __version__,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "n_patients": n_patients,
                "decisions_per_second": round(rate),
            }
        )
    assert rate > 0  # sanity only; speed is logged, not enforced


def test_log_env_steps_per_second(example_scenario: Scenario) -> None:
    """Full env steps (action, engine, observation, mask) under a random masked policy."""
    import time

    import numpy as np

    from hospital_sim.env import EnvConfig, HospitalEnv

    env = HospitalEnv(EnvConfig(example_scenario))
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    steps, started = 0, time.perf_counter()
    while steps < 4000:
        *_, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        steps += 1
        if terminated or truncated:
            env.reset()
    rate = steps / (time.perf_counter() - started)
    RESULTS.append((-10, rate))
    with HISTORY.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "version": __version__,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "n_patients": "env-10",
                "decisions_per_second": round(rate),
            }
        )
    assert rate > 0
