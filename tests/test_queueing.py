"""The engine against closed-form queueing results.

One operation per patient, exponential service (gamma with cv = 1), Poisson
arrivals and FIFO dispatch make the simulator an M/M/c queue. Agreement with
the formulas checks the event loop, arrivals, sampling and the waiting-time
book-keeping together.

A single run of 40,000 patients estimates the mean wait to within roughly
3-6% (one standard deviation, measured over 12 seeds), and starting from an
empty system biases it slightly low. Each test therefore averages a few
fixed seeds, which keeps it deterministic and the tolerance honest.
"""

from __future__ import annotations

import math

import pytest

from hospital_sim.generators import generate_instance
from hospital_sim.sim import Simulator, compute_metrics, run

from .conftest import make_scenario

pytestmark = pytest.mark.slow

N_PATIENTS = 40_000
SEEDS = (0, 1, 2, 3)


def mean_wait_once(servers: int, arrival_rate: float, service_rate: float, seed: int) -> float:
    scenario = make_scenario(
        resources={"Server": servers},
        op_types={
            "Service": (
                {"Server": 1},
                {"distribution": "gamma", "mean": 1 / service_rate, "cv": 1.0},
                True,
            )
        },
        pathways={"p": [("s", "Service", [])]},
        arrivals={"kind": "poisson", "n_patients": N_PATIENTS, "rate": arrival_rate},
    )
    sim = run(Simulator(generate_instance(scenario, seed)))
    return compute_metrics(sim).mean_op_wait


def mean_wait(servers: int, arrival_rate: float, service_rate: float) -> float:
    return math.fsum(mean_wait_once(servers, arrival_rate, service_rate, s) for s in SEEDS) / len(
        SEEDS
    )


def erlang_c_wait(servers: int, arrival_rate: float, service_rate: float) -> float:
    """Mean time in queue of an M/M/c system."""
    a = arrival_rate / service_rate
    rho = a / servers
    tail = a**servers / (math.factorial(servers) * (1 - rho))
    p_wait = tail / (sum(a**k / math.factorial(k) for k in range(servers)) + tail)
    return p_wait / (servers * service_rate - arrival_rate)


@pytest.mark.parametrize("rho", [0.5, 0.7])
def test_mm1_mean_wait(rho: float) -> None:
    expected = rho / (1.0 - rho)  # Wq = rho / (mu - lambda) with mu = 1
    assert erlang_c_wait(1, rho, 1.0) == pytest.approx(expected)
    assert mean_wait(1, rho, 1.0) == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize(("servers", "arrival_rate"), [(2, 1.4), (3, 2.4)])
def test_mmc_mean_wait(servers: int, arrival_rate: float) -> None:
    expected = erlang_c_wait(servers, arrival_rate, 1.0)
    assert mean_wait(servers, arrival_rate, 1.0) == pytest.approx(expected, rel=0.06)


def test_utilization_matches_offered_load() -> None:
    scenario = make_scenario(
        resources={"Server": 3},
        op_types={
            "Service": ({"Server": 1}, {"distribution": "gamma", "mean": 1.0, "cv": 1.0}, True)
        },
        pathways={"p": [("s", "Service", [])]},
        arrivals={"kind": "poisson", "n_patients": N_PATIENTS, "rate": 2.4},
    )
    sim = run(Simulator(generate_instance(scenario, 3)))
    assert compute_metrics(sim).utilization["Server"] == pytest.approx(0.8, rel=0.02)
