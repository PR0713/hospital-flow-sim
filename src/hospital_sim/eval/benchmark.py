"""Throughput measurement of the core engine."""

from __future__ import annotations

import dataclasses
import time

from hospital_sim.domain.model import Scenario
from hospital_sim.generators.instance import generate_instance
from hospital_sim.sim.runner import run
from hospital_sim.sim.simulator import Simulator


def decisions_per_second(scenario: Scenario, n_patients: int, episodes: int) -> float:
    """Operation starts per second under FIFO dispatch, over ``episodes``
    fixed-size episodes of the scenario's first arrival stream. Instance
    generation and simulator construction are not timed."""
    arrivals = dataclasses.replace(scenario.arrivals[0], n_patients=n_patients, horizon=None)
    sims = [
        Simulator(generate_instance(scenario, seed, arrivals=arrivals)) for seed in range(episodes)
    ]
    started = time.perf_counter()
    for sim in sims:
        run(sim)
    elapsed = time.perf_counter() - started
    return sum(sim.n_ops for sim in sims) / elapsed
