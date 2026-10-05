"""Throughput of the core engine under FIFO dispatch.

Run with: .venv/bin/python benchmarks/bench_core.py
The same measurement runs in the test suite (tests/test_performance.py),
which appends to benchmarks/history.csv.
"""

from __future__ import annotations

from hospital_sim.eval.benchmark import decisions_per_second
from hospital_sim.generators import load_scenario

if __name__ == "__main__":
    scenario = load_scenario("configs/example_hospital.yaml")
    for n_patients, episodes in ((10, 500), (50, 200), (200, 50), (1000, 5)):
        rate = decisions_per_second(scenario, n_patients, episodes)
        print(f"{n_patients:>5} patients x {episodes:>4} episodes: {rate:>9,.0f} decisions/s")
