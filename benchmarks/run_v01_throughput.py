"""Throughput baseline per tier (single process). Run on an otherwise idle machine:

    caffeinate -ims .venv/bin/python benchmarks/run_v01_throughput.py

Measures, for each tier, over ``STEPS`` steps under a random masked policy
(``action_space.sample()``), resets included:

* base env: steps/s and start decisions/s;
* the same through ``PointerActionWrapper`` (default settings);
* the bare simulator (FIFO dispatch, first-idle teams, no observation),
  decisions/s including instance generation.

Writes ``benchmarks/results/v0.1/throughput.json`` and regenerates
``docs/BASELINE_RESULTS_v0.1.md``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

from hospital_sim.env import EnvConfig, HospitalEnv, PointerActionWrapper, instance_seed
from hospital_sim.eval import TIERS, load_tier
from hospital_sim.generators import generate_instance
from hospital_sim.sim import Simulator

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "v0.1"
STEPS = 10_000


def _env_rate(env: gym.Env[Any, Any], wait_index: int) -> tuple[float, float]:
    env.reset(seed=0)
    env.action_space.seed(0)
    starts = 0
    started = time.perf_counter()
    for _ in range(STEPS):
        action = int(env.action_space.sample())  # a valid action (MaskedDiscrete)
        starts += action != wait_index
        *_, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    seconds = time.perf_counter() - started
    return STEPS / seconds, starts / seconds


def _bare_rate(tier: str) -> float:
    scenario = load_tier(tier).scenario
    decisions, episode = 0, 0
    started = time.perf_counter()
    while decisions < STEPS:
        sim = Simulator(generate_instance(scenario, instance_seed(0, "train", episode, 0)))
        while sim.run_until_decision():
            sim.start(sim.startable_ops()[0])
            decisions += 1
        episode += 1
    return decisions / (time.perf_counter() - started)


def _machine() -> str:
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        brand = ""
    return f"{brand or platform.processor()} ({platform.system()} {platform.machine()})"


def main() -> None:
    rows = []
    for tier in TIERS:
        scenario = load_tier(tier).scenario
        base = HospitalEnv(EnvConfig(scenario, split="train"))
        assert base.wait_action is not None
        env_steps, env_decisions = _env_rate(base, base.wait_action)
        pointer = PointerActionWrapper(HospitalEnv(EnvConfig(scenario, split="train")))
        pointer_steps, pointer_decisions = _env_rate(pointer, pointer.wait_index)
        row = {
            "tier": tier,
            "env_steps": env_steps,
            "env_decisions": env_decisions,
            "pointer_steps": pointer_steps,
            "pointer_decisions": pointer_decisions,
            "bare_decisions": _bare_rate(tier),
        }
        rows.append(row)
        print({k: (round(v) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "throughput.json").write_text(
        json.dumps(
            {
                "machine": _machine(),
                "python": platform.python_version(),
                "steps": STEPS,
                "rows": rows,
            },
            indent=1,
        )
    )
    sys.path.insert(0, str(Path(__file__).parent))
    from v01_report import write_report

    write_report()


if __name__ == "__main__":
    main()
