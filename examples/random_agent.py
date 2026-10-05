"""The smallest possible agent loop: random valid actions.

Run from the repository root:

    .venv/bin/python examples/random_agent.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import load_scenario

CONFIG = Path(__file__).parent.parent / "configs" / "example_hospital_dynamic.yaml"


def main(episodes: int = 5, seed: int = 0) -> list[float]:
    env = HospitalEnv(EnvConfig(load_scenario(CONFIG), split="train", observation="graph"))
    rng = np.random.default_rng(seed)
    returns = []
    observation, info = env.reset(seed=seed)
    for episode in range(episodes):
        total, done = 0.0, False
        while not done:
            mask = env.action_masks()  # True where an action is valid right now
            action = int(rng.choice(np.flatnonzero(mask)))
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        summary = info["episode"]
        print(
            f"episode {episode}: return {total:8.1f}, "
            f"{summary['patients_finished']} patients, makespan {summary['makespan']:.0f} min"
        )
        returns.append(total)
        observation, info = env.reset()  # next episode, a new instance
    return returns


if __name__ == "__main__":
    main()
