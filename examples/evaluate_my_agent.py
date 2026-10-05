"""Template: plug in your own agent and compare it with the baselines.

1. Put your policy in ``MyAgent.act``. It receives the observation dict and
   the boolean action mask, and must return the index of a valid action.
2. Train it however you like against ``make_env("train")``.
3. Run this file. It evaluates on the ``test`` split, on exactly the same
   instances as the baselines, and prints paired differences.

    .venv/bin/python examples/evaluate_my_agent.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hospital_sim.baselines import Policy
from hospital_sim.env import OBSERVATION_SCHEMA_VERSION, EnvConfig, HospitalEnv
from hospital_sim.env.observation import OP_FEATURES
from hospital_sim.eval import EnvAgent, Evaluation, evaluate
from hospital_sim.generators import load_scenario

CONFIG = Path(__file__).parent.parent / "configs" / "example_hospital_dynamic.yaml"
SCENARIO = load_scenario(CONFIG)

# The observation layout this agent was written for; fail loudly if it changes.
assert OBSERVATION_SCHEMA_VERSION == 1


def env_config(split: str) -> EnvConfig:
    """One place for every environment setting, so training and evaluation match."""
    return EnvConfig(
        SCENARIO,
        split=split,  # "train" to learn, "eval" to tune, "test" to report
        observation="graph",  # adds precedence edges; see env.graph() for the unpadded form
        allocation_rules=("flexible",),  # add more rules to let the agent choose the team
        reward="weighted_flow_time",
    )


def make_env(split: str) -> HospitalEnv:
    return HospitalEnv(env_config(split))


class MyAgent:
    """Replace the body of ``act`` with your model.

    This placeholder starts the valid operation that has waited longest, and
    never waits. It is here only so that the file runs end to end.
    """

    def act(self, observation: dict[str, Any], mask: np.ndarray) -> int:
        n_ops = observation["ops"].shape[0]
        startable = np.flatnonzero(mask[:n_ops])  # with one allocation rule, action = row
        waited = observation["ops"][startable, OP_FEATURES.index("waited")]
        return int(startable[np.argmax(waited)])

    def train(self, steps: int = 0) -> None:
        """Your training loop goes here; it should only ever touch the train split."""
        env = make_env("train")
        env.reset(seed=0)
        # ... collect experience with env.step(...) and env.action_masks(), update the model ...


def main(n_episodes: int = 100) -> Evaluation:
    agent = MyAgent()
    agent.train()

    candidates = [
        Policy("fifo"),
        Policy("urgent_first"),  # the baselines worth beating on this config
        Policy("weight_aware"),
        Policy("acuity_aware"),
        EnvAgent("my_agent", agent.act, env_config("test")),
    ]
    result = evaluate(SCENARIO, candidates, n_episodes=n_episodes, split="test")

    columns = ("weighted_mean_flow_time", "mean_flow_time", "makespan", "left")
    print(result.table(columns))
    print()
    for baseline in ("urgent_first+flexible", "weight_aware+flexible"):
        difference, half_width = result.paired_difference(
            "weighted_mean_flow_time", "my_agent", baseline
        )
        verdict = "better" if difference + half_width < 0 else "not shown to be better"
        print(f"my_agent minus {baseline}: {difference:+.1f} ± {half_width:.1f}  ({verdict})")
    return result


if __name__ == "__main__":
    main()
