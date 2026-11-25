"""v0.1, task 1: a truncated episode is scored with the cost it has accrued.

Before v0.1 the harness and ``info["episode"]`` counted finished patients
only, so an agent that waited until truncation scored a perfect zero.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hospital_sim.baselines import Policy
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.eval import EnvAgent, evaluate
from hospital_sim.eval.harness import record_episode
from hospital_sim.generators import load_scenario

CONFIGS = Path(__file__).parent.parent / "configs"

FLOW_METRICS = ("mean_flow_time", "weighted_mean_flow_time", "total_weighted_flow_time")


def _always_wait(observation: dict[str, Any], mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[-1])  # wait is the last action whenever it is valid


@pytest.fixture(scope="module")
def dynamic() -> Any:
    return load_scenario(CONFIGS / "example_hospital_dynamic.yaml")


def test_always_wait_scores_worse_than_weight_aware_on_every_flow_metric(dynamic: Any) -> None:
    agent = EnvAgent("always_wait", _always_wait, EnvConfig(dynamic, split="eval"))
    result = evaluate(dynamic, [Policy("weight_aware"), agent], 6)
    good = result.summary
    for metric in FLOW_METRICS:
        assert good(metric)["always_wait"][0] > good(metric)["weight_aware+flexible"][0], (
            metric,
            good(metric),
        )
    assert result.truncation_rate("always_wait") == 1.0
    assert result.truncation_rate("weight_aware+flexible") == 0.0
    assert result.summary("unfinished")["always_wait"][0] > 0
    assert "truncated" in result.table().splitlines()[0]


def test_episode_info_counts_unfinished_patients(dynamic: Any) -> None:
    env = HospitalEnv(EnvConfig(dynamic, reward_scale=1.0))
    env.reset(seed=0)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(_always_wait({}, env.action_masks()))
        total += reward
        if terminated or truncated:
            break
    episode = info["episode"]
    assert truncated and episode["truncated"] is True
    assert episode["unfinished_patients"] > 0
    assert episode["unfinished_weighted_flow_time"] > 0
    # The reported cost is the cost the rewards charged (no leavers here).
    assert episode["total_weighted_flow_time"] == pytest.approx(-total, rel=1e-9)
    assert episode["makespan"] == pytest.approx(episode["time"])


def test_terminated_episodes_are_scored_exactly_as_before(dynamic: Any) -> None:
    from hospital_sim.sim import episode_outcome

    sim = Policy("weight_aware").run(env_instance(dynamic))
    outcome = episode_outcome(sim)
    record = record_episode(sim)
    assert outcome.unfinished == 0 and record.values["truncated"] == 0.0
    flows = [sim.completion_time[p.id] - p.arrival_time for p in sim.instance.patients]
    assert record.values["mean_flow_time"] == math.fsum(flows) / len(flows)


def env_instance(scenario: Any) -> Any:
    from hospital_sim.generators import generate_instance

    return generate_instance(scenario, 11)
