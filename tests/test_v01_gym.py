"""v0.1, task 4: gymnasium compatibility with the default configuration.

Before v0.1 ``check_env`` failed on the default ``EnvConfig`` (it samples
unmasked random actions, and masked actions raise), every float Box was
``(-inf, inf)``, and there was no registered environment id.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import hospital_sim  # noqa: F401  (registers SchedSim-v0)
from hospital_sim.env import EnvConfig, HospitalEnv, InvalidActionError
from hospital_sim.generators import load_scenario
from hospital_sim.sim import OpState

CONFIGS = Path(__file__).parent.parent / "configs"
NAMES = ("example_hospital", "example_hospital_dynamic", "example_hospital_full")


def scenario(name: str) -> Any:
    return load_scenario(CONFIGS / f"{name}.yaml")


def _check(env: Any) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env)
    return [str(w.message) for w in caught]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("observation", ["dict", "graph"])
def test_check_env_passes_with_the_default_config(name: str, observation: str) -> None:
    messages = _check(HospitalEnv(EnvConfig(scenario(name), observation=observation)))
    # Lower bounds are all finite now; only unbounded times remain open above.
    assert not any("minimum value is -infinity" in m for m in messages)


def test_gym_make_passes_check_env() -> None:
    env = gym.make("SchedSim-v0")
    assert isinstance(env.unwrapped, HospitalEnv)
    _check(env)
    env.close()


def test_gym_make_takes_a_scenario_and_config_fields() -> None:
    env = gym.make("SchedSim-v0", scenario="example_hospital", observation="graph")
    obs, info = env.reset(seed=0)
    assert "precedence_edges" in obs and info["episode_index"] == 0
    with pytest.raises(FileNotFoundError, match="no scenario named"):
        gym.make("SchedSim-v0", scenario="no_such_scenario")


def test_action_space_samples_only_valid_actions() -> None:
    env = HospitalEnv(EnvConfig(scenario("example_hospital_dynamic")))
    env.reset(seed=0)
    for _ in range(50):
        mask = env.action_masks()
        assert all(mask[env.action_space.sample()] for _ in range(20))
        *_, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()


def test_other_masked_actions_still_raise() -> None:
    env = HospitalEnv(EnvConfig(scenario("example_hospital_dynamic")))
    env.reset(seed=0)
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(InvalidActionError):
        env.step(invalid)


def test_a_sample_drawn_before_a_reset_runs_the_lowest_valid_action() -> None:
    """What check_env does: sample, reset, then step with the old sample."""
    env = HospitalEnv(EnvConfig(scenario("example_hospital_dynamic")))
    env.reset(seed=0)
    for seed in range(1, 200):
        env.step(env.action_space.sample())
        stale = int(env.action_space.sample())
        env.reset(seed=seed)
        if not env.action_masks()[stale]:
            lowest = int(np.flatnonzero(env.action_masks())[0])
            assert lowest != env.wait_action  # wait is the last index
            sim = env.sim
            assert sim is not None
            expected = sim.visible_ops[lowest]
            *_, info = env.step(stale)
            assert info["invalid_action"] is True
            assert sim.state[expected] != OpState.READY  # the lowest valid op was started
            return
    raise AssertionError("no stale sample was ever invalid")


def test_spaces_pickle_and_async_vector_env_works() -> None:
    env = HospitalEnv(EnvConfig(scenario("example_hospital_dynamic")))
    env.reset(seed=0)
    space = pickle.loads(pickle.dumps(env.action_space))
    assert space == env.action_space and space.valid.any()
    assert pickle.loads(pickle.dumps(env.observation_space)) == env.observation_space

    # Worker processes start fresh: the "module:id" form imports hospital_sim
    # (and so registers the id) inside each worker.
    venv = gym.vector.AsyncVectorEnv(
        [lambda: gym.make("hospital_sim:SchedSim-v0") for _ in range(2)]
    )
    try:
        obs, _ = venv.reset(seed=0)
        assert obs["ops"].shape[0] == 2
        for _ in range(20):
            masks = np.stack(venv.call("action_masks"))
            actions = np.array([int(np.flatnonzero(m)[0]) for m in masks])
            obs, *_ = venv.step(actions)
        assert venv.single_observation_space.contains(
            {key: value[0] for key, value in obs.items()}
        )
    finally:
        venv.close()


@pytest.mark.parametrize(
    "name",
    [
        "example_hospital",
        "example_hospital_dynamic",
        "example_hospital_full",
        "example_hospital_patience",
        "example_hospital_stress",
    ],
)
def test_observations_stay_inside_the_bounded_space(name: str) -> None:
    env = HospitalEnv(EnvConfig(scenario(name), observation="graph"))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(3):
        while True:
            for key, space in env.observation_space.spaces.items():
                assert space.contains(obs[key]), key
            obs, *_, terminated, truncated, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            if terminated or truncated:
                obs, _ = env.reset()
                break
