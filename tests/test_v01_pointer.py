"""v0.1, task 5: the optional pointer-style action wrapper."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hospital_sim.env import EnvConfig, HospitalEnv, InvalidActionError
from hospital_sim.env.observation import OP_FEATURES
from hospital_sim.env.pointer import PointerActionWrapper
from hospital_sim.generators import load_scenario

CONFIGS = Path(__file__).parent.parent / "configs"
WAITED = OP_FEATURES.index("waited")


def scenario(name: str = "example_hospital_dynamic") -> Any:
    return load_scenario(CONFIGS / f"{name}.yaml")


def wrapped(**kwargs: Any) -> PointerActionWrapper:
    rules = kwargs.pop("rules", ("flexible",))
    name = kwargs.pop("name", "example_hospital_dynamic")
    return PointerActionWrapper(
        HospitalEnv(EnvConfig(scenario(name), allocation_rules=rules)), **kwargs
    )


def test_candidates_are_exactly_the_valid_base_actions() -> None:
    env = wrapped(rules=("flexible", "fastest"))
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    for _ in range(300):
        base_mask = info["base_action_mask"]
        mask = env.action_masks()
        n = int(mask[:-1].sum())
        listed = sorted(slot * 2 + rule for slot, rule in env.candidates())
        assert listed == sorted(np.flatnonzero(base_mask[:-1]).tolist())
        assert mask[-1] == base_mask[-1] and (obs["candidate_mask"] == mask).all()
        assert (obs["candidate_slot"][:n] >= 0).all() and (obs["candidate_slot"][n:] == -1).all()
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        if terminated or truncated:
            obs, info = env.reset()


def test_wrapper_trajectory_equals_the_base_env_with_mapped_actions() -> None:
    env = wrapped()
    rng = np.random.default_rng(1)
    _, _ = env.reset(seed=5)
    base_actions, rewards = [], []
    while True:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        if action == env.wait_index:
            base_actions.append(env.base.wait_action)
        else:
            slot, rule = env.candidates()[action]
            base_actions.append(slot * env.n_rules + rule)
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    base = HospitalEnv(EnvConfig(scenario()))
    base.reset(seed=5)
    replay = [base.step(a)[1] for a in base_actions]
    assert replay == rewards


def test_ordering_without_shuffle_is_oldest_ready_first() -> None:
    env = wrapped(shuffle_candidates=False)
    rng = np.random.default_rng(2)
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        n = int(env.action_masks()[:-1].sum())
        waited = obs["candidate_features"][:n, WAITED]
        assert (np.diff(waited) <= 0).all()
        obs, *_, terminated, truncated, _ = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            obs, _ = env.reset()


def test_shuffle_is_on_by_default_and_reproducible() -> None:
    """Shuffled and unshuffled wrappers driven by the same base actions hold
    the same candidates, in a different order at least sometimes; the
    shuffle repeats exactly under the same seed."""

    def run(seed: int, shuffle: bool) -> list[list[tuple[int, int]]]:
        env = wrapped(shuffle_candidates=shuffle)
        env.reset(seed=seed)
        rng = np.random.default_rng(0)
        seen = []
        for _ in range(80):
            candidates = env.candidates()
            seen.append(candidates)
            # Choose by base action (lowest row among a random rule-free pick),
            # so both wrappers take the same base action whatever the order.
            if candidates and rng.random() < 0.8:
                action = candidates.index(min(candidates))
            else:
                mask = env.action_masks()
                action = env.wait_index if mask[env.wait_index] else 0
            *_, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        return seen

    assert wrapped().shuffle_candidates is True
    shuffled, again, plain = run(3, True), run(3, True), run(3, False)
    assert shuffled == again
    assert [sorted(c) for c in shuffled] == [sorted(c) for c in plain]
    assert any(a != b for a, b in zip(shuffled, plain, strict=True))


def test_overflow_keeps_the_oldest_and_reports_the_rest() -> None:
    env = wrapped(max_candidates=2, shuffle_candidates=False)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    for _ in range(400):
        base_valid = int(info["base_action_mask"][:-1].sum())
        assert info["candidates_dropped"] == max(base_valid - 2, 0)
        if info["candidates_dropped"]:
            base = env.base
            ops = obs["ops"]
            startable = np.flatnonzero(ops[: base.max_ops, OP_FEATURES.index("startable")])
            kept = obs["candidate_features"][:2, WAITED]
            assert kept.min() >= np.sort(ops[startable, WAITED])[-2] - 1e-6
            return
        obs, _, terminated, truncated, info = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            obs, info = env.reset()
    raise AssertionError("never more than two candidates")


def test_masked_pointer_actions_raise_and_compact_observation() -> None:
    env = wrapped(keep_base_observation=False)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {
        "global",
        "candidate_features",
        "candidate_mask",
        "candidate_slot",
        "candidate_rule",
        "candidate_op_type",
    }
    assert env.observation_space.contains(obs)
    padding = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(InvalidActionError, match="pointer action"):
        env.step(padding)


@pytest.mark.parametrize("keep", [True, False])
def test_wrapper_passes_check_env(keep: bool) -> None:
    env = wrapped(keep_base_observation=keep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env)


def test_wraps_a_gym_make_env() -> None:
    env = PointerActionWrapper(gym.make("hospital_sim:SchedSim-v0", scenario="example_hospital"))
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 129 and obs["candidate_mask"].any()
