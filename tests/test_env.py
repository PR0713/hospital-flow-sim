"""M3: the Gymnasium environment."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hospital_sim.domain import Scenario
from hospital_sim.env import (
    INFO_KEYS,
    Combined,
    EnvConfig,
    FlowTime,
    HospitalEnv,
    InvalidActionError,
    Makespan,
    Randomization,
    instance_seed,
)
from hospital_sim.env.observation import OP_FEATURES, PATIENT_FEATURES, RESOURCE_FEATURES
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import OpState

from .conftest import make_scenario

CONFIGS = Path(__file__).parent.parent / "configs"


def scenario(name: str) -> Scenario:
    return load_scenario(CONFIGS / f"example_hospital{name}.yaml")


def small(n_patients: int = 4) -> Scenario:
    base = scenario("")
    return dataclasses.replace(
        base, arrivals=(dataclasses.replace(base.arrivals[0], n_patients=n_patients),)
    )


def random_valid(env: HospitalEnv, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def play(env: HospitalEnv, seed: int | None, policy_seed: int = 0, **reset: Any) -> dict[str, Any]:
    """One episode under a random masked policy; returns what happened."""
    rng = np.random.default_rng(policy_seed)
    obs, info = env.reset(seed=seed, **reset)
    rewards, steps = [], 0
    while True:
        obs, reward, terminated, truncated, info = env.step(random_valid(env, rng))
        rewards.append(reward)
        steps += 1
        if terminated or truncated:
            return {
                "rewards": rewards,
                "steps": steps,
                "terminated": terminated,
                "info": info,
                "obs": obs,
            }


# --- gymnasium compatibility and smoke ------------------------------------------


@pytest.mark.parametrize("name", ["", "_dynamic", "_stress"])
@pytest.mark.parametrize("observation", ["dict", "graph"])
def test_passes_gymnasium_check_env(name: str, observation: str) -> None:
    env = HospitalEnv(
        EnvConfig(scenario(name), observation=observation, on_invalid_action="fallback")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # infinite Box bounds are intentional
        check_env(env, skip_render_check=True)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "episodes"), [("", 150), ("_dynamic", 150), ("_stress", 40)])
def test_random_masked_policy_runs_many_episodes(name: str, episodes: int) -> None:
    env = HospitalEnv(EnvConfig(scenario(name), observation="graph", debug=True))
    rng = np.random.default_rng(1)
    obs, info = env.reset(seed=7)
    finished = terminated_count = 0
    while finished < episodes:
        assert set(info) <= INFO_KEYS
        assert env.action_masks().any(), "a live episode must offer an action"
        obs, reward, terminated, truncated, info = env.step(random_valid(env, rng))
        assert np.isfinite(reward) and reward <= 0
        if terminated or truncated:
            assert env.observation_space.contains(obs)
            assert not env.action_masks().any()
            finished += 1
            terminated_count += terminated
            obs, info = env.reset()
            assert env.observation_space.contains(obs)
    assert terminated_count > 0.7 * episodes


# --- action masks ---------------------------------------------------------------


@pytest.mark.parametrize("rules", [("flexible",), ("flexible", "fastest")])
def test_every_unmasked_action_runs_and_every_masked_one_is_rejected(
    rules: tuple[str, ...],
) -> None:
    sc = small(3)
    instance = generate_instance(sc, 5)

    def fresh(history: list[int]) -> HospitalEnv:
        env = HospitalEnv(EnvConfig(sc, allocation_rules=rules, debug=True))
        env.reset(seed=0, options={"instance": instance})
        for action in history:
            env.step(action)
        return env

    rng = np.random.default_rng(3)
    history: list[int] = []
    checked = 0
    while True:
        env = fresh(history)
        mask = env.action_masks()
        for action in np.flatnonzero(mask):
            fresh(history).step(int(action))  # must not raise
            checked += 1
        for action in np.flatnonzero(~mask)[:: max(1, (~mask).sum() // 5)]:
            with pytest.raises(InvalidActionError):
                fresh(history).step(int(action))
        choice = int(rng.choice(np.flatnonzero(mask)))
        *_, terminated, truncated, _ = env.step(choice)
        history.append(choice)
        if terminated or truncated:
            break
    assert checked > len(history)


@pytest.mark.parametrize("name", ["", "_dynamic"])
def test_random_masked_policy_smoke(name: str) -> None:
    """Quick version of the long smoke test below: a few episodes in debug mode."""
    env = HospitalEnv(EnvConfig(scenario(name), observation="graph", debug=True))
    rng = np.random.default_rng(1)
    obs, info = env.reset(seed=7)
    finished = 0
    while finished < 5:
        assert set(info) <= INFO_KEYS and env.action_masks().any()
        obs, reward, terminated, truncated, info = env.step(random_valid(env, rng))
        assert np.isfinite(reward) and reward <= 0
        if terminated or truncated:
            assert env.observation_space.contains(obs)
            finished += 1
            obs, info = env.reset()


def test_out_of_range_and_fallback() -> None:
    env = HospitalEnv(EnvConfig(small()))
    env.reset(seed=0)
    with pytest.raises(InvalidActionError):
        env.step(env.n_actions + 3)
    lenient = HospitalEnv(EnvConfig(small(), on_invalid_action="fallback"))
    lenient.reset(seed=0)
    masked = int(np.flatnonzero(~lenient.action_masks())[0])
    *_, info = lenient.step(masked)
    assert info["invalid_action"] is True


def test_mask_marks_exactly_the_startable_operations() -> None:
    env = HospitalEnv(EnvConfig(small(), debug=True))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=2)
    for _ in range(40):
        mask = env.action_masks()
        startable = obs["ops"][:, OP_FEATURES.index("startable")] > 0
        assert (mask[: env.max_ops] == startable).all()
        assert env.sim is not None
        assert sorted(env.sim.visible_ops[s] for s in np.flatnonzero(startable)) == sorted(
            env.sim.startable_ops()
        )
        obs, _, terminated, truncated, _ = env.step(random_valid(env, rng))
        if terminated or truncated:
            break


# --- waiting --------------------------------------------------------------------


def test_wait_is_masked_in_a_static_batch_when_nothing_is_running() -> None:
    env = HospitalEnv(EnvConfig(small()))
    env.reset(seed=0)
    assert env.wait_action is not None and not env.action_masks()[env.wait_action]
    env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.action_masks()[env.wait_action]  # something is running now


def test_wait_can_be_disabled() -> None:
    env = HospitalEnv(EnvConfig(small(), allow_wait=False))
    assert env.wait_action is None and env.action_space.n == env.max_ops
    assert play(env, seed=0)["terminated"]


def test_too_many_consecutive_waits_truncates() -> None:
    env = HospitalEnv(EnvConfig(scenario("_stress"), max_consecutive_waits=5))
    env.reset(seed=0)
    for i in range(5):
        assert env.wait_action is not None and env.action_masks()[env.wait_action]
        _, _, terminated, truncated, info = env.step(env.wait_action)
        assert not terminated
        assert truncated == (i == 4)
    assert info["truncation_reason"] == "max_consecutive_waits"
    assert info["episode"]["terminated"] is False
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(env.wait_action)


def test_time_limit_truncates() -> None:
    env = HospitalEnv(EnvConfig(small(8), max_episode_time=60.0))
    result = play(env, seed=0)
    assert not result["terminated"]
    assert result["info"]["truncation_reason"] == "max_episode_time"
    assert result["info"]["time"] >= 60.0


# --- rewards --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "_dynamic"])
@pytest.mark.parametrize(
    ("reward", "metric"),
    [
        ("weighted_flow_time", "total_weighted_flow_time"),
        ("flow_time", "total_flow_time"),
        ("makespan", "makespan"),
    ],
)
def test_rewards_sum_exactly_to_the_final_metric(name: str, reward: str, metric: str) -> None:
    env = HospitalEnv(EnvConfig(scenario(name), reward=reward, reward_scale=1.0))
    checked = 0
    for seed in range(12):
        result = play(env, seed=seed, policy_seed=seed)
        if result["terminated"]:
            final = result["info"]["episode"][metric]
            assert -sum(result["rewards"]) == pytest.approx(final, rel=1e-9, abs=1e-9)
            checked += 1
    assert checked >= 8


def test_weights_enter_the_default_reward() -> None:
    env = HospitalEnv(EnvConfig(scenario("_dynamic"), reward_scale=1.0))
    for seed in range(30):
        episode = play(env, seed=seed)["info"]["episode"]
        if episode["total_weighted_flow_time"] > episode["total_flow_time"]:
            return  # an urgent (weight 5) patient was present and counted more
    raise AssertionError("no episode with an urgent patient in 30 seeds")


def test_combined_reward_and_scale() -> None:
    combo = Combined([(1.0, Makespan()), (0.25, FlowTime(weighted=False))])
    env = HospitalEnv(EnvConfig(small(6), reward=combo, reward_scale=0.5))
    result = play(env, seed=3)
    episode = result["info"]["episode"]
    expected = 0.5 * (episode["makespan"] + 0.25 * episode["total_flow_time"])
    assert -sum(result["rewards"]) == pytest.approx(expected, rel=1e-9)
    assert episode["reward_potential"] == pytest.approx(2 * expected, rel=1e-9)


# --- reset, seeds, splits -------------------------------------------------------


def patients_of(env: HospitalEnv) -> Any:
    assert env.sim is not None
    return env.sim.instance.patients


def test_reset_with_a_seed_is_reproducible_and_later_resets_are_fresh() -> None:
    env = HospitalEnv(EnvConfig(scenario("")))
    first_obs, _ = env.reset(seed=11)
    first = patients_of(env)
    second_obs, _ = env.reset()
    second = patients_of(env)
    assert first != second, "each episode must get a fresh instance"

    again_obs, _ = env.reset(seed=11)
    assert patients_of(env) == first
    assert all(np.array_equal(first_obs[k], again_obs[k]) for k in first_obs)
    env.reset()
    assert patients_of(env) == second  # the whole sequence replays

    other = HospitalEnv(EnvConfig(scenario("")))
    other.reset(seed=11)
    assert patients_of(other) == first  # and across env objects


def test_unseeded_env_still_gives_distinct_episodes() -> None:
    env = HospitalEnv(EnvConfig(scenario("")))
    env.reset()
    first = patients_of(env)
    env.reset()
    assert patients_of(env) != first


def test_splits_are_disjoint_by_construction() -> None:
    seeds = {
        split: {instance_seed(42, split, episode, 0) for episode in range(10_000)}
        for split in ("train", "eval", "test")
    }
    assert all(len(s) == 10_000 for s in seeds.values())
    assert not seeds["train"] & seeds["eval"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["eval"] & seeds["test"]
    # Also across base seeds and resamples.
    assert instance_seed(1, "train", 0, 1) != instance_seed(1, "train", 1, 0)
    assert instance_seed(1, "eval", 5, 0) != instance_seed(2, "train", 5, 0)
    with pytest.raises(ValueError, match="split must be one of"):
        EnvConfig(scenario(""), split="validation")


def test_same_seed_in_different_splits_gives_different_instances() -> None:
    instances = {}
    for split in ("train", "eval", "test"):
        env = HospitalEnv(EnvConfig(scenario(""), split=split))
        env.reset(seed=3)
        instances[split] = patients_of(env)
        again = HospitalEnv(EnvConfig(scenario(""), split=split))
        again.reset(seed=3)
        assert patients_of(again) == instances[split]
    assert len({id(v) for v in instances.values()}) == 3
    assert instances["train"] != instances["eval"] != instances["test"]


def sparse() -> Scenario:
    return make_scenario(
        resources={"Nurse": 1},
        op_types={"T": ({"Nurse": 1}, 5, True)},
        pathways={"p": [("t", "T", [])]},
        arrivals={"kind": "poisson", "horizon": 100, "rate": 0.01},  # mean one patient
    )


def test_empty_episodes_are_resampled_and_counted() -> None:
    assert any(len(generate_instance(sparse(), s).patients) == 0 for s in range(20))
    env = HospitalEnv(EnvConfig(sparse()))
    per_reset = []
    _, info = env.reset(seed=0)
    for _ in range(60):
        per_reset.append(info["n_resamples"])
        assert len(patients_of(env)) >= 1
        _, info = env.reset()
    assert sum(per_reset) > 5 and env.total_resamples >= sum(per_reset)

    strict = HospitalEnv(EnvConfig(sparse(), min_patients=3))
    for _ in range(10):
        strict.reset()
        assert len(patients_of(strict)) >= 3
    assert strict.total_resamples > env.total_resamples


def test_per_episode_randomization_of_noise_parameters() -> None:
    ranges = Randomization(rho=(0.1, 0.8), cv_scale=(0.5, 1.5), arrival_rate_scale=(0.5, 1.0))
    nominal = scenario("_dynamic")

    def drawn(env: HospitalEnv) -> tuple[float, float, float]:
        assert env.sim is not None
        sc = env.sim.instance.scenario
        cv = sc.operation_type_by_name["Surgery"].duration.cv
        rate = sc.arrivals[1].rate
        assert rate is not None
        return sc.complexity.rho, cv / 0.4, rate / 0.003

    env = HospitalEnv(EnvConfig(nominal, randomize=ranges))
    env.reset(seed=5)
    draws = [drawn(env)]
    baseline_expected = None
    for _ in range(40):
        obs, _ = env.reset()
        draws.append(drawn(env))
        # The agent's duration knowledge stays nominal whatever was drawn.
        valid = obs["op_valid"] > 0
        column = obs["ops"][valid, OP_FEATURES.index("expected_duration")]
        types = obs["op_type"][valid]
        table = {int(t): float(v) for t, v in zip(types, column, strict=True)}
        baseline_expected = baseline_expected or {}
        for t, v in table.items():
            assert baseline_expected.setdefault(t, v) == v
    rho, cv, rate = (np.array(x) for x in zip(*draws, strict=True))
    assert 0.1 <= rho.min() < 0.3 and 0.6 < rho.max() <= 0.8
    assert 0.5 <= cv.min() < 0.8 and 1.2 < cv.max() <= 1.5
    assert rate.min() >= 0.5 and rate.max() <= 1.0 and rate.std() > 0.05

    replay = HospitalEnv(EnvConfig(nominal, randomize=ranges))
    replay.reset(seed=5)
    assert drawn(replay) == draws[0]


# --- (operation, rule) actions --------------------------------------------------


def test_rule_choice_changes_the_team() -> None:
    sc = make_scenario(
        resources={"Nurse": 1},
        op_types={"Surgery": ({"Lead": 1, "Nurse": 1}, 60, True)},
        pathways={"p": [("s", "Surgery", [])]},
        extra_resources=[
            {"id": "slow", "roles": ["Lead"], "attributes": {"speed": 0.5}},
            {"id": "fast", "roles": ["Lead", "Scrub"], "attributes": {"speed": 2.0}},
        ],
        speed_roles={"Surgery": ["Lead"]},
    )
    # "Scrub" is unused by Surgery but makes the fast surgeon the flexible one.
    env = HospitalEnv(EnvConfig(sc, allocation_rules=("flexible", "fastest"), reward="makespan"))
    assert env.action_space.n == 2 * env.max_ops + 1
    outcomes = {}
    for rule in (0, 1):
        env.reset(seed=0)
        *_, terminated, _, info = env.step(0 * 2 + rule)
        assert terminated
        outcomes[rule] = info["episode"]["makespan"]
    assert outcomes == {0: 120.0, 1: 30.0}  # flexible keeps "fast" free; fastest uses it


# --- observation content --------------------------------------------------------


def test_observation_fields_and_growth_during_an_episode() -> None:
    env = HospitalEnv(EnvConfig(scenario("_dynamic"), observation="graph", debug=True))
    rng = np.random.default_rng(4)
    obs, _ = env.reset(seed=9)
    assert env.sim is not None
    total_patients = len(env.sim.instance.patients)
    seen_ops, seen_patients, realized_checked = [], [], 0
    col = OP_FEATURES.index
    while True:
        sim = env.sim
        k, m = int(obs["op_valid"].sum()), int(obs["patient_valid"].sum())
        seen_ops.append(k)
        seen_patients.append(m)
        assert m == sum(sim.arrived) and k == len(sim.visible_ops)
        assert (obs["ops"][k:] == 0).all() and (obs["op_type"][k:] == -1).all()
        assert set(np.unique(obs["patients"][:m, PATIENT_FEATURES.index("acuity")])) <= {
            0.0,
            0.5,
            1.0,
        }

        graph = env.graph()
        assert graph["nodes"]["op"].shape == (k, len(OP_FEATURES))
        assert graph["nodes"]["patient"].shape == (m, len(PATIENT_FEATURES))
        edges = graph["edges"]["op__precedes__op"]
        assert edges.size == 0 or edges.max() < k
        padded = obs["precedence_edges"]
        assert (padded[:, : edges.shape[1]] == edges).all() and (
            padded[:, edges.shape[1] :] == -1
        ).all()
        for src, dst in edges.T:
            assert sim.visible_ops[dst] in sim.succs[sim.visible_ops[src]]
        uses = graph["edges"]["op__uses__resource"]
        assert all(sim.resources.holder[r] == sim.visible_ops[s] for s, r in uses.T)

        for slot in range(k):
            op = sim.visible_ops[slot]
            if sim.state[op] == OpState.DONE:
                truth = (sim.end_time[op] - sim.start_time[op]) / env.time_scale
                assert obs["ops"][slot, col("realized_duration")] == pytest.approx(truth, rel=1e-5)
                realized_checked += 1
            else:
                assert obs["ops"][slot, col("realized_duration")] == 0

        obs, _, terminated, truncated, _ = env.step(random_valid(env, rng))
        if terminated or truncated:
            break
    assert seen_ops == sorted(seen_ops) and seen_ops[-1] > seen_ops[0]  # the graph grew
    assert seen_patients[0] < total_patients  # later patients were not visible at first
    assert realized_checked > 0


def test_patient_class_features_are_visible() -> None:
    env = HospitalEnv(EnvConfig(scenario("_dynamic")))
    weight, urgent = PATIENT_FEATURES.index("weight"), PATIENT_FEATURES.index("urgent")
    for seed in range(40):
        result = play(env, seed=seed)
        patients = result["obs"]["patients"][result["obs"]["patient_valid"] > 0]
        if (patients[:, urgent] == 1).any():
            assert set(patients[patients[:, urgent] == 1][:, weight]) == {5.0}
            assert set(patients[patients[:, urgent] == 0][:, weight]) <= {1.0}
            return
    raise AssertionError("no urgent patient seen")


def test_roster_features_come_from_the_calendar() -> None:
    env = HospitalEnv(EnvConfig(scenario("_dynamic")))
    obs, info = env.reset(seed=0)
    assert env.sim is not None
    ids = env.sim.resources.resource_ids
    f = RESOURCE_FEATURES.index
    nurse, surgeon = ids.index("nurse_a_1"), ids.index("surgeon_junior")
    # nurse_a: on duty 0-720 with a break at 240.
    expected = (240.0 - info["time"]) / env.time_scale
    assert obs["resources"][nurse, f("rostered_on")] == 1
    assert obs["resources"][nurse, f("time_to_roster_change")] == pytest.approx(expected, rel=1e-5)
    assert obs["resources"][surgeon, f("time_to_roster_change")] == 1e3  # no calendar: capped
    assert obs["resources"][surgeon, f("speed")] == pytest.approx(0.85)
