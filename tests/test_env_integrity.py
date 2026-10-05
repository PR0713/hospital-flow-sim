"""M3 integrity: nothing hidden reaches the agent.

Twin tests: two instances that differ only in something the agent must not
know are driven with the same policy. Observation, mask, reward and info must
be identical at every step before the first event that reveals the
difference, and must differ afterwards (so each test is known to have teeth).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hospital_sim.env as env_package
from hospital_sim.domain import ProblemInstance, Scenario
from hospital_sim.env import INFO_KEYS, EnvConfig, HospitalEnv
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import OpState

CONFIGS = Path(__file__).parent.parent / "configs"


def scenario(name: str) -> Scenario:
    return load_scenario(CONFIGS / f"example_hospital{name}.yaml")


Record = tuple[float, dict[str, Any], np.ndarray, float, dict[str, Any]]


def trace(sc: Scenario, instance: ProblemInstance) -> tuple[list[Record], HospitalEnv]:
    """Run one episode with a policy that depends only on the mask."""
    env = HospitalEnv(EnvConfig(sc, observation="graph"))
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0, options={"instance": instance})
    records: list[Record] = [(info["time"], obs, env.action_masks(), 0.0, info)]
    while True:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, reward, terminated, truncated, info = env.step(action)
        records.append((info["time"], obs, env.action_masks(), reward, info))
        if terminated or truncated:
            return records, env


def same(a: Record, b: Record) -> bool:
    def equal(x: Any, y: Any) -> bool:
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, np.ndarray):
            return bool(np.array_equal(x, y))
        return bool(x == y)

    return all(equal(x, y) for x, y in zip(a, b, strict=True))


def assert_identical_until(a: list[Record], b: list[Record], reveal: float, min_steps: int) -> None:
    compared = 0
    for ra, rb in zip(a, b, strict=False):
        if ra[0] >= reveal or rb[0] >= reveal:
            break
        assert same(ra, rb), f"records differ at t={ra[0]}, before the reveal at t={reveal}"
        compared += 1
    assert compared >= min_steps, f"only {compared} steps before the reveal; pick another case"
    diverged = len(a) != len(b) or any(not same(x, y) for x, y in zip(a, b, strict=True))
    assert diverged, "the twins never diverged, so this test checks nothing"


def with_hidden(instance: ProblemInstance, **changes: Any) -> ProblemInstance:
    return dataclasses.replace(instance, hidden=dataclasses.replace(instance.hidden, **changes))


def test_duration_quantile_of_an_unfinished_operation_does_not_leak() -> None:
    sc = scenario("")
    a = generate_instance(sc, 3)
    surgery = next(op.id for op in a.operations if op.op_type == "Surgery")
    quantiles = list(a.hidden.duration_quantiles)
    quantiles[surgery] = 0.999 if quantiles[surgery] < 0.5 else 0.001
    b = with_hidden(a, duration_quantiles=tuple(quantiles))

    trace_a, env_a = trace(sc, a)
    trace_b, env_b = trace(sc, b)
    assert env_a.sim is not None and env_b.sim is not None
    # The difference shows when the operation completes in either world.
    reveal = min(env_a.sim.end_time[surgery], env_b.sim.end_time[surgery])
    assert env_a.sim.start_time[surgery] == env_b.sim.start_time[surgery]
    assert_identical_until(trace_a, trace_b, reveal, min_steps=15)


def downtime_events(env: HospitalEnv) -> set[tuple[int, float, str]]:
    assert env.sim is not None
    events = set()
    for r, start, end, reason in env.sim.downtime_log:  # ground truth: test use only
        if reason in ("absent", "breakdown"):
            events.add((r, start, "down"))
            events.add((r, end, "up"))
    return events


def test_future_absences_and_breakdowns_do_not_leak() -> None:
    sc = scenario("_stress")
    checked = 0
    for seed in range(40):
        a = generate_instance(sc, seed)
        b = with_hidden(a, availability_seed=10_000 + seed)
        trace_a, env_a = trace(sc, a)
        trace_b, env_b = trace(sc, b)
        horizon = min(trace_a[-1][0], trace_b[-1][0])
        differing = [
            t for _, t, _ in downtime_events(env_a) ^ downtime_events(env_b) if t <= horizon
        ]
        if not differing or min(differing) < 150:
            continue  # same truth within the episode, or it differs too early to test
        assert_identical_until(trace_a, trace_b, min(differing), min_steps=10)
        checked += 1
        if checked == 4:
            return
    raise AssertionError(f"only {checked} usable twin pairs found")


def test_a_future_arrival_does_not_leak() -> None:
    sc = scenario("_dynamic")
    for seed in range(40):
        a = generate_instance(sc, seed)
        if len(a.patients) < 6:
            continue
        last = a.patients[-1]
        if last.arrival_time < 150:
            continue
        # Same patient, arriving 40 minutes later.
        b = dataclasses.replace(
            a,
            patients=(
                *a.patients[:-1],
                dataclasses.replace(last, arrival_time=last.arrival_time + 40),
            ),
        )
        trace_a, _ = trace(sc, a)
        trace_b, _ = trace(sc, b)
        assert_identical_until(trace_a, trace_b, last.arrival_time, min_steps=10)
        return
    raise AssertionError("no suitable instance")


def test_the_latent_factor_never_reaches_the_agent() -> None:
    sc = scenario("_dynamic")
    a = generate_instance(sc, 2)
    b = with_hidden(a, patient_factors=tuple(-f for f in a.hidden.patient_factors))
    trace_a, _ = trace(sc, a)
    trace_b, _ = trace(sc, b)
    assert len(trace_a) == len(trace_b) > 10
    assert all(same(x, y) for x, y in zip(trace_a, trace_b, strict=True))


# --- whitelist scans ------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "_stress"])
def test_keys_are_whitelisted_and_no_value_equals_a_hidden_time(name: str) -> None:
    sc = scenario(name)
    env = HospitalEnv(EnvConfig(sc, observation="graph"))
    rng = np.random.default_rng(1)
    obs, info = env.reset(seed=4)
    assert env.sim is not None
    running_seen = 0
    for _ in range(400):
        assert set(obs) == set(env.observation_space.spaces)
        assert set(info) <= INFO_KEYS
        sim = env.sim
        # Ground truth read straight from the engine, for the test only.
        completions = {e[3]: e[0] for e in sim._events if e[1] == 0}
        floats = (
            np.concatenate([v.ravel() for v in obs.values() if v.dtype == np.float32]).astype(
                np.float64
            )
            * env.time_scale
        )
        for op, completes in completions.items():
            assert sim.state[op] == OpState.RUNNING
            running_seen += 1
            for secret in (completes, completes - sim.now, completes - sim.start_time[op]):
                if secret > 1e-6:
                    assert not np.isclose(floats, secret, rtol=1e-6, atol=1e-6).any(), (
                        f"observation contains hidden time {secret} of running op {op}"
                    )
        obs, _, terminated, truncated, info = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            obs, info = env.reset()
    assert running_seen > 200


FORBIDDEN = {
    "hidden",
    "downtime_log",
    "duration_quantiles",
    "patient_factors",
    "availability_seed",
    "_events",
    "_quantiles",
    "_down_end",
    "_dist",
    "_streams",
    "_overtime_from",
    "has_pending_events",
}


def test_env_code_never_touches_hidden_state() -> None:
    """Static check: no module of the env package reads an attribute that
    holds hidden data. (Docstrings may mention them; code may not.)"""
    package_dir = Path(env_package.__file__).parent
    offenders = []
    for path in sorted(package_dir.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(node, "attr", None) or getattr(node, "id", None)
            if name in FORBIDDEN:
                offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert offenders == []


def test_expected_remaining_is_an_estimate_not_the_truth() -> None:
    sc = scenario("")
    env = HospitalEnv(EnvConfig(sc))
    rng = np.random.default_rng(2)
    obs, _ = env.reset(seed=1)
    assert env.sim is not None
    from hospital_sim.env.observation import OP_FEATURES

    errors = []
    for _ in range(150):
        sim = env.sim
        completions = {e[3]: e[0] for e in sim._events if e[1] == 0}
        for slot, op in enumerate(sim.visible_ops):
            if op in completions:
                shown = obs["ops"][slot, OP_FEATURES.index("expected_remaining")] * env.time_scale
                errors.append(shown - (completions[op] - sim.now))
        obs, _, terminated, truncated, _ = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            obs, _ = env.reset()
    errors_array = np.array(errors)
    assert len(errors_array) > 100
    assert np.abs(errors_array).min() > 1e-3  # never the true value
    # ...but an honest estimate: errors are small on average relative to their spread.
    assert abs(errors_array.mean()) < 0.25 * errors_array.std()
