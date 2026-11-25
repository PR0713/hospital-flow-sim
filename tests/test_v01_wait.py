"""v0.1, task 3: waiting always lets time pass, and truncation can be charged.

Before v0.1, in a scenario with arrivals over time but no rosters or
breakdowns, wait stayed valid once the event list was empty and then did
nothing: time stood still, the reward was 0, and 100 such waits truncated
the episode with patients unserved (audit 11.3).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hospital_sim.domain.model import DurationKind, Scenario
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import OpState, PatientStatus

CONFIGS = Path(__file__).parent.parent / "configs"


def dynamic() -> Scenario:
    return load_scenario(CONFIGS / "example_hospital_dynamic.yaml")


def no_rosters(scenario: Scenario) -> Scenario:
    """Arrivals over time, but no calendars or breakdowns: the event list can run dry."""
    return dataclasses.replace(
        scenario,
        calendars=(),
        resources=tuple(
            dataclasses.replace(r, calendar=None, breakdown=None) for r in scenario.resources
        ),
    )


def deterministic(scenario: Scenario) -> Scenario:
    return dataclasses.replace(
        scenario,
        operation_types=tuple(
            dataclasses.replace(
                t,
                duration=dataclasses.replace(
                    t.duration,
                    kind=DurationKind.DETERMINISTIC,
                    cv=0.0,
                    shift=0.0,
                    complication_probability=0.0,
                    complication_multiplier=1.0,
                ),
            )
            for t in scenario.operation_types
        ),
    )


def first_start(env: HospitalEnv) -> int:
    valid = np.flatnonzero(env.action_masks())
    starts = [a for a in valid if a != env.wait_action]
    return int(starts[0] if starts else valid[0])


def test_every_wait_lets_time_pass() -> None:
    """A valid wait is never a no-op, including when the event list is empty."""
    env = HospitalEnv(EnvConfig(no_rosters(dynamic()), split="train"))
    rng = np.random.default_rng(0)
    empty_queue_waits = 0
    for seed in range(40):
        env.reset(seed=seed)
        while True:
            sim = env.sim
            assert sim is not None
            mask = env.action_masks()
            if mask[env.wait_action] and rng.random() < 0.3:
                before, empty = sim.now, not sim.has_pending_events()
                *_, terminated, truncated, _ = env.step(env.wait_action)
                assert sim.now > before
                empty_queue_waits += empty
            else:
                *_, terminated, truncated, _ = env.step(first_start(env))
            if terminated or truncated:
                assert terminated  # never truncated by a run of no-op waits
                break
    assert empty_queue_waits > 0  # the case that used to be a no-op was exercised


def test_wait_with_an_empty_event_list_runs_to_the_public_horizon() -> None:
    sc = no_rosters(dynamic())
    horizon = max(a.horizon or 0.0 for a in sc.arrivals)
    env = HospitalEnv(EnvConfig(sc, split="train", reward_scale=1.0))
    for seed in range(200):
        env.reset(seed=seed)
        while True:
            sim = env.sim
            assert sim is not None
            if env.action_masks()[env.wait_action] and not sim.has_pending_events():
                assert sim.now < horizon
                present = sum(s == PatientStatus.PRESENT for s in sim.patient_status)
                start = sim.now
                _, reward, *_ = env.step(env.wait_action)
                assert sim.now == horizon
                # Nothing happened in between, so each present patient accrued
                # the whole interval (weights 1 and 5).
                assert reward <= -present * (horizon - start) + 1e-9
                # Arrivals are now provably over and nothing is running: no wait.
                running = any(s == OpState.RUNNING for s in sim.state)
                assert env.action_masks()[env.wait_action] == (
                    running or any(sim.in_turnover) or sim.has_pending_events()
                )
                return
            *_, terminated, truncated, _ = env.step(first_start(env))
            if terminated or truncated:
                break
    raise AssertionError("no state with an empty event list before the horizon")


def test_wait_mask_does_not_reveal_that_arrivals_are_over() -> None:
    """Two instances that differ only in a hidden future arrival get the same
    masks until that arrival happens, even when one event list is empty."""
    sc = no_rosters(dynamic())
    horizon = max(a.horizon or 0.0 for a in sc.arrivals)
    for seed in range(60):
        a = generate_instance(sc, seed)
        if len(a.patients) < 4 or a.patients[-1].arrival_time > horizon - 30:
            continue
        last = a.patients[-1]
        extra = dataclasses.replace(
            last,
            id=len(a.patients),
            arrival_time=horizon - 1.0,
            operations=tuple(
                dataclasses.replace(
                    op,
                    id=op.id + a.n_operations - last.operations[0].id,
                    patient_id=len(a.patients),
                    predecessors=tuple(
                        p + a.n_operations - last.operations[0].id for p in op.predecessors
                    ),
                    successors=tuple(
                        s + a.n_operations - last.operations[0].id for s in op.successors
                    ),
                )
                for op in last.operations
            ),
        )
        hidden = a.hidden
        own = [hidden.duration_quantiles[op.id] for op in last.operations]
        b = dataclasses.replace(
            a,
            patients=(*a.patients, extra),
            hidden=dataclasses.replace(
                hidden,
                duration_quantiles=(*hidden.duration_quantiles, *own),
                patient_factors=(*hidden.patient_factors, hidden.patient_factors[-1]),
            ),
        )
        masks: list[list[Any]] = []
        for instance in (a, b):
            env = HospitalEnv(EnvConfig(sc, split="eval"))
            env.reset(options={"instance": instance})
            trace = []
            while env.sim is not None and env.sim.now < horizon - 1.0:
                mask = env.action_masks()
                trace.append((env.sim.now, mask.tobytes()))
                action = env.wait_action if mask[env.wait_action] else first_start(env)
                *_, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            masks.append(trace)
        assert masks[0] == masks[1]
        if any(t for t, _ in masks[0]):
            return
    raise AssertionError("no suitable instance")


# --- truncation cost -----------------------------------------------------------


def _wait_until_truncated(cost: str) -> tuple[list[float], dict[str, Any], HospitalEnv]:
    env = HospitalEnv(
        EnvConfig(dynamic(), max_consecutive_waits=5, truncation_cost=cost, reward_scale=1.0)
    )
    env.reset(seed=1)
    rewards = []
    while True:
        mask = env.action_masks()
        action = env.wait_action if mask[env.wait_action] else int(np.flatnonzero(mask)[0])
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            assert truncated
            return rewards, info, env


def test_truncation_cost_defaults_to_none() -> None:
    assert EnvConfig(dynamic()).truncation_cost == "none"
    with pytest.raises(ValueError, match="truncation_cost"):
        EnvConfig(dynamic(), truncation_cost="accrued")


def test_expected_remaining_truncation_cost_is_one_final_reward_term() -> None:
    plain, info_none, _ = _wait_until_truncated("none")
    charged, info, _ = _wait_until_truncated("expected_remaining")
    assert charged[:-1] == plain[:-1]  # same trajectory, same rewards until the end
    cost = info["episode"]["truncation_cost"]
    assert cost > 0 and info_none["episode"]["truncation_cost"] == 0.0
    assert charged[-1] == pytest.approx(plain[-1] - cost, rel=1e-12)
    # The undiscounted return is minus (accrued cost + truncation cost).
    assert -sum(charged) == pytest.approx(info["episode"]["reward_potential"] + cost, rel=1e-9)


def test_remaining_flow_bound_never_exceeds_the_true_remaining_flow() -> None:
    """With deterministic durations the bound must hold on every sample path."""
    env = HospitalEnv(EnvConfig(deterministic(no_rosters(dynamic())), split="train"))
    for seed in range(8):
        env.reset(seed=seed)
        checkpoints: list[tuple[float, dict[int, float]]] = []
        while True:
            builder, sim = env._builder, env.sim
            assert builder is not None and sim is not None
            checkpoints.append((sim.now, builder.remaining_flow_lower_bound()))
            *_, terminated, truncated, _ = env.step(first_start(env))
            if terminated or truncated:
                break
        assert terminated
        for now, bounds in checkpoints:
            for patient, bound in bounds.items():
                assert sim.completion_time[patient] - now >= bound - 1e-6
        assert any(b > 0 for _, bs in checkpoints for b in bs.values())
