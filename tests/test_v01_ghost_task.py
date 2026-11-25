"""v0.1, task 2: nothing of a patient who has left can start afterwards.

Before v0.1, cancelling a departed patient's ready operation released its
successors, and a pure delay among them (no resources, not occupying the
patient) started on its own. The departed patient then stayed "open" until
it finished, so the observation carried a NaN time in system and the flow
time was inflated. Reproduction: ``example_hospital_patience``, seed 0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import load_scenario
from hospital_sim.sim import InvariantViolation, OpState, PatientStatus, check_invariants

PATIENCE = Path(__file__).parent.parent / "configs" / "example_hospital_patience.yaml"
_LIVE = (OpState.HIDDEN, OpState.WAITING, OpState.READY, OpState.RUNNING)


@pytest.fixture(scope="module")
def patience() -> Any:
    return load_scenario(PATIENCE)


def _withdrawn_with_live_ops(env: HospitalEnv) -> list[int]:
    sim = env.sim
    assert sim is not None
    return [
        p.id
        for p in sim.instance.patients
        if sim.patient_status[p.id] in (PatientStatus.LEFT, PatientStatus.NO_SHOW)
        and any(sim.state[op.id] in _LIVE for op in p.operations)
    ]


def _play(env: HospitalEnv, episodes: int, check: Any) -> int:
    """Random masked play as in the audit reproduction; returns patients who left."""
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    left = 0
    for _ in range(episodes):
        while True:
            check(env, obs)
            obs, _, terminated, truncated, info = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            if terminated or truncated:
                check(env, obs)
                left += info["episode"]["patients_left"]
                obs, _ = env.reset()
                break
    return left


def test_departed_patients_never_have_live_operations(patience: Any) -> None:
    env = HospitalEnv(EnvConfig(patience, split="train"))

    def check(env: HospitalEnv, obs: dict[str, Any]) -> None:
        assert _withdrawn_with_live_ops(env) == []
        assert env.observation_space.contains(obs)
        for key, value in obs.items():
            assert np.isfinite(value).all(), key

    assert _play(env, 3, check) > 0  # the scenario really exercised leaving


def test_departed_patient_flow_ends_when_they_leave(patience: Any) -> None:
    env = HospitalEnv(EnvConfig(patience, split="train"))

    def check(env: HospitalEnv, obs: dict[str, Any]) -> None:
        sim = env.sim
        assert sim is not None
        for p in sim.instance.patients:
            if sim.patient_status[p.id] == PatientStatus.LEFT:
                # Left unseen: completion is the leave time, i.e. arrival + patience.
                assert sim.completion_time[p.id] == pytest.approx(
                    p.arrival_time + sim.instance.hidden.patience[p.id]
                )

    assert _play(env, 3, check) > 0


def test_debug_run_with_leavers_keeps_every_invariant(patience: Any) -> None:
    env = HospitalEnv(EnvConfig(patience, split="train", debug=True))
    assert _play(env, 3, lambda env, obs: None) > 0


def test_invariant_flags_a_withdrawn_patient_with_a_live_operation(patience: Any) -> None:
    env = HospitalEnv(EnvConfig(patience, split="train"))
    env.reset(seed=0)
    sim = env.sim
    assert sim is not None
    check_invariants(sim)
    patient = sim.visible_patients[0]
    sim.patient_status[patient] = PatientStatus.LEFT  # corrupt: left, but ops still open
    with pytest.raises(InvariantViolation, match="withdrawn"):
        check_invariants(sim)


def _corrupt_arrival_times(env: HospitalEnv) -> None:
    builder = env._builder
    assert builder is not None
    builder.arrival[:] = np.nan  # makes every time_in_system non-finite


def test_non_finite_values_are_replaced_counted_and_logged_once(
    patience: Any, caplog: pytest.LogCaptureFixture
) -> None:
    env = HospitalEnv(EnvConfig(patience, split="train"))
    _, info = env.reset(seed=0)
    assert info["nonfinite_obs_count"] == 0
    _corrupt_arrival_times(env)
    with caplog.at_level(logging.WARNING, logger="hospital_sim.env"):
        obs, *_, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
        first = info["nonfinite_obs_count"]
        obs, *_, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert first > 0 and info["nonfinite_obs_count"] > first  # cumulative per episode
    assert all(np.isfinite(v).all() for v in obs.values())
    assert sum("non-finite" in r.message for r in caplog.records) == 1


def test_non_finite_values_raise_in_debug_mode(patience: Any) -> None:
    from hospital_sim.env import NonFiniteObservationError

    env = HospitalEnv(EnvConfig(patience, split="train", debug=True))
    env.reset(seed=0)
    _corrupt_arrival_times(env)
    with pytest.raises(NonFiniteObservationError, match="patients"):
        env.step(int(np.flatnonzero(env.action_masks())[0]))
