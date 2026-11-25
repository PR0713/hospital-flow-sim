"""Gymnasium environment for external RL agents (no training code)."""

from hospital_sim.env.config import EnvConfig, Randomization, instance_seed
from hospital_sim.env.hospital_env import INFO_KEYS, HospitalEnv, InvalidActionError
from hospital_sim.env.observation import OBSERVATION_SCHEMA_VERSION, NonFiniteObservationError
from hospital_sim.env.registration import ENV_ID, make_env, resolve_scenario
from hospital_sim.env.rewards import REWARDS, Combined, FlowTime, Makespan, Reward
from hospital_sim.env.spaces import MaskedDiscrete

__all__ = [
    "ENV_ID",
    "INFO_KEYS",
    "OBSERVATION_SCHEMA_VERSION",
    "REWARDS",
    "Combined",
    "EnvConfig",
    "FlowTime",
    "HospitalEnv",
    "InvalidActionError",
    "Makespan",
    "MaskedDiscrete",
    "NonFiniteObservationError",
    "Randomization",
    "Reward",
    "instance_seed",
    "make_env",
    "resolve_scenario",
]
