"""Environment configuration, seed derivation and per-episode randomization."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from hospital_sim.domain.model import ArrivalKind, ArrivalSpec, Scenario
from hospital_sim.env.rewards import Reward

SPLITS = {"train": 0, "eval": 1, "test": 2}


@dataclass(frozen=True)
class Randomization:
    """Ranges from which noise parameters are drawn anew for every episode.

    ``None`` leaves a parameter at its configured value. The drawn values are
    never shown to the agent: what it knows about durations is the nominal
    scenario.
    """

    rho: tuple[float, float] | None = None
    # Multiplies the CV of every lognormal, gamma and truncated-normal duration.
    cv_scale: tuple[float, float] | None = None
    # Multiplies every Poisson arrival rate (constant or profile).
    arrival_rate_scale: tuple[float, float] | None = None


@dataclass
class EnvConfig:
    scenario: Scenario
    # Which family of instance seeds to draw from; the three are disjoint.
    split: str = "train"
    # One rule: the agent picks an operation. Several: it picks (operation, rule).
    allocation_rules: tuple[str, ...] = ("flexible",)
    allow_wait: bool = True
    # Reaching either limit ends the episode as truncated.
    max_consecutive_waits: int = 100
    max_episode_time: float | None = None
    # A name from ``rewards.REWARDS`` or any object with ``potential(sim)``.
    reward: str | Reward = "weighted_flow_time"
    # Rewards are multiplied by this; ``None`` means 1 / time_scale.
    reward_scale: float | None = None
    # "dict": padded feature tables. "graph": the same plus precedence edges.
    observation: str = "dict"
    # Times in observations are divided by this; ``None`` = mean expected
    # operation duration of the scenario.
    time_scale: float | None = None
    # Padding sizes. ``None`` derives them from the scenario; an instance that
    # does not fit is resampled.
    max_patients: int | None = None
    max_ops: int | None = None
    max_edges: int | None = None
    # Episodes with fewer patients are resampled.
    min_patients: int = 1
    randomize: Randomization | None = None
    # Show, per operation, the expected work its completion may reveal (from
    # the pathway template). Switch off to ablate that prior.
    gated_work_prior: bool = True
    # Added to the cost (times the patient's weight) for every patient who
    # leaves unseen. ``None`` = twice the expected work of the longest pathway,
    # so that letting a patient leave is never cheaper than treating them.
    leave_penalty: float | None = None
    # "raise" on a masked action, or "fallback" to the lowest valid action.
    on_invalid_action: str = "raise"
    debug: bool = False
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, got {self.split!r}")
        if self.observation not in ("dict", "graph"):
            raise ValueError("observation must be 'dict' or 'graph'")
        if self.on_invalid_action not in ("raise", "fallback"):
            raise ValueError("on_invalid_action must be 'raise' or 'fallback'")
        if self.min_patients < 1:
            raise ValueError("min_patients must be >= 1")
        if not self.allocation_rules:
            raise ValueError("allocation_rules must name at least one rule")


def instance_seed(base_seed: int, split: str, episode: int, resample: int) -> int:
    """A distinct integer for every (base seed, split, episode, resample).

    The fields occupy separate bit ranges, so two splits can never produce the
    same instance seed, whatever the base seed and episode index.
    """
    if not (0 <= episode < 1 << 52 and 0 <= resample < 1 << 8 and base_seed >= 0):
        raise ValueError("seed component out of range")
    return (base_seed << 64) | (SPLITS[split] << 60) | (episode << 8) | resample


def _scaled_arrivals(spec: ArrivalSpec, scale: float) -> ArrivalSpec:
    if spec.kind is not ArrivalKind.POISSON:
        return spec
    return dataclasses.replace(
        spec,
        rate=None if spec.rate is None else spec.rate * scale,
        rate_profile=tuple(r * scale for r in spec.rate_profile),
    )


def randomized_scenario(scenario: Scenario, spec: Randomization | None, seed: int) -> Scenario:
    """The scenario with this episode's noise parameters drawn from ``spec``."""
    if spec is None:
        return scenario
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(7,)))
    # Always three draws, in this order, so one range does not shift another.
    u_rho, u_cv, u_rate = rng.random(3)

    def draw(bounds: tuple[float, float] | None, u: float, default: float) -> float:
        return default if bounds is None else bounds[0] + u * (bounds[1] - bounds[0])

    rho = draw(spec.rho, u_rho, scenario.complexity.rho)
    cv_scale = draw(spec.cv_scale, u_cv, 1.0)
    rate_scale = draw(spec.arrival_rate_scale, u_rate, 1.0)
    return dataclasses.replace(
        scenario,
        complexity=dataclasses.replace(scenario.complexity, rho=rho),
        operation_types=tuple(
            dataclasses.replace(
                t, duration=dataclasses.replace(t.duration, cv=t.duration.cv * cv_scale)
            )
            for t in scenario.operation_types
        ),
        arrivals=tuple(_scaled_arrivals(a, rate_scale) for a in scenario.arrivals),
    )


def expected_patients_bound(scenario: Scenario, rate_scale: float = 1.0) -> int:
    """A generous upper bound on patients per episode, used to size padding."""
    total = 0.0
    for spec in scenario.arrivals:
        if spec.kind is ArrivalKind.STATIC or spec.horizon is None:
            assert spec.n_patients is not None
            total += spec.n_patients
            continue
        rate = (
            sum(spec.rate_profile) / len(spec.rate_profile)
            if spec.rate_profile
            else spec.rate or 0.0
        )
        mean = rate * rate_scale * spec.horizon
        # Mean plus six standard deviations of a compound Poisson count.
        bound = mean + 6.0 * math.sqrt(mean * (2.0 * spec.batch_mean - 1.0)) + 5.0
        total += min(bound, spec.n_patients) if spec.n_patients is not None else bound
    return math.ceil(total)
