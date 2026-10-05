"""Evaluation harness: many seeded episodes per policy, on common instances.

Every policy runs on exactly the same instances (common random numbers), so
differences between policies are paired and far less noisy than the raw
spread of each. Instances are drawn from the ``eval`` or ``test`` seed split;
the ``train`` split is refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from hospital_sim.baselines.dispatching import Policy
from hospital_sim.domain.model import ProblemInstance, Scenario
from hospital_sim.env.config import EnvConfig, Randomization, instance_seed, randomized_scenario
from hospital_sim.env.hospital_env import HospitalEnv
from hospital_sim.generators.instance import generate_instance
from hospital_sim.sim.metrics import compute_metrics
from hospital_sim.sim.simulator import OpState, Simulator

METRICS = (
    "makespan",
    "mean_flow_time",
    "weighted_mean_flow_time",
    "mean_op_wait",
    "total_wait",
    "overtime",
    "left",
    "no_show",
    "total_weighted_flow_time",
    "left_weight",
    "arrived_weight",
)


@dataclass(frozen=True)
class EpisodeRecord:
    """Outcome of one episode under one policy."""

    values: dict[str, float]  # the METRICS above
    utilization: dict[str, float]  # per role
    # patient class -> {"patients", "mean_flow_time", "mean_op_wait"}
    by_class: dict[str, dict[str, float]]


@dataclass(frozen=True)
class EnvAgent:
    """An agent that acts through the environment: ``act(observation, mask)``
    returns an action index. ``config`` must not use the train split."""

    name: str
    act: Callable[[dict[str, Any], np.ndarray], int]
    config: EnvConfig


def record_episode(sim: Simulator) -> EpisodeRecord:
    metrics = compute_metrics(sim)
    by_class: dict[str, dict[str, float]] = {}
    for patient in sim.instance.patients:
        if not sim.arrived[patient.id] or math.isnan(sim.completion_time[patient.id]):
            continue
        entry = by_class.setdefault(
            patient.patient_class, {"patients": 0.0, "flow": 0.0, "wait": 0.0, "ops": 0.0}
        )
        entry["patients"] += 1
        entry["flow"] += sim.completion_time[patient.id] - patient.arrival_time
        for op in patient.operations:
            if sim.state[op.id] == OpState.DONE:
                entry["wait"] += sim.start_time[op.id] - sim.ready_time[op.id]
                entry["ops"] += 1
    return EpisodeRecord(
        values={
            "makespan": metrics.makespan,
            "mean_flow_time": metrics.mean_flow_time,
            "weighted_mean_flow_time": metrics.weighted_mean_flow_time,
            "mean_op_wait": metrics.mean_op_wait,
            "total_wait": metrics.total_wait,
            "overtime": math.fsum(sim.overtime),
            "left": float(sim.n_left),
            "no_show": float(sim.n_no_show),
            "total_weighted_flow_time": math.fsum(
                w * f for w, f in zip(metrics.weights, metrics.flow_times, strict=True)
            ),
            "left_weight": sim.left_weight,
            "arrived_weight": math.fsum(metrics.weights),
        },
        utilization=metrics.utilization,
        by_class={
            name: {
                "patients": e["patients"],
                "mean_flow_time": e["flow"] / e["patients"],
                "mean_op_wait": e["wait"] / e["ops"] if e["ops"] else 0.0,
            }
            for name, e in by_class.items()
        },
    )


def sample_instances(
    scenario: Scenario,
    n_episodes: int,
    *,
    base_seed: int = 0,
    split: str = "eval",
    randomize: Randomization | None = None,
    min_patients: int = 1,
) -> list[ProblemInstance]:
    """The evaluation instances: the same list for the same arguments."""
    if split == "train":
        raise ValueError("evaluation must not use the train split; use 'eval' or 'test'")
    instances = []
    for episode in range(n_episodes):
        for resample in range(256):
            seed = instance_seed(base_seed, split, episode, resample)
            instance = generate_instance(randomized_scenario(scenario, randomize, seed), seed)
            if len(instance.patients) >= min_patients:
                instances.append(instance)
                break
        else:
            raise RuntimeError(f"no instance with at least {min_patients} patients in 256 draws")
    return instances


def _mean_ci(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Mean, standard deviation and half-width of the t confidence interval."""
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, float("nan")
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sd / math.sqrt(n)
    return mean, sd, half


@dataclass
class Evaluation:
    """Per-episode records for each policy, aligned by instance."""

    scenario: str
    n_episodes: int
    records: dict[str, list[EpisodeRecord]] = field(default_factory=dict)
    confidence: float = 0.95

    def series(self, policy: str, metric: str) -> np.ndarray:
        return np.array([r.values[metric] for r in self.records[policy]])

    def summary(self, metric: str) -> dict[str, tuple[float, float, float]]:
        """policy -> (mean, standard deviation, CI half-width)."""
        return {p: _mean_ci(self.series(p, metric), self.confidence) for p in self.records}

    def paired_difference(self, metric: str, policy: str, reference: str) -> tuple[float, float]:
        """Mean of (policy - reference) over the common instances, and the
        half-width of its confidence interval."""
        mean, _, half = _mean_ci(
            self.series(policy, metric) - self.series(reference, metric), self.confidence
        )
        return mean, half

    def utilization(self, policy: str) -> dict[str, tuple[float, float, float]]:
        roles = self.records[policy][0].utilization
        return {
            role: _mean_ci(
                np.array([r.utilization[role] for r in self.records[policy]]), self.confidence
            )
            for role in roles
        }

    def by_class(self, policy: str, metric: str) -> dict[str, tuple[float, float, float, int]]:
        """patient class -> (mean, sd, CI half-width, episodes containing the class)."""
        values: dict[str, list[float]] = {}
        for record in self.records[policy]:
            for name, entry in record.by_class.items():
                values.setdefault(name, []).append(entry[metric])
        return {
            name: (*_mean_ci(np.array(v), self.confidence), len(v))
            for name, v in sorted(values.items())
        }

    def ranking(self, metric: str) -> list[str]:
        summary = self.summary(metric)
        return sorted(summary, key=lambda p: summary[p][0])

    def table(
        self, metrics: Sequence[str] = ("mean_flow_time", "makespan", "mean_op_wait", "left")
    ) -> str:
        """Markdown table: mean ± CI half-width per policy, best first by the
        first metric. ``left`` (patients who gave up waiting) is always worth
        including: a policy can look good on flow time by letting people leave."""
        lines = [
            "| Policy | " + " | ".join(metrics) + " |",
            "|---|" + "---|" * len(metrics),
        ]
        summaries = {m: self.summary(m) for m in metrics}
        for policy in self.ranking(metrics[0]):
            cells = [
                f"{summaries[m][policy][0]:.1f} ± {summaries[m][policy][2]:.1f}" for m in metrics
            ]
            lines.append(f"| {policy} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _run_env_agent(agent: EnvAgent, scenario: Scenario, instance: ProblemInstance) -> Simulator:
    if agent.config.split == "train":
        raise ValueError(f"agent '{agent.name}' uses a train-split env; evaluation refuses it")
    env = HospitalEnv(agent.config)
    observation, _ = env.reset(seed=0, options={"instance": instance})
    while True:
        observation, _, terminated, truncated, _ = env.step(
            agent.act(observation, env.action_masks())
        )
        if terminated or truncated:
            assert env.sim is not None
            return env.sim


def evaluate(
    scenario: Scenario,
    policies: Sequence[Policy | EnvAgent],
    n_episodes: int,
    *,
    base_seed: int = 0,
    split: str = "eval",
    randomize: Randomization | None = None,
    min_patients: int = 1,
    confidence: float = 0.95,
    debug: bool = False,
) -> Evaluation:
    """Run every policy on the same ``n_episodes`` instances."""
    instances = sample_instances(
        scenario,
        n_episodes,
        base_seed=base_seed,
        split=split,
        randomize=randomize,
        min_patients=min_patients,
    )
    result = Evaluation(scenario=scenario.name, n_episodes=n_episodes, confidence=confidence)
    for policy in policies:
        if policy.name in result.records:
            raise ValueError(f"duplicate policy name '{policy.name}'")
        result.records[policy.name] = [
            record_episode(
                policy.run(instance, nominal=scenario, debug=debug)
                if isinstance(policy, Policy)
                else _run_env_agent(policy, scenario, instance)
            )
            for instance in instances
        ]
    return result
