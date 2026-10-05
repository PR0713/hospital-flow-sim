"""Gymnasium environment over the core simulator. No training code."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from hospital_sim.domain.model import ArrivalKind, ProblemInstance
from hospital_sim.env.config import (
    EnvConfig,
    expected_patients_bound,
    instance_seed,
    randomized_scenario,
)
from hospital_sim.env.observation import (
    GLOBAL_FEATURES,
    OBSERVATION_SCHEMA_VERSION,
    OP_FEATURES,
    PATIENT_FEATURES,
    RESOURCE_FEATURES,
    ROLE_FEATURES,
    ObservationBuilder,
)
from hospital_sim.env.rewards import REWARDS, Reward
from hospital_sim.generators.instance import generate_instance
from hospital_sim.sim.allocation import RULES
from hospital_sim.sim.durations import StandardDurationModel
from hospital_sim.sim.metrics import compute_metrics
from hospital_sim.sim.simulator import OpState, Simulator

_MAX_RESAMPLES = 1000
# The only keys ``info`` may ever contain.
INFO_KEYS = frozenset(
    {
        "time",
        "episode_index",
        "observation_schema",
        "n_resamples",
        "action_mask",
        "invalid_action",
        "truncation_reason",
        "episode",
    }
)


class InvalidActionError(ValueError):
    """The agent chose an action that the mask rules out."""


class HospitalEnv(gym.Env[dict[str, Any], Any]):
    """Pick which ready operation to start next (and optionally by which
    allocation rule), or wait for the next event.

    Action ``slot * n_rules + rule`` starts the operation in observation row
    ``slot`` with ``config.allocation_rules[rule]``; the last action is "wait"
    when waiting is enabled. ``action_masks()`` says which actions are valid.
    """

    metadata = {"render_modes": []}
    observation_schema_version = OBSERVATION_SCHEMA_VERSION

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        scenario = config.scenario
        self._rules = [RULES[name] for name in config.allocation_rules]
        self._reward: Reward = (
            REWARDS[config.reward] if isinstance(config.reward, str) else config.reward
        )

        nominal = StandardDurationModel(scenario)
        means = [nominal.base(i).mean for i in range(len(scenario.operation_types))]
        self.time_scale = config.time_scale or float(np.mean(means))
        self._reward_scale = (
            config.reward_scale if config.reward_scale is not None else 1.0 / self.time_scale
        )

        rate_hi = 1.0
        if config.randomize and config.randomize.arrival_rate_scale:
            rate_hi = config.randomize.arrival_rate_scale[1]
        longest = max(sum(1 + s.repeat_max for s in p.steps) for p in scenario.pathways)
        by_name = {t.name: m for t, m in zip(scenario.operation_types, means, strict=True)}
        longest_work = max(
            sum(by_name[s.op_type] * (1 + s.repeat_max) for s in p.steps) for p in scenario.pathways
        )
        self.leave_penalty = (
            config.leave_penalty if config.leave_penalty is not None else 2.0 * longest_work
        )
        self.max_patients = config.max_patients or expected_patients_bound(scenario, rate_hi)
        self.max_ops = config.max_ops or self.max_patients * longest
        self.max_edges = config.max_edges or 3 * self.max_ops

        # Public reasons for which waiting can change something. Whether an
        # arrival is actually still to come is hidden, so it is not used.
        self._always_waitable = (
            any(a.kind is not ArrivalKind.STATIC for a in scenario.arrivals)
            or bool(scenario.calendars)
            or any(r.breakdown is not None for r in scenario.resources)
        )

        n_rules = len(self._rules)
        self.wait_action = self.max_ops * n_rules if config.allow_wait else None
        self.n_actions = self.max_ops * n_rules + int(config.allow_wait)
        self.action_space = spaces.Discrete(self.n_actions)

        n_res, n_roles = len(scenario.resources), len(scenario.roles)
        n_types = len(scenario.operation_types)

        def table(rows: int, columns: int) -> spaces.Box:
            return spaces.Box(-np.inf, np.inf, (rows, columns), np.float32)

        obs_spaces: dict[str, spaces.Space[Any]] = {
            "ops": table(self.max_ops, len(OP_FEATURES)),
            "op_valid": spaces.Box(0, 1, (self.max_ops,), np.int8),
            "op_type": spaces.Box(-1, n_types - 1, (self.max_ops,), np.int32),
            "op_patient": spaces.Box(-1, self.max_patients - 1, (self.max_ops,), np.int32),
            "op_requirements": table(self.max_ops, n_roles),
            "patients": table(self.max_patients, len(PATIENT_FEATURES)),
            "patient_valid": spaces.Box(0, 1, (self.max_patients,), np.int8),
            "resources": table(n_res, len(RESOURCE_FEATURES)),
            "resource_roles": spaces.Box(0, 1, (n_res, n_roles), np.int8),
            "resource_op": spaces.Box(-1, self.max_ops - 1, (n_res,), np.int32),
            "roles": table(n_roles, len(ROLE_FEATURES)),
            "global": spaces.Box(-np.inf, np.inf, (len(GLOBAL_FEATURES),), np.float32),
        }
        if config.observation == "graph":
            obs_spaces["precedence_edges"] = spaces.Box(
                -1, self.max_ops - 1, (2, self.max_edges), np.int32
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self._base_seed: int | None = None
        self._episode = -1
        self.total_resamples = 0
        self.sim: Simulator | None = None
        self._builder: ObservationBuilder | None = None
        self._mask = np.zeros(self.n_actions, dtype=np.bool_)
        self._observation: dict[str, Any] = {}
        self._startable: list[int] = []
        self._waits = 0
        self._potential = 0.0
        self._over = True

    # --- reset ---------------------------------------------------------------

    def _fits(self, instance: ProblemInstance) -> bool:
        n_edges = sum(len(op.predecessors) for op in instance.operations)
        return (
            self.config.min_patients <= len(instance.patients) <= self.max_patients
            and instance.n_operations <= self.max_ops
            and n_edges <= self.max_edges
        )

    def _sample_instance(self) -> tuple[ProblemInstance, int]:
        assert self._base_seed is not None
        for resample in range(_MAX_RESAMPLES):
            seed = instance_seed(
                self._base_seed, self.config.split, self._episode, min(resample, 255)
            )
            if resample > 255:  # beyond the reserved bits: keep drawing new seeds
                seed = instance_seed(
                    self._base_seed + resample, self.config.split, self._episode, 255
                )
            scenario = randomized_scenario(self.config.scenario, self.config.randomize, seed)
            instance = generate_instance(scenario, seed)
            if self._fits(instance):
                return instance, resample
        raise RuntimeError(
            f"no instance with {self.config.min_patients}..{self.max_patients} patients and at "
            f"most {self.max_ops} operations after {_MAX_RESAMPLES} draws; check the padding sizes"
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode on a fresh instance.

        ``reset(seed=s)`` restarts the sequence of episodes: the same ``s``
        always gives the same first instance, and later ``reset()`` calls give
        the next ones. ``options={"instance": ...}`` runs a given instance.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._base_seed, self._episode = seed, -1
        elif self._base_seed is None:
            self._base_seed = int(self.np_random.integers(0, 2**62))
        self._episode += 1

        if options and "instance" in options:
            instance, resamples = options["instance"], 0
            if not self._fits(instance):
                raise ValueError("the given instance does not fit the padding sizes")
        else:
            instance, resamples = self._sample_instance()
        self.total_resamples += resamples

        self.sim = Simulator(instance, debug=self.config.debug)
        self._builder = ObservationBuilder(
            self.config.scenario,
            instance,
            self.sim,
            max_patients=self.max_patients,
            max_ops=self.max_ops,
            max_edges=self.max_edges,
            time_scale=self.time_scale,
            graph=self.config.observation == "graph",
            gated_work_prior=self.config.gated_work_prior,
        )
        self._waits = 0
        self._potential = 0.0  # every potential is zero at time zero
        self._over = not self.sim.run_until_decision()
        self._refresh()
        info = self._info()
        info["n_resamples"] = resamples
        info["observation_schema"] = OBSERVATION_SCHEMA_VERSION
        return self._observation, info

    # --- step ----------------------------------------------------------------

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        sim, builder = self.sim, self._builder
        if sim is None or builder is None:
            raise RuntimeError("call reset() before step()")
        if self._over:
            raise RuntimeError("the episode is over; call reset()")
        action = int(action)
        invalid = not (0 <= action < self._mask.size and self._mask[action])
        if invalid:
            if self.config.on_invalid_action == "raise":
                raise InvalidActionError(f"action {action} is masked")
            action = int(np.flatnonzero(self._mask)[0])

        if action == self.wait_action:
            sim.advance()  # no-op if nothing is scheduled
            self._waits += 1
        else:
            slot, rule = divmod(action, len(self._rules))
            op = builder.op_at(slot)
            sim.start(op, self._rules[rule](sim, op))
            self._waits = 0

        terminated = not sim.run_until_decision()
        reason = None
        if not terminated:
            if self._waits >= self.config.max_consecutive_waits:
                reason = "max_consecutive_waits"
            elif (
                self.config.max_episode_time is not None and sim.now >= self.config.max_episode_time
            ):
                reason = "max_episode_time"
        self._over = terminated or reason is not None

        potential = self._reward.potential(sim) + self.leave_penalty * sim.left_weight
        reward = -(potential - self._potential) * self._reward_scale
        self._potential = potential

        self._refresh()
        info = self._info()
        if invalid:
            info["invalid_action"] = True
        if reason is not None:
            info["truncation_reason"] = reason
        if self._over:
            metrics = compute_metrics(sim)
            info["episode"] = {
                "terminated": terminated,
                "time": sim.now,
                "makespan": metrics.makespan,
                "patients_finished": len(metrics.flow_times),
                "total_flow_time": math.fsum(metrics.flow_times),
                "total_weighted_flow_time": math.fsum(
                    w * f for w, f in zip(metrics.weights, metrics.flow_times, strict=True)
                ),
                "total_wait": metrics.total_wait,
                "patients_left": sim.n_left,
                "no_shows": sim.n_no_show,
                "leave_penalty": self.leave_penalty * sim.left_weight,
                "reward_potential": potential,
            }
        return self._observation, float(reward), terminated, reason is not None, info

    # --- masks and views -----------------------------------------------------

    def _refresh(self) -> None:
        sim, builder = self.sim, self._builder
        assert sim is not None and builder is not None
        self._startable = [] if self._over else sim.startable_ops()
        self._observation = builder.build(
            self._startable, self._waits, self.config.max_consecutive_waits
        )
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        if not self._over:
            n_rules = len(self._rules)
            for op in self._startable:
                slot = int(builder.slot_of[op])
                mask[slot * n_rules : (slot + 1) * n_rules] = True
            if self.wait_action is not None:
                running = any(s == OpState.RUNNING for s in sim.state)
                mask[self.wait_action] = self._always_waitable or running or any(sim.in_turnover)
        self._mask = mask

    def action_masks(self) -> NDArray[np.bool_]:
        """Boolean array over actions; ``True`` means the action is valid now."""
        return self._mask.copy()

    def graph(self) -> dict[str, Any]:
        """The current observation as an unpadded heterogeneous graph."""
        assert self._builder is not None
        return self._builder.graph_view(self._observation)

    def _info(self) -> dict[str, Any]:
        assert self.sim is not None
        return {
            "time": self.sim.now,
            "episode_index": self._episode,
            "action_mask": self._mask.copy(),
        }
