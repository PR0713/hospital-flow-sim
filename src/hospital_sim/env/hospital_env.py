"""Gymnasium environment over the core simulator. No training code."""

from __future__ import annotations

import logging
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
    feature_bounds,
)
from hospital_sim.env.rewards import REWARDS, Reward
from hospital_sim.env.spaces import MaskedDiscrete
from hospital_sim.generators.instance import generate_instance
from hospital_sim.sim.allocation import RULES
from hospital_sim.sim.durations import StandardDurationModel
from hospital_sim.sim.metrics import compute_metrics, episode_outcome
from hospital_sim.sim.simulator import OpState, PatientStatus, Simulator

_MAX_RESAMPLES = 1000
_log = logging.getLogger("hospital_sim.env")
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
        # Observation cells this episode that were NaN or infinite and replaced
        # by 0 (always 0 unless something is wrong; debug mode raises instead).
        "nonfinite_obs_count",
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
    action_space: MaskedDiscrete
    observation_space: spaces.Dict

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

        # Waiting is offered only for public reasons that something can still
        # happen (see ``_wait_is_valid``). Rosters and breakdowns go on forever.
        self._rosters = bool(scenario.calendars) or any(
            r.breakdown is not None for r in scenario.resources
        )

        n_rules = len(self._rules)
        self.wait_action = self.max_ops * n_rules if config.allow_wait else None
        self.n_actions = self.max_ops * n_rules + int(config.allow_wait)
        # ``sample()`` draws a currently valid action (see env/spaces.py).
        self.action_space = MaskedDiscrete(self.n_actions)

        n_res, n_roles = len(scenario.resources), len(scenario.roles)
        n_types = len(scenario.operation_types)

        scheduled = [a for a in scenario.arrivals if a.kind is ArrivalKind.SCHEDULED]
        bounds = feature_bounds(
            max_ops=self.max_ops,
            max_patients=self.max_patients,
            max_speed=max(float(r.attributes.get("speed", 1.0)) for r in scenario.resources),
            max_members=max(len(m) for m in scenario.resources_by_role.values()),
            min_scheduled_in=-max((a.no_show_timeout for a in scheduled), default=0.0)
            / self.time_scale,
        )

        def table(rows: int, name: str, features: tuple[str, ...]) -> spaces.Box:
            low = np.array([bounds[name][f][0] for f in features], dtype=np.float32)
            high = np.array([bounds[name][f][1] for f in features], dtype=np.float32)
            return spaces.Box(
                np.tile(low, (rows, 1)), np.tile(high, (rows, 1)), dtype=np.float32
            )

        max_quantity = max(
            (req.quantity for t in scenario.operation_types for req in t.requirements),
            default=0,
        )

        obs_spaces: dict[str, spaces.Space[Any]] = {
            "ops": table(self.max_ops, "ops", OP_FEATURES),
            "op_valid": spaces.Box(0, 1, (self.max_ops,), np.int8),
            "op_type": spaces.Box(-1, n_types - 1, (self.max_ops,), np.int32),
            "op_patient": spaces.Box(-1, self.max_patients - 1, (self.max_ops,), np.int32),
            "op_requirements": spaces.Box(
                0.0, float(max_quantity), (self.max_ops, n_roles), np.float32
            ),
            "patients": table(self.max_patients, "patients", PATIENT_FEATURES),
            "patient_valid": spaces.Box(0, 1, (self.max_patients,), np.int8),
            "resources": table(n_res, "resources", RESOURCE_FEATURES),
            "resource_roles": spaces.Box(0, 1, (n_res, n_roles), np.int8),
            "resource_op": spaces.Box(-1, self.max_ops - 1, (n_res,), np.int32),
            "roles": table(n_roles, "roles", ROLE_FEATURES),
            "global": spaces.Box(
                np.array([bounds["global"][f][0] for f in GLOBAL_FEATURES], dtype=np.float32),
                np.array([bounds["global"][f][1] for f in GLOBAL_FEATURES], dtype=np.float32),
                dtype=np.float32,
            ),
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
        self._nonfinite = 0
        self._warned_nonfinite = False

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
            debug=self.config.debug,
        )
        self._nonfinite = 0
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
            # An action drawn by ``action_space.sample()`` for an earlier state
            # (sampled, then reset, then stepped - as gymnasium's check_env
            # does) is not an agent error: run the lowest valid action. Any
            # other masked action raises in the default mode.
            stale = self.action_space.is_stale_sample(action)
            if self.config.on_invalid_action == "raise" and not stale:
                raise InvalidActionError(f"action {action} is masked")
            action = int(np.flatnonzero(self._mask)[0])

        if action == self.wait_action:
            if not sim.advance():
                # Nothing is scheduled, yet by public information an arrival
                # may still come: let time pass to the earliest public bound
                # (an arrival horizon). Never a no-op, and it reveals nothing
                # that waiting in the real system would not.
                bound = self._public_arrival_bound()
                if bound is None:  # the mask rules this out
                    raise RuntimeError("wait with nothing that could happen")
                sim.idle_until(bound)
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
        truncation_cost = 0.0
        if reason is not None and self.config.truncation_cost == "expected_remaining":
            # One final term for the cost still to come (lower bound, public
            # information only). Only for learners that treat truncation as
            # terminal; a bootstrapping learner should keep the default "none".
            truncation_cost = math.fsum(
                sim.weight[patient] * bound
                for patient, bound in builder.remaining_flow_lower_bound().items()
            )
            reward -= truncation_cost * self._reward_scale
        info = self._info()
        if invalid:
            info["invalid_action"] = True
        if reason is not None:
            info["truncation_reason"] = reason
        if self._over:
            # Costs count patients still in the system with the time they have
            # spent so far, so a truncated episode is never scored as cheaper
            # than what it has accrued. Unchanged for terminated episodes.
            metrics = compute_metrics(sim)
            outcome = episode_outcome(sim, metrics)
            info["episode"] = {
                "terminated": terminated,
                "truncated": reason is not None,
                "time": sim.now,
                "makespan": outcome.makespan,
                "patients_finished": len(metrics.flow_times),
                "unfinished_patients": outcome.unfinished,
                "total_flow_time": outcome.flow_time,
                "total_weighted_flow_time": outcome.weighted_flow_time,
                "unfinished_flow_time": outcome.unfinished_flow_time,
                "unfinished_weighted_flow_time": outcome.unfinished_weighted_flow_time,
                "total_wait": outcome.total_wait,
                "patients_left": sim.n_left,
                "no_shows": sim.n_no_show,
                "leave_penalty": self.leave_penalty * sim.left_weight,
                "reward_potential": potential,
                # Unscaled; the final reward already includes -scale * this.
                "truncation_cost": truncation_cost,
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
        if builder.nonfinite:
            self._nonfinite += builder.nonfinite
            if not self._warned_nonfinite:
                self._warned_nonfinite = True
                _log.warning(
                    "observation had %d non-finite values (replaced by 0) at t=%s; "
                    "counted in info['nonfinite_obs_count']. Run with debug=True to locate.",
                    builder.nonfinite,
                    sim.now,
                )
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        if not self._over:
            n_rules = len(self._rules)
            for op in self._startable:
                slot = int(builder.slot_of[op])
                mask[slot * n_rules : (slot + 1) * n_rules] = True
            if self.wait_action is not None:
                mask[self.wait_action] = self._wait_is_valid()
        self._mask = mask
        self.action_space.set_valid(mask)

    def _wait_is_valid(self) -> bool:
        """Waiting can change something, for a reason the agent can know:
        an operation is running, a resource is in turnover, the scenario has
        rosters or breakdowns, or an arrival stream is not provably over."""
        sim = self.sim
        assert sim is not None
        return (
            self._rosters
            or any(sim.in_turnover)
            or any(s == OpState.RUNNING for s in sim.state)
            or self._arrivals_may_come()
        )

    def _arrivals_may_come(self) -> bool:
        """From public information only: may a patient still arrive?

        Static patients are all there at time zero. A booked stream is over
        once every appointment has resolved (arrived or no-show). A Poisson
        stream is over past its horizon or once its patient cap has arrived.
        Whether an arrival is actually still drawn is hidden and not used.
        """
        sim = self.sim
        assert sim is not None
        patients = sim.instance.patients
        known: dict[str, int] = {}
        expected: set[str] = set()
        for p in sim.visible_patients:  # public: the booking list and arrived walk-ins
            name = patients[p].patient_class
            if sim.patient_status[p] == PatientStatus.EXPECTED:
                expected.add(name)
            elif sim.arrived[p]:
                known[name] = known.get(name, 0) + 1
        for spec in sim.instance.scenario.arrivals:
            if spec.kind is ArrivalKind.SCHEDULED and spec.name in expected:
                return True
            if spec.kind is ArrivalKind.POISSON:
                past_horizon = spec.horizon is not None and sim.now >= spec.horizon
                capped = spec.n_patients is not None and known.get(spec.name, 0) >= spec.n_patients
                if not (past_horizon or capped):
                    return True
        return False

    def _public_arrival_bound(self) -> float | None:
        """The earliest horizon of a Poisson stream that is not provably over."""
        sim = self.sim
        assert sim is not None
        if not self._arrivals_may_come():
            return None
        horizons = [
            spec.horizon
            for spec in sim.instance.scenario.arrivals
            if spec.kind is ArrivalKind.POISSON
            and spec.horizon is not None
            and spec.horizon > sim.now
        ]
        return min(horizons) if horizons else None

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
            "nonfinite_obs_count": self._nonfinite,
        }
