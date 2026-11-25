"""Pointer-style actions: choose among the currently valid (operation, rule) pairs.

The base ``HospitalEnv`` action is ``slot * n_rules + rule`` over every padded
observation row, of which well under 1% are valid at a time. This optional
wrapper turns the valid pairs into a short, padded candidate list and makes
the action an index into it:

* actions ``0 .. K-1`` pick candidate ``k``; action ``K`` is wait;
* ``obs["candidate_features"][k]`` describes candidate ``k`` (its operation
  row, its patient row, its requirement vector and a one-hot of its rule);
  ``obs["candidate_mask"]`` (length ``K + 1``, wait last) says which actions
  are valid, as does ``action_masks()``.

**Ordering.** Candidates are first ordered oldest-ready first (largest
``waited``; ties by observation row, then rule). If there are more than
``K``, the ``K`` oldest are kept and ``info["candidates_dropped"]`` counts the
rest. With ``shuffle_candidates=True`` (default) the kept candidates are then
put in a random order at every step, so a policy cannot learn a position
bias such as "always take the first". The shuffle is seeded by ``reset(seed)``
and is reproducible. With ``shuffle_candidates=False`` the order above is
kept, which makes index 0 the FIFO choice.

The base environment is unchanged; its own mask is in
``info["base_action_mask"]``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from hospital_sim.env.hospital_env import HospitalEnv, InvalidActionError
from hospital_sim.env.observation import OP_FEATURES, PATIENT_FEATURES
from hospital_sim.env.spaces import MaskedDiscrete

_WAITED = OP_FEATURES.index("waited")


class PointerActionWrapper(gym.Wrapper[dict[str, Any], Any, dict[str, Any], Any]):
    """See the module docstring. ``env`` may be wrapped (e.g. by ``gym.make``)
    as long as ``env.unwrapped`` is a ``HospitalEnv``."""

    def __init__(
        self,
        env: gym.Env[dict[str, Any], Any],
        max_candidates: int = 128,
        shuffle_candidates: bool = True,
        keep_base_observation: bool = True,
    ) -> None:
        super().__init__(env)
        base = env.unwrapped
        if not isinstance(base, HospitalEnv):
            raise TypeError("PointerActionWrapper needs a HospitalEnv underneath")
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.base = base
        self.max_candidates = max_candidates
        self.shuffle_candidates = shuffle_candidates
        self.keep_base_observation = keep_base_observation
        self.n_rules = len(base.config.allocation_rules)
        self.wait_index = max_candidates
        self._n_start_actions = base.max_ops * self.n_rules

        scenario = base.config.scenario
        self.feature_names = (
            *(f"op.{f}" for f in OP_FEATURES),
            *(f"patient.{f}" for f in PATIENT_FEATURES),
            *(f"requires.{role.name}" for role in scenario.roles),
            *(f"rule.{rule}" for rule in base.config.allocation_rules),
        )
        base_space = base.observation_space

        def box(key: str) -> spaces.Box:
            space = base_space[key]
            assert isinstance(space, spaces.Box)
            return space

        low = np.concatenate(
            [
                box("ops").low[0],
                box("patients").low[0],
                box("op_requirements").low[0],
                np.zeros(self.n_rules, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                box("ops").high[0],
                box("patients").high[0],
                box("op_requirements").high[0],
                np.ones(self.n_rules, dtype=np.float32),
            ]
        )
        k = max_candidates
        n_types = len(scenario.operation_types)
        candidate_spaces: dict[str, spaces.Space[Any]] = {
            "candidate_features": spaces.Box(
                np.tile(low, (k, 1)), np.tile(high, (k, 1)), dtype=np.float32
            ),
            "candidate_mask": spaces.Box(0, 1, (k + 1,), np.int8),
            "candidate_slot": spaces.Box(-1, base.max_ops - 1, (k,), np.int32),
            "candidate_rule": spaces.Box(-1, self.n_rules - 1, (k,), np.int32),
            "candidate_op_type": spaces.Box(-1, n_types - 1, (k,), np.int32),
        }
        kept = (
            dict(base_space.spaces)
            if keep_base_observation
            else {"global": box("global")}
        )
        self.observation_space = spaces.Dict({**kept, **candidate_spaces})
        # Kept separately: Wrapper exposes ``action_space`` as a plain Space.
        self._space = MaskedDiscrete(k + 1)
        self.action_space = self._space
        self._feature_shape = (k, len(self.feature_names))

        self._rng: np.random.Generator | None = None
        self._actions = np.zeros(0, dtype=np.int64)  # base action per candidate
        self._mask = np.zeros(k + 1, dtype=np.bool_)

    # --- gymnasium API -----------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(11,)))
        elif self._rng is None:
            self._rng = np.random.default_rng(int(self.base.np_random.integers(2**63)))
        return self._wrap(obs, info)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = int(action)
        invalid = not (0 <= action < self._mask.size and self._mask[action])
        if invalid:
            stale = self._space.is_stale_sample(action)
            if self.base.config.on_invalid_action == "raise" and not stale:
                raise InvalidActionError(f"pointer action {action} is masked")
            action = int(np.flatnonzero(self._mask)[0])
        if action == self.wait_index:
            assert self.base.wait_action is not None
            base_action = self.base.wait_action
        else:
            base_action = int(self._actions[action])
        obs, reward, terminated, truncated, info = self.env.step(base_action)
        wrapped, info = self._wrap(obs, info)
        if invalid:
            info["invalid_action"] = True
        return wrapped, float(reward), terminated, truncated, info

    def action_masks(self) -> NDArray[np.bool_]:
        """Boolean array over the ``K + 1`` pointer actions (wait last)."""
        return self._mask.copy()

    def candidates(self) -> list[tuple[int, int]]:
        """(observation row, rule index) of each current candidate, in position order."""
        return [(int(a) // self.n_rules, int(a) % self.n_rules) for a in self._actions]

    # --- internals -------------------------------------------------------------

    def _wrap(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_mask = self.base.action_masks()
        actions = np.flatnonzero(base_mask[: self._n_start_actions])
        slots, rules = actions // self.n_rules, actions % self.n_rules
        waited = obs["ops"][slots, _WAITED] if len(actions) else np.zeros(0)
        order = np.lexsort((rules, slots, -waited))  # last key is the primary one
        k = self.max_candidates
        dropped = max(len(actions) - k, 0)
        actions = actions[order][:k]
        if self.shuffle_candidates and len(actions) > 1:
            assert self._rng is not None
            actions = actions[self._rng.permutation(len(actions))]
        self._actions = actions
        slots, rules = actions // self.n_rules, actions % self.n_rules
        n = len(actions)

        features = np.zeros(self._feature_shape, np.float32)
        if n:
            patient_rows = obs["op_patient"][slots]
            features[:n] = np.concatenate(
                [
                    obs["ops"][slots],
                    obs["patients"][patient_rows],
                    obs["op_requirements"][slots],
                    np.eye(self.n_rules, dtype=np.float32)[rules],
                ],
                axis=1,
            )
        mask = np.zeros(k + 1, dtype=np.bool_)
        mask[:n] = True
        if self.base.wait_action is not None:
            mask[self.wait_index] = bool(base_mask[self.base.wait_action])
        self._mask = mask
        self._space.set_valid(mask)

        def padded(values: NDArray[Any]) -> NDArray[np.int32]:
            out = np.full(k, -1, dtype=np.int32)
            out[:n] = values
            return out

        wrapped = dict(obs) if self.keep_base_observation else {"global": obs["global"]}
        wrapped.update(
            candidate_features=features,
            candidate_mask=mask.astype(np.int8),
            candidate_slot=padded(slots),
            candidate_rule=padded(rules),
            candidate_op_type=padded(obs["op_type"][slots] if n else np.zeros(0)),
        )
        info = dict(info)
        info["base_action_mask"] = info.get("action_mask", base_mask)
        info["action_mask"] = mask.copy()
        info["candidates_dropped"] = dropped
        return wrapped, info
