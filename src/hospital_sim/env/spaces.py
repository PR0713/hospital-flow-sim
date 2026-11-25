"""Action space whose ``sample()`` respects the current action mask."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class MaskedDiscrete(spaces.Discrete):
    """A ``Discrete`` space whose ``sample()`` draws a currently valid action.

    The environment calls ``set_valid`` after every reset and step, so
    ``env.action_space.sample()`` is a valid action for the state it was drawn
    in. An explicit ``mask`` or ``probability`` behaves exactly as in
    ``Discrete``; before the first reset (no valid action known) sampling is
    uniform. Equality, batching and pickling are those of ``Discrete``: the
    space holds plain arrays, never a reference to the environment.

    ``is_stale_sample`` lets the environment recognise one specific mistake
    that tools such as ``gymnasium.utils.env_checker.check_env`` make: drawing
    ``sample()`` and then resetting before stepping with it.
    """

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, seed=seed)
        self.valid: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
        # Bumped by every ``set_valid``; tells which state a sample was drawn for.
        self.version = 0
        self._last_sample: tuple[int, int] | None = None  # (action, version)

    def set_valid(self, mask: NDArray[np.bool_]) -> None:
        self.valid = mask
        self.version += 1

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> np.int64:
        if mask is None and probability is None and self.valid.any():
            action = super().sample(mask=self.valid.astype(np.int8))
            self._last_sample = (int(action), self.version)
            return np.int64(action)
        return np.int64(super().sample(mask=mask, probability=probability))

    def is_stale_sample(self, action: int) -> bool:
        """``action`` is what ``sample()`` returned for an earlier state."""
        last = self._last_sample
        return last is not None and last[0] == action and last[1] != self.version

    def __repr__(self) -> str:
        return f"MaskedDiscrete({self.n})"
