"""Pluggable rewards.

Every reward is defined by a potential: a quantity that only grows as the
episode unfolds and whose final value is the metric being minimised. The step
reward is minus its increase since the previous decision, so the rewards of
an episode sum exactly to minus the final metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hospital_sim.sim.simulator import Simulator


class Reward(Protocol):
    def potential(self, sim: Simulator) -> float:
        """Cost accrued from time zero up to ``sim.now``."""
        ...


@dataclass(frozen=True)
class FlowTime:
    """Total time patients spend in the system, weighted by patient weight
    (so urgent classes count more). Final value: sum of weight x flow time."""

    weighted: bool = True

    def potential(self, sim: Simulator) -> float:
        return sim.flow_accrued(self.weighted)


@dataclass(frozen=True)
class Makespan:
    """Elapsed time. Final value: the makespan. Meant for static batches."""

    def potential(self, sim: Simulator) -> float:
        return sim.now


@dataclass(frozen=True)
class Combined:
    """Weighted sum of other rewards: ``terms = [(coefficient, reward), ...]``."""

    terms: Sequence[tuple[float, Reward]]

    def potential(self, sim: Simulator) -> float:
        return sum(coefficient * reward.potential(sim) for coefficient, reward in self.terms)


REWARDS: dict[str, Reward] = {
    "weighted_flow_time": FlowTime(weighted=True),
    "flow_time": FlowTime(weighted=False),
    "makespan": Makespan(),
}
