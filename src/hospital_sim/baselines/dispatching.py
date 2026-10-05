"""Dispatching rules: which startable operation goes next.

All rules are non-idling (they never wait while something can start) and use
only what a scheduler may know: visible operations, expected durations,
ready times and the public attributes of patients who have arrived.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import special

from hospital_sim.domain.model import ProblemInstance, Scenario
from hospital_sim.sim.allocation import RULES
from hospital_sim.sim.simulator import OpState, Simulator

# (simulator, startable operations in ready order) -> operation to start
Dispatch = Callable[[Simulator, list[int]], int]
_NOT_STARTED = (OpState.WAITING, OpState.READY)


def remaining_work(sim: Simulator, patient: int) -> float:
    """Expected duration of the patient's visible operations not yet started."""
    return sum(
        sim.expected_duration[sim.type_of[op.id]]
        for op in sim.instance.patients[patient].operations
        if sim.state[op.id] in _NOT_STARTED
    )


def fifo(sim: Simulator, startable: list[int]) -> int:
    """Longest-ready operation first."""
    return startable[0]


def spt(sim: Simulator, startable: list[int]) -> int:
    """Shortest expected processing time first."""
    return min(startable, key=lambda op: sim.expected_duration[sim.type_of[op]])


def longest_remaining_work(sim: Simulator, startable: list[int]) -> int:
    """The patient with the most expected work still to do goes first."""
    return max(startable, key=lambda op: remaining_work(sim, sim.patient_of[op]))


def most_constrained_first(sim: Simulator, startable: list[int]) -> int:
    """The operation that is hardest to staff goes first: the largest team,
    then the least spare capacity in its tightest role. Large teams starve
    if small ones keep taking their members."""

    def key(op: int) -> tuple[int, float]:
        reqs = sim.resources.requirements[sim.type_of[op]]
        idle = sim.resources.idle_count
        slack = min((idle[role] / qty for role, qty in reqs), default=float("inf"))
        return (-sum(qty for _, qty in reqs), slack)

    return min(startable, key=key)


def urgent_first(sim: Simulator, startable: list[int]) -> int:
    """Urgent patients first; longest-ready within each group."""
    patients = sim.instance.patients
    return min(startable, key=lambda op: not patients[sim.patient_of[op]].urgent)


def weight_aware(sim: Simulator, startable: list[int]) -> int:
    """Weighted shortest remaining work: highest patient weight per unit of
    expected work left (Smith's rule applied to the patient)."""

    def key(op: int) -> float:
        patient = sim.patient_of[op]
        return -sim.weight[patient] / max(remaining_work(sim, patient), 1e-9)

    return min(startable, key=key)


def acuity_duration_factors(scenario: Scenario) -> list[list[float]]:
    """``factors[acuity class][operation type]``: how much longer than its
    configured mean an operation is expected to take for a patient of that
    visible class.

    Derived from public scenario parameters only. The class is a noisy reading
    of the hidden complexity, so E[complexity | class] is known; complexity
    shifts the log-duration of every operation by sigma * sqrt(rho) per unit.
    """
    complexity = scenario.complexity
    k = complexity.acuity_classes
    if k < 2:
        return []
    edges = [
        float(special.ndtri(i / k)) if 0 < i < k else math.copysign(math.inf, i - 0.5)
        for i in range(k + 1)
    ]

    def pdf(x: float) -> float:
        return 0.0 if math.isinf(x) else math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    # Mean of a standard normal within each equal-probability class, times
    # the correlation between the reading and the true complexity.
    expected_complexity = [
        complexity.acuity_correlation * (pdf(edges[c]) - pdf(edges[c + 1])) * k for c in range(k)
    ]
    factors = []
    for z in expected_complexity:
        row = []
        for op_type in scenario.operation_types:
            cv = op_type.duration.cv
            sigma = math.sqrt(math.log1p(cv * cv)) if cv > 0 else 0.0
            row.append(math.exp(sigma * math.sqrt(complexity.rho) * z))
        factors.append(row)
    return factors


def acuity_aware(scenario: Scenario) -> Dispatch:
    """``weight_aware`` with remaining work predicted from the visible acuity
    class: a high-acuity patient is expected to be slower at everything.

    Comparing it with ``weight_aware`` measures what the observed class is
    worth to a simple rule. Without acuity classes, or with rho = 0, the two
    are identical.
    """
    factors = acuity_duration_factors(scenario)

    def choose(sim: Simulator, startable: list[int]) -> int:
        patients = sim.instance.patients

        def key(op: int) -> float:
            patient = sim.patient_of[op]
            acuity = patients[patient].acuity_class
            if acuity is None or not factors:
                work = remaining_work(sim, patient)
            else:
                work = sum(
                    sim.expected_duration[sim.type_of[o.id]] * factors[acuity][sim.type_of[o.id]]
                    for o in patients[patient].operations
                    if sim.state[o.id] in _NOT_STARTED
                )
            return -sim.weight[patient] / max(work, 1e-9)

        return min(startable, key=key)

    return choose


def random_dispatch(seed: int) -> Dispatch:
    """Uniformly random among startable operations; reproducible per seed."""
    rng = np.random.default_rng(seed)

    def choose(sim: Simulator, startable: list[int]) -> int:
        return startable[int(rng.integers(len(startable)))]

    return choose


DISPATCHERS: dict[str, Dispatch] = {
    "fifo": fifo,
    "spt": spt,
    "longest_remaining_work": longest_remaining_work,
    "most_constrained_first": most_constrained_first,
    "urgent_first": urgent_first,
    "weight_aware": weight_aware,
}
# Rules that need the scenario's public parameters to be built.
DISPATCHER_FACTORIES: dict[str, Callable[[Scenario], Dispatch]] = {
    "acuity_aware": acuity_aware,
}


@dataclass(frozen=True)
class Policy:
    """A dispatching rule paired with an allocation rule."""

    dispatcher: str  # a key of DISPATCHERS or DISPATCHER_FACTORIES, or "random"
    allocation: str = "flexible"  # a key of sim.allocation.RULES

    @property
    def name(self) -> str:
        return f"{self.dispatcher}+{self.allocation}"

    def run(
        self, instance: ProblemInstance, *, nominal: Scenario | None = None, debug: bool = False
    ) -> Simulator:
        """Simulate ``instance`` to completion under this policy.

        ``nominal`` is the scenario as configured. Pass it when the instance
        was generated with randomized noise parameters, so that a rule which
        uses scenario parameters sees the configured ones, not the drawn ones.
        """
        sim = Simulator(instance, debug=debug)
        if self.dispatcher == "random":
            # Seeded from the instance, so the random policy is reproducible
            # and sees the same draws whatever it is compared with.
            dispatch = random_dispatch(instance.seed % (2**63))
        elif self.dispatcher in DISPATCHER_FACTORIES:
            dispatch = DISPATCHER_FACTORIES[self.dispatcher](nominal or instance.scenario)
        else:
            dispatch = DISPATCHERS[self.dispatcher]
        allocate = RULES[self.allocation]
        while sim.run_until_decision():
            op = dispatch(sim, sim.startable_ops())
            sim.start(op, allocate(sim, op))
        return sim


def all_policies(allocation: str = "flexible") -> list[Policy]:
    return [Policy(name, allocation) for name in ("random", *DISPATCHERS, *DISPATCHER_FACTORIES)]
