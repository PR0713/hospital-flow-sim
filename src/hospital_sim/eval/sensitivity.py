"""Noise-source sensitivity analysis.

Start from a scenario with every source of randomness on, switch one source
off at a time, and run the same policies on the same seeds. For each source
this shows how much the outcomes move, and whether the ranking of policies
changes. A source that moves neither is cosmetic for policy comparison.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from hospital_sim.baselines.dispatching import Policy
from hospital_sim.domain.model import (
    ArrivalKind,
    ArrivalSpec,
    DurationKind,
    DurationSpec,
    Scenario,
)
from hospital_sim.eval.harness import Evaluation, _mean_ci, evaluate
from hospital_sim.sim.durations import make_distribution


def _no_duration_noise(scenario: Scenario) -> Scenario:
    """Every operation takes exactly its expected duration."""
    return dataclasses.replace(
        scenario,
        operation_types=tuple(
            dataclasses.replace(
                t,
                duration=DurationSpec(
                    DurationKind.DETERMINISTIC, make_distribution(t.duration).mean
                ),
            )
            for t in scenario.operation_types
        ),
    )


def _no_latent_factor(scenario: Scenario) -> Scenario:
    """Durations of one patient are independent, and so are revealed steps."""
    return dataclasses.replace(
        scenario,
        complexity=dataclasses.replace(scenario.complexity, rho=0.0, reveal_correlation=0.0),
    )


def _regular(spec: ArrivalSpec) -> ArrivalSpec:
    if spec.kind is ArrivalKind.SCHEDULED:
        return dataclasses.replace(spec, punctuality_sd=0.0)
    if spec.kind is not ArrivalKind.POISSON:
        return spec
    rate = sum(spec.rate_profile) / len(spec.rate_profile) if spec.rate_profile else spec.rate
    assert rate is not None
    if spec.horizon is not None:
        n = max(round(rate * spec.horizon), 1)
        interval = spec.horizon / n
    else:
        assert spec.n_patients is not None
        n, interval = spec.n_patients, 1.0 / rate
    # Same expected number of patients, evenly spaced and exactly on time.
    return dataclasses.replace(
        spec,
        kind=ArrivalKind.SCHEDULED,
        n_patients=n,
        horizon=None,
        rate=None,
        rate_profile=(),
        profile_period=None,
        batch_mean=1.0,
        appointment_start=interval / 2.0,
        appointment_interval=interval,
        punctuality_sd=0.0,
        no_show_probability=0.0,
    )


def _no_arrival_variability(scenario: Scenario) -> Scenario:
    """Arrivals evenly spaced and punctual, with the same expected numbers.

    Side effect: these patients become a visible booking list. The baselines
    do not use that list, so it does not affect this analysis.
    """
    return dataclasses.replace(scenario, arrivals=tuple(_regular(a) for a in scenario.arrivals))


def _steady_arrivals(scenario: Scenario) -> Scenario:
    """Cleaner partial variant: arrivals stay random and unannounced, but
    without bursts or a time-of-day profile (a constant-rate Poisson stream
    with the same mean)."""

    def steady(spec: ArrivalSpec) -> ArrivalSpec:
        if spec.kind is not ArrivalKind.POISSON:
            return spec
        rate = sum(spec.rate_profile) / len(spec.rate_profile) if spec.rate_profile else spec.rate
        return dataclasses.replace(
            spec, rate=rate, rate_profile=(), profile_period=None, batch_mean=1.0
        )

    return dataclasses.replace(scenario, arrivals=tuple(steady(a) for a in scenario.arrivals))


def _no_urgent_stream(scenario: Scenario) -> Scenario:
    return dataclasses.replace(
        scenario, arrivals=tuple(a for a in scenario.arrivals if not a.urgent)
    )


def _no_rosters(scenario: Scenario) -> Scenario:
    """Everyone is on duty around the clock: no shifts, breaks or absences."""
    return dataclasses.replace(
        scenario,
        resources=tuple(dataclasses.replace(r, calendar=None) for r in scenario.resources),
    )


def _no_breakdowns(scenario: Scenario) -> Scenario:
    return dataclasses.replace(
        scenario,
        resources=tuple(dataclasses.replace(r, breakdown=None) for r in scenario.resources),
    )


def _no_reveal(scenario: Scenario) -> Scenario:
    """Nothing is discovered mid-stay: optional steps are known on arrival and
    there are no repeats (so the repeat workload disappears too)."""
    return dataclasses.replace(
        scenario,
        pathways=tuple(
            dataclasses.replace(
                p,
                steps=tuple(
                    dataclasses.replace(s, revealed_by=None, repeat_probability=0.0, repeat_max=0)
                    for s in p.steps
                ),
            )
            for p in scenario.pathways
        ),
    )


def _reveal_visible(scenario: Scenario) -> Scenario:
    """Cleaner partial variant: the information effect alone. Gated steps are
    known on arrival; repeats (and their workload) stay."""
    return dataclasses.replace(
        scenario,
        pathways=tuple(
            dataclasses.replace(
                p, steps=tuple(dataclasses.replace(s, revealed_by=None) for s in p.steps)
            )
            for p in scenario.pathways
        ),
    )


def _no_repeats(scenario: Scenario) -> Scenario:
    """Cleaner partial variant: the workload effect alone. No repeats; gated
    steps stay hidden."""
    return dataclasses.replace(
        scenario,
        pathways=tuple(
            dataclasses.replace(
                p,
                steps=tuple(
                    dataclasses.replace(s, repeat_probability=0.0, repeat_max=0) for s in p.steps
                ),
            )
            for p in scenario.pathways
        ),
    )


def _no_no_shows(scenario: Scenario) -> Scenario:
    return dataclasses.replace(
        scenario,
        arrivals=tuple(dataclasses.replace(a, no_show_probability=0.0) for a in scenario.arrivals),
    )


def _no_leaving(scenario: Scenario) -> Scenario:
    return dataclasses.replace(
        scenario,
        arrivals=tuple(dataclasses.replace(a, patience_mean=None) for a in scenario.arrivals),
    )


# Each entry switches one source off. The results are not re-validated; they
# only remove things from a scenario that was valid.
SOURCES: dict[str, Callable[[Scenario], Scenario]] = {
    "duration_noise": _no_duration_noise,
    "latent_factor": _no_latent_factor,
    "arrival_variability": _no_arrival_variability,
    "urgent_stream": _no_urgent_stream,
    "rosters_and_absences": _no_rosters,
    "breakdowns": _no_breakdowns,
    "reveal": _no_reveal,
    "no_shows": _no_no_shows,
    "leaving": _no_leaving,
}
# Narrower switches that separate the two effects mixed in "arrival
# variability" and in "reveal". Not part of the default study.
PARTIAL_SOURCES: dict[str, Callable[[Scenario], Scenario]] = {
    "arrival_bursts_and_profile": _steady_arrivals,
    "reveal_information_only": _reveal_visible,
    "repeat_workload_only": _no_repeats,
}
# Not a property of the world but of the objective: handled without a re-run.
LEAVE_PENALTY = "leave_penalty"

KPIS = ("cost", "mean_flow_time", "makespan", "mean_op_wait", "left")


def ablate(scenario: Scenario, source: str) -> Scenario:
    """``scenario`` with one noise source switched off."""
    return {**SOURCES, **PARTIAL_SOURCES}[source](scenario)


def kpi(result: Evaluation, policy: str, name: str, leave_penalty: float) -> np.ndarray:
    """Per-episode values. ``cost`` is the default objective per unit of
    patient weight: weighted mean flow time plus the leave penalty. Dividing
    by weight keeps it comparable when a source changes how many patients come."""
    if name != "cost":
        return result.series(policy, name)
    weight = np.maximum(result.series(policy, "arrived_weight"), 1e-9)
    total = result.series(policy, "total_weighted_flow_time") + leave_penalty * result.series(
        policy, "left_weight"
    )
    return np.asarray(total / weight)


@dataclass(frozen=True)
class SourceEffect:
    source: str
    # KPI -> (mean of [on - off] for the reference policy, CI half-width). A
    # positive number means the source makes that KPI larger.
    effect: dict[str, tuple[float, float]]
    # Kendall's tau between the policies' mean cost with the source on and off.
    tau: float
    best_when_off: str
    # Mean cost of each policy with the source off.
    cost_when_off: dict[str, float]


@dataclass(frozen=True)
class SensitivityResult:
    scenario: str
    n_episodes: int
    leave_penalty: float
    reference: str  # best policy by cost with every source on
    cost_all_on: dict[str, float]
    effects: list[SourceEffect]


def run_sensitivity(
    scenario: Scenario,
    policies: Sequence[Policy],
    n_episodes: int,
    leave_penalty: float,
    *,
    base_seed: int = 0,
    sources: Sequence[str] | None = None,
) -> SensitivityResult:
    names = [p.name for p in policies]
    base = evaluate(scenario, policies, n_episodes, base_seed=base_seed)

    def mean_cost(result: Evaluation, penalty: float) -> dict[str, float]:
        return {n: float(kpi(result, n, "cost", penalty).mean()) for n in names}

    on = mean_cost(base, leave_penalty)
    reference = min(on, key=lambda n: on[n])

    def compare(source: str, off: Evaluation, off_penalty: float) -> SourceEffect:
        cost_off = mean_cost(off, off_penalty)
        effect = {}
        for name in KPIS:
            diff = kpi(base, reference, name, leave_penalty) - kpi(
                off, reference, name, off_penalty
            )
            mean, _, half = _mean_ci(diff, base.confidence)
            effect[name] = (mean, half)
        tau = float(
            stats.kendalltau([on[n] for n in names], [cost_off[n] for n in names]).statistic
        )
        return SourceEffect(source, effect, tau, min(cost_off, key=lambda n: cost_off[n]), cost_off)

    effects = []
    for source in sources if sources is not None else SOURCES:
        if source == LEAVE_PENALTY:
            continue
        off = evaluate(ablate(scenario, source), policies, n_episodes, base_seed=base_seed)
        effects.append(compare(source, off, leave_penalty))
    if sources is None or LEAVE_PENALTY in sources:
        effects.append(compare(LEAVE_PENALTY, base, 0.0))  # same episodes, penalty removed
    return SensitivityResult(scenario.name, n_episodes, leave_penalty, reference, on, effects)


def format_report(result: SensitivityResult) -> str:
    """Markdown tables for ``docs/SENSITIVITY.md``."""
    lines = [
        f"Reference policy (lowest cost with everything on): `{result.reference}`.",
        "",
        "Effect of each source on the reference policy: mean of (source on − source off)",
        "over the same seeds, ± 95% CI half-width. Positive means the source increases the KPI.",
        "",
        "| Source | " + " | ".join(KPIS) + " |",
        "|---|" + "---|" * len(KPIS),
    ]
    for e in result.effects:
        cells = [f"{e.effect[k][0]:+.2f} ± {e.effect[k][1]:.2f}" for k in KPIS]
        lines.append(f"| {e.source} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Effect on the ranking of policies by mean cost. Kendall's tau compares the order",
        "with the source on against the order with it off (1 = same order).",
        "",
        "| Source | Kendall's tau | Best policy when off | Cost spread across policies when off |",
        "|---|---|---|---|",
    ]
    spread_on = max(result.cost_all_on.values()) - min(result.cost_all_on.values())
    for e in result.effects:
        spread = max(e.cost_when_off.values()) - min(e.cost_when_off.values())
        lines.append(f"| {e.source} | {e.tau:.2f} | {e.best_when_off} | {spread:.1f} |")
    lines += ["", f"Cost spread across policies with everything on: {spread_on:.1f}.", ""]
    return "\n".join(lines)
