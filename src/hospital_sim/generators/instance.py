"""Turn a ``Scenario`` plus a seed into a concrete ``ProblemInstance``."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import special

from hospital_sim.domain.dag import contract
from hospital_sim.domain.model import (
    ArrivalKind,
    ArrivalSpec,
    ComplexitySpec,
    HiddenRealization,
    Operation,
    PathwayTemplate,
    Patient,
    ProblemInstance,
    Scenario,
)
from hospital_sim.rng import make_streams


@dataclass(slots=True)
class _Draft:
    """A patient before ids are assigned (ids follow global arrival order)."""

    arrival_time: float
    spec: ArrivalSpec
    pathway: PathwayTemplate
    preds: dict[str, tuple[str, ...]]
    # step id -> (template step id, step id of its reveal trigger or None)
    steps: dict[str, tuple[str, str | None]]
    scheduled_time: float | None
    appointment: int | None
    patience: float
    raw_quantiles: np.ndarray  # independent uniforms, one per operation
    factor: float  # latent complexity, standard normal
    acuity_noise: float  # standard normal


def _included_steps(
    pathway: PathwayTemplate, rng: np.random.Generator, factor: float, reveal_correlation: float
) -> set[str]:
    """Resolve optional steps and branch groups for one patient."""
    included: set[str] = set()
    groups: dict[str, list[tuple[str, float]]] = {}
    for step in pathway.steps:
        if step.branch_group is not None:
            groups.setdefault(step.branch_group, []).append((step.id, step.branch_weight))
        elif step.probability >= 1:
            included.add(step.id)
        else:
            u = rng.random()
            if step.revealed_by is None or reveal_correlation == 0:
                happens = u < step.probability
            else:
                # A revealed step is likelier for a complex patient: mix the
                # hidden factor into its propensity. The marginal probability
                # stays ``step.probability``.
                a = reveal_correlation
                propensity = a * factor + math.sqrt(1.0 - a * a) * float(special.ndtri(1.0 - u))
                happens = propensity > float(special.ndtri(1.0 - step.probability))
            if happens:
                included.add(step.id)
    for members in groups.values():
        weights = np.array([w for _, w in members])
        included.add(members[int(rng.choice(len(members), p=weights / weights.sum()))][0])
    # A step cannot be revealed by something that does not happen.
    trigger = {step.id: step.revealed_by for step in pathway.steps}
    changed = True
    while changed:
        changed = False
        for step_id in list(included):
            if trigger[step_id] is not None and trigger[step_id] not in included:
                included.discard(step_id)
                changed = True
    return included


def _expand_repeats(
    pathway: PathwayTemplate,
    preds: dict[str, tuple[str, ...]],
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, str | None]]]:
    """Unroll repeated steps into a chain of copies, each revealed by the
    previous one. Whatever followed the step now follows its last copy."""
    by_id = {step.id: step for step in pathway.steps}
    out_preds: dict[str, tuple[str, ...]] = {}
    steps: dict[str, tuple[str, str | None]] = {}
    last_copy: dict[str, str] = {}
    for step_id in preds:
        step = by_id[step_id]
        out_preds[step_id] = preds[step_id]
        steps[step_id] = (step_id, step.revealed_by)
        last_copy[step_id] = step_id
        if step.repeat_max > 0:
            draws = rng.random(step.repeat_max)  # always all of them: stable streams
            for k, u in enumerate(draws, start=2):
                if u >= step.repeat_probability:
                    break
                copy = f"{step_id}#{k}"
                out_preds[copy] = (last_copy[step_id],)
                steps[copy] = (step_id, last_copy[step_id])
                last_copy[step_id] = copy
    for step_id, step_preds in out_preds.items():
        if steps[step_id][0] == step_id:  # an original step, not a copy
            out_preds[step_id] = tuple(last_copy[p] for p in step_preds)
    return out_preds, steps


def _arrival_times(arrivals: ArrivalSpec, rng: np.random.Generator) -> list[float]:
    n = arrivals.n_patients
    if arrivals.kind is ArrivalKind.STATIC:
        assert n is not None
        return [0.0] * n
    if arrivals.kind is ArrivalKind.SCHEDULED:
        assert n is not None
        lateness = arrivals.punctuality_sd * rng.standard_normal(n)
        absent = rng.random(n) < arrivals.no_show_probability
        # Nobody arrives after being declared a no-show.
        lateness = np.clip(
            lateness, -3.0 * arrivals.punctuality_sd, 0.999 * arrivals.no_show_timeout
        )
        booked = arrivals.appointment_start + arrivals.appointment_interval * np.arange(n)
        return [
            math.inf if absent[i] else max(float(booked[i] + lateness[i]), 0.0) for i in range(n)
        ]

    profile = arrivals.rate_profile
    if not profile and arrivals.horizon is None and arrivals.batch_mean == 1:
        # Plain homogeneous Poisson with a fixed count: one vectorised draw.
        assert arrivals.rate is not None and n is not None
        gaps = rng.exponential(1.0 / arrivals.rate, size=n)
        return [float(t) for t in np.cumsum(gaps)]

    # General case: thinning (Lewis and Shedler) of a homogeneous process at
    # the peak rate, with each accepted event bringing a geometric batch.
    peak = max(profile) if profile else arrivals.rate
    assert peak is not None
    event_rate = peak / arrivals.batch_mean
    period = arrivals.profile_period or 1.0
    times: list[float] = []
    t = 0.0
    while n is None or len(times) < n:
        t += float(rng.exponential(1.0 / event_rate))
        if arrivals.horizon is not None and t > arrivals.horizon:
            break
        if profile:
            rate_now = profile[int((t % period) / period * len(profile)) % len(profile)]
            if rng.random() >= rate_now / peak:
                continue
        batch = 1 if arrivals.batch_mean == 1 else int(rng.geometric(1.0 / arrivals.batch_mean))
        times.extend([t] * batch)
    return times if n is None else times[:n]


def _draft_stream(scenario: Scenario, spec: ArrivalSpec, seed: int, index: int) -> list[_Draft]:
    streams = make_streams(seed, index)
    names = [name for name, _ in spec.pathway_mix] or [p.name for p in scenario.pathways]
    weights = np.array([w for _, w in spec.pathway_mix] or [1.0] * len(names))
    weights = weights / weights.sum()

    times = _arrival_times(spec, streams.arrivals)
    n = len(times)
    patience_u = streams.arrivals.random(n)
    if spec.patience_mean is None:
        patience = [math.inf] * n
    else:
        sigma2 = math.log1p(spec.patience_cv**2)
        mu = math.log(spec.patience_mean) - 0.5 * sigma2
        patience = [
            math.exp(mu + math.sqrt(sigma2) * float(special.ndtri(min(max(u, 1e-12), 1 - 1e-12))))
            for u in patience_u
        ]
    factors = streams.latent.standard_normal(n)
    noise = streams.latent.standard_normal(n)

    reveal_correlation = scenario.complexity.reveal_correlation
    shapes = []
    for i in range(n):
        pathway = scenario.pathway_by_name[
            names[int(streams.pathways.choice(len(names), p=weights))]
        ]
        included = _included_steps(pathway, streams.pathways, float(factors[i]), reveal_correlation)
        preds = contract({step.id: step.predecessors for step in pathway.steps}, included)
        preds, steps = _expand_repeats(pathway, preds, streams.pathways)
        shapes.append((pathway, preds, steps))

    raw = streams.durations.random(sum(len(preds) for _, preds, _ in shapes))

    scheduled = spec.kind is ArrivalKind.SCHEDULED
    drafts: list[_Draft] = []
    offset = 0
    for i, (time, (pathway, preds, steps)) in enumerate(zip(times, shapes, strict=True)):
        drafts.append(
            _Draft(
                arrival_time=time,
                spec=spec,
                pathway=pathway,
                preds=preds,
                steps=steps,
                scheduled_time=spec.appointment_start + spec.appointment_interval * i
                if scheduled
                else None,
                appointment=i if scheduled else None,
                patience=patience[i],
                raw_quantiles=raw[offset : offset + len(preds)],
                factor=float(factors[i]),
                acuity_noise=float(noise[i]),
            )
        )
        offset += len(preds)
    return drafts


def _correlated_quantiles(raw: np.ndarray, factor: float, rho: float) -> list[float]:
    """Gaussian copula: mix the patient's factor into each operation's score.

    Each result is still uniform(0, 1), so every duration keeps its configured
    distribution; only the dependence between a patient's operations changes.
    """
    if rho == 0:
        return [float(u) for u in raw]
    scores = math.sqrt(rho) * factor + math.sqrt(1.0 - rho) * special.ndtri(raw)
    return [float(u) for u in special.ndtr(scores)]


def _acuity_class(draft: _Draft, complexity: ComplexitySpec) -> int | None:
    k = complexity.acuity_classes
    if k < 2:
        return None
    c = complexity.acuity_correlation
    reading = c * draft.factor + math.sqrt(1.0 - c * c) * draft.acuity_noise
    # Equal-probability classes: the reading is standard normal.
    return min(int(float(special.ndtr(reading)) * k), k - 1)


def generate_instance(
    scenario: Scenario,
    seed: int,
    arrivals: ArrivalSpec | Sequence[ArrivalSpec] | None = None,
) -> ProblemInstance:
    """Sample patients, their pathways and the hidden duration quantiles.

    All randomness of an episode is fixed here. The simulator itself is then
    deterministic given the instance and the policy, so two policies run on the
    same instance face exactly the same patients and the same per-operation
    "luck". Each arrival stream draws from its own generators, so adding or
    changing one patient class leaves the others untouched.

    ``arrivals`` overrides ``scenario.arrivals`` (e.g. to vary the batch size);
    it is not re-validated, so prefer values derived from a validated scenario.
    """
    if arrivals is None:
        specs: Sequence[ArrivalSpec] = scenario.arrivals
    elif isinstance(arrivals, ArrivalSpec):
        specs = (arrivals,)
    else:
        specs = arrivals

    drafts = [
        draft
        for index, spec in enumerate(specs)
        for draft in _draft_stream(scenario, spec, seed, index)
    ]
    drafts.sort(key=lambda d: d.arrival_time)  # stable: ties keep stream order

    complexity = scenario.complexity
    patients: list[Patient] = []
    quantiles: list[float] = []
    triggers: list[int] = []
    next_op_id = 0
    for patient_id, draft in enumerate(drafts):
        preds = draft.preds
        acuity = _acuity_class(draft, complexity)
        op_id = {step_id: next_op_id + i for i, step_id in enumerate(preds)}
        succs: dict[str, list[int]] = {step_id: [] for step_id in preds}
        for step_id, step_preds in preds.items():
            for pred in step_preds:
                succs[pred].append(op_id[step_id])
        base_type = {step.id: step.op_type for step in draft.pathway.steps}

        operations = tuple(
            Operation(
                id=op_id[step_id],
                patient_id=patient_id,
                step_id=step_id,
                op_type=scenario.resolve_op_type(
                    base_type[draft.steps[step_id][0]], draft.spec.name, acuity
                ),
                predecessors=tuple(op_id[p] for p in preds[step_id]),
                successors=tuple(succs[step_id]),
            )
            for step_id in preds
        )
        next_op_id += len(operations)
        for step_id in preds:
            trigger = draft.steps[step_id][1]
            triggers.append(-1 if trigger is None else op_id[trigger])
        quantiles.extend(_correlated_quantiles(draft.raw_quantiles, draft.factor, complexity.rho))
        patients.append(
            Patient(
                id=patient_id,
                pathway=draft.pathway.name,
                arrival_time=draft.arrival_time,
                operations=operations,
                patient_class=draft.spec.name,
                weight=draft.spec.weight,
                urgent=draft.spec.urgent,
                acuity_class=acuity,
                scheduled_time=draft.scheduled_time,
                appointment=draft.appointment,
            )
        )

    return ProblemInstance(
        scenario=scenario,
        patients=tuple(patients),
        hidden=HiddenRealization(
            duration_quantiles=tuple(quantiles),
            patient_factors=tuple(d.factor for d in drafts),
            reveal_trigger=tuple(triggers) if any(t >= 0 for t in triggers) else (),
            patience=tuple(d.patience for d in drafts)
            if any(d.patience < math.inf for d in drafts)
            else (),
        ),
        seed=seed,
    )


def without_operations(instance: ProblemInstance, drop: set[int]) -> ProblemInstance:
    """A copy of ``instance`` in which the given operations never existed.

    Operations revealed (directly or indirectly) by a dropped one are dropped
    too; whatever followed a dropped operation follows its predecessors
    instead. Everything else, including every remaining duration quantile, is
    unchanged. Used to build "what if that step had not been drawn" twins.
    """
    triggers = instance.hidden.reveal_trigger or (-1,) * instance.n_operations
    drop = set(drop)
    changed = True
    while changed:
        changed = False
        for op in instance.operations:
            if op.id not in drop and triggers[op.id] in drop:
                drop.add(op.id)
                changed = True

    kept = [op for op in instance.operations if op.id not in drop]
    new_id = {op.id: i for i, op in enumerate(kept)}
    preds = contract({op.id: op.predecessors for op in instance.operations}, set(new_id))
    succs: dict[int, list[int]] = {op.id: [] for op in kept}
    for op_id, op_preds in preds.items():
        for pred in op_preds:
            succs[pred].append(op_id)

    patients = []
    for patient in instance.patients:
        operations = tuple(
            dataclasses.replace(
                op,
                id=new_id[op.id],
                predecessors=tuple(new_id[p] for p in preds[op.id]),
                successors=tuple(new_id[s] for s in succs[op.id]),
            )
            for op in patient.operations
            if op.id in new_id
        )
        patients.append(dataclasses.replace(patient, operations=operations))
    new_triggers = tuple(-1 if triggers[op.id] < 0 else new_id[triggers[op.id]] for op in kept)
    return dataclasses.replace(
        instance,
        patients=tuple(patients),
        hidden=dataclasses.replace(
            instance.hidden,
            duration_quantiles=tuple(instance.hidden.duration_quantiles[op.id] for op in kept),
            reveal_trigger=new_triggers if any(t >= 0 for t in new_triggers) else (),
        ),
    )
