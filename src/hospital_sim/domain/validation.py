"""Semantic validation of a ``Scenario``.

Structural checks (types, required fields) belong to the config schema; this
module checks meaning: references resolve, pathways are DAGs, and every
operation type can actually be staffed by the resource pool.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from hospital_sim.domain.dag import CycleError, topological_order
from hospital_sim.domain.matching import find_team
from hospital_sim.domain.model import (
    ArrivalKind,
    Calendar,
    DurationKind,
    DurationSpec,
    OperationType,
    PathwayTemplate,
    Scenario,
)


class ScenarioError(ValueError):
    """A scenario is invalid. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("invalid scenario:\n" + "\n".join(f"  - {e}" for e in errors))


def _duplicates(names: Iterable[str]) -> list[str]:
    return [name for name, count in Counter(names).items() if count > 1]


def _check_duration(owner: str, spec: DurationSpec, errors: list[str]) -> None:
    if not 0 <= spec.complication_probability < 1:
        errors.append(f"{owner}: complication probability must be in [0, 1)")
    if spec.complication_multiplier < 1:
        errors.append(f"{owner}: complication multiplier must be >= 1")
    if spec.kind is DurationKind.EMPIRICAL:
        if not spec.samples:
            errors.append(f"{owner}: empirical duration needs at least one sample")
        elif min(spec.samples) < 0:
            errors.append(f"{owner}: empirical duration samples must be non-negative")
        return
    if spec.kind is DurationKind.LOGLOGISTIC:
        if not (math.isfinite(spec.mean) and spec.mean > 0):
            errors.append(f"{owner}: duration mean must be positive, got {spec.mean}")
        if spec.tail_index <= 1:
            errors.append(f"{owner}: loglogistic tail_index must be > 1 (finite mean)")
        if spec.cv:
            errors.append(f"{owner}: loglogistic takes tail_index, not cv")
        return
    if spec.tail_index:
        errors.append(f"{owner}: tail_index is only valid for the loglogistic distribution")
    if not (math.isfinite(spec.mean) and spec.mean > 0):
        errors.append(f"{owner}: duration mean must be positive, got {spec.mean}")
    if spec.cv < 0:
        errors.append(f"{owner}: duration cv must be non-negative, got {spec.cv}")
    if spec.kind is DurationKind.DETERMINISTIC:
        if spec.cv != 0:
            errors.append(f"{owner}: deterministic duration cannot have cv > 0")
    elif spec.cv == 0:
        errors.append(f"{owner}: {spec.kind.value} duration needs cv > 0")
    if spec.kind is DurationKind.TRUNCATED_NORMAL and not 0 <= spec.low < spec.high:
        errors.append(f"{owner}: truncated normal needs 0 <= low < high")
    if spec.shift:
        if spec.kind not in (DurationKind.LOGNORMAL, DurationKind.GAMMA):
            errors.append(f"{owner}: shift is only valid for lognormal and gamma durations")
        elif not 0 < spec.shift < spec.mean:
            errors.append(f"{owner}: shift must be in (0, mean), got {spec.shift}")


def _check_operation_type(op_type: OperationType, scenario: Scenario, errors: list[str]) -> None:
    owner = f"operation type '{op_type.name}'"
    pool = scenario.resources_by_role
    known_roles = {role.name for role in scenario.roles}

    for role in _duplicates(req.role for req in op_type.requirements):
        errors.append(f"{owner}: role '{role}' listed more than once; merge the quantities")

    resolvable = True
    for req in op_type.requirements:
        if req.role not in known_roles:
            errors.append(f"{owner}: unknown role '{req.role}'")
            resolvable = False
        elif req.quantity < 1:
            errors.append(f"{owner}: quantity for '{req.role}' must be >= 1, got {req.quantity}")
            resolvable = False
        elif req.quantity > len(pool[req.role]):
            errors.append(
                f"{owner}: needs {req.quantity} x '{req.role}' but only {len(pool[req.role])} exist"
            )
            resolvable = False
    if resolvable and find_team(op_type.requirements, pool) is None:
        errors.append(
            f"{owner}: each role has enough resources on its own, but no full team exists "
            "because some resources are shared between the required roles"
        )

    required = {req.role for req in op_type.requirements}
    for role in op_type.speed_roles:
        if role not in required:
            errors.append(f"{owner}: speed role '{role}' is not one of its required roles")

    for role, time in op_type.turnover:
        if role not in required:
            errors.append(f"{owner}: turnover role '{role}' is not one of its required roles")
        if time <= 0:
            errors.append(f"{owner}: turnover time for '{role}' must be positive")

    if op_type.base is not None:
        base = scenario.operation_type_by_name.get(op_type.base)
        if base is None or base.base is not None:
            errors.append(f"{owner}: overrides unknown base type '{op_type.base}'")
        if op_type.when_class is None and op_type.when_acuity is None:
            errors.append(f"{owner}: an override needs a patient_class or an acuity_class")
        if op_type.when_class is not None and op_type.when_class not in {
            a.name for a in scenario.arrivals
        }:
            errors.append(f"{owner}: unknown patient class '{op_type.when_class}'")
        if op_type.when_acuity is not None and not (
            0 <= op_type.when_acuity < scenario.complexity.acuity_classes
        ):
            errors.append(
                f"{owner}: acuity class {op_type.when_acuity} does not exist "
                f"(the scenario has {scenario.complexity.acuity_classes})"
            )

    _check_duration(owner, op_type.duration, errors)


def _check_pathway(
    pathway: PathwayTemplate, scenario: Scenario, errors: list[str], warnings: list[str]
) -> None:
    owner = f"pathway '{pathway.name}'"
    if not pathway.steps:
        errors.append(f"{owner}: has no steps")
        return
    for step_id in _duplicates(step.id for step in pathway.steps):
        errors.append(f"{owner}: duplicate step id '{step_id}'")

    step_ids = {step.id for step in pathway.steps}
    edges_ok = True
    for step in pathway.steps:
        where = f"{owner}, step '{step.id}'"
        if step.op_type not in scenario.operation_type_by_name:
            errors.append(f"{where}: unknown operation type '{step.op_type}'")
        elif scenario.operation_type_by_name[step.op_type].base is not None:
            errors.append(f"{where}: '{step.op_type}' is a derived type; name its base type")
        if step.revealed_by is not None and (
            step.revealed_by not in step_ids or step.revealed_by == step.id
        ):
            errors.append(f"{where}: revealed_by must name another step of the pathway")
        if not 0 <= step.repeat_probability < 1 or step.repeat_max < 0:
            errors.append(f"{where}: repeat needs 0 <= probability < 1 and max >= 0")
        if (step.repeat_probability > 0) != (step.repeat_max > 0):
            errors.append(f"{where}: repeat needs both a probability and a max")
        for pred in step.predecessors:
            if pred not in step_ids:
                errors.append(f"{where}: unknown predecessor '{pred}'")
                edges_ok = False
        if not 0 < step.probability <= 1:
            errors.append(f"{where}: probability must be in (0, 1], got {step.probability}")
        if step.branch_group is not None:
            if step.probability != 1:
                errors.append(f"{where}: cannot be both optional and in a branch group")
            if step.branch_weight <= 0:
                errors.append(f"{where}: branch_weight must be positive")

    if edges_ok and len(step_ids) == len(pathway.steps):
        try:
            topological_order({step.id: step.predecessors for step in pathway.steps})
        except CycleError as exc:
            errors.append(f"{owner}: {exc}")

    by_group: dict[str, set[str | None]] = {}
    for step in pathway.steps:
        if step.branch_group is not None:
            by_group.setdefault(step.branch_group, set()).add(step.revealed_by)
    for group, triggers in by_group.items():
        if len(triggers) > 1:
            errors.append(f"{owner}: steps of branch group '{group}' must share one revealed_by")
    groups = Counter(step.branch_group for step in pathway.steps if step.branch_group is not None)
    for group, size in groups.items():
        if size == 1:
            warnings.append(f"{owner}: branch group '{group}' has a single step (always chosen)")
    # A patient with no operations has no meaning in the simulator. Something
    # must be certain and visible on arrival.
    certain = [
        step
        for step in pathway.steps
        if step.revealed_by is None and (step.probability >= 1 or step.branch_group is not None)
    ]
    if not certain:
        errors.append(f"{owner}: every step is optional, so a patient could have no operations")


def _check_calendar(calendar: Calendar, errors: list[str]) -> None:
    owner = f"calendar '{calendar.name}'"
    if calendar.period <= 0:
        errors.append(f"{owner}: period must be positive")
        return
    if not calendar.shifts:
        errors.append(f"{owner}: needs at least one shift")
    if not 0 <= calendar.absence_probability < 1:
        errors.append(f"{owner}: absence_probability must be in [0, 1)")
    previous_end = 0.0
    for start, end in calendar.shifts:
        if not 0 <= start < end <= calendar.period:
            errors.append(
                f"{owner}: shift ({start}, {end}) must satisfy 0 <= start < end <= period"
            )
        elif start < previous_end:
            errors.append(f"{owner}: shifts must be listed in time order and not overlap")
        previous_end = max(previous_end, end)
    previous_end = 0.0
    for start, duration in sorted(calendar.breaks):
        if duration <= 0:
            errors.append(f"{owner}: break at {start} must have a positive duration")
        elif not any(s <= start and start + duration <= e for s, e in calendar.shifts):
            errors.append(f"{owner}: break at {start} must lie inside one shift")
        elif start < previous_end:
            errors.append(f"{owner}: breaks must not overlap")
        previous_end = max(previous_end, start + duration)


def planned_on_duty(calendar: Calendar, time: float) -> bool:
    """Whether the roster has a resource on duty at ``time`` (absences aside)."""
    t = time % calendar.period
    if not any(start <= t < end for start, end in calendar.shifts):
        return False
    return not any(start <= t < start + duration for start, duration in calendar.breaks)


def _check_staffable_under_calendars(scenario: Scenario, errors: list[str]) -> None:
    """Every operation type needs some instant at which a full team is rostered."""
    horizon = 2 * max(c.period for c in scenario.calendars)
    instants = {0.0}
    for calendar in scenario.calendars:
        marks = [t for shift in calendar.shifts for t in shift]
        marks += [t for start, duration in calendar.breaks for t in (start, start + duration)]
        cycles = int(horizon // calendar.period) + 1
        instants.update(c * calendar.period + t for c in range(cycles) for t in marks)

    pools = []
    for time in sorted(t for t in instants if t < horizon):
        on_duty = {
            r.id
            for r in scenario.resources
            if r.calendar is None or planned_on_duty(scenario.calendar_by_name[r.calendar], time)
        }
        pools.append(
            {
                role: [r for r in ids if r in on_duty]
                for role, ids in scenario.resources_by_role.items()
            }
        )
    for op_type in scenario.operation_types:
        if not any(find_team(op_type.requirements, pool) is not None for pool in pools):
            errors.append(
                f"operation type '{op_type.name}': the calendars never have a full team on duty "
                "at the same time"
            )


def _check_arrivals(scenario: Scenario, errors: list[str], warnings: list[str]) -> None:
    if not scenario.arrivals:
        errors.append("scenario defines no arrival stream")
    for name in _duplicates(a.name for a in scenario.arrivals):
        errors.append(f"duplicate arrival stream '{name}'")

    for arrivals in scenario.arrivals:
        owner = f"arrival stream '{arrivals.name}'"
        if arrivals.n_patients is not None and arrivals.n_patients < 1:
            errors.append(f"{owner}: n_patients must be >= 1, got {arrivals.n_patients}")
        if arrivals.horizon is not None and arrivals.horizon <= 0:
            errors.append(f"{owner}: horizon must be positive")
        if arrivals.batch_mean < 1:
            errors.append(f"{owner}: batch_mean must be >= 1, got {arrivals.batch_mean}")
        if arrivals.weight <= 0:
            errors.append(f"{owner}: weight must be positive")

        if arrivals.patience_mean is not None and (
            arrivals.patience_mean <= 0 or arrivals.patience_cv <= 0
        ):
            errors.append(f"{owner}: patience needs a positive mean and cv")
        if arrivals.kind is not ArrivalKind.SCHEDULED and (
            arrivals.appointment_interval or arrivals.no_show_probability or arrivals.punctuality_sd
        ):
            errors.append(f"{owner}: appointments need kind: scheduled")
        if arrivals.kind is ArrivalKind.SCHEDULED:
            if arrivals.n_patients is None:
                errors.append(f"{owner}: scheduled arrivals need n_patients")
            if arrivals.horizon is not None or arrivals.rate is not None or arrivals.rate_profile:
                errors.append(f"{owner}: scheduled arrivals take neither horizon nor a rate")
            if arrivals.appointment_start < 0 or arrivals.appointment_interval < 0:
                errors.append(f"{owner}: appointment start and interval must be non-negative")
            if arrivals.punctuality_sd < 0 or arrivals.no_show_timeout <= 0:
                errors.append(f"{owner}: punctuality_sd must be >= 0 and no_show_timeout > 0")
            if not 0 <= arrivals.no_show_probability < 1:
                errors.append(f"{owner}: no_show_probability must be in [0, 1)")
        elif arrivals.kind is ArrivalKind.STATIC:
            if arrivals.n_patients is None:
                errors.append(f"{owner}: static arrivals need n_patients")
            if arrivals.horizon is not None or arrivals.rate is not None or arrivals.rate_profile:
                errors.append(f"{owner}: static arrivals take neither horizon nor a rate")
        else:
            if (arrivals.rate is None) == (not arrivals.rate_profile):
                errors.append(f"{owner}: poisson arrivals need exactly one of rate, rate_profile")
            if arrivals.rate is not None and arrivals.rate <= 0:
                errors.append(f"{owner}: rate must be positive")
            if arrivals.rate_profile:
                if min(arrivals.rate_profile) < 0 or max(arrivals.rate_profile) <= 0:
                    errors.append(f"{owner}: rate_profile needs non-negative rates, one positive")
                if not (arrivals.profile_period and arrivals.profile_period > 0):
                    errors.append(f"{owner}: rate_profile needs a positive profile_period")
            if arrivals.n_patients is None and arrivals.horizon is None:
                errors.append(f"{owner}: poisson arrivals need n_patients, horizon, or both")

        for name in _duplicates(name for name, _ in arrivals.pathway_mix):
            errors.append(f"{owner}: pathway '{name}' listed more than once in pathway_mix")
        for name, weight in arrivals.pathway_mix:
            if name not in scenario.pathway_by_name:
                errors.append(f"{owner}: unknown pathway '{name}' in pathway_mix")
            if weight <= 0:
                errors.append(f"{owner}: pathway_mix weight for '{name}' must be positive")

    if scenario.arrivals and all(a.pathway_mix for a in scenario.arrivals):
        mixed = {name for a in scenario.arrivals for name, _ in a.pathway_mix}
        for pathway in scenario.pathways:
            if pathway.name not in mixed:
                warnings.append(f"pathway '{pathway.name}' is not in any pathway_mix; never used")

    complexity = scenario.complexity
    if not 0 <= complexity.rho <= 1:
        errors.append(f"complexity: rho must be in [0, 1], got {complexity.rho}")
    if complexity.acuity_classes < 0 or complexity.acuity_classes == 1:
        errors.append("complexity: acuity_classes must be 0 (off) or >= 2")
    if not 0 <= complexity.acuity_correlation <= 1:
        errors.append("complexity: acuity_correlation must be in [0, 1]")
    if not 0 <= complexity.reveal_correlation <= 1:
        errors.append("complexity: reveal_correlation must be in [0, 1]")


def validate_scenario(scenario: Scenario) -> list[str]:
    """Raise ``ScenarioError`` listing every problem; return non-fatal warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    for label, names in (
        ("role", [r.name for r in scenario.roles]),
        ("resource", [r.id for r in scenario.resources]),
        ("operation type", [t.name for t in scenario.operation_types]),
        ("pathway", [p.name for p in scenario.pathways]),
    ):
        for name in _duplicates(names):
            errors.append(f"duplicate {label} '{name}'")

    known_roles = {role.name for role in scenario.roles}
    for resource in scenario.resources:
        if not resource.roles:
            errors.append(f"resource '{resource.id}': must have at least one role")
        for role in sorted(resource.roles - known_roles):
            errors.append(f"resource '{resource.id}': unknown role '{role}'")
        if resource.attributes.get("speed", 1.0) <= 0:
            errors.append(f"resource '{resource.id}': speed must be positive")
        if resource.calendar is not None and resource.calendar not in scenario.calendar_by_name:
            errors.append(f"resource '{resource.id}': unknown calendar '{resource.calendar}'")
        if resource.breakdown is not None and (
            resource.breakdown.mtbf <= 0 or resource.breakdown.mttr <= 0
        ):
            errors.append(f"resource '{resource.id}': breakdown mtbf and mttr must be positive")

    for name in _duplicates(c.name for c in scenario.calendars):
        errors.append(f"duplicate calendar '{name}'")
    calendars_ok = True
    for calendar in scenario.calendars:
        before = len(errors)
        _check_calendar(calendar, errors)
        calendars_ok = calendars_ok and len(errors) == before

    if not scenario.operation_types:
        errors.append("scenario defines no operation types")
    if not scenario.pathways:
        errors.append("scenario defines no pathways")

    for op_type in scenario.operation_types:
        _check_operation_type(op_type, scenario, errors)
    for pathway in scenario.pathways:
        _check_pathway(pathway, scenario, errors, warnings)
    _check_arrivals(scenario, errors, warnings)

    used_types = {step.op_type for pathway in scenario.pathways for step in pathway.steps}
    for op_type in scenario.operation_types:
        if op_type.base is None and op_type.name not in used_types:
            warnings.append(f"operation type '{op_type.name}' is not used by any pathway")
    required_roles = {req.role for t in scenario.operation_types for req in t.requirements}
    for declared in scenario.roles:
        if declared.name not in required_roles:
            warnings.append(f"role '{declared.name}' is not required by any operation type")

    if not errors and calendars_ok and scenario.calendars:
        _check_staffable_under_calendars(scenario, errors)

    if errors:
        raise ScenarioError(errors)
    return warnings
