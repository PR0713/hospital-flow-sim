"""Pure-Python domain model. No simulation state lives here.

Two levels:

* ``Scenario`` is the static description of a hospital: roles, resources,
  operation types, pathway templates and the arrival process.
* ``ProblemInstance`` is one realisation of a scenario: concrete patients with
  arrival times, concrete operation DAGs and the hidden duration quantiles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property


class RoleKind(StrEnum):
    STAFF = "staff"
    ROOM = "room"
    EQUIPMENT = "equipment"


class DurationKind(StrEnum):
    DETERMINISTIC = "deterministic"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    TRUNCATED_NORMAL = "truncated_normal"
    EMPIRICAL = "empirical"
    LOGLOGISTIC = "loglogistic"


class ArrivalKind(StrEnum):
    STATIC = "static"
    POISSON = "poisson"
    SCHEDULED = "scheduled"


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    kind: RoleKind = RoleKind.STAFF


@dataclass(frozen=True, slots=True)
class Calendar:
    """A repeating roster. Times are within ``[0, period]``.

    A resource on this calendar is on duty during ``shifts`` except during
    ``breaks`` (start, duration). Each shift occurrence is missed entirely
    with ``absence_probability``.
    """

    name: str
    period: float
    shifts: tuple[tuple[float, float], ...]
    breaks: tuple[tuple[float, float], ...] = ()
    absence_probability: float = 0.0


@dataclass(frozen=True, slots=True)
class BreakdownSpec:
    """Mean time between failures and mean time to repair (both exponential)."""

    mtbf: float
    mttr: float


@dataclass(frozen=True, slots=True)
class Resource:
    """One identifiable, capacity-1 resource. May be eligible for several roles."""

    id: str
    roles: frozenset[str]
    # Heterogeneity, e.g. {"speed": 1.2} (see OperationType.speed_roles).
    attributes: Mapping[str, float] = field(default_factory=dict)
    # Name of the roster this resource follows; ``None`` = always on duty.
    calendar: str | None = None
    breakdown: BreakdownSpec | None = None


@dataclass(frozen=True, slots=True)
class RequirementSpec:
    role: str
    quantity: int = 1


@dataclass(frozen=True, slots=True)
class DurationSpec:
    """Parameters of a duration distribution; sampling lives in the simulator.

    ``mean`` and ``cv`` (coefficient of variation) parameterise lognormal and
    gamma directly. For ``truncated_normal`` they describe the normal *before*
    truncation to ``[low, high]``. ``empirical`` resamples from ``samples``.
    """

    kind: DurationKind
    mean: float
    cv: float = 0.0
    low: float = 0.0
    high: float = math.inf
    samples: tuple[float, ...] = ()
    # Lognormal/gamma only: a minimum duration. ``mean`` and ``cv`` still
    # describe the whole duration, shift included.
    shift: float = 0.0
    # With this probability the operation hits a complication and takes
    # ``complication_multiplier`` times as long. ``mean`` and ``cv`` describe
    # the uncomplicated case.
    complication_probability: float = 0.0
    complication_multiplier: float = 1.0
    # Log-logistic only: P(X > t) falls like t ** -tail_index. Must exceed 1
    # (finite mean); at 2 or below the variance is infinite. ``cv`` is unused.
    tail_index: float = 0.0


@dataclass(frozen=True, slots=True)
class OperationType:
    name: str
    # Empty for a pure delay (transport, waiting for a result).
    requirements: tuple[RequirementSpec, ...]
    duration: DurationSpec
    # If True the patient must be physically present, so two such operations of
    # the same patient cannot overlap (the patient is an implicit resource).
    requires_patient: bool = True
    # Roles whose assigned members' ``speed`` attribute scales the duration
    # (e.g. the lead surgeon). Empty: the duration ignores the team.
    speed_roles: tuple[str, ...] = ()
    # (role, time): after completion, the resources filling that role stay
    # unavailable this long (cleaning), while the rest of the team is released.
    turnover: tuple[tuple[str, float], ...] = ()
    # Set on a derived type: the type it replaces for patients matching the
    # conditions below (each ``None`` means "any"). Pathways name base types;
    # the generator swaps in the first matching derived type per patient.
    base: str | None = None
    when_class: str | None = None
    when_acuity: int | None = None


@dataclass(frozen=True, slots=True)
class PathwayStep:
    """One node of a pathway template.

    A step is always included unless it is *optional* (``probability < 1``,
    included independently) or belongs to a ``branch_group`` (exactly one step
    of the group is chosen per patient, with odds ``branch_weight``).
    """

    id: str
    op_type: str
    predecessors: tuple[str, ...] = ()
    probability: float = 1.0
    branch_group: str | None = None
    branch_weight: float = 1.0
    # If set, the step (when it happens at all) stays unknown to the scheduler
    # until the named step of the same patient completes.
    revealed_by: str | None = None
    # After each execution the step is repeated with this probability, at most
    # ``repeat_max`` extra times. Each repeat is revealed by the one before it.
    repeat_probability: float = 0.0
    repeat_max: int = 0


@dataclass(frozen=True, slots=True)
class PathwayTemplate:
    name: str
    steps: tuple[PathwayStep, ...]


@dataclass(frozen=True, slots=True)
class ArrivalSpec:
    """One arrival stream, i.e. one patient class.

    Episode shape: ``n_patients`` alone gives a fixed-size episode;
    ``horizon`` generates arrivals until that time, after which the simulation
    drains. With both, whichever limit is hit first applies.
    """

    kind: ArrivalKind
    n_patients: int | None
    # Poisson only: mean patients per time unit (constant rate).
    rate: float | None = None
    # (pathway name, weight); empty means uniform over all pathways.
    pathway_mix: tuple[tuple[str, float], ...] = ()
    name: str = "default"
    horizon: float | None = None
    # Poisson only, instead of ``rate``: piecewise-constant patients per time
    # unit over equal bins of ``profile_period``, repeating (24 values over
    # 1440 minutes = hour of day; 168 over 10080 = hour of week).
    rate_profile: tuple[float, ...] = ()
    profile_period: float | None = None
    # Mean patients per arrival event (geometric). 1 = no bursts. The mean
    # patient rate is unchanged; arrivals just come in clumps.
    batch_mean: float = 1.0
    # Importance of this class in weighted objectives.
    weight: float = 1.0
    urgent: bool = False
    # Scheduled only: appointment i is at ``appointment_start + i * interval``.
    # The booking list is public; actual arrival is the appointment plus a
    # normal lateness (``punctuality_sd``), and a patient fails to show with
    # ``no_show_probability``. A no-show becomes known ``no_show_timeout``
    # after the appointment.
    appointment_start: float = 0.0
    appointment_interval: float = 0.0
    punctuality_sd: float = 0.0
    no_show_probability: float = 0.0
    no_show_timeout: float = 30.0
    # A patient whose first operation has not started this long after arrival
    # leaves. Lognormal with this mean and CV; ``None`` = infinitely patient.
    patience_mean: float | None = None
    patience_cv: float = 0.5


@dataclass(frozen=True, slots=True)
class ComplexitySpec:
    """Per-patient latent complexity ("complex patients are slow at everything").

    ``rho`` is the correlation between the normal scores of any two operation
    durations of the same patient (Gaussian copula); 0 means independent. The
    factor itself is hidden. If ``acuity_classes >= 2`` each patient shows a
    class in ``0..acuity_classes-1`` (higher = more complex) derived from a
    noisy reading of the factor; ``acuity_correlation`` is the correlation
    between that reading and the true factor.
    """

    rho: float = 0.0
    acuity_classes: int = 0
    acuity_correlation: float = 0.7
    # How strongly the hidden factor drives whether a revealed (gated) optional
    # step happens: correlation between the factor and the step's latent
    # propensity. 0 = independent. The step's marginal probability is unchanged.
    reveal_correlation: float = 0.5


@dataclass(frozen=True)
class Scenario:
    name: str
    roles: tuple[Role, ...]
    resources: tuple[Resource, ...]
    operation_types: tuple[OperationType, ...]
    pathways: tuple[PathwayTemplate, ...]
    arrivals: tuple[ArrivalSpec, ...]
    time_unit: str = "minute"
    complexity: ComplexitySpec = ComplexitySpec()
    calendars: tuple[Calendar, ...] = ()

    @cached_property
    def calendar_by_name(self) -> dict[str, Calendar]:
        return {c.name: c for c in self.calendars}

    def resolve_op_type(self, name: str, patient_class: str, acuity_class: int | None) -> str:
        """The operation type a patient of this class and acuity actually gets
        for base type ``name``: the first matching derived type, else ``name``."""
        for candidate in self.operation_types:
            if (
                candidate.base == name
                and candidate.when_class in (None, patient_class)
                and candidate.when_acuity in (None, acuity_class)
            ):
                return candidate.name
        return name

    @cached_property
    def operation_type_by_name(self) -> dict[str, OperationType]:
        return {t.name: t for t in self.operation_types}

    @cached_property
    def pathway_by_name(self) -> dict[str, PathwayTemplate]:
        return {p.name: p for p in self.pathways}

    @cached_property
    def resources_by_role(self) -> dict[str, tuple[str, ...]]:
        """Eligible resource ids per role, in declaration order."""
        by_role: dict[str, list[str]] = {role.name: [] for role in self.roles}
        for resource in self.resources:
            for role in resource.roles:
                by_role.setdefault(role, []).append(resource.id)
        return {role: tuple(ids) for role, ids in by_role.items()}


@dataclass(frozen=True, slots=True)
class Operation:
    """One concrete step of one patient. ``id`` is unique across the instance."""

    id: int
    patient_id: int
    step_id: str
    op_type: str
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Patient:
    id: int
    pathway: str
    arrival_time: float
    operations: tuple[Operation, ...]
    # Name of the arrival stream that produced this patient.
    patient_class: str = "default"
    weight: float = 1.0
    urgent: bool = False
    # Visible, noisy indication of complexity; ``None`` if not configured.
    acuity_class: int | None = None
    # Booked appointment time and position in the booking list (public), for
    # scheduled arrivals. ``arrival_time`` is infinite for a no-show.
    scheduled_time: float | None = None
    appointment: int | None = None


@dataclass(frozen=True, slots=True)
class HiddenRealization:
    """Everything about an instance that a scheduling policy must never see.

    Only the simulator reads this. Observation builders get no access to it.
    """

    # One uniform(0, 1) draw per operation, indexed by Operation.id. The
    # simulator turns it into a duration through the inverse CDF when the
    # operation starts.
    duration_quantiles: tuple[float, ...]
    # Latent complexity per patient (standard normal), indexed by Patient.id.
    patient_factors: tuple[float, ...] = ()
    # Seed of the absence and breakdown streams; ``None`` means the instance seed.
    availability_seed: int | None = None
    # Per operation: the operation whose completion reveals it, or -1 if it is
    # visible from the patient's arrival. Empty = nothing is gated.
    reveal_trigger: tuple[int, ...] = ()
    # Per patient: time after arrival at which they leave if not yet seen
    # (infinite = never). Empty = nobody leaves.
    patience: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProblemInstance:
    scenario: Scenario
    patients: tuple[Patient, ...]
    hidden: HiddenRealization
    seed: int

    @cached_property
    def operations(self) -> tuple[Operation, ...]:
        """All operations, ordered so that ``operations[i].id == i``."""
        return tuple(op for patient in self.patients for op in patient.operations)

    @property
    def n_operations(self) -> int:
        return len(self.operations)
