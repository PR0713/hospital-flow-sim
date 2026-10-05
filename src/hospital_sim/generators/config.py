"""Config schema (pydantic) and YAML/JSON loader for scenarios.

The schema checks structure and value ranges; ``ScenarioConfig.to_scenario``
then builds the domain ``Scenario`` and runs the semantic validation.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hospital_sim.domain.model import (
    ArrivalKind,
    ArrivalSpec,
    BreakdownSpec,
    Calendar,
    ComplexitySpec,
    DurationKind,
    DurationSpec,
    OperationType,
    PathwayStep,
    PathwayTemplate,
    RequirementSpec,
    Resource,
    Role,
    RoleKind,
    Scenario,
)
from hospital_sim.domain.validation import validate_scenario
from hospital_sim.generators.pathways import random_pathway


class _Model(BaseModel):
    # Unknown keys are almost always typos; fail loudly.
    model_config = ConfigDict(extra="forbid")


class RoleConfig(_Model):
    name: str
    kind: RoleKind = RoleKind.STAFF


class ShiftConfig(_Model):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class BreakConfig(_Model):
    start: float = Field(ge=0)
    duration: float = Field(gt=0)


class CalendarConfig(_Model):
    """A repeating roster: on duty during ``shifts``, except during ``breaks``."""

    name: str
    period: float = Field(default=1440.0, gt=0)
    shifts: list[ShiftConfig] = Field(min_length=1)
    breaks: list[BreakConfig] = Field(default_factory=list)
    # Chance that a resource misses a whole shift.
    absence_probability: float = Field(default=0.0, ge=0, lt=1)


class BreakdownConfig(_Model):
    """Mean time between failures and mean time to repair."""

    mtbf: float = Field(gt=0)
    mttr: float = Field(gt=0)


class ResourceConfig(_Model):
    """One resource, or ``count`` interchangeable ones sharing the same roles.

    With ``count > 1`` (or no ``id``) the ids are ``<base>_1 .. <base>_<count>``
    where ``base`` is ``id`` if given, else the first role in lower case.
    """

    id: str | None = None
    roles: list[str] = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    attributes: dict[str, float] = Field(default_factory=dict)
    calendar: str | None = None
    breakdown: BreakdownConfig | None = None

    def expand(self) -> list[Resource]:
        roles = frozenset(self.roles)
        breakdown = (
            BreakdownSpec(self.breakdown.mtbf, self.breakdown.mttr) if self.breakdown else None
        )
        if self.id is not None and self.count == 1:
            ids = [self.id]
        else:
            base = self.id or self.roles[0].lower()
            ids = [f"{base}_{i}" for i in range(1, self.count + 1)]
        return [
            Resource(rid, roles, dict(self.attributes), self.calendar, breakdown) for rid in ids
        ]


class RequirementConfig(_Model):
    role: str
    quantity: int = Field(default=1, ge=1)


class ComplicationConfig(_Model):
    """Rare long case: with ``probability`` the duration is multiplied."""

    probability: float = Field(gt=0, lt=1)
    multiplier: float = Field(gt=1)


class DurationConfig(_Model):
    distribution: DurationKind = DurationKind.LOGNORMAL
    mean: float | None = Field(default=None, gt=0)
    cv: float | None = Field(default=None, ge=0)
    low: float = Field(default=0.0, ge=0)
    high: float = math.inf
    samples: list[float] = Field(default_factory=list)
    # Lognormal/gamma: minimum duration (mean and cv still describe the total).
    shift: float = Field(default=0.0, ge=0)
    complication: ComplicationConfig | None = None
    # Log-logistic only: power-law tail exponent (> 1; smaller is heavier).
    tail_index: float | None = Field(default=None, gt=1)

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        if self.distribution is DurationKind.EMPIRICAL:
            if not self.samples:
                raise ValueError("empirical duration needs 'samples'")
            if self.mean is not None or self.cv is not None:
                raise ValueError("empirical duration takes 'samples', not 'mean'/'cv'")
            return self
        if self.samples:
            raise ValueError("'samples' is only valid for the empirical distribution")
        if self.mean is None:
            raise ValueError(f"{self.distribution.value} duration needs 'mean'")
        if self.distribution is DurationKind.LOGLOGISTIC:
            if self.tail_index is None or self.cv is not None:
                raise ValueError("loglogistic duration takes 'mean' and 'tail_index', not 'cv'")
            return self
        if self.tail_index is not None:
            raise ValueError("'tail_index' is only valid for the loglogistic distribution")
        if self.distribution is DurationKind.DETERMINISTIC:
            if self.cv:
                raise ValueError("deterministic duration cannot have cv > 0")
        elif not self.cv:
            raise ValueError(f"{self.distribution.value} duration needs 'cv' > 0")
        return self

    def to_domain(self) -> DurationSpec:
        probability = self.complication.probability if self.complication else 0.0
        multiplier = self.complication.multiplier if self.complication else 1.0
        if self.distribution is DurationKind.EMPIRICAL:
            return DurationSpec(
                kind=self.distribution,
                mean=float(np.mean(self.samples)),
                samples=tuple(self.samples),
                complication_probability=probability,
                complication_multiplier=multiplier,
            )
        assert self.mean is not None
        return DurationSpec(
            kind=self.distribution,
            mean=self.mean,
            cv=self.cv or 0.0,
            low=self.low,
            high=self.high,
            shift=self.shift,
            complication_probability=probability,
            complication_multiplier=multiplier,
            tail_index=self.tail_index or 0.0,
        )


class WhenConfig(_Model):
    """Which patients an override applies to; at least one condition."""

    patient_class: str | None = None
    acuity_class: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _some_condition(self) -> Self:
        if self.patient_class is None and self.acuity_class is None:
            raise ValueError("'when' needs patient_class, acuity_class, or both")
        return self


class OverrideConfig(_Model):
    """Different requirements and/or duration for some patients.

    Give ``requirements`` to replace the whole requirement vector, and either
    ``duration`` (a full replacement) or ``duration_scale`` (a multiplier on
    the base duration). The first matching override in the list applies.
    """

    when: WhenConfig
    requirements: list[RequirementConfig] | None = None
    duration: DurationConfig | None = None
    duration_scale: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _one_duration_change(self) -> Self:
        if self.duration is not None and self.duration_scale != 1.0:
            raise ValueError("give 'duration' or 'duration_scale', not both")
        return self


def _scaled(spec: DurationSpec, factor: float) -> DurationSpec:
    """The same distribution stretched in time: CV and tail shape unchanged."""
    if factor == 1.0:
        return spec
    return dataclasses.replace(
        spec,
        mean=spec.mean * factor,
        shift=spec.shift * factor,
        low=spec.low * factor,
        high=spec.high * factor,
        samples=tuple(x * factor for x in spec.samples),
    )


class OperationTypeConfig(_Model):
    name: str
    # Empty for a pure delay (transport, waiting for a result).
    requirements: list[RequirementConfig] = Field(default_factory=list)
    duration: DurationConfig
    requires_patient: bool = True
    # Roles whose members' ``speed`` attribute scales this operation's duration.
    speed_roles: list[str] = Field(default_factory=list)
    # role -> time the resources in that role stay unavailable after completion.
    turnover: dict[str, float] = Field(default_factory=dict)
    overrides: list[OverrideConfig] = Field(default_factory=list)

    def to_domain(self) -> list[OperationType]:
        """The base type followed by one derived type per override."""
        base = OperationType(
            name=self.name,
            requirements=tuple(RequirementSpec(r.role, r.quantity) for r in self.requirements),
            duration=self.duration.to_domain(),
            requires_patient=self.requires_patient,
            speed_roles=tuple(self.speed_roles),
            turnover=tuple(self.turnover.items()),
        )
        types = [base]
        for override in self.overrides:
            when = override.when
            label = ",".join(
                part
                for part in (
                    when.patient_class,
                    None if when.acuity_class is None else f"acuity={when.acuity_class}",
                )
                if part
            )
            types.append(
                dataclasses.replace(
                    base,
                    name=f"{self.name}[{label}]",
                    requirements=base.requirements
                    if override.requirements is None
                    else tuple(RequirementSpec(r.role, r.quantity) for r in override.requirements),
                    duration=override.duration.to_domain()
                    if override.duration is not None
                    else _scaled(base.duration, override.duration_scale),
                    base=self.name,
                    when_class=when.patient_class,
                    when_acuity=when.acuity_class,
                )
            )
        return types


class RepeatConfig(_Model):
    """After each execution, repeat with ``probability``, at most ``max`` times."""

    probability: float = Field(gt=0, lt=1)
    max: int = Field(ge=1)


class StepConfig(_Model):
    id: str
    op_type: str
    after: list[str] = Field(default_factory=list)
    probability: float = Field(default=1.0, gt=0, le=1)
    branch_group: str | None = None
    branch_weight: float = Field(default=1.0, gt=0)
    # Unknown to the scheduler until this other step of the patient completes.
    revealed_by: str | None = None
    repeat: RepeatConfig | None = None


class RandomPathwayConfig(_Model):
    n_ops: int = Field(ge=1)
    density: float = Field(ge=0, le=1)
    # Operation types to draw from; empty means all of them.
    op_types: list[str] = Field(default_factory=list)
    seed: int = 0


class PathwayConfig(_Model):
    """Either hand-written ``steps`` or a ``random`` generator spec."""

    name: str
    steps: list[StepConfig] | None = None
    random: RandomPathwayConfig | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.steps is None) == (self.random is None):
            raise ValueError("a pathway needs exactly one of 'steps' or 'random'")
        return self


class RateProfileConfig(_Model):
    """Piecewise-constant arrival rate over equal bins of ``period``, repeating."""

    period: float = Field(gt=0)
    rates: list[float] = Field(min_length=1)


class AppointmentsConfig(_Model):
    """A booking list: appointment i is at ``start + i * interval``."""

    start: float = Field(default=0.0, ge=0)
    interval: float = Field(ge=0)
    # Standard deviation of (arrival - appointment); 0 = perfectly punctual.
    punctuality_sd: float = Field(default=0.0, ge=0)
    no_show_probability: float = Field(default=0.0, ge=0, lt=1)
    # A missing patient is declared a no-show this long after the appointment.
    no_show_timeout: float = Field(default=30.0, gt=0)


class PatienceConfig(_Model):
    """How long a patient waits to be seen before leaving (lognormal)."""

    mean: float = Field(gt=0)
    cv: float = Field(default=0.5, gt=0)


class ArrivalConfig(_Model):
    """One arrival stream (one patient class)."""

    name: str = "default"
    kind: ArrivalKind = ArrivalKind.STATIC
    n_patients: int | None = Field(default=None, ge=1)
    horizon: float | None = Field(default=None, gt=0)
    rate: float | None = Field(default=None, gt=0)
    rate_profile: RateProfileConfig | None = None
    batch_mean: float = Field(default=1.0, ge=1)
    weight: float = Field(default=1.0, gt=0)
    urgent: bool = False
    pathway_mix: dict[str, float] = Field(default_factory=dict)
    # Required for kind: scheduled, and only valid there.
    appointments: AppointmentsConfig | None = None
    patience: PatienceConfig | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if (self.kind is ArrivalKind.SCHEDULED) != (self.appointments is not None):
            raise ValueError("'appointments' is required for kind: scheduled, and only valid there")
        if self.kind is not ArrivalKind.POISSON:
            if self.n_patients is None:
                raise ValueError(f"{self.kind.value} arrivals need 'n_patients'")
            if self.rate is not None or self.rate_profile is not None or self.horizon is not None:
                raise ValueError("'rate', 'rate_profile' and 'horizon' need kind: poisson")
            return self
        if (self.rate is None) == (self.rate_profile is None):
            raise ValueError("poisson arrivals need exactly one of 'rate' and 'rate_profile'")
        if self.n_patients is None and self.horizon is None:
            raise ValueError("poisson arrivals need 'n_patients', 'horizon', or both")
        return self

    def to_domain(self) -> ArrivalSpec:
        return ArrivalSpec(
            kind=self.kind,
            n_patients=self.n_patients,
            rate=self.rate,
            pathway_mix=tuple(self.pathway_mix.items()),
            name=self.name,
            horizon=self.horizon,
            rate_profile=tuple(self.rate_profile.rates) if self.rate_profile else (),
            profile_period=self.rate_profile.period if self.rate_profile else None,
            batch_mean=self.batch_mean,
            weight=self.weight,
            urgent=self.urgent,
            appointment_start=self.appointments.start if self.appointments else 0.0,
            appointment_interval=self.appointments.interval if self.appointments else 0.0,
            punctuality_sd=self.appointments.punctuality_sd if self.appointments else 0.0,
            no_show_probability=self.appointments.no_show_probability if self.appointments else 0.0,
            no_show_timeout=self.appointments.no_show_timeout if self.appointments else 30.0,
            patience_mean=self.patience.mean if self.patience else None,
            patience_cv=self.patience.cv if self.patience else 0.5,
        )


class ComplexityConfig(_Model):
    """Hidden per-patient complexity and its optional visible acuity class."""

    rho: float = Field(default=0.0, ge=0, le=1)
    # 0 switches the visible class off.
    acuity_classes: int = Field(default=0, ge=0)
    acuity_correlation: float = Field(default=0.7, ge=0, le=1)
    # How strongly the hidden factor drives revealed optional steps (0 = not at all).
    reveal_correlation: float = Field(default=0.5, ge=0, le=1)


class ScenarioConfig(_Model):
    name: str
    time_unit: str = "minute"
    roles: list[RoleConfig] = Field(min_length=1)
    resources: list[ResourceConfig] = Field(min_length=1)
    operation_types: list[OperationTypeConfig] = Field(min_length=1)
    pathways: list[PathwayConfig] = Field(min_length=1)
    # One stream, or a list of streams (one per patient class).
    arrivals: ArrivalConfig | list[ArrivalConfig]
    calendars: list[CalendarConfig] = Field(default_factory=list)
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)

    def to_scenario(self) -> tuple[Scenario, list[str]]:
        """Build and validate the domain scenario. Returns it with any warnings."""
        op_type_names = [t.name for t in self.operation_types]
        streams = self.arrivals if isinstance(self.arrivals, list) else [self.arrivals]
        pathways: list[PathwayTemplate] = []
        for pathway in self.pathways:
            if pathway.random is not None:
                spec = pathway.random
                pathways.append(
                    random_pathway(
                        name=pathway.name,
                        op_types=spec.op_types or op_type_names,
                        n_ops=spec.n_ops,
                        density=spec.density,
                        rng=np.random.default_rng(spec.seed),
                    )
                )
            else:
                assert pathway.steps is not None
                pathways.append(
                    PathwayTemplate(
                        name=pathway.name,
                        steps=tuple(
                            PathwayStep(
                                id=s.id,
                                op_type=s.op_type,
                                predecessors=tuple(s.after),
                                probability=s.probability,
                                branch_group=s.branch_group,
                                branch_weight=s.branch_weight,
                                revealed_by=s.revealed_by,
                                repeat_probability=s.repeat.probability if s.repeat else 0.0,
                                repeat_max=s.repeat.max if s.repeat else 0,
                            )
                            for s in pathway.steps
                        ),
                    )
                )

        scenario = Scenario(
            name=self.name,
            time_unit=self.time_unit,
            roles=tuple(Role(r.name, r.kind) for r in self.roles),
            resources=tuple(res for cfg in self.resources for res in cfg.expand()),
            operation_types=tuple(d for t in self.operation_types for d in t.to_domain()),
            pathways=tuple(pathways),
            arrivals=tuple(a.to_domain() for a in streams),
            calendars=tuple(
                Calendar(
                    name=c.name,
                    period=c.period,
                    shifts=tuple((s.start, s.end) for s in c.shifts),
                    breaks=tuple((b.start, b.duration) for b in c.breaks),
                    absence_probability=c.absence_probability,
                )
                for c in self.calendars
            ),
            complexity=ComplexitySpec(
                rho=self.complexity.rho,
                acuity_classes=self.complexity.acuity_classes,
                acuity_correlation=self.complexity.acuity_correlation,
                reveal_correlation=self.complexity.reveal_correlation,
            ),
        )
        return scenario, validate_scenario(scenario)


def load_config(path: str | Path) -> ScenarioConfig:
    """Parse a ``.yaml``/``.yml``/``.json`` scenario file into the schema."""
    path = Path(path)
    text = path.read_text()
    data: Any
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config format '{path.suffix}' (use .yaml, .yml or .json)")
    return ScenarioConfig.model_validate(data)


def load_scenario(path: str | Path) -> Scenario:
    """Load and fully validate a scenario file. Warnings are discarded; use
    ``load_config(path).to_scenario()`` to see them."""
    scenario, _ = load_config(path).to_scenario()
    return scenario
