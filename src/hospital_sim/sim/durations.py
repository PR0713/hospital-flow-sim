"""Duration distributions and the pluggable duration model.

Every distribution exposes the same things:

* ``mean`` - the expected duration (what a scheduler is allowed to know);
* ``ppf(u)`` - turns an operation's hidden quantile into its realized duration;
* ``sf(t)`` - the probability of lasting longer than ``t``;
* ``expected_remaining(elapsed)`` - ``E[X - t | X > t]``, the honest estimate
  of how much longer a running operation will take.

Base families (lognormal, gamma, ...) are combined with three wrappers:
``Shifted`` (a minimum duration), ``Complicated`` (a rare long case, for heavy
tails) and ``Scaled`` (a faster or slower team).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from scipy import special

from hospital_sim.domain.model import DurationKind, DurationSpec, Scenario

if TYPE_CHECKING:
    from hospital_sim.sim.resources import ResourceManager, Team

# Keeps ppf away from 0 and infinity.
_EPS = 1e-12
_TINY = 1e-300


def _clip(u: float) -> float:
    return min(max(u, _EPS), 1.0 - _EPS)


def _pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class Distribution(Protocol):
    @property
    def mean(self) -> float: ...
    def ppf(self, u: float) -> float: ...
    def sf(self, t: float) -> float: ...
    def expected_remaining(self, elapsed: float) -> float: ...


# --- base families --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Deterministic:
    value: float

    @property
    def mean(self) -> float:
        return self.value

    def ppf(self, u: float) -> float:
        return self.value

    def sf(self, t: float) -> float:
        return 1.0 if t < self.value else 0.0

    def expected_remaining(self, elapsed: float) -> float:
        return max(self.value - elapsed, 0.0)


@dataclass(frozen=True, slots=True)
class Lognormal:
    mu: float
    sigma: float

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float) -> Lognormal:
        sigma2 = math.log1p((sd / mean) ** 2)
        return cls(mu=math.log(mean) - 0.5 * sigma2, sigma=math.sqrt(sigma2))

    @property
    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma**2)

    def ppf(self, u: float) -> float:
        return math.exp(self.mu + self.sigma * float(special.ndtri(_clip(u))))

    def sf(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return float(special.ndtr(-(math.log(t) - self.mu) / self.sigma))

    def expected_remaining(self, elapsed: float) -> float:
        if elapsed <= 0:
            return self.mean - elapsed
        d = (math.log(elapsed) - self.mu) / self.sigma
        survival = float(special.ndtr(-d))
        if survival < _TINY:
            return 0.0
        return self.mean * float(special.ndtr(self.sigma - d)) / survival - elapsed


@dataclass(frozen=True, slots=True)
class Gamma:
    shape: float
    scale: float

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float) -> Gamma:
        return cls(shape=(mean / sd) ** 2, scale=sd * sd / mean)

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    def ppf(self, u: float) -> float:
        return self.scale * float(special.gammaincinv(self.shape, _clip(u)))

    def sf(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return float(special.gammaincc(self.shape, t / self.scale))

    def expected_remaining(self, elapsed: float) -> float:
        if elapsed <= 0:
            return self.mean - elapsed
        x = elapsed / self.scale
        survival = float(special.gammaincc(self.shape, x))
        if survival < _TINY:
            return 0.0
        return self.mean * float(special.gammaincc(self.shape + 1.0, x)) / survival - elapsed


@dataclass(frozen=True, slots=True)
class TruncatedNormal:
    """A normal(mu, sigma) restricted to ``[low, high]``."""

    mu: float
    sigma: float
    low: float
    high: float

    def _cdf(self, x: float) -> float:
        return float(special.ndtr((x - self.mu) / self.sigma))

    def _mean_above(self, low: float) -> float:
        a = (low - self.mu) / self.sigma
        b = (self.high - self.mu) / self.sigma
        mass = float(special.ndtr(b) - special.ndtr(a))
        if mass < _TINY:
            return low
        return self.mu + self.sigma * (_pdf(a) - _pdf(b)) / mass

    @property
    def mean(self) -> float:
        return self._mean_above(self.low)

    def ppf(self, u: float) -> float:
        lo, hi = self._cdf(self.low), self._cdf(self.high)
        x = self.mu + self.sigma * float(special.ndtri(lo + _clip(u) * (hi - lo)))
        return min(max(x, self.low), self.high)

    def sf(self, t: float) -> float:
        if t < self.low:
            return 1.0
        if t >= self.high:
            return 0.0
        lo, hi = self._cdf(self.low), self._cdf(self.high)
        return (hi - self._cdf(t)) / (hi - lo)

    def expected_remaining(self, elapsed: float) -> float:
        if elapsed >= self.high:
            return 0.0
        return max(self._mean_above(max(self.low, elapsed)) - elapsed, 0.0)


@dataclass(frozen=True, slots=True)
class Empirical:
    """Resamples the given values: ``ppf`` is the inverse of the empirical CDF."""

    sorted_samples: tuple[float, ...]

    @property
    def mean(self) -> float:
        return math.fsum(self.sorted_samples) / len(self.sorted_samples)

    def ppf(self, u: float) -> float:
        n = len(self.sorted_samples)
        return self.sorted_samples[min(int(u * n), n - 1)]

    def sf(self, t: float) -> float:
        n = len(self.sorted_samples)
        return (n - bisect.bisect_right(self.sorted_samples, t)) / n

    def expected_remaining(self, elapsed: float) -> float:
        longer = self.sorted_samples[bisect.bisect_right(self.sorted_samples, elapsed) :]
        if not longer:
            return 0.0
        return math.fsum(longer) / len(longer) - elapsed


@dataclass(frozen=True, slots=True)
class LogLogistic:
    """Genuinely heavy-tailed: ``P(X > t) = 1 / (1 + (t / alpha) ** beta)``,
    which falls like a power law ``t ** -beta`` instead of exponentially in
    ``log(t) ** 2`` as a lognormal does. ``alpha`` is the median."""

    alpha: float
    beta: float

    @classmethod
    def from_mean(cls, mean: float, tail_index: float) -> LogLogistic:
        b = math.pi / tail_index
        return cls(alpha=mean * math.sin(b) / b, beta=tail_index)

    @property
    def mean(self) -> float:
        b = math.pi / self.beta
        return self.alpha * b / math.sin(b)

    def ppf(self, u: float) -> float:
        u = _clip(u)
        return self.alpha * math.pow(u / (1.0 - u), 1.0 / self.beta)

    def sf(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 / (1.0 + math.pow(t / self.alpha, self.beta))

    def expected_remaining(self, elapsed: float) -> float:
        if elapsed <= 0:
            return self.mean - elapsed
        survival = self.sf(elapsed)
        if survival < _TINY:
            return 0.0
        # E[X; X > t] = mean * (1 - I_F(t)(1 + 1/beta, 1 - 1/beta)), I = regularised
        # incomplete beta, from the substitution u = F(x).
        k = 1.0 / self.beta
        upper = float(special.betaincc(1.0 + k, 1.0 - k, 1.0 - survival))
        return self.mean * upper / survival - elapsed


# --- wrappers -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Shifted:
    """``shift + X``: an operation that can never be shorter than ``shift``."""

    base: Distribution
    shift: float

    @property
    def mean(self) -> float:
        return self.shift + self.base.mean

    def ppf(self, u: float) -> float:
        return self.shift + self.base.ppf(u)

    def sf(self, t: float) -> float:
        return 1.0 if t <= self.shift else self.base.sf(t - self.shift)

    def expected_remaining(self, elapsed: float) -> float:
        if elapsed <= self.shift:
            return self.mean - elapsed
        return self.base.expected_remaining(elapsed - self.shift)


@dataclass(frozen=True, slots=True)
class Scaled:
    """``scale * X``: the same case done by a slower (>1) or faster (<1) team."""

    base: Distribution
    scale: float

    @property
    def mean(self) -> float:
        return self.scale * self.base.mean

    def ppf(self, u: float) -> float:
        return self.scale * self.base.ppf(u)

    def sf(self, t: float) -> float:
        return self.base.sf(t / self.scale)

    def expected_remaining(self, elapsed: float) -> float:
        return self.scale * self.base.expected_remaining(elapsed / self.scale)


@dataclass(frozen=True, slots=True)
class Complicated:
    """A two-component mixture: with ``probability`` the case hits a
    complication and takes ``multiplier`` times as long.

    This models a distinct cluster of long cases, not a heavy tail: beyond the
    cluster the tail is as thin as the base distribution's. For a power-law
    tail use ``LogLogistic``.

    The top ``probability`` share of quantiles are the complicated cases, so a
    patient with a high hidden quantile (for instance through a high latent
    complexity) is the one who gets the complication.
    """

    base: Distribution
    probability: float
    multiplier: float

    @property
    def mean(self) -> float:
        return self.base.mean * (1.0 - self.probability + self.probability * self.multiplier)

    def ppf(self, u: float) -> float:
        routine = 1.0 - self.probability
        if u < routine:
            return self.base.ppf(u / routine)
        return self.multiplier * self.base.ppf((u - routine) / self.probability)

    def sf(self, t: float) -> float:
        return (1.0 - self.probability) * self.base.sf(t) + self.probability * self.base.sf(
            t / self.multiplier
        )

    def expected_remaining(self, elapsed: float) -> float:
        # Posterior-weighted average over "routine" and "complicated".
        w_routine = (1.0 - self.probability) * self.base.sf(elapsed)
        w_complicated = self.probability * self.base.sf(elapsed / self.multiplier)
        total = w_routine + w_complicated
        if total < _TINY:
            return 0.0
        routine = self.base.expected_remaining(elapsed) if w_routine > 0 else 0.0
        complicated = self.multiplier * self.base.expected_remaining(elapsed / self.multiplier)
        return (w_routine * routine + w_complicated * complicated) / total


def make_distribution(spec: DurationSpec) -> Distribution:
    dist: Distribution
    sd = spec.mean * spec.cv
    match spec.kind:
        case DurationKind.DETERMINISTIC:
            dist = Deterministic(spec.mean)
        case DurationKind.LOGNORMAL:
            dist = Lognormal.from_mean_sd(spec.mean - spec.shift, sd)
        case DurationKind.GAMMA:
            dist = Gamma.from_mean_sd(spec.mean - spec.shift, sd)
        case DurationKind.TRUNCATED_NORMAL:
            dist = TruncatedNormal(spec.mean, sd, spec.low, spec.high)
        case DurationKind.EMPIRICAL:
            dist = Empirical(tuple(sorted(spec.samples)))
        case DurationKind.LOGLOGISTIC:
            dist = LogLogistic.from_mean(spec.mean, spec.tail_index)
        case _:
            raise ValueError(f"unknown duration kind {spec.kind!r}")
    if spec.shift:
        dist = Shifted(dist, spec.shift)
    if spec.complication_probability:
        dist = Complicated(dist, spec.complication_probability, spec.complication_multiplier)
    return dist


# --- duration model -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StartContext:
    """What a duration model may look at when an operation starts.

    ``resources`` gives read access to resource state at the start instant,
    e.g. ``resources.work_time[r]`` (cumulative busy time, for fatigue).
    """

    op: int
    op_type: int
    team: Team
    now: float
    resources: ResourceManager


class DurationModel(Protocol):
    """Decides which distribution an operation's duration follows.

    ``base`` is what a scheduler may know before a team is chosen. ``at_start``
    is called once when the operation starts and may depend on the team and on
    resource state; the simulator then applies the operation's hidden quantile
    to the returned distribution. Nothing here may use hidden data.
    """

    def base(self, op_type: int) -> Distribution: ...

    def at_start(self, ctx: StartContext) -> Distribution: ...


class StandardDurationModel:
    """Type-level distributions, scaled by the speed of the governing team members.

    For an operation type with ``speed_roles``, the duration is divided by the
    geometric mean of the ``speed`` attribute (default 1) of the resources
    assigned to those roles. Types without ``speed_roles`` ignore the team.
    """

    def __init__(self, scenario: Scenario) -> None:
        self._base = [make_distribution(t.duration) for t in scenario.operation_types]
        self._speed = [float(r.attributes.get("speed", 1.0)) for r in scenario.resources]
        # Per type: positions of the requirement groups that govern speed.
        self._speed_groups = [
            tuple(i for i, req in enumerate(t.requirements) if req.role in t.speed_roles)
            for t in scenario.operation_types
        ]

    def base(self, op_type: int) -> Distribution:
        return self._base[op_type]

    def team_speed(self, op_type: int, team: Team) -> float:
        speeds = [self._speed[r] for g in self._speed_groups[op_type] for r in team[g]]
        if not speeds:
            return 1.0
        return math.exp(math.fsum(math.log(s) for s in speeds) / len(speeds))

    def at_start(self, ctx: StartContext) -> Distribution:
        base = self._base[ctx.op_type]
        if not self._speed_groups[ctx.op_type]:
            return base
        speed = self.team_speed(ctx.op_type, ctx.team)
        return base if speed == 1.0 else Scaled(base, 1.0 / speed)
