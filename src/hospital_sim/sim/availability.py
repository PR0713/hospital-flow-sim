"""Streams of unavailable intervals for one resource.

Each stream is a lazy iterator of ``(start, end)`` in increasing time order,
never overlapping itself. The three streams of a resource (planned roster,
absences, breakdowns) are independent and may overlap one another; the
resource is unavailable while any of them is.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import count

import numpy as np

from hospital_sim.domain.model import BreakdownSpec, Calendar

Interval = tuple[float, float]

# Stream ids, also used to key the random generators.
PLANNED, ABSENCE, BREAKDOWN = 0, 1, 2
STREAM_NAMES = ("off duty", "absent", "breakdown")


def off_duty_within_period(calendar: Calendar) -> list[Interval]:
    """Off-duty intervals of one calendar period: gaps between shifts, plus breaks."""
    off: list[Interval] = []
    t = 0.0
    for start, end in calendar.shifts:
        if start > t:
            off.append((t, start))
        t = end
    if t < calendar.period:
        off.append((t, calendar.period))
    off.extend((start, start + duration) for start, duration in calendar.breaks)
    return sorted(off)


def planned_status(calendar: Calendar, time: float) -> tuple[bool, float]:
    """From the public roster alone: is a resource on duty at ``time``, and
    how long until the roster next changes that?"""
    off = off_duty_within_period(calendar)
    if not off:
        return True, float("inf")
    # Two periods, with intervals that touch across the boundary merged.
    merged: list[Interval] = []
    for start, end in off + [(s + calendar.period, e + calendar.period) for s, e in off]:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    t = time % calendar.period
    for start, end in merged:
        if start <= t < end:
            return False, end - t
        if t < start:
            return True, start - t
    return True, float("inf")


def planned_off_duty(calendar: Calendar) -> Iterator[Interval]:
    """The public roster: known to everyone in advance."""
    base = off_duty_within_period(calendar)
    if not base:
        return
    for cycle in count():
        shift = cycle * calendar.period
        for start, end in base:
            yield (shift + start, shift + end)


def absences(calendar: Calendar, rng: np.random.Generator) -> Iterator[Interval]:
    """Whole shifts missed, each with the calendar's absence probability."""
    if calendar.absence_probability <= 0:
        return
    for cycle in count():
        shift = cycle * calendar.period
        for start, end in calendar.shifts:
            if rng.random() < calendar.absence_probability:
                yield (shift + start, shift + end)


def breakdowns(spec: BreakdownSpec, rng: np.random.Generator) -> Iterator[Interval]:
    """Alternating exponential up and down times, on calendar time."""
    t = 0.0
    while True:
        t += float(rng.exponential(spec.mtbf))
        repair = float(rng.exponential(spec.mttr))
        yield (t, t + repair)
        t += repair
