"""Metrics of a finished (or partially run) simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from hospital_sim.sim.simulator import OpState, Simulator


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    # Time the last patient finished.
    makespan: float
    # Per patient: completion time minus arrival time.
    flow_times: tuple[float, ...]
    # Objective weight of each of those patients (from their arrival stream).
    weights: tuple[float, ...]
    # Per started operation: start time minus the time it became ready.
    op_waits: tuple[float, ...]
    # Per role: busy time of the eligible resources divided by the time they
    # were on duty in [0, makespan] (overtime included). A resource in several
    # roles counts towards each of them.
    utilization: dict[str, float]
    # Per role: time worked after the resource should have gone off duty.
    overtime: dict[str, float]
    # Per role: time spent unavailable in turnover (cleaning) after operations.
    turnover: dict[str, float]
    # Patients who gave up before being seen, and booked patients who never came.
    n_left: int = 0
    n_no_show: int = 0

    @property
    def mean_flow_time(self) -> float:
        return math.fsum(self.flow_times) / len(self.flow_times) if self.flow_times else 0.0

    @property
    def weighted_mean_flow_time(self) -> float:
        """Flow time averaged with patient weights (urgent classes count more)."""
        total = math.fsum(self.weights)
        if total == 0:
            return 0.0
        return math.fsum(w * f for w, f in zip(self.weights, self.flow_times, strict=True)) / total

    @property
    def total_wait(self) -> float:
        return math.fsum(self.op_waits)

    @property
    def mean_op_wait(self) -> float:
        return math.fsum(self.op_waits) / len(self.op_waits) if self.op_waits else 0.0


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """The cost of an episode so far, counting patients still in the system.

    ``EpisodeMetrics`` covers finished patients only, which is right for a
    run to completion but flatters one that was cut off. Here a patient who
    arrived and is still in the system contributes the time spent so far
    (``now - arrival``), and a ready operation that has not started
    contributes the wait so far. For a finished episode (``unfinished == 0``)
    every value equals its ``EpisodeMetrics`` counterpart exactly.

    Patients who have not arrived yet are not counted: they are not in the
    system, and how many are still to come is hidden information.
    """

    end_time: float  # sim.now when the outcome was taken
    # Last completion, or ``end_time`` while anyone is still in the system.
    makespan: float
    # Patients who arrived and are still in the system.
    unfinished: int
    # Finished (including left) plus unfinished patients.
    arrived: int
    arrived_weight: float
    # Totals over finished patients plus the accrued time of unfinished ones.
    flow_time: float
    weighted_flow_time: float
    # The unfinished patients' share of the totals above.
    unfinished_flow_time: float
    unfinished_weighted_flow_time: float
    # Waits of started operations plus the waits so far of ready ones.
    total_wait: float
    n_waits: int
    # Overtime including stretches still running at ``end_time``.
    overtime: float

    @property
    def mean_flow_time(self) -> float:
        return self.flow_time / self.arrived if self.arrived else 0.0

    @property
    def weighted_mean_flow_time(self) -> float:
        return self.weighted_flow_time / self.arrived_weight if self.arrived_weight else 0.0

    @property
    def mean_op_wait(self) -> float:
        return self.total_wait / self.n_waits if self.n_waits else 0.0


def unfinished_patients(sim: Simulator) -> list[int]:
    """Patients who have arrived and are still in the system."""
    return [
        p.id
        for p in sim.instance.patients
        if sim.arrived[p.id] and math.isnan(sim.completion_time[p.id])
    ]


def episode_outcome(sim: Simulator, metrics: EpisodeMetrics | None = None) -> EpisodeOutcome:
    """Cost so far of ``sim``, finished or not. See ``EpisodeOutcome``."""
    metrics = metrics or compute_metrics(sim)
    now = sim.now
    patients = sim.instance.patients
    unfinished = unfinished_patients(sim)
    open_flows = [now - patients[p].arrival_time for p in unfinished]
    open_weights = [patients[p].weight for p in unfinished]
    pending = [
        now - sim.ready_time[op]
        for p in unfinished
        for op in (o.id for o in patients[p].operations)
        if sim.state[op] == OpState.READY
    ]
    finished_weighted = [w * f for w, f in zip(metrics.weights, metrics.flow_times, strict=True)]
    open_weighted = [w * f for w, f in zip(open_weights, open_flows, strict=True)]
    return EpisodeOutcome(
        end_time=now,
        makespan=max(metrics.makespan, now) if unfinished else metrics.makespan,
        unfinished=len(unfinished),
        arrived=len(metrics.flow_times) + len(unfinished),
        arrived_weight=math.fsum((*metrics.weights, *open_weights)),
        flow_time=math.fsum((*metrics.flow_times, *open_flows)),
        weighted_flow_time=math.fsum((*finished_weighted, *open_weighted)),
        unfinished_flow_time=math.fsum(open_flows),
        unfinished_weighted_flow_time=math.fsum(open_weighted),
        total_wait=math.fsum((*metrics.op_waits, *pending)),
        n_waits=len(metrics.op_waits) + len(pending),
        overtime=math.fsum(sim.overtime_so_far()),
    )


def compute_metrics(sim: Simulator) -> EpisodeMetrics:
    """Metrics over finished patients and finished operations."""
    patients = sim.instance.patients
    # Everyone who arrived and is no longer in the system, including patients
    # who left unseen (their flow time is the time they waited). No-shows never
    # arrived and are not counted.
    finished_patients = [
        p for p in patients if sim.arrived[p.id] and not math.isnan(sim.completion_time[p.id])
    ]
    flow_times = tuple(sim.completion_time[p.id] - p.arrival_time for p in finished_patients)
    finished = [op for op in range(sim.n_ops) if sim.state[op] == OpState.DONE]
    makespan = max((sim.completion_time[p.id] for p in finished_patients), default=0.0)

    busy = [0.0] * len(sim.resources.resource_ids)
    for op in finished:
        team = sim.team[op]
        assert team is not None
        for group in team:
            for r in group:
                busy[r] += sim.end_time[op] - sim.start_time[op]
    # Time each resource was rostered off (or absent, or broken) in [0, makespan].
    off = [0.0] * len(busy)
    turnover = [0.0] * len(busy)
    down: dict[int, list[tuple[float, float]]] = {}
    for r, start, end, reason in sim.downtime_log:
        if reason == "turnover":
            turnover[r] += max(min(end, makespan) - start, 0.0)
        else:
            down.setdefault(r, []).append((start, min(end, makespan)))
    for r, spans in down.items():
        reach = 0.0  # union of possibly overlapping spans
        for start, end in sorted(spans):
            if end > max(start, reach):
                off[r] += end - max(start, reach)
                reach = end

    def per_role(values: list[float]) -> dict[str, float]:
        return {
            name: math.fsum(values[r] for r in members)
            for name, members in zip(sim.resources.role_names, sim.resources.members, strict=True)
        }

    worked, turned, over = per_role(busy), per_role(turnover), per_role(sim.overtime)
    on_duty = per_role([makespan - off[r] + sim.overtime[r] for r in range(len(busy))])
    utilization = {
        name: worked[name] / on_duty[name] if on_duty[name] > 0 else 0.0 for name in worked
    }
    return EpisodeMetrics(
        makespan=makespan,
        flow_times=flow_times,
        weights=tuple(p.weight for p in finished_patients),
        op_waits=tuple(sim.start_time[op] - sim.ready_time[op] for op in finished),
        utilization=utilization,
        overtime=over,
        turnover=turned,
        n_left=sim.n_left,
        n_no_show=sim.n_no_show,
    )
