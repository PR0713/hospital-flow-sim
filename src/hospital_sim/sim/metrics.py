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
