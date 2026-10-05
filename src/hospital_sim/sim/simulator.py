"""Event-driven core simulator. Knows nothing about RL.

Usage::

    sim = Simulator(instance)
    while sim.run_until_decision():
        op = choose(sim.startable_ops())   # any policy
        sim.start(op)                      # or sim.start(op, team)
    # sim.done is now True

Time only moves in ``advance`` / ``run_until_decision``; ``start`` and
``cancel`` act at the current time. Operations, patients, resources and
operation types are addressed by integer index.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from enum import IntEnum

from hospital_sim.domain.model import ProblemInstance
from hospital_sim.rng import availability_rng
from hospital_sim.sim import availability
from hospital_sim.sim.durations import (
    Distribution,
    DurationModel,
    StandardDurationModel,
    StartContext,
)
from hospital_sim.sim.resources import FREE, ResourceManager, Team


class OpState(IntEnum):
    HIDDEN = 0  # the scheduler does not know this operation exists yet
    WAITING = 1  # visible, predecessors unfinished
    READY = 2  # may start as soon as a team (and the patient) is free
    RUNNING = 3
    DONE = 4
    CANCELLED = 5


class EventKind(IntEnum):
    # Order at equal times. It is immaterial for decisions (the whole
    # timestamp is drained first) but must be fixed for reproducibility.
    COMPLETION = 0
    TURNOVER_END = 1
    RESOURCE_UP = 2
    RESOURCE_DOWN = 3
    ARRIVAL = 4
    LEAVE = 5
    NO_SHOW = 6


class PatientStatus(IntEnum):
    UNKNOWN = 0  # the scheduler does not know this patient exists
    EXPECTED = 1  # booked appointment, not yet arrived
    PRESENT = 2
    FINISHED = 3
    LEFT = 4  # gave up waiting before being seen
    NO_SHOW = 5


# Events that represent pending work, as opposed to roster changes.
_WORK_EVENTS = (
    EventKind.COMPLETION,
    EventKind.TURNOVER_END,
    EventKind.ARRIVAL,
    EventKind.LEAVE,
    EventKind.NO_SHOW,
)
_N_STREAMS = len(availability.STREAM_NAMES)


class SimulationError(RuntimeError):
    """An operation was started or cancelled in a state that does not allow it."""


class DeadlockError(SimulationError):
    """Nothing can start, nothing is pending, yet patients remain."""


class Simulator:
    def __init__(
        self,
        instance: ProblemInstance,
        *,
        duration_model: DurationModel | None = None,
        debug: bool = False,
        stall_limit: int = 10_000,
    ) -> None:
        scenario = instance.scenario
        self.instance = instance
        self.debug = debug
        # Roster-only event batches tolerated while nothing can start and no
        # work is pending, before the run is declared deadlocked.
        self.stall_limit = stall_limit
        self.resources = ResourceManager(scenario)
        self.duration_model: DurationModel = duration_model or StandardDurationModel(scenario)

        self.op_types = scenario.operation_types
        type_index = {t.name: i for i, t in enumerate(self.op_types)}
        ops = instance.operations
        self.n_ops = len(ops)
        self.n_patients = len(instance.patients)
        self.type_of = [type_index[op.op_type] for op in ops]
        self.patient_of = [op.patient_id for op in ops]
        self.preds = [op.predecessors for op in ops]
        self.succs = [op.successors for op in ops]
        self.needs_patient = [t.requires_patient for t in self.op_types]
        # Pure delays that do not occupy the patient start without a decision.
        self.auto_start = [not t.requirements and not t.requires_patient for t in self.op_types]
        # Mean duration per type before a team is known (what a scheduler may use).
        self.expected_duration = [
            self.duration_model.base(i).mean for i in range(len(self.op_types))
        ]
        self.weight = [p.weight for p in instance.patients]
        role_index = {name: i for i, name in enumerate(self.resources.role_names)}
        self.speed_role_indices = [
            frozenset(role_index[name] for name in t.speed_roles) for t in self.op_types
        ]
        self.resource_speed = [float(r.attributes.get("speed", 1.0)) for r in scenario.resources]
        # Per type: (position of the requirement group, turnover time).
        self.turnover = [
            tuple(
                (i, dict(t.turnover)[req.role])
                for i, req in enumerate(t.requirements)
                if req.role in dict(t.turnover)
            )
            for t in self.op_types
        ]
        self._quantiles = instance.hidden.duration_quantiles
        # Reveal gating: trigger[op] is the operation whose completion makes
        # ``op`` visible (-1: visible on arrival); gated_by is the inverse.
        self._trigger = instance.hidden.reveal_trigger or (-1,) * self.n_ops
        self._gated_by: list[list[int]] = [[] for _ in range(self.n_ops)]
        for op_id, trigger in enumerate(self._trigger):
            if trigger >= 0:
                self._gated_by[trigger].append(op_id)
        self._patience = instance.hidden.patience or (math.inf,) * self.n_patients
        timeout = {a.name: a.no_show_timeout for a in scenario.arrivals}
        self._no_show_timeout = [timeout.get(p.patient_class, 30.0) for p in instance.patients]

        self.reset()

    def reset(self) -> None:
        """Back to time zero with nothing arrived."""
        n = self.n_ops
        self.now = 0.0
        self.resources.reset()
        self.state = [OpState.HIDDEN] * n
        self.pending_preds = [len(p) for p in self.preds]
        self.ready_time = [math.nan] * n
        self.start_time = [math.nan] * n
        self.end_time = [math.nan] * n  # published on completion only
        self.team: list[Team | None] = [None] * n
        # Distribution each started operation was drawn from (team-dependent).
        self._dist: list[Distribution | None] = [None] * n
        self.arrived = [False] * self.n_patients
        self.patient_busy = [False] * self.n_patients
        self.open_ops = [len(p.operations) for p in self.instance.patients]
        self.completion_time = [math.nan] * self.n_patients
        self.n_patients_done = 0
        self._ready: dict[int, None] = {}  # insertion-ordered set
        # Operations the scheduler knows about, in the order they became visible.
        self.visible_ops: list[int] = []
        # Running sums behind ``flow_accrued``: over patients in the system,
        # and over finished patients.
        self._in_system = [0.0, 0.0, 0.0, 0.0]  # count, sum a, sum w, sum w*a
        self._flow_done = [0.0, 0.0]  # plain, weighted
        self._seq = 0
        self._events: list[tuple[float, int, int, int]] = []
        self._n_work_events = 0
        self.patient_status = [PatientStatus.UNKNOWN] * self.n_patients
        # Patients the scheduler knows about, in the order it learned of them:
        # the booking list first, then walk-ins as they arrive.
        booked = [p for p in self.instance.patients if p.scheduled_time is not None]
        booked.sort(key=lambda p: (p.scheduled_time, p.patient_class, p.appointment))
        self.visible_patients = [p.id for p in booked]
        for patient in booked:
            self.patient_status[patient.id] = PatientStatus.EXPECTED
        self.n_left = 0
        self.n_no_show = 0
        self.left_weight = 0.0
        for patient in self.instance.patients:
            if math.isfinite(patient.arrival_time):
                self._push(patient.arrival_time, EventKind.ARRIVAL, patient.id)
            else:
                assert patient.scheduled_time is not None
                self._push(
                    patient.scheduled_time + self._no_show_timeout[patient.id],
                    EventKind.NO_SHOW,
                    patient.id,
                )

        n_resources = len(self.resources.resource_ids)
        # Number of availability streams currently holding each resource down.
        self.down_count = [0] * n_resources
        # Bit k set: availability stream k (planned, absence, breakdown) is active.
        self.down_mask = [0] * n_resources
        self.in_turnover = [False] * n_resources
        # Time worked after the resource should have gone off duty.
        self.overtime = [0.0] * n_resources
        self._overtime_from = [math.nan] * n_resources
        # (resource, start, end, reason). Ground truth, including the realized
        # end of breakdowns: for analysis after the episode, never for a policy.
        self.downtime_log: list[tuple[int, float, float, str]] = []
        self._streams: dict[int, Iterator[availability.Interval]] = {}
        self._down_end: dict[int, float] = {}
        self._start_availability_streams()
        self._check()

    def _start_availability_streams(self) -> None:
        scenario = self.instance.scenario
        seed = self.instance.hidden.availability_seed
        if seed is None:
            seed = self.instance.seed
        for r, resource in enumerate(scenario.resources):
            if resource.calendar is not None:
                calendar = scenario.calendar_by_name[resource.calendar]
                self._streams[r * _N_STREAMS + availability.PLANNED] = (
                    availability.planned_off_duty(calendar)
                )
                self._streams[r * _N_STREAMS + availability.ABSENCE] = availability.absences(
                    calendar, availability_rng(seed, r, availability.ABSENCE)
                )
            if resource.breakdown is not None:
                self._streams[r * _N_STREAMS + availability.BREAKDOWN] = availability.breakdowns(
                    resource.breakdown, availability_rng(seed, r, availability.BREAKDOWN)
                )
        for key in list(self._streams):
            self._schedule_next_down(key)

    def _schedule_next_down(self, key: int) -> None:
        interval = next(self._streams[key], None)
        if interval is not None:
            self._down_end[key] = interval[1]
            self._push(interval[0], EventKind.RESOURCE_DOWN, key)

    # --- queries -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.n_patients_done == self.n_patients

    @property
    def ready_ops(self) -> list[int]:
        """Visible operations whose predecessors are finished, oldest first."""
        return list(self._ready)

    def can_start(self, op: int) -> bool:
        if self.state[op] != OpState.READY:
            return False
        op_type = self.type_of[op]
        if self.needs_patient[op_type] and self.patient_busy[self.patient_of[op]]:
            return False
        return self.resources.can_staff(op_type)

    def startable_ops(self) -> list[int]:
        """Ready operations that could start right now, oldest first."""
        staffable: dict[int, bool] = {}
        result = []
        for op in self._ready:
            op_type = self.type_of[op]
            if self.needs_patient[op_type] and self.patient_busy[self.patient_of[op]]:
                continue
            ok = staffable.get(op_type)
            if ok is None:
                ok = staffable[op_type] = self.resources.can_staff(op_type)
            if ok:
                result.append(op)
        return result

    def has_pending_events(self) -> bool:
        return bool(self._events)

    def flow_accrued(self, weighted: bool = True) -> float:
        """Total (weighted) time patients have spent in the system so far:
        finished patients in full, present patients up to now. At the end of an
        episode this is the total (weighted) flow time."""
        count, sum_a, sum_w, sum_wa = self._in_system
        if weighted:
            return self._flow_done[1] + self.now * sum_w - sum_wa
        return self._flow_done[0] + self.now * count - sum_a

    def expected_remaining(self, op: int) -> float:
        """Expected time left for a running operation, given how long it has
        already run. Uses the distribution only, never the realized duration."""
        dist = self._dist[op]
        if self.state[op] != OpState.RUNNING or dist is None:
            raise SimulationError(f"operation {op} is not running")
        return dist.expected_remaining(self.now - self.start_time[op])

    # --- actions -----------------------------------------------------------

    def start(self, op: int, team: Team | None = None) -> None:
        """Start a ready operation now, on ``team`` or on the first idle team."""
        if self.state[op] != OpState.READY:
            raise SimulationError(f"operation {op} is {self.state[op].name}, not READY")
        op_type = self.type_of[op]
        patient = self.patient_of[op]
        if self.needs_patient[op_type] and self.patient_busy[patient]:
            raise SimulationError(f"patient {patient} is busy with another operation")
        if team is None:
            team = self.resources.first_idle_team(op_type)
            if team is None:
                raise SimulationError(f"no idle team for operation {op}")
        self._begin(op, team)
        self._drain()
        self._check()

    def cancel(self, op: int) -> None:
        """Cancel an operation that has not started. Successors are not blocked."""
        state = self.state[op]
        if state not in (OpState.HIDDEN, OpState.WAITING, OpState.READY):
            raise SimulationError(f"operation {op} is {state.name} and cannot be cancelled")
        self._cancel(op)
        self._drain()
        self._check()

    def _cancel(self, op: int) -> None:
        self._ready.pop(op, None)
        self.state[op] = OpState.CANCELLED
        self._close(op)
        if self.pending_preds[op] == 0:
            self._release_successors(op)
        # What this operation would have revealed will now never happen.
        for gated in self._gated_by[op]:
            if self.state[gated] == OpState.HIDDEN:
                self._cancel(gated)

    def _cancel_patient(self, patient: int) -> None:
        for op in self.instance.patients[patient].operations:
            if self.state[op.id] in (OpState.HIDDEN, OpState.WAITING, OpState.READY):
                self._cancel(op.id)

    # --- time --------------------------------------------------------------

    def advance(self) -> bool:
        """Jump to the next event time and process everything scheduled then.

        Returns ``False`` if there was no event left.
        """
        if not self._events:
            return False
        self.now = self._events[0][0]
        self._drain()
        self._check()
        return True

    def run_until_decision(self) -> bool:
        """Advance until some operation can start. ``False`` once the episode is over."""
        stalled = 0
        while not self._any_startable():
            if self.done:
                return False
            # Roster changes alone can go on forever; only they do not count as progress.
            stalled = stalled + 1 if self._n_work_events == 0 else 0
            if stalled > self.stall_limit or not self.advance():
                raise DeadlockError(
                    f"t={self.now}: no startable operation and no pending work, "
                    f"but {self.n_patients - self.n_patients_done} patients remain"
                )
        return True

    # --- internals ---------------------------------------------------------

    def _any_startable(self) -> bool:
        # Staffing depends on the operation type only, so ask once per type.
        staffable: dict[int, bool] = {}
        for op in self._ready:
            op_type = self.type_of[op]
            if self.needs_patient[op_type] and self.patient_busy[self.patient_of[op]]:
                continue
            ok = staffable.get(op_type)
            if ok is None:
                ok = staffable[op_type] = self.resources.can_staff(op_type)
            if ok:
                return True
        return False

    def _push(self, time: float, kind: EventKind, payload: int) -> None:
        heapq.heappush(self._events, (time, kind, self._seq, payload))
        self._seq += 1
        if kind in _WORK_EVENTS:
            self._n_work_events += 1

    def _drain(self) -> None:
        """Process every event due at the current time."""
        events = self._events
        while events and events[0][0] <= self.now:
            _, kind, _, payload = heapq.heappop(events)
            if kind in _WORK_EVENTS:
                self._n_work_events -= 1
            if kind == EventKind.COMPLETION:
                self._complete(payload)
            elif kind == EventKind.ARRIVAL:
                self._arrive(payload)
            elif kind == EventKind.RESOURCE_DOWN:
                self._resource_down(payload)
            elif kind == EventKind.RESOURCE_UP:
                self._resource_up(payload)
            elif kind == EventKind.TURNOVER_END:
                self._turnover_end(payload)
            elif kind == EventKind.LEAVE:
                self._leave(payload)
            else:
                self._no_show(payload)

    def _leave(self, patient: int) -> None:
        """The patient's patience has run out: they go, unless already seen."""
        if self.patient_status[patient] != PatientStatus.PRESENT:
            return
        seen = any(
            self.state[op.id] in (OpState.RUNNING, OpState.DONE)
            for op in self.instance.patients[patient].operations
        )
        if seen:
            return
        self.patient_status[patient] = PatientStatus.LEFT
        self.n_left += 1
        self.left_weight += self.weight[patient]
        self._cancel_patient(patient)

    def _no_show(self, patient: int) -> None:
        self.patient_status[patient] = PatientStatus.NO_SHOW
        self.n_no_show += 1
        self._cancel_patient(patient)

    def _resource_down(self, key: int) -> None:
        r, stream = divmod(key, _N_STREAMS)
        end = self._down_end[key]
        self.downtime_log.append((r, self.now, end, availability.STREAM_NAMES[stream]))
        self.down_count[r] += 1
        self.down_mask[r] |= 1 << stream
        self.resources.block(r)
        # Lazy removal: a busy resource finishes its operation first.
        if self.down_count[r] == 1 and self.resources.holder[r] != FREE:
            self._overtime_from[r] = self.now
        self._push(end, EventKind.RESOURCE_UP, key)
        self._schedule_next_down(key)

    def _resource_up(self, key: int) -> None:
        r, stream = divmod(key, _N_STREAMS)
        self.down_count[r] -= 1
        self.down_mask[r] &= ~(1 << stream)
        self.resources.unblock(r)
        if self.down_count[r] == 0:
            self._stop_overtime(r)

    def _stop_overtime(self, r: int) -> None:
        if not math.isnan(self._overtime_from[r]):
            self.overtime[r] += self.now - self._overtime_from[r]
            self._overtime_from[r] = math.nan

    def _turnover_end(self, r: int) -> None:
        self.in_turnover[r] = False
        self.resources.unblock(r)

    def _arrive(self, patient: int) -> None:
        self.arrived[patient] = True
        if self.patient_status[patient] == PatientStatus.UNKNOWN:
            self.visible_patients.append(patient)
        self.patient_status[patient] = (
            PatientStatus.PRESENT if self.open_ops[patient] > 0 else PatientStatus.FINISHED
        )
        if math.isfinite(self._patience[patient]):
            self._push(self.now + self._patience[patient], EventKind.LEAVE, patient)
        if self.open_ops[patient] > 0:
            arrival, weight = self.instance.patients[patient].arrival_time, self.weight[patient]
            acc = self._in_system
            acc[0] += 1.0
            acc[1] += arrival
            acc[2] += weight
            acc[3] += weight * arrival
        for op in self.instance.patients[patient].operations:
            if self.state[op.id] == OpState.HIDDEN and self._trigger[op.id] < 0:
                self._reveal(op.id)

    def _reveal(self, op: int) -> None:
        self.visible_ops.append(op)
        if self.pending_preds[op] == 0:
            self._make_ready(op)
        else:
            self.state[op] = OpState.WAITING

    def _make_ready(self, op: int) -> None:
        self.state[op] = OpState.READY
        self.ready_time[op] = self.now
        self._ready[op] = None
        if self.auto_start[self.type_of[op]]:
            self._begin(op, ())

    def _begin(self, op: int, team: Team) -> None:
        op_type = self.type_of[op]
        self.resources.acquire(op, op_type, team)
        if self.needs_patient[op_type]:
            self.patient_busy[self.patient_of[op]] = True
        del self._ready[op]
        self.state[op] = OpState.RUNNING
        self.start_time[op] = self.now
        self.team[op] = team
        dist = self.duration_model.at_start(
            StartContext(op=op, op_type=op_type, team=team, now=self.now, resources=self.resources)
        )
        self._dist[op] = dist
        self._push(self.now + dist.ppf(self._quantiles[op]), EventKind.COMPLETION, op)

    def _complete(self, op: int) -> None:
        op_type = self.type_of[op]
        team = self.team[op]
        assert team is not None
        hold: dict[int, float] = {
            r: time for group, time in self.turnover[op_type] for r in team[group]
        }
        self.resources.release(team, self.now - self.start_time[op], hold)
        for group in team:
            for r in group:
                self._stop_overtime(r)
        for r, time in hold.items():
            self.in_turnover[r] = True
            self.downtime_log.append((r, self.now, self.now + time, "turnover"))
            self._push(self.now + time, EventKind.TURNOVER_END, r)
        if self.needs_patient[op_type]:
            self.patient_busy[self.patient_of[op]] = False
        self.state[op] = OpState.DONE
        self.end_time[op] = self.now
        # Reveal first, in the same event: a successor that was waiting on a
        # hidden operation must never be seen without it.
        for gated in self._gated_by[op]:
            if self.state[gated] == OpState.HIDDEN:
                self._reveal(gated)
        self._close(op)
        self._release_successors(op)

    def _close(self, op: int) -> None:
        """Book-keeping for an operation that will never (again) need service."""
        patient = self.patient_of[op]
        self.open_ops[patient] -= 1
        if self.open_ops[patient] == 0:
            self.completion_time[patient] = self.now
            self.n_patients_done += 1
            if self.patient_status[patient] == PatientStatus.PRESENT:
                self.patient_status[patient] = PatientStatus.FINISHED
            if self.arrived[patient]:
                arrival, weight = self.instance.patients[patient].arrival_time, self.weight[patient]
                acc = self._in_system
                acc[0] -= 1.0
                acc[1] -= arrival
                acc[2] -= weight
                acc[3] -= weight * arrival
                self._flow_done[0] += self.now - arrival
                self._flow_done[1] += weight * (self.now - arrival)

    def _release_successors(self, op: int) -> None:
        """``op`` no longer holds anything back: it is done, or cancelled with
        all of its own predecessors resolved."""
        for succ in self.succs[op]:
            self.pending_preds[succ] -= 1
            if self.pending_preds[succ] == 0:
                state = self.state[succ]
                if state == OpState.WAITING:
                    self._make_ready(succ)
                elif state == OpState.CANCELLED:
                    self._release_successors(succ)

    def _check(self) -> None:
        if self.debug:
            from hospital_sim.sim.invariants import check_invariants

            check_invariants(self)
