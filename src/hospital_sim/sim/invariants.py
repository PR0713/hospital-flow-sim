"""Debug-mode invariant checker.

``check_invariants`` recomputes every piece of derived state from scratch and
compares it with what the simulator maintains incrementally. It is O(size of
the instance) per call, so it only runs with ``Simulator(debug=True)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hospital_sim.sim.availability import STREAM_NAMES
from hospital_sim.sim.resources import FREE

if TYPE_CHECKING:
    from hospital_sim.sim.simulator import Simulator


class InvariantViolation(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def check_invariants(sim: Simulator) -> None:
    # Imported here: simulator.py imports this module lazily, and vice versa.
    from hospital_sim.sim.simulator import EventKind, OpState

    res = sim.resources
    state = sim.state

    # --- resources: no double booking, holders and teams agree ---------------
    held_by: dict[int, list[int]] = {}
    for r, holder in enumerate(res.holder):
        if holder != FREE:
            held_by.setdefault(holder, []).append(r)
    for op in held_by:
        _require(state[op] == OpState.RUNNING, f"op {op} holds resources but is {state[op].name}")
    for role, members in enumerate(res.members):
        idle = sum(res.holder[r] == FREE and res.blocked[r] == 0 for r in members)
        _require(res.idle_count[role] == idle, f"idle count of role {role} is stale")

    running_per_patient = [0] * sim.n_patients
    for op in range(sim.n_ops):
        st = state[op]
        op_type = sim.type_of[op]
        patient = sim.patient_of[op]
        team = sim.team[op]

        if st == OpState.RUNNING:
            _require(team is not None, f"running op {op} has no team")
            assert team is not None
            flat = [r for group in team for r in group]
            # All-or-nothing: the op holds exactly its team, nothing more or less.
            _require(sorted(flat) == sorted(held_by.get(op, [])), f"op {op} team/holders differ")
            _require(len(flat) == len(set(flat)), f"op {op} uses a resource twice")
            reqs = res.requirements[op_type]
            _require(len(team) == len(reqs), f"op {op} team shape differs from requirements")
            for (role, qty), group in zip(reqs, team, strict=True):
                _require(len(group) == qty, f"op {op} has wrong quantity for role {role}")
                for r in group:
                    _require(r in res.members[role], f"op {op}: resource {r} not eligible")
            if sim.needs_patient[op_type]:
                running_per_patient[patient] += 1
        else:
            _require(op not in held_by, f"op {op} is {st.name} but holds resources")

        # --- precedence ---------------------------------------------------------
        unresolved = sum(not _resolved(sim, p) for p in sim.preds[op])
        _require(sim.pending_preds[op] == unresolved, f"pending count of op {op} is stale")
        if st in (OpState.READY, OpState.RUNNING, OpState.DONE):
            _require(unresolved == 0, f"op {op} is {st.name} with unfinished predecessors")
            _require(sim.arrived[patient], f"op {op} is {st.name} before its patient arrived")
        if st == OpState.WAITING:
            _require(unresolved > 0, f"op {op} is WAITING with nothing to wait for")
            _require(sim.arrived[patient], f"op {op} is WAITING before its patient arrived")
        _require((st == OpState.READY) == (op in sim._ready), f"ready set wrong for op {op}")

        # --- times --------------------------------------------------------------
        arrival = sim.instance.patients[patient].arrival_time
        if st in (OpState.RUNNING, OpState.DONE):
            _require(
                arrival <= sim.ready_time[op] <= sim.start_time[op] <= sim.now, f"op {op} times"
            )
        if st == OpState.DONE:
            _require(sim.start_time[op] <= sim.end_time[op] <= sim.now, f"op {op} end time")
        else:
            _require(math.isnan(sim.end_time[op]), f"op {op} publishes an end time early")

    # --- patients ---------------------------------------------------------------
    done = 0
    for record in sim.instance.patients:
        pid = record.id
        running = running_per_patient[pid]
        _require(running <= 1, f"patient {pid} is in {running} places at once")
        _require(sim.patient_busy[pid] == (running == 1), f"patient {pid} busy flag is stale")
        open_ops = sum(
            state[op.id] not in (OpState.DONE, OpState.CANCELLED) for op in record.operations
        )
        _require(sim.open_ops[pid] == open_ops, f"open-op count of patient {pid} is stale")
        done += open_ops == 0
        _require(
            (open_ops == 0) != math.isnan(sim.completion_time[pid]),
            f"completion time of patient {pid} is inconsistent",
        )
    _require(sim.n_patients_done == done, "finished-patient count is stale")

    # --- event list: exactly one completion per running op, nothing in the past ---
    completions = sorted(e[3] for e in sim._events if e[1] == EventKind.COMPLETION)
    running_ops = [op for op in range(sim.n_ops) if state[op] == OpState.RUNNING]
    _require(completions == running_ops, "completion events and running ops differ")
    arrivals = sorted(e[3] for e in sim._events if e[1] == EventKind.ARRIVAL)
    _require(
        arrivals
        == [
            p.id
            for p in sim.instance.patients
            if not sim.arrived[p.id] and math.isfinite(p.arrival_time)
        ],
        "arrival events and unarrived patients differ",
    )
    _require(all(e[0] >= sim.now for e in sim._events), "an event is scheduled in the past")

    # --- availability: every block has a reason and an event that will lift it ---
    turnover_ends = sorted(e[3] for e in sim._events if e[1] == EventKind.TURNOVER_END)
    _require(
        turnover_ends == [r for r, flag in enumerate(sim.in_turnover) if flag],
        "turnover flags and turnover events differ",
    )
    ups = [0] * len(res.holder)
    for e in sim._events:
        if e[1] == EventKind.RESOURCE_UP:
            ups[e[3] // len(STREAM_NAMES)] += 1
    for r in range(len(res.holder)):
        _require(sim.down_count[r] == ups[r], f"resource {r}: down count and pending ups differ")
        _require(
            res.blocked[r] == sim.down_count[r] + sim.in_turnover[r],
            f"resource {r}: block count has no matching reason",
        )
        _require(
            not sim.in_turnover[r] or res.holder[r] == FREE,
            f"resource {r} is in turnover while held",
        )
    work = sum(e[1] not in (EventKind.RESOURCE_UP, EventKind.RESOURCE_DOWN) for e in sim._events)
    _require(sim._n_work_events == work, "pending-work counter is stale")


def _resolved(sim: Simulator, op: int) -> bool:
    """Done, or cancelled with everything before it resolved."""
    from hospital_sim.sim.simulator import OpState

    if sim.state[op] == OpState.DONE:
        return True
    if sim.state[op] == OpState.CANCELLED:
        return all(_resolved(sim, p) for p in sim.preds[op])
    return False
