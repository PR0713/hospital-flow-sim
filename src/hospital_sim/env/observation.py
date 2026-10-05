"""Builds observations from what a scheduler is allowed to know.

Rules this module must keep (tests enforce them):

* only operations in ``sim.visible_ops`` and patients who have arrived appear;
* duration knowledge comes from the *nominal* scenario: expected durations
  and expected remaining time, never a realized completion time;
* roster information comes from the public calendar; an absence or breakdown
  shows only as "unavailable now", never as a known end time;
* it never touches ``instance.hidden``, the event list or ``downtime_log``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hospital_sim.domain.dag import contract
from hospital_sim.domain.model import ProblemInstance, Scenario
from hospital_sim.sim import availability
from hospital_sim.sim.durations import StandardDurationModel, StartContext
from hospital_sim.sim.resources import FREE
from hospital_sim.sim.simulator import OpState, PatientStatus, Simulator

# Bump when the meaning, order or number of observation fields changes, so an
# agent trained against one layout is never silently fed another.
OBSERVATION_SCHEMA_VERSION = 1

OP_FEATURES = (
    "is_waiting",
    "is_ready",
    "is_running",
    "is_done",
    "is_cancelled",
    "startable",
    "expected_duration",
    "waited",
    "elapsed",
    "expected_remaining",
    "realized_duration",
    "open_predecessors",
    "successors",
    "requires_patient",
    "expected_gated_work",
)
PATIENT_FEATURES = (
    "in_system",
    "finished",
    "expected",
    "left",
    "no_show",
    "weight",
    "urgent",
    "acuity",
    "time_in_system",
    "scheduled_in",
    "open_operations",
    "expected_remaining_work",
    "busy",
)
RESOURCE_FEATURES = (
    "available",
    "busy",
    "unavailable",
    "rostered_on",
    "unplanned_down",
    "in_turnover",
    "time_to_roster_change",
    "expected_remaining",
    "work_done",
    "speed",
)
ROLE_FEATURES = ("idle", "members", "demand")
GLOBAL_FEATURES = (
    "time",
    "patients_in_system",
    "ready",
    "startable",
    "running",
    "consecutive_waits",
)

_UNPLANNED = (1 << availability.ABSENCE) | (1 << availability.BREAKDOWN)
# Cap on "time to roster change", in units of time_scale.
_ROSTER_CAP = 1e3

FloatArray = NDArray[np.float32]


class ObservationBuilder:
    """Per-episode observation builder. ``nominal`` is the configured scenario,
    which may differ from the randomized one the simulator runs."""

    def __init__(
        self,
        nominal: Scenario,
        instance: ProblemInstance,
        sim: Simulator,
        *,
        max_patients: int,
        max_ops: int,
        max_edges: int,
        time_scale: float,
        graph: bool,
        gated_work_prior: bool = True,
    ) -> None:
        self.sim = sim
        self.max_patients, self.max_ops, self.max_edges = max_patients, max_ops, max_edges
        self.scale = time_scale
        self.graph = graph
        self.model = StandardDurationModel(nominal)

        ops = instance.operations
        n = len(ops)
        self.type_of = np.array(sim.type_of, dtype=np.int32).reshape(n)
        self.patient_of = np.array(sim.patient_of, dtype=np.int32).reshape(n)
        expected = np.array([self.model.base(i).mean for i in range(len(sim.op_types))])
        self.expected = expected[self.type_of] if n else np.zeros(0)
        self.needs_patient = np.array(sim.needs_patient, dtype=np.float32)[self.type_of]
        edges = [(pred, op.id) for op in ops for pred in op.predecessors]
        self.edges = np.array(edges, dtype=np.int32).reshape(-1, 2)
        n_roles = len(sim.resources.role_names)
        requirements = np.zeros((len(sim.op_types), n_roles), dtype=np.float32)
        for t, reqs in enumerate(sim.resources.requirements):
            for role, qty in reqs:
                requirements[t, role] = qty
        self.type_requirements = requirements

        patients = instance.patients
        self.arrival = np.array([p.arrival_time for p in patients], dtype=np.float64)
        self.weight = np.array([p.weight for p in patients], dtype=np.float32)
        self.urgent = np.array([p.urgent for p in patients], dtype=np.float32)
        classes = max(nominal.complexity.acuity_classes - 1, 1)
        self.acuity = np.array(
            [-1.0 if p.acuity_class is None else p.acuity_class / classes for p in patients],
            dtype=np.float32,
        )

        n_res = len(sim.resources.resource_ids)
        self.resource_roles = np.zeros((n_res, n_roles), dtype=np.int8)
        for r, roles in enumerate(sim.resources.roles_of):
            self.resource_roles[r, list(roles)] = 1
        self.members = np.array([len(m) for m in sim.resources.members], dtype=np.float32)
        self.speed = np.array(sim.resource_speed, dtype=np.float32)
        self.calendars = [
            None if r.calendar is None else nominal.calendar_by_name[r.calendar]
            for r in nominal.resources
        ]

        self.scheduled = np.array(
            [np.nan if p.scheduled_time is None else p.scheduled_time for p in patients],
            dtype=np.float64,
        )
        # Template knowledge: can a pathway hide steps at all? If not, the
        # visible precedence edges are simply the edges between visible nodes.
        self.gating = any(
            step.revealed_by is not None or step.repeat_max > 0
            for pathway in nominal.pathways
            for step in pathway.steps
        )
        self.preds_of = {op.id: op.predecessors for op in ops}
        self.ops_of_patient = [[op.id for op in p.operations] for p in patients]
        self.gated_work = (
            self._gated_work_prior(nominal, instance) if gated_work_prior else np.zeros(n)
        )

        # Engine ids -> rows. Rows are handed out in the order things become
        # visible, so an index never reveals a hidden operation or patient.
        self.slot_of = np.full(n, -1, dtype=np.int32)
        self.n_slots = 0
        self.patient_slot = np.full(len(patients), -1, dtype=np.int32)
        self.n_patient_slots = 0
        self._visible_edges = np.zeros((2, 0), dtype=np.int32)

    # --- slots ---------------------------------------------------------------

    def _gated_work_prior(self, nominal: Scenario, instance: ProblemInstance) -> np.ndarray:
        """Per operation: expected work its completion may reveal, from the
        pathway template alone. It is the same number whether or not the
        revealed steps were actually drawn for this patient."""
        type_index = {t.name: i for i, t in enumerate(nominal.operation_types)}
        prior = np.zeros(instance.n_operations)
        for patient in instance.patients:
            steps = {s.id: s for s in nominal.pathway_by_name[patient.pathway].steps}
            group_weight: dict[str, float] = {}
            for step in steps.values():
                if step.branch_group is not None:
                    group_weight[step.branch_group] = (
                        group_weight.get(step.branch_group, 0.0) + step.branch_weight
                    )

            def mean_of(step_id: str, patient: Any = patient, steps: Any = steps) -> float:
                name = nominal.resolve_op_type(
                    steps[step_id].op_type, patient.patient_class, patient.acuity_class
                )
                return self.model.base(type_index[name]).mean

            for op in patient.operations:
                base, _, copy = op.step_id.partition("#")
                step = steps[base]
                total = 0.0
                if step.repeat_max > 0:
                    done_copies = int(copy) if copy else 1
                    p = step.repeat_probability
                    total += mean_of(base) * sum(
                        p**j for j in range(1, step.repeat_max - done_copies + 2)
                    )
                if not copy:
                    for other in steps.values():
                        if other.revealed_by != base:
                            continue
                        chance = (
                            other.branch_weight / group_weight[other.branch_group]
                            if other.branch_group is not None
                            else other.probability
                        )
                        total += chance * mean_of(other.id)
                prior[op.id] = total
        return prior

    def sync_slots(self) -> None:
        sim = self.sim
        visible = sim.visible_ops
        grew = len(visible) > self.n_slots
        for slot in range(self.n_slots, len(visible)):
            self.slot_of[visible[slot]] = slot
        self.n_slots = len(visible)
        known = sim.visible_patients
        for slot in range(self.n_patient_slots, len(known)):
            self.patient_slot[known[slot]] = slot
        self.n_patient_slots = len(known)
        if grew:
            self._visible_edges = self._edges_between_visible()

    def _edges_between_visible(self) -> np.ndarray:
        """Precedence among visible operations, in row numbering.

        If a hidden operation sits between two visible ones, the edge bridges
        over it, so the graph looks exactly as if the hidden one did not exist.
        """
        if not len(self.edges):
            return np.zeros((2, 0), dtype=np.int32)
        if not self.gating:
            src, dst = self.slot_of[self.edges[:, 0]], self.slot_of[self.edges[:, 1]]
            keep = (src >= 0) & (dst >= 0)
            return np.stack([src[keep], dst[keep]]).astype(np.int32)
        pairs: list[tuple[int, int]] = []
        for patient in self.sim.visible_patients:
            own = self.ops_of_patient[patient]
            shown = {op for op in own if self.slot_of[op] >= 0}
            if not shown:
                continue
            bridged = contract({op: self.preds_of[op] for op in own}, shown)
            pairs.extend(
                (int(self.slot_of[pred]), int(self.slot_of[op]))
                for op, op_preds in bridged.items()
                for pred in op_preds
            )
        return np.array(pairs, dtype=np.int32).reshape(-1, 2).T

    def op_at(self, slot: int) -> int:
        return self.sim.visible_ops[slot]

    # --- pieces --------------------------------------------------------------

    def _expected_remaining(self, op: int) -> float:
        """Conditional expectation under the nominal model and the actual team."""
        sim = self.sim
        team = sim.team[op]
        assert team is not None
        dist = self.model.at_start(
            StartContext(
                op=op,
                op_type=sim.type_of[op],
                team=team,
                now=sim.start_time[op],
                resources=sim.resources,
            )
        )
        return dist.expected_remaining(sim.now - sim.start_time[op])

    def build(self, startable: list[int], consecutive_waits: int, max_waits: int) -> dict[str, Any]:
        sim, scale = self.sim, self.scale
        self.sync_slots()
        k = self.n_slots
        vis = np.array(sim.visible_ops, dtype=np.int64)
        now = sim.now

        state = np.array(sim.state, dtype=np.int64)[vis]
        ready_time = np.array(sim.ready_time)[vis]
        start_time = np.array(sim.start_time)[vis]
        end_time = np.array(sim.end_time)[vis]
        waiting, ready = state == OpState.WAITING, state == OpState.READY
        running, done = state == OpState.RUNNING, state == OpState.DONE
        cancelled = state == OpState.CANCELLED
        open_ = waiting | ready | running

        expected = self.expected[vis]
        remaining = np.where(waiting | ready, expected, 0.0)
        for i in np.flatnonzero(running):
            remaining[i] = self._expected_remaining(int(vis[i]))
        with np.errstate(invalid="ignore"):
            waited = np.where(
                ready, now - ready_time, np.where(running | done, start_time - ready_time, 0.0)
            )
            elapsed = np.where(running, now - start_time, 0.0)
            realized = np.where(done, end_time - start_time, 0.0)

        is_startable = np.zeros(k, dtype=bool)
        if startable:
            is_startable[self.slot_of[startable]] = True

        src, dst = self._visible_edges
        unresolved_src = ~(done | cancelled)[src] if len(src) else np.zeros(0, dtype=bool)
        open_preds = np.bincount(dst[unresolved_src], minlength=k) if k else np.zeros(0)
        n_succ = np.bincount(src, minlength=k) if k else np.zeros(0)

        ops = np.zeros((self.max_ops, len(OP_FEATURES)), dtype=np.float32)
        ops[:k] = np.column_stack(
            [
                waiting,
                ready,
                running,
                done,
                cancelled,
                is_startable,
                expected / scale,
                np.nan_to_num(waited) / scale,
                elapsed / scale,
                remaining / scale,
                np.nan_to_num(realized) / scale,
                open_preds,
                n_succ,
                self.needs_patient[vis],
                np.where(open_, self.gated_work[vis], 0.0) / scale,
            ]
        ).reshape(k, len(OP_FEATURES))

        op_type = np.full(self.max_ops, -1, dtype=np.int32)
        op_patient = np.full(self.max_ops, -1, dtype=np.int32)
        op_type[:k] = self.type_of[vis]
        op_patient[:k] = self.patient_slot[self.patient_of[vis]]
        op_valid = np.zeros(self.max_ops, dtype=np.int8)
        op_valid[:k] = 1
        op_requirements = np.zeros(
            (self.max_ops, self.type_requirements.shape[1]), dtype=np.float32
        )
        op_requirements[:k] = self.type_requirements[self.type_of[vis]]

        # --- patients the scheduler knows about: the booking list, then
        # walk-ins as they arrive.
        known = np.array(sim.visible_patients, dtype=np.int64)
        m = len(known)
        status = np.array(sim.patient_status, dtype=np.int64)[known]
        present = status == PatientStatus.PRESENT
        expected_soon = status == PatientStatus.EXPECTED
        left, no_show = status == PatientStatus.LEFT, status == PatientStatus.NO_SHOW
        finished = status == PatientStatus.FINISHED
        gone = finished | left
        completion = np.array(sim.completion_time)[known]
        time_in_system = np.zeros(m)
        time_in_system[present] = now - self.arrival[known[present]]
        time_in_system[gone] = completion[gone] - self.arrival[known[gone]]
        scheduled_in = np.zeros(m)
        scheduled_in[expected_soon] = self.scheduled[known[expected_soon]] - now
        row_of_vis = self.patient_slot[self.patient_of[vis]]
        open_ops = np.bincount(row_of_vis[open_], minlength=m)[:m] if k else np.zeros(m)
        work_left = (
            np.bincount(row_of_vis, weights=np.where(open_, remaining, 0.0), minlength=m)[:m]
            if k
            else np.zeros(m)
        )
        patients = np.zeros((self.max_patients, len(PATIENT_FEATURES)), dtype=np.float32)
        patients[:m] = np.column_stack(
            [
                present,
                finished,
                expected_soon,
                left,
                no_show,
                self.weight[known],
                self.urgent[known],
                self.acuity[known],
                time_in_system / scale,
                scheduled_in / scale,
                open_ops,
                work_left / scale,
                np.array(sim.patient_busy, dtype=np.float32)[known],
            ]
        ).reshape(m, len(PATIENT_FEATURES))
        patient_valid = np.zeros(self.max_patients, dtype=np.int8)
        patient_valid[:m] = 1

        # --- resources
        res = sim.resources
        n_res = len(res.holder)
        resources = np.zeros((n_res, len(RESOURCE_FEATURES)), dtype=np.float32)
        resource_op = np.full(n_res, -1, dtype=np.int32)
        roster: dict[str, tuple[bool, float]] = {}  # one lookup per calendar
        for r in range(n_res):
            holder = res.holder[r]
            busy = holder != FREE
            calendar = self.calendars[r]
            on, until = True, float("inf")
            if calendar is not None:
                if calendar.name not in roster:
                    roster[calendar.name] = availability.planned_status(calendar, now)
                on, until = roster[calendar.name]
            row = resources[r]
            row[0] = not busy and res.blocked[r] == 0
            row[1] = busy
            row[2] = not busy and res.blocked[r] > 0
            row[3] = on
            row[4] = bool(sim.down_mask[r] & _UNPLANNED)
            row[5] = sim.in_turnover[r]
            row[6] = min(until / scale, _ROSTER_CAP)
            if busy:
                slot = int(self.slot_of[holder])
                resource_op[r] = slot
                row[7] = ops[slot, OP_FEATURES.index("expected_remaining")]
            row[8] = res.work_time[r] / scale
            row[9] = self.speed[r]

        # --- roles
        demand = (
            self.type_requirements[self.type_of[vis[ready]]].sum(axis=0)
            if ready.any()
            else np.zeros(len(self.members))
        )
        roles = np.column_stack(
            [np.array(res.idle_count, dtype=np.float32), self.members, demand]
        ).astype(np.float32)

        observation: dict[str, Any] = {
            "ops": ops,
            "op_valid": op_valid,
            "op_type": op_type,
            "op_patient": op_patient,
            "op_requirements": op_requirements,
            "patients": patients,
            "patient_valid": patient_valid,
            "resources": resources,
            "resource_roles": self.resource_roles.copy(),
            "resource_op": resource_op,
            "roles": roles,
            "global": np.array(
                [
                    now / scale,
                    float(present.sum()),
                    float(ready.sum()),
                    float(len(startable)),
                    float(running.sum()),
                    consecutive_waits / max(max_waits, 1),
                ],
                dtype=np.float32,
            ),
        }
        if self.graph:
            if len(src) > self.max_edges:
                raise ValueError(f"{len(src)} precedence edges exceed max_edges={self.max_edges}")
            edges = np.full((2, self.max_edges), -1, dtype=np.int32)
            edges[0, : len(src)] = src
            edges[1, : len(dst)] = dst
            observation["precedence_edges"] = edges
        return observation

    def graph_view(self, observation: dict[str, Any]) -> dict[str, Any]:
        """The same information without padding, as typed nodes and edge lists
        (``edge_index`` arrays of shape ``[2, E]``, source row first)."""
        k = int(observation["op_valid"].sum())
        m = int(observation["patient_valid"].sum())
        self.sync_slots()
        precedes = self._visible_edges.copy()
        requires = np.argwhere(observation["op_requirements"][:k] > 0).T
        uses = np.argwhere(observation["resource_op"] >= 0).ravel()
        return {
            "nodes": {
                "op": observation["ops"][:k],
                "patient": observation["patients"][:m],
                "resource": observation["resources"],
                "role": observation["roles"],
            },
            "op_type": observation["op_type"][:k],
            "edges": {
                "op__precedes__op": precedes,
                "op__requires__role": requires,
                "op__of__patient": np.stack([np.arange(k), observation["op_patient"][:k]]),
                "resource__fills__role": np.argwhere(observation["resource_roles"] > 0).T,
                "op__uses__resource": np.stack([observation["resource_op"][uses], uses]),
            },
            "edge_attr": {
                "op__requires__role": observation["op_requirements"][:k][tuple(requires)],
            },
            "global": observation["global"],
        }
