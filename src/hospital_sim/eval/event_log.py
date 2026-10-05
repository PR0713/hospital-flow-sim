"""Event log: one row per finished operation.

This is the format the calibration functions read. A hospital extract with
the same columns can be fitted in exactly the same way.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hospital_sim.sim.simulator import OpState, Simulator

COLUMNS = (
    "episode",
    "patient",
    "patient_class",
    "arrival_time",
    "op_type",
    "step",
    "ready_time",
    "start_time",
    "end_time",
    "resources",
)
_NUMERIC = ("arrival_time", "ready_time", "start_time", "end_time")


def event_log_rows(sim: Simulator, episode: int = 0) -> list[dict[str, Any]]:
    """Rows for every finished operation of ``sim``. For analysis after an
    episode; it contains realized times and must not be shown to a policy."""
    ids = sim.resources.resource_ids
    rows = []
    for op in sim.instance.operations:
        if sim.state[op.id] != OpState.DONE:
            continue
        patient = sim.instance.patients[op.patient_id]
        team = sim.team[op.id] or ()
        rows.append(
            {
                "episode": episode,
                "patient": patient.id,
                "patient_class": patient.patient_class,
                "arrival_time": patient.arrival_time,
                "op_type": op.op_type,
                "step": op.step_id,
                "ready_time": sim.ready_time[op.id],
                "start_time": sim.start_time[op.id],
                "end_time": sim.end_time[op.id],
                "resources": ";".join(ids[r] for group in team for r in group),
            }
        )
    return rows


def write_event_log(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_event_log(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["episode"], row["patient"] = int(raw["episode"]), int(raw["patient"])
            for column in _NUMERIC:
                row[column] = float(raw[column])
            rows.append(row)
        return rows
