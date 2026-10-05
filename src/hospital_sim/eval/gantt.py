"""Gantt chart of a simulation: one row per resource.

A multi-resource operation is drawn on every resource row it occupies, so a
surgery shows up as aligned bars on the operating room, both surgeons, the
anaesthetist and two nurses. Colour encodes the operation type; each bar is
labelled with its patient.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hospital_sim.sim.simulator import OpState, Simulator

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Fixed categorical order; operation types beyond it are drawn in OTHER.
PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
OTHER = "#898781"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Fills dark enough to need a light label.
_DARK_FILLS = {"#2a78d6", "#008300", "#4a3aa7", "#e34948", OTHER}


def plot_gantt(sim: Simulator, path: str | Path | None = None, title: str | None = None) -> Figure:
    """Draw every finished operation of ``sim``; save to ``path`` if given."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    res = sim.resources
    # Rows grouped by role, in declaration order; a resource is listed once,
    # under the first role it belongs to.
    rows: list[int] = []
    for members in res.members:
        rows.extend(r for r in members if r not in rows)
    row_of = {r: i for i, r in enumerate(rows)}

    type_names = [t.name for t in sim.op_types]
    # Colours go to the operation types that occupy a resource, in declaration
    # order. Pure delays never appear on a resource row and take none.
    drawn = [i for i, t in enumerate(sim.op_types) if t.requirements]
    colour = {i: PALETTE[k] if k < len(PALETTE) else OTHER for k, i in enumerate(drawn)}

    finished = [op for op in range(sim.n_ops) if sim.state[op] == OpState.DONE]
    horizon = max((sim.end_time[op] for op in finished), default=1.0) or 1.0

    fig, ax = plt.subplots(figsize=(12, 0.34 * len(rows) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Unavailability first, behind the operations: roster gaps, absences and
    # breakdowns as pale spans, turnover as a grey bar after the operation.
    seen_reasons: set[str] = set()
    for r, start, end, reason in sim.downtime_log:
        if start >= horizon:
            continue
        seen_reasons.add("turnover" if reason == "turnover" else "unavailable")
        ax.barh(
            row_of[r],
            min(end, horizon) - start,
            left=start,
            height=0.62 if reason == "turnover" else 0.9,
            color=AXIS if reason == "turnover" else GRID,
            edgecolor=SURFACE if reason == "turnover" else "none",
            linewidth=1.5 if reason == "turnover" else 0,
            zorder=1.5 if reason == "turnover" else 0.5,
            gid=reason,
        )
    used_types: set[int] = set()
    for op in finished:
        team = sim.team[op]
        assert team is not None
        op_type = sim.type_of[op]
        start, width = sim.start_time[op], sim.end_time[op] - sim.start_time[op]
        for group in team:
            for r in group:
                used_types.add(op_type)
                ax.barh(
                    row_of[r],
                    width,
                    left=start,
                    height=0.62,
                    color=colour[op_type],
                    edgecolor=SURFACE,  # surface gap between adjacent bars
                    linewidth=1.5,
                    zorder=2,
                    gid="operation",
                )
                # Label only where it fits; the legend and table carry the rest.
                if width / horizon > 0.022:
                    ax.text(
                        start + width / 2,
                        row_of[r],
                        f"P{sim.patient_of[op]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#ffffff" if colour[op_type] in _DARK_FILLS else INK,
                        zorder=3,
                    )

    ax.set_yticks(range(len(rows)), [res.resource_ids[r] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, horizon * 1.01)
    ax.set_xlabel(f"time ({sim.instance.scenario.time_unit}s)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(
        title or f"{sim.instance.scenario.name}, seed {sim.instance.seed}",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(
        handles=[Patch(color=colour[i], label=type_names[i]) for i in sorted(used_types)]
        + [
            Patch(color=fill, label=label)
            for key, fill, label in (
                ("turnover", AXIS, "turnover"),
                ("unavailable", GRID, "off duty / absent / down"),
            )
            if key in seen_reasons
        ],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig
