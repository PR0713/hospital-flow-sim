from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hospital_sim.domain import Scenario  # noqa: E402
from hospital_sim.eval import plot_gantt  # noqa: E402
from hospital_sim.generators import generate_instance  # noqa: E402
from hospital_sim.sim import Simulator, run  # noqa: E402


def test_gantt_draws_one_bar_per_occupied_resource(
    example_scenario: Scenario, tmp_path: Path
) -> None:
    sim = run(Simulator(generate_instance(example_scenario, 3)))
    target = tmp_path / "gantt.png"
    fig = plot_gantt(sim, target)

    ax = fig.axes[0]
    expected_bars = sum(len(group) for team in sim.team if team for group in team)
    assert sum(p.get_gid() == "operation" for p in ax.patches) == expected_bars
    assert [label.get_text() for label in ax.get_yticklabels()] == sorted(
        sim.resources.resource_ids, key=[t.get_text() for t in ax.get_yticklabels()].index
    )
    assert len(ax.get_yticklabels()) == len(sim.resources.resource_ids)
    assert target.stat().st_size > 0


def test_surgery_appears_on_all_six_team_rows(example_scenario: Scenario) -> None:
    sim = run(Simulator(generate_instance(example_scenario, 3)))
    surgery = next(o for o in range(sim.n_ops) if sim.op_types[sim.type_of[o]].name == "Surgery")
    fig = plot_gantt(sim)
    start, width = sim.start_time[surgery], sim.end_time[surgery] - sim.start_time[surgery]
    rows = [
        p for p in fig.axes[0].patches if p.get_x() == start and abs(p.get_width() - width) < 1e-9
    ]
    assert len(rows) == 6
