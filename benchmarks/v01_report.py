"""Render docs/BASELINE_RESULTS_v0.1.md from benchmarks/results/v0.1/*.json.

    .venv/bin/python benchmarks/v01_report.py

Sections appear for whichever result files exist: baselines (run_v01_baselines.py),
PPO smoke test (ppo_smoke.py) and throughput (run_v01_throughput.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "v0.1"
DOC = ROOT / "docs" / "BASELINE_RESULTS_v0.1.md"
TIERS = ("T1", "T2", "T3", "T4")
COLUMNS = (
    ("cost", "episode cost", 0),
    ("weighted_mean_flow_time", "weighted mean flow", 1),
    ("mean_flow_time", "mean flow", 1),
    ("makespan", "makespan", 0),
    ("overtime", "overtime", 1),
    ("left", "leavers", 2),
)


def _load(name: str) -> Any:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _cell(value: list[float], digits: int) -> str:
    mean, half = value
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _gap(value: list[float]) -> str:
    return f"{value[0]:+.0f} ± {value[1]:.0f}"


def _baselines(lines: list[str]) -> None:
    results = {t: _load(f"baselines_{t}.json") for t in TIERS}
    results = {t: r for t, r in results.items() if r}
    if not results:
        return
    any_result = next(iter(results.values()))
    lines += [
        "## Heuristic baselines",
        "",
        f"Every dispatching rule (including `random`) with every allocation rule, on the "
        f"{any_result['episodes']} frozen **test** seeds of each tier, through "
        "`hospital_sim.eval.evaluate` (all policies on identical instances). Cells are "
        "mean ± half-width of the 95% t confidence interval over episodes. Times are in "
        "minutes.",
        "",
        "- **episode cost** is what the default reward sums to, before scaling: total "
        "weighted flow time plus `leave_penalty` × weight of patients who left unseen "
        "(minutes × weight). The best heuristic is chosen on it.",
        "- **weighted mean flow** is total weighted flow time over the arrived weight. "
        "Leavers count with the time they waited, so read it together with **leavers**.",
        "- **overtime** is resource-minutes worked past the end of a shift (tiers with "
        "rosters only).",
        "- The truncation rate is 0% for every heuristic (they always run to completion), "
        "so it is not repeated per row. The column exists for RL agents in §PPO.",
        "",
        "### Summary",
        "",
        "| Tier | Best heuristic (episode cost) | its episode cost | `random+flexible` | "
        "best − random (paired) | best by weighted mean flow |",
        "|---|---|---|---|---|---|",
    ]
    for tier, r in results.items():
        best, ref = r["best"], r["reference"]
        s = r["summary"]
        gap, half = r["gap_best_minus_reference"]["cost"]
        relative = 100 * gap / s[ref]["cost"][0]
        lines.append(
            f"| {tier} | **{best}** | {_cell(s[best]['cost'], 0)} | {_cell(s[ref]['cost'], 0)} | "
            f"{gap:+.0f} ± {half:.0f} ({relative:+.1f}%) | {r['best_by_weighted_mean_flow_time']} |"
        )
    lines += [""]
    for tier, r in results.items():
        s = r["summary"]
        canonical = list(s)  # written in the runner's fixed order
        order = sorted(s, key=lambda name: (s[name]["cost"][0], canonical.index(name)))
        lines += [
            f"### {tier}",
            "",
            f"Leave penalty {r['leave_penalty']:.0f} min × weight. Best heuristic in bold; "
            f"paired gap of the best heuristic to `{r['reference']}`: episode cost "
            f"{r['gap_best_minus_reference']['cost'][0]:+.0f} ± "
            f"{r['gap_best_minus_reference']['cost'][1]:.0f}, weighted mean flow "
            f"{r['gap_best_minus_reference']['weighted_mean_flow_time'][0]:+.1f} ± "
            f"{r['gap_best_minus_reference']['weighted_mean_flow_time'][1]:.1f}. "
            f"Best random allocation: `{r['best_random']}`."
            + (
                " Identical on every episode to the best: "
                + ", ".join(f"`{n}`" for n in r["tied_with_best"])
                + "."
                if r.get("tied_with_best")
                else ""
            ),
            "",
            "| Policy | " + " | ".join(label for _, label, _ in COLUMNS) + " | truncated |",
            "|---|" + "---|" * (len(COLUMNS) + 1),
        ]
        for name in order:
            label = f"**{name}**" if name == r["best"] else name
            cells = [_cell(s[name][key], digits) for key, _, digits in COLUMNS]
            truncated = f"{100 * s[name]['truncated'][0]:.0f}%"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {truncated} |")
        if r.get("figure"):
            lines += [
                "",
                f"![{tier} Gantt chart]({Path(r['figure']).relative_to('docs')})",
                "",
                f"*{tier}: Gantt chart of `{r['best']}` on the first test instance "
                "(one row per resource; colour = operation type; label = patient).*",
            ]
        lines.append("")


def _ppo(lines: list[str]) -> None:
    runs = [r for t in TIERS if (r := _load(f"ppo_{t}.json"))]
    if not runs:
        return
    lines += [
        "## Learnability smoke test (MaskablePPO)",
        "",
        "Not a tuned agent: the aim is to show that the environment is learnable and that "
        "the learner finds no way around the cost. Setup, budget and curves are below; "
        "evaluation is on the same 200 test seeds as the heuristics, through the harness.",
        "",
    ]
    first = runs[0]
    lines += [f"- {line}" for line in first["setup"]] + [""]
    lines += [
        "| Tier | budget (env steps) | wall time | trained policy: episode cost | "
        "random+flexible | best heuristic | truncated | wait share | leavers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['tier']} | {r['timesteps']:,} | {r['wall_minutes']:.0f} min | "
            f"{_cell(r['ppo']['cost'], 0)} | {_cell(r['random']['cost'], 0)} | "
            f"{_cell(r['best_heuristic']['cost'], 0)} (`{r['best_heuristic']['name']}`) | "
            f"{100 * r['ppo']['truncation_rate']:.0f}% | {100 * r['ppo']['wait_share']:.1f}% | "
            f"{r['ppo']['left'][0]:.2f} |"
        )
    lines += [""]
    for r in runs:
        lines += [
            f"### {r['tier']}",
            "",
            "Paired difference on the test seeds, episode cost: PPO − random "
            f"{_gap(r['paired']['ppo_minus_random'])}; PPO − best heuristic "
            f"{_gap(r['paired']['ppo_minus_best'])}. Weighted mean flow: PPO "
            f"{_cell(r['ppo']['weighted_mean_flow_time'], 1)}, random "
            f"{_cell(r['random']['weighted_mean_flow_time'], 1)}, best heuristic "
            f"{_cell(r['best_heuristic']['weighted_mean_flow_time'], 1)}.",
            "",
            *[f"- {note}" for note in r.get("notes", [])],
            "",
            f"![{r['tier']} learning curve]({Path(r['curve']).relative_to('docs')})",
            "",
        ]


def _throughput(lines: list[str]) -> None:
    t = _load("throughput.json")
    if not t:
        return
    lines += [
        "## Throughput baseline",
        "",
        f"Single process, {t['machine']}, Python {t['python']}, {t['steps']:,} steps per "
        "row under a random masked policy (`action_space.sample()`), resets included. "
        "*env steps/s* counts every step (starts and waits); *decisions/s* counts starts "
        "only. The bare simulator runs FIFO dispatch with first-idle teams and no "
        "observation, instance generation included. Measured with `benchmarks/"
        "run_v01_throughput.py`; compare later optimisations against these rows on the "
        "same machine.",
        "",
        "| Tier | env steps/s | env decisions/s | pointer wrapper steps/s | "
        "pointer decisions/s | bare simulator decisions/s |",
        "|---|---|---|---|---|---|",
    ]
    for row in t["rows"]:
        lines.append(
            f"| {row['tier']} | {row['env_steps']:,.0f} | {row['env_decisions']:,.0f} | "
            f"{row['pointer_steps']:,.0f} | {row['pointer_decisions']:,.0f} | "
            f"{row['bare_decisions']:,.0f} |"
        )
    lines.append("")


def write_report() -> None:
    lines = [
        "# Baseline results, v0.1",
        "",
        "Reference numbers for `hospital_sim` 0.1.0 on the frozen benchmark tiers "
        "(`benchmarks/tiers/`, see its README). Regenerate with "
        "`benchmarks/run_v01_baselines.py`, `benchmarks/ppo_smoke.py` and "
        "`benchmarks/run_v01_throughput.py`; this page is rendered by "
        "`benchmarks/v01_report.py` from `benchmarks/results/v0.1/`.",
        "",
        "**These are results on illustrative scenarios, not on a model of any real "
        "hospital.** They compare methods under controlled conditions only.",
        "",
    ]
    _baselines(lines)
    _ppo(lines)
    _throughput(lines)
    DOC.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    write_report()
