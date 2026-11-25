"""v0.1 baseline results on the frozen tiers' test seeds.

    caffeinate -ims .venv/bin/python benchmarks/run_v01_baselines.py   # ~10-20 min

Runs every dispatching rule (including random) with every allocation rule on
the 200 test seeds of T1-T4 through ``hospital_sim.eval.evaluate`` (common
instances, paired comparisons), writes compact summaries to
``benchmarks/results/v0.1/baselines_<tier>.json``, draws a Gantt chart of the
best heuristic on the first test instance of each tier into
``docs/figures/v0.1/``, and regenerates ``docs/BASELINE_RESULTS_v0.1.md``.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from hospital_sim.baselines import DISPATCHER_FACTORIES, DISPATCHERS, Policy
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.eval import TIERS, evaluate, instances_from_seeds, load_tier
from hospital_sim.sim.allocation import RULES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "v0.1"
FIGURES = ROOT / "docs" / "figures" / "v0.1"
METRICS = (
    "cost",
    "weighted_mean_flow_time",
    "total_weighted_flow_time",
    "mean_flow_time",
    "makespan",
    "overtime",
    "left",
    "truncated",
)
DISPATCH = ("random", *DISPATCHERS, *DISPATCHER_FACTORIES)
REFERENCE = "random+flexible"


def leave_penalty(tier: str) -> float:
    return HospitalEnv(EnvConfig(load_tier(tier).scenario)).leave_penalty


def _run(tier: str, dispatcher: str, allocation: str) -> tuple[str, dict[str, list[float]]]:
    """Per-episode values of one policy on one tier (runs in a worker)."""
    t = load_tier(tier)
    result = evaluate(t.scenario, [Policy(dispatcher, allocation)], seeds=t.test_seeds)
    name = f"{dispatcher}+{allocation}"
    penalty = leave_penalty(tier)
    values = {m: result.series(name, m).tolist() for m in METRICS if m != "cost"}
    # The env's episode cost: total weighted flow time plus the leave penalty,
    # in minutes x weight. This is what the default reward sums to (unscaled).
    values["cost"] = (
        result.series(name, "total_weighted_flow_time")
        + penalty * result.series(name, "left_weight")
    ).tolist()
    return name, values


def _ci(x: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), half


def summarise(tier: str, series: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    # Fixed order for ties (several policies often give identical episodes):
    # dispatchers as listed, allocation rules with the default "flexible" first.
    canonical = [f"{d}+{a}" for d in DISPATCH for a in RULES if f"{d}+{a}" in series]
    summary = {
        name: {m: _ci(np.array(series[name][m])) for m in METRICS} for name in canonical
    }
    heuristics = [n for n in canonical if n in series and not n.startswith("random+")]
    best = min(heuristics, key=lambda n: summary[n]["cost"][0])
    best_wmft = min(heuristics, key=lambda n: summary[n]["weighted_mean_flow_time"][0])
    tied_with_best = [
        n for n in heuristics if n != best and series[n]["cost"] == series[best]["cost"]
    ]
    gaps = {}
    for metric in ("cost", "weighted_mean_flow_time"):
        diff = np.array(series[best][metric]) - np.array(series[REFERENCE][metric])
        gaps[metric] = _ci(diff)
    best_random = min(
        (n for n in canonical if n.startswith("random+")), key=lambda n: summary[n]["cost"][0]
    )
    t = load_tier(tier)
    return {
        "tier": tier,
        "episodes": len(t.test_seeds),
        "leave_penalty": leave_penalty(tier),
        "summary": summary,
        "best": best,
        "best_by_weighted_mean_flow_time": best_wmft,
        "reference": REFERENCE,
        "best_random": best_random,
        "tied_with_best": tied_with_best,
        "gap_best_minus_reference": gaps,
    }


def gantt(tier: str, policy: str) -> Path:
    from hospital_sim.eval.gantt import plot_gantt

    t = load_tier(tier)
    dispatcher, allocation = policy.split("+")
    sim = Policy(dispatcher, allocation).run(
        instances_from_seeds(t.scenario, t.test_seeds[:1])[0], nominal=t.scenario
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"gantt_{tier}.png"
    figure = plot_gantt(sim, path, title=f"{tier}: {policy}, first test instance")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def main(tiers: tuple[str, ...] = TIERS, workers: int = 8) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = [(t, d, a) for t in tiers for d in DISPATCH for a in RULES]
    started = time.perf_counter()
    series: dict[str, dict[str, dict[str, list[float]]]] = {t: {} for t in tiers}
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_run, *job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            tier = futures[future][0]
            name, values = future.result()
            series[tier][name] = values
            if done % 16 == 0:
                print(f"{done}/{len(jobs)} policies done, {time.perf_counter() - started:.0f}s")
    for tier in tiers:
        result = summarise(tier, series[tier])
        result["figure"] = str(gantt(tier, result["best"]).relative_to(ROOT))
        result["wall_seconds_all_tiers"] = time.perf_counter() - started
        (RESULTS / f"baselines_{tier}.json").write_text(json.dumps(result, indent=1))
        gap = result["gap_best_minus_reference"]
        print(f"{tier}: best {result['best']}, gap vs {REFERENCE} {gap}")
    from v01_report import write_report

    write_report()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(tuple(sys.argv[1:]) or TIERS)
