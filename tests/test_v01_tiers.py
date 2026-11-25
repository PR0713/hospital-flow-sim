"""v0.1, task 9: frozen benchmark tiers with committed seed lists."""

from __future__ import annotations

import shutil
from pathlib import Path

import gymnasium as gym
import pytest

import hospital_sim  # noqa: F401  (registers SchedSim-v0)
from hospital_sim.baselines import Policy
from hospital_sim.domain.model import ArrivalKind
from hospital_sim.env import EnvConfig, HospitalEnv, instance_seed
from hospital_sim.eval import TIERS, evaluate, instances_from_seeds, load_tier
from hospital_sim.eval.tiers import N_EVAL, N_TEST, TIER_ROOT
from hospital_sim.generators.config import load_config


@pytest.mark.parametrize("name", TIERS)
def test_tier_loads_validates_and_has_frozen_disjoint_seeds(name: str) -> None:
    tier = load_tier(name)
    _, warnings = load_config(tier.path / "config.yaml").to_scenario()
    assert warnings == []
    assert len(tier.eval_seeds) == N_EVAL and len(tier.test_seeds) == N_TEST
    assert len(set(tier.eval_seeds) | set(tier.test_seeds)) == N_EVAL + N_TEST
    assert {(s >> 60) & 0xF for s in tier.eval_seeds} == {1}  # eval split
    assert {(s >> 60) & 0xF for s in tier.test_seeds} == {2}  # test split
    env = HospitalEnv(EnvConfig(tier.scenario))
    for instance in instances_from_seeds(tier.scenario, tier.test_seeds[:20]):
        assert instance.patients and env._fits(instance)


def test_tiers_increase_in_difficulty_as_specified() -> None:
    t1, t2, t3, t4 = (load_tier(n).scenario for n in TIERS)
    # T1: static, single-skill, low variance, no rosters or breakdowns.
    assert all(a.kind is ArrivalKind.STATIC for a in t1.arrivals)
    assert all(len(r.roles) == 1 for r in t1.resources)
    assert max(t.duration.cv for t in t1.operation_types) <= 0.1
    assert not t1.calendars and not any(r.breakdown for r in t1.resources)
    # T2: Poisson arrivals, multi-role resources, lognormal durations, no rosters.
    assert all(a.kind is ArrivalKind.POISSON for a in t2.arrivals)
    assert any(len(r.roles) > 1 for r in t2.resources)
    assert {t.duration.kind.value for t in t2.operation_types} == {"lognormal"}
    assert not t2.calendars and t2.complexity.rho == 0
    # T3: T2 plus rosters, absences, breakdowns and correlated durations.
    assert t3.calendars and any(c.absence_probability > 0 for c in t3.calendars)
    assert any(r.breakdown for r in t3.resources) and t3.complexity.rho > 0
    # T4: hidden, branching and repeating steps, patience, scheduled patients.
    steps = [s for p in t4.pathways for s in p.steps]
    assert any(s.revealed_by for s in steps) and any(s.repeat_max for s in steps)
    assert any(s.branch_group for s in steps)
    assert any(a.patience_mean for a in t4.arrivals)


def test_evaluate_on_explicit_seeds_matches_the_same_instances() -> None:
    tier = load_tier("T2")
    seeds = tier.test_seeds[:5]
    by_seed = evaluate(tier.scenario, [Policy("fifo")], seeds=seeds)
    assert by_seed.n_episodes == 5
    sims = [Policy("fifo").run(i) for i in instances_from_seeds(tier.scenario, seeds)]
    assert [r.values["makespan"] for r in by_seed.records["fifo+flexible"]] == [
        max(s.completion_time) for s in sims
    ]
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(tier.scenario, [Policy("fifo")], 5, seeds=seeds)


def test_training_seeds_are_refused() -> None:
    tier = load_tier("T1")
    with pytest.raises(ValueError, match="not an eval or test"):
        evaluate(tier.scenario, [Policy("fifo")], seeds=[instance_seed(0, "train", 0, 0)])


def test_an_edited_tier_config_is_refused(tmp_path: Path) -> None:
    shutil.copytree(TIER_ROOT / "T1", tmp_path / "T1")
    config = tmp_path / "T1" / "config.yaml"
    config.write_text(config.read_text().replace("count: 6", "count: 7"))
    with pytest.raises(ValueError, match="frozen"):
        load_tier("T1", root=tmp_path)


def test_gym_make_accepts_tier_names() -> None:
    env = gym.make("SchedSim-v0", scenario="T3")
    assert env.unwrapped.config.scenario.name == "T3"
