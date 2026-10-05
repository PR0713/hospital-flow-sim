"""M5: sensitivity ablations, calibration round trip, schema version."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hospital_sim.baselines import Policy, all_policies
from hospital_sim.domain import ArrivalKind, DurationKind, Scenario
from hospital_sim.env import OBSERVATION_SCHEMA_VERSION, EnvConfig, HospitalEnv
from hospital_sim.env import observation as obs_module
from hospital_sim.eval import evaluate, sample_instances
from hospital_sim.eval.calibration import fit_arrival_profile, fit_durations
from hospital_sim.eval.event_log import COLUMNS, event_log_rows, read_event_log, write_event_log
from hospital_sim.eval.sensitivity import (
    KPIS,
    LEAVE_PENALTY,
    SOURCES,
    ablate,
    format_report,
    kpi,
    run_sensitivity,
)
from hospital_sim.generators import generate_instance, load_scenario
from hospital_sim.sim import Simulator, run

from .conftest import make_scenario

FULL = Path(__file__).parent.parent / "configs" / "example_hospital_full.yaml"


@pytest.fixture(scope="module")
def full() -> Scenario:
    return load_scenario(FULL)


def simulate(scenario: Scenario, seeds: range) -> list[Simulator]:
    return [run(Simulator(generate_instance(scenario, seed), debug=seed < 3)) for seed in seeds]


# --- each ablation removes exactly its source -----------------------------------


@pytest.mark.slow
def test_every_listed_source_has_an_ablation(full: Scenario) -> None:
    assert set(SOURCES) == {
        "duration_noise",
        "latent_factor",
        "arrival_variability",
        "urgent_stream",
        "rosters_and_absences",
        "breakdowns",
        "reveal",
        "no_shows",
        "leaving",
    }
    for source in SOURCES:
        sims = simulate(ablate(full, source), range(4))
        assert all(sim.done for sim in sims), source


def test_duration_noise_off_means_expected_durations(full: Scenario) -> None:
    off = ablate(full, "duration_noise")
    assert all(t.duration.kind is DurationKind.DETERMINISTIC for t in off.operation_types)
    by_name = {t.name: t.duration.mean for t in off.operation_types}
    # The scheduler-visible expectation is unchanged...
    before = Simulator(generate_instance(full, 0))
    after = Simulator(generate_instance(off, 0))
    assert after.expected_duration == pytest.approx(before.expected_duration)
    # ...and realized durations equal it (surgery is scaled by the lead's speed).
    sim = run(after)
    for op in sim.instance.operations:
        finished = not math.isnan(sim.end_time[op.id])  # no-shows and leavers are cancelled
        if finished and not op.op_type.startswith("Surgery"):
            assert sim.end_time[op.id] - sim.start_time[op.id] == pytest.approx(by_name[op.op_type])


@pytest.mark.slow
def test_latent_factor_off_removes_within_patient_correlation(full: Scenario) -> None:
    from scipy import special

    def within(scenario: Scenario) -> float:
        pairs = []
        for seed in range(150):
            instance = generate_instance(scenario, seed)
            q = special.ndtri(np.array(instance.hidden.duration_quantiles))
            pairs += [(q[p.operations[0].id], q[p.operations[1].id]) for p in instance.patients]
        a, b = np.array(pairs).T
        return float(np.corrcoef(a, b)[0, 1])

    assert within(full) == pytest.approx(0.4, abs=0.07)
    assert abs(within(ablate(full, "latent_factor"))) < 0.07


@pytest.mark.slow
def test_arrival_variability_off_gives_regular_punctual_arrivals(full: Scenario) -> None:
    off = ablate(full, "arrival_variability")
    assert all(a.kind is ArrivalKind.SCHEDULED for a in off.arrivals)
    counts = {len(generate_instance(off, seed).patients) for seed in range(10)}
    assert len(counts) == 1  # no randomness left in who comes when (no-shows aside)
    expected = np.mean([len(generate_instance(full, s).patients) for s in range(300)])
    assert counts.pop() == pytest.approx(expected, rel=0.1)
    walk_in = next(a for a in off.arrivals if a.name == "walk_in")
    instance = generate_instance(off, 0)
    times = sorted(p.arrival_time for p in instance.patients if p.patient_class == "walk_in")
    assert np.diff(times) == pytest.approx(walk_in.appointment_interval)


@pytest.mark.slow
def test_structural_sources_are_removed(full: Scenario) -> None:
    def reasons(scenario: Scenario) -> set[str]:
        return {x[3] for sim in simulate(scenario, range(25)) for x in sim.downtime_log}

    assert {"off duty", "absent", "breakdown"} <= reasons(full)
    assert reasons(ablate(full, "rosters_and_absences")) == {"breakdown", "turnover"}
    assert "breakdown" not in reasons(ablate(full, "breakdowns"))

    def count(scenario: Scenario, what: str) -> int:
        sims = simulate(scenario, range(40))
        if what == "gated":
            return sum(t >= 0 for sim in sims for t in sim.instance.hidden.reveal_trigger)
        if what == "urgent":
            return sum(p.urgent for sim in sims for p in sim.instance.patients)
        return sum(getattr(sim, what) for sim in sims)

    assert count(full, "gated") > 0 and count(ablate(full, "reveal"), "gated") == 0
    assert count(full, "urgent") > 0 and count(ablate(full, "urgent_stream"), "urgent") == 0
    assert count(full, "n_no_show") > 0 and count(ablate(full, "no_shows"), "n_no_show") == 0
    assert count(ablate(full, "leaving"), "n_left") == 0


def test_an_ablation_leaves_unrelated_patients_untouched(full: Scenario) -> None:
    """Common random numbers across variants: removing the urgent stream or
    the breakdowns does not change the elective patients."""

    def elective(scenario: Scenario) -> list[Any]:
        instance = generate_instance(scenario, 5)
        q = instance.hidden.duration_quantiles
        return [
            (p.arrival_time, p.pathway, tuple(q[op.id] for op in p.operations))
            for p in instance.patients
            if p.patient_class == "elective"
        ]

    assert elective(full) == elective(ablate(full, "urgent_stream"))
    assert elective(full) == elective(ablate(full, "breakdowns"))
    assert elective(full) == elective(ablate(full, "leaving"))


# --- the study itself -----------------------------------------------------------


@pytest.fixture(scope="module")
def study(full: Scenario) -> Any:
    policies = [Policy("fifo"), Policy("urgent_first"), Policy("spt"), Policy("weight_aware")]
    return run_sensitivity(full, policies, 40, leave_penalty=500.0)


@pytest.mark.slow
def test_study_covers_every_source_and_kpi(study: Any) -> None:
    assert [e.source for e in study.effects] == [*SOURCES, LEAVE_PENALTY]
    for effect in study.effects:
        assert set(effect.effect) == set(KPIS)
        assert -1.0 <= effect.tau <= 1.0
        assert effect.best_when_off in study.cost_all_on
    assert study.reference == min(study.cost_all_on, key=study.cost_all_on.get)


@pytest.mark.slow
def test_effects_have_the_expected_signs(study: Any) -> None:
    by_source = {e.source: e for e in study.effects}
    # Rosters remove capacity: flow time and makespan are higher with them on.
    rosters = by_source["rosters_and_absences"].effect
    assert rosters["mean_flow_time"][0] > rosters["mean_flow_time"][1] > 0
    # The penalty cannot change the schedule, only the cost.
    penalty = by_source[LEAVE_PENALTY].effect
    assert penalty["mean_flow_time"] == (0.0, 0.0) and penalty["makespan"] == (0.0, 0.0)
    assert penalty["cost"][0] >= 0
    # Nobody leaves once leaving is switched off.
    leaving = by_source["leaving"].effect
    assert leaving["left"][0] >= 0


def test_cost_is_weighted_flow_plus_penalty_per_unit_weight(full: Scenario) -> None:
    result = evaluate(full, [Policy("fifo")], 15)
    name = "fifo+flexible"
    cost = kpi(result, name, "cost", 300.0)
    by_hand = (
        result.series(name, "total_weighted_flow_time") + 300.0 * result.series(name, "left_weight")
    ) / result.series(name, "arrived_weight")
    assert cost == pytest.approx(by_hand)
    assert kpi(result, name, "cost", 0.0) == pytest.approx(
        result.series(name, "weighted_mean_flow_time")
    )


@pytest.mark.slow
def test_report_formats(study: Any) -> None:
    report = format_report(study)
    assert "| Source | cost | mean_flow_time | makespan | mean_op_wait | left |" in report
    assert "Kendall's tau" in report and report.count("| leave_penalty |") == 2


def test_leavers_appear_in_tables_and_env_info(full: Scenario) -> None:
    result = evaluate(full, all_policies()[:2], 10)
    assert "left" in result.table().splitlines()[0]
    env = HospitalEnv(EnvConfig(full, leave_penalty=123.0))
    assert env.leave_penalty == 123.0
    rng = np.random.default_rng(0)
    _, info = env.reset(seed=0)
    assert info["observation_schema"] == OBSERVATION_SCHEMA_VERSION
    while True:
        *_, terminated, truncated, info = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            break
    assert {"patients_left", "no_shows", "leave_penalty"} <= set(info["episode"])


# --- event log and calibration, by round trip -----------------------------------

KNOWN = {
    "Triage": {"mean": 12.0, "cv": 0.35},
    "Scan": {"distribution": "gamma", "mean": 40.0, "cv": 0.5},
    "Form": {"distribution": "deterministic", "mean": 7.0},
}
PROFILE = [0.02, 0.06, 0.10, 0.04, 0.0, 0.03]


def known_scenario() -> Scenario:
    return make_scenario(
        resources={"Staff": 8},
        op_types={name: ({"Staff": 1}, spec, True) for name, spec in KNOWN.items()},
        pathways={"p": [("t", "Triage", []), ("s", "Scan", ["t"]), ("f", "Form", ["s"])]},
        arrivals={
            "kind": "poisson",
            "horizon": 600,
            "rate_profile": {"period": 600, "rates": PROFILE},
        },
    )


@pytest.fixture(scope="module")
def log() -> list[dict[str, Any]]:
    scenario = known_scenario()
    rows: list[dict[str, Any]] = []
    for episode, instance in enumerate(sample_instances(scenario, 150)):
        rows += event_log_rows(Policy("fifo").run(instance), episode)
    return rows


def test_event_log_round_trips_through_csv(log: list[dict[str, Any]], tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    write_event_log(path, log[:500])
    assert path.read_text().splitlines()[0] == ",".join(COLUMNS)
    back = read_event_log(path)
    assert len(back) == 500
    for original, loaded in zip(log[:500], back, strict=True):
        assert loaded["op_type"] == original["op_type"]
        assert loaded["resources"] == original["resources"] and loaded["resources"].startswith(
            "staff_"
        )
        assert loaded["end_time"] == pytest.approx(original["end_time"], rel=1e-12)


def test_fit_durations_recovers_family_and_parameters(log: list[dict[str, Any]]) -> None:
    fits = fit_durations(log)
    assert set(fits) == set(KNOWN)
    assert fits["Triage"].distribution == "lognormal"
    assert fits["Scan"].distribution == "gamma"
    assert fits["Form"].distribution == "deterministic" and fits["Form"].mean == pytest.approx(7.0)
    for name in ("Triage", "Scan"):
        assert fits[name].mean == pytest.approx(KNOWN[name]["mean"], rel=0.03)
        assert fits[name].cv == pytest.approx(KNOWN[name]["cv"], rel=0.06)
        assert fits[name].ks_pvalue > 0.01 and fits[name].n > 1000
    assert fits["Scan"].to_config() == {
        "distribution": "gamma",
        "mean": round(fits["Scan"].mean, 4),
        "cv": round(fits["Scan"].cv, 4),
    }


def test_fitted_config_is_accepted_by_the_loader(log: list[dict[str, Any]]) -> None:
    fits = fit_durations(log)
    refit = make_scenario(
        resources={"Staff": 8},
        op_types={name: ({"Staff": 1}, fit.to_config(), True) for name, fit in fits.items()},
        pathways={"p": [("t", "Triage", []), ("s", "Scan", ["t"]), ("f", "Form", ["s"])]},
    )
    assert refit.operation_type_by_name["Scan"].duration.kind is DurationKind.GAMMA


def test_fit_arrival_profile_recovers_the_rates(log: list[dict[str, Any]]) -> None:
    fitted = fit_arrival_profile(log, period=600, n_bins=6)
    assert fitted["period"] == 600
    rates = np.array(fitted["rates"])
    assert rates[4] == 0.0  # nobody arrives in the empty bin
    assert rates == pytest.approx(PROFILE, rel=0.12, abs=0.004)
    # 150 episodes of about 25 patients: the total is tight.
    assert rates.sum() == pytest.approx(sum(PROFILE), rel=0.04)
    with pytest.raises(ValueError, match="empty"):
        fit_arrival_profile([], period=600, n_bins=6)


def test_too_few_samples_are_skipped(log: list[dict[str, Any]]) -> None:
    assert fit_durations(log[:40], min_samples=30) == {} or len(fit_durations(log[:40])) <= 3
    assert fit_durations(log[:20], min_samples=30) == {}


# --- observation schema ---------------------------------------------------------


def test_schema_version_is_bumped_when_the_layout_changes() -> None:
    """The fingerprint covers every feature name in order. If this fails you
    changed the observation layout: bump OBSERVATION_SCHEMA_VERSION, update
    the fingerprint here, and say so in docs/ENV_API.md."""
    layout = repr(
        (
            obs_module.OP_FEATURES,
            obs_module.PATIENT_FEATURES,
            obs_module.RESOURCE_FEATURES,
            obs_module.ROLE_FEATURES,
            obs_module.GLOBAL_FEATURES,
        )
    )
    fingerprint = hashlib.sha256(layout.encode()).hexdigest()[:16]
    assert (OBSERVATION_SCHEMA_VERSION, fingerprint) == (1, FINGERPRINT), fingerprint
    assert HospitalEnv.observation_schema_version == 1
    assert math.isfinite(len(layout))


FINGERPRINT = "67b71b629d4e93d0"


# --- narrower sensitivity switches, the patience config, the acuity baseline ----

PATIENCE = Path(__file__).parent.parent / "configs" / "example_hospital_patience.yaml"


def test_partial_switches_separate_the_mixed_effects(full: Scenario) -> None:
    steady = ablate(full, "arrival_bursts_and_profile")
    walk_in = next(a for a in steady.arrivals if a.name == "walk_in")
    assert walk_in.kind is ArrivalKind.POISSON and not walk_in.rate_profile
    assert walk_in.batch_mean == 1.0 and walk_in.rate == pytest.approx(0.13 / 6)
    assert next(a for a in steady.arrivals if a.name == "elective") == full.arrivals[0]

    def shape(scenario: Scenario) -> tuple[int, int, int]:
        """(operations, gated non-repeat operations, repeat copies) over a few seeds."""
        total = gated = repeats = 0
        for seed in range(30):
            instance = generate_instance(scenario, seed)
            triggers = instance.hidden.reveal_trigger or (-1,) * instance.n_operations
            for op in instance.operations:
                total += 1
                repeats += "#" in op.step_id
                gated += triggers[op.id] >= 0 and "#" not in op.step_id
        return total, gated, repeats

    _, gated, repeats = shape(full)
    assert gated > 0 and repeats > 0
    # Information only: nothing is gated any more, the repeats are all still there.
    _, gated_visible, repeats_visible = shape(ablate(full, "reveal_information_only"))
    assert gated_visible == 0 and repeats_visible == repeats
    # Workload only: the repeats are gone, the gated steps are still hidden.
    _, gated_kept, repeats_removed = shape(ablate(full, "repeat_workload_only"))
    assert repeats_removed == 0 and gated_kept > 0


def test_patience_config_makes_leaving_matter() -> None:
    scenario = load_scenario(PATIENCE)
    result = evaluate(scenario, [Policy("fifo"), Policy("weight_aware")], 30)
    left_fifo = result.series("fifo+flexible", "left").mean()
    assert left_fifo > 1.0  # patients really do give up here
    assert result.series("weight_aware+flexible", "left").mean() < left_fifo
    full_left = evaluate(load_scenario(FULL), [Policy("fifo")], 30).series("fifo+flexible", "left")
    assert full_left.mean() < 0.2  # ...whereas in the full example they hardly ever do


def test_acuity_factors_follow_from_public_parameters(full: Scenario) -> None:
    from hospital_sim.baselines import acuity_duration_factors

    factors = np.array(acuity_duration_factors(full))
    assert factors.shape == (3, len(full.operation_types))
    assert (factors[0] <= 1).all() and (factors[2] >= 1).all()  # low acuity is quicker
    assert factors[1] == pytest.approx(1.0)  # the middle class is average
    assert factors[0] * factors[2] == pytest.approx(1.0)  # symmetric classes
    surgery = [t.name for t in full.operation_types].index("Surgery")
    # sigma(cv 0.4) * sqrt(rho 0.4) * E[z | top third] = 0.385 * 0.632 * 0.7 * 1.091
    assert factors[2, surgery] == pytest.approx(math.exp(0.3853 * 0.6325 * 0.7 * 1.0908), rel=1e-3)

    no_classes = make_scenario(
        resources={"Nurse": 1},
        op_types={"T": ({"Nurse": 1}, {"mean": 5, "cv": 0.5}, True)},
        pathways={"p": [("t", "T", [])]},
    )
    assert acuity_duration_factors(no_classes) == []


def test_acuity_aware_uses_the_class_and_nothing_hidden() -> None:
    """Two patients with identical visible work; only the acuity class differs."""
    scenario = make_scenario(
        resources={"Nurse": 1},
        op_types={"T": ({"Nurse": 1}, {"mean": 30, "cv": 0.8}, True)},
        pathways={"p": [("t", "T", [])]},
        arrivals={"kind": "static", "n_patients": 40},
        complexity={"rho": 0.6, "acuity_classes": 3, "acuity_correlation": 0.9},
    )
    instance = generate_instance(scenario, 0)
    sim = Policy("acuity_aware").run(instance, debug=True)
    order = sorted(range(sim.n_ops), key=lambda op: sim.start_time[op])
    classes = [instance.patients[sim.patient_of[op]].acuity_class for op in order]
    assert classes == sorted(classes)  # expected-quickest patients first

    # Without classes it is exactly weight_aware.
    plain = make_scenario(
        resources={"Nurse": 1},
        op_types={"T": ({"Nurse": 1}, {"mean": 30, "cv": 0.8}, True)},
        pathways={"p": [("t", "T", [])]},
        arrivals={"kind": "static", "n_patients": 20},
    )
    twin = generate_instance(plain, 1)
    assert (
        Policy("acuity_aware").run(twin).start_time == Policy("weight_aware").run(twin).start_time
    )


def test_acuity_rule_sees_the_nominal_scenario_under_randomization(full: Scenario) -> None:
    """With randomized noise the drawn rho is hidden: the rule must be built
    from the configured scenario, so its choices do not depend on the draw."""
    from hospital_sim.env import Randomization

    ranges = Randomization(rho=(0.0, 0.9))
    instance = sample_instances(full, 1, randomize=ranges)[0]
    assert instance.scenario.complexity.rho != full.complexity.rho
    with_nominal = Policy("acuity_aware").run(instance, nominal=full)
    # Same instance, but pretend the true rho had been something else.
    relabelled = dataclasses.replace(
        instance,
        scenario=dataclasses.replace(
            instance.scenario,
            complexity=dataclasses.replace(instance.scenario.complexity, rho=0.123),
        ),
    )
    assert (
        Policy("acuity_aware").run(relabelled, nominal=full).start_time == with_nominal.start_time
    )
