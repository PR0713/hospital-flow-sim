# Assumptions and design decisions

Every domain or design decision that the brief did not fix is recorded here.
Status is one of:

- **Brief**: stated in the project brief.
- **Accepted**: proposed as a default and accepted ("go ahead") on 2026-10-05.
- **Mine**: decided during implementation without explicit sign-off. Please review these.
- **Mine (approved 2026-10-05)**: proposed in the realism audit and approved in the review
  of that audit. Logged as "Mine" at the reviewer's request.

Entries marked with a milestone, e.g. *(M2b)*, are agreed but not implemented yet.

Milestones: M1 domain and config (done) · M2a core engine · M2b realism layer ·
M2c pathway reveal and resource-side randomness · M3 Gymnasium env · M4 baselines
and evaluation · M5 documentation. Review stop after each.

## Model

| # | Decision | Status |
|---|---|---|
| A1 | Every resource has capacity 1. A ward with N beds is N resources. | Brief |
| A2 | Resources are individually identifiable. Instances within a role are interchangeable in v1; `Resource.attributes` is the hook for heterogeneity. | Brief |
| A3 | A resource may belong to several roles, but fills at most one slot of a given operation. Team feasibility is therefore a bipartite matching, not a per-role count. | Accepted |
| A4 | Operations are non-preemptive. | Brief |
| A5 | All-or-nothing acquisition; all resources of an operation are released together on completion. Early release per role is a hook only. *(M2)* | Accepted |
| A6 | `requires_patient` (default true) on an operation type makes the patient an implicit capacity-1 resource: two such operations of one patient cannot overlap. Operations with `requires_patient: false` (e.g. lab analysis) can run in parallel with anything. Enforcement is *(M2)*. | Accepted |
| A7 | Time is a float in minutes. `time_unit` in the config is a label only. | Accepted |
| A8 | No setup/transfer times, breakdowns, shifts, priorities or due dates in v1. | Brief |
| A10 | An operation type may have no requirements (a pure delay: transport, waiting for a result). | Mine (approved 2026-10-05) |
| A11 | A pure delay with `requires_patient: false` starts automatically the moment it is ready, with no decision. A pure delay with `requires_patient: true` is still a decision, because starting it occupies the patient and can therefore displace another operation of that patient. | Mine |
| A12 | No preemption, in any form: no interrupted operations, no mid-operation breakdowns. | Mine (approved 2026-10-05) |
| A9 | An operation type may list each role only once; a duplicate is an error ("merge the quantities") rather than being summed silently. | Mine |

## Operation lifecycle (M2a)

| # | Decision | Status |
|---|---|---|
| S1 | States: `HIDDEN -> WAITING -> READY -> RUNNING -> DONE`, plus `CANCELLED` (reachable from `HIDDEN`, `WAITING`, `READY`; never from `RUNNING`). | Mine (approved 2026-10-05) |
| S2 | `HIDDEN` means "the scheduler does not know this operation exists". Operations of patients who have not arrived are `HIDDEN`; from M2c so are steps awaiting a reveal. | Mine |
| S3 | A cancelled operation never blocks its successors, and never loses precedence either: a successor waits for the cancelled operation's own unfinished predecessors (same rule as P4). | Mine |
| S4 | A patient is complete when every operation is `DONE` or `CANCELLED`; the completion time is when the last one resolved. | Mine |
| S5 | Decision epoch: at least one `READY` operation whose team can be formed from idle resources (and whose patient is free, if required). All events sharing a timestamp are processed before a decision is requested. | Brief |
| S6 | The realized completion time of a running operation is private to the engine. `end_time` is published only when the operation completes. | Mine (approved 2026-10-05) |
| S7 | If nothing can start, no event is pending and patients remain, the engine raises `DeadlockError`. Validation guarantees every operation type can be staffed by an idle hospital, so this indicates a bug, not a schedule. | Mine |

## Engine defaults and metrics (M2a)

| # | Decision | Status |
|---|---|---|
| E1 | `Simulator.start(op)` without a team uses the first idle eligible resources in declaration order (matching-aware when roles overlap). Other allocation rules arrive in M4. | Mine |
| E2 | A team is addressed as one tuple of resource indices per requirement, in requirement order, so the role each resource fills is explicit. | Mine |
| E3 | Waiting time of an operation = start minus the time it became ready. Flow time of a patient = completion minus arrival. Makespan = last patient completion. | Accepted |
| E4 | Per-role utilization = busy time of the role's eligible resources over `[0, makespan]`, divided by their number. A resource in several roles counts towards each. | Mine |
| E5 | Queueing sanity tests average four fixed seeds of 40,000 patients and accept 5-6% error. A single run is only accurate to about 3-6%, and starting empty biases the mean wait slightly low; no warm-up is discarded. | Mine |

## Pathways

| # | Decision | Status |
|---|---|---|
| P1 | Precedence exists only within a patient. | Brief |
| P2 | Optional and branching steps are resolved when the patient is generated (this keeps common random numbers). Until M2c the resulting DAG is fully visible from arrival. **Superseded in part by V1**: from M2c, steps can stay hidden until a trigger operation completes. | Accepted, then revised |
| P3 | Two mechanisms: `probability < 1` includes a step independently; `branch_group` picks exactly one step of the group, with odds `branch_weight`. A step cannot use both. | Mine |
| P4 | When a step is left out, its successors inherit its predecessors (`a -> x -> b` becomes `a -> b`). Precedence is never lost by skipping a step. | Mine |
| P5 | A pathway in which every step is optional is rejected, because a patient with no operations has no meaning. | Mine |
| P6 | Generated DAGs are stored transitively reduced. This removes redundant edges and does not change the partial order. | Mine |
| P7 | Random pathways: steps are laid out in a fixed order and each forward pair gets an edge with probability `density` (0 = fully parallel, 1 = chain). Operation types are drawn uniformly with replacement. A random template is built once per scenario from its own `seed`, not per patient. | Mine |

## Randomness

| # | Decision | Status |
|---|---|---|
| R1 | Three independent streams from one seed: arrivals, pathways, durations. Changing the arrival process does not change the patients' pathways or duration luck (tested). | Brief, extended |
| R2 | Each operation gets one uniform(0,1) quantile when the instance is generated. The simulator converts it to a duration through the inverse CDF at start time, using the parameters that apply then (including the team, once durations depend on it). The same operation therefore has the same luck under every policy, whatever order the policy starts things in. This is equivalent to "drawn at start time" from the scheduler's point of view. *(conversion is M2)* | Accepted |
| R3 | `ProblemInstance.duration_quantiles` must never appear in an observation. *(enforced in M3)* | Brief |
| R4 | Poisson arrivals generate a fixed number of patients (`n_patients`); the first arrival is one exponential gap after t = 0. The episode ends when all are done, so makespan stays well defined. | Accepted |
| R5 | Pathway choice per patient follows `pathway_mix` weights; empty means uniform over all pathways. | Mine |

## Realism layer and visibility (approved after the audit)

| # | Decision | Status |
|---|---|---|
| V1 | Mid-stay pathway reveal: steps may stay `HIDDEN` until a trigger operation completes (repeat tests, added operations, complications). Outcomes are still pre-drawn per patient. *(M2c)* | Mine (approved 2026-10-05) |
| V2 | Per-patient latent complexity factor via a Gaussian copula on the duration quantiles, with configurable rho. Marginal distributions are unchanged. *(M2b)* | Mine (approved 2026-10-05) |
| V3 | The latent factor is hidden by default; a configurable noisy acuity class is visible at arrival. *(M2b)* | Mine (approved 2026-10-05) |
| V4 | Heavy-tail samplers (complication mixture, shifted lognormal). *(M2b)* | Mine (approved 2026-10-05) |
| V5 | Seniority speed multiplier, and a duration interface that can see resource state (for fatigue later). *(M2b)* | Mine (approved 2026-10-05) |
| V6 | Non-stationary arrivals (rate profile by hour and weekday), bursts, per-class arrival streams. *(M2b)* | Mine (approved 2026-10-05) |
| V7 | Two episode modes: static batch with fixed N (makespan setting), and horizon-based dynamic arrivals followed by a drain phase. Extends R4. *(M2b)* | Mine (approved 2026-10-05) |
| V8 | Patient weight and an urgent class; the weight enters the objective in M3. *(M2b / M3)* | Mine (approved 2026-10-05) |
| V9 | Shifts, breaks, absences, between-operation breakdowns ("finish the current operation, then leave"), no-shows, patients leaving, fatigue. *(M2c)* | Mine (approved 2026-10-05) |
| V10 | Observations expose the *expected* remaining time of a running operation given the time elapsed, never the true value. The samplers provide this from M2a; the observation is *(M3)*. | Mine (approved 2026-10-05) |
| V11 | Observations include realized durations of *finished* operations, so an agent can infer a patient's hidden complexity. *(M3)* | Mine |
| V12 | Per-episode randomization of noise parameters (rho, CVs, arrival rates drawn from configured ranges at reset). *(M3)* | Mine (approved 2026-10-05) |
| V13 | Hidden data lives in `ProblemInstance.hidden`, separate from everything a policy may see; observation builders will receive a view without it and `info` a key whitelist. *(split done in M2a, enforcement and leak tests M3)* | Mine (approved 2026-10-05) |
| V14 | Instance seeds derive from `(base_seed, split, episode_index)` with split in train / eval / test, so splits are disjoint by construction; the evaluation harness refuses a train-split env. *(M3 / M4)* | Mine (approved 2026-10-05) |
| V15 | No real data is available. Calibration is a minimal hook only *(M4)*; all parameters remain illustrative. | Mine (approved 2026-10-05) |

## Realism layer as built (M2b)

| # | Decision | Status |
|---|---|---|
| B1 | `complexity.rho` is the correlation between the normal scores of any two operations of one patient. `rho = 0` skips the transform entirely, so quantiles are bit-identical to M2a (a golden-value test guards this). | Mine (approved 2026-10-05) |
| B2 | The visible acuity class comes from a noisy reading `c * factor + sqrt(1 - c^2) * noise`, cut into `acuity_classes` equal-probability classes (0 = least complex). `acuity_correlation` is `c`. Off by default. | Mine |
| B3 | The latent factor and the acuity noise are drawn for every patient whether or not they are used, so switching them on does not shift any other draw. | Mine |
| B4 | A complication is a two-component mixture: with probability p the duration is multiplied by m. The top p share of an operation's quantiles are the complicated cases, so under `rho > 0` it is the complex patients who get complications. `mean` and `cv` describe the routine case; the scheduler-visible expected duration includes the complication. | Mine |
| B5 | A complication mixture is not much heavier in the far tail than a single lognormal with the same mean and CV (about 1.5x at four times the routine mean in the tested case). What it adds is a distinct cluster of long cases. For a genuinely heavier tail, raise `cv`. | Mine (finding) |
| B6 | `shift` is a minimum duration for lognormal and gamma; `mean` and `cv` still describe the whole duration. | Mine |
| B7 | Team speed: an operation type lists `speed_roles`; its duration is divided by the geometric mean of the `speed` attribute (default 1) of the resources assigned to those roles. No `speed_roles` means the team is ignored. | Mine |
| B8 | The scheduler-visible expected duration of a not-yet-started operation assumes speed 1, because the team is not known. Once started, `Simulator.expected_remaining(op)` uses the actual team. | Mine |
| B9 | Duration model interface: `base(op_type)` and `at_start(ctx)`, where `ctx` carries the team, the time and read access to resource state (`work_time`, `jobs_done`). A model returns a distribution; the engine applies the hidden quantile. Fatigue (M2c) plugs in here. | Mine (approved 2026-10-05) |
| B10 | Each arrival stream is a patient class with its own `weight`, `urgent` flag and pathway mix. Patients are numbered in global arrival order. | Mine (approved 2026-10-05) |
| B11 | Every arrival stream has its own family of random generators, so adding or changing a class leaves the other classes' patients, pathways and duration luck unchanged (tested). | Mine |
| B12 | `rate` and `rate_profile` are in patients per time unit. `batch_mean > 1` groups arrivals into geometric batches without changing the mean patient rate; the index of dispersion of counts becomes `2 * batch_mean - 1`. | Mine |
| B13 | `rate_profile` is piecewise constant over equal bins of `period` and repeats (24 values over 1440 minutes for hour of day, 168 over 10080 for hour of week). Generated by thinning. | Mine |
| B14 | `urgent` is a label plus a weight. It gives no dispatching priority and no preemption inside the engine; acting on it is the policy's job, and the weight enters the reward in M3. | Mine |
| B15 | In horizon mode an episode can contain zero patients. The engine handles it (done immediately, metrics are zero); M3 must decide whether to resample such episodes. | Mine (open for M3) |
| B16 | Distributional tests use fixed seeds and accept p > 0.01. One test checks that the same KS procedure rejects a wrong distribution. | Mine |
| B17 | The performance test never fails on speed. It prints decisions per second for 10, 50 and 200 patients in the pytest summary and appends them to `benchmarks/history.csv`. | Mine (approved 2026-10-05) |

## Decisions from the M2b review

| # | Decision | Status |
|---|---|---|
| T1 | Team selection in M3: formulation A with a flexibility-preserving allocation rule as the default; an optional (operation, rule) action behind a flag; sequential slot filling only if M4 shows it is needed. *(M3)* | Mine (approved 2026-10-05) |
| T2 | M4 measures the cap: the same dispatching policy under several allocation rules, plus a search-based best-allocation reference on small instances, with the gap reported. *(M4)* | Mine (approved 2026-10-05) |
| T3 | Horizon mode with too few patients: the env resamples until an instance has at least a configurable minimum (default 1), and logs the number of resamples. *(M3)* | Mine (approved 2026-10-05) |
| T4 | Urgent stays a label plus a weight in the engine. M4 includes at least one priority-aware baseline (urgent-first and weight-aware). *(M4)* | Mine (approved 2026-10-05) |
| T5 | `loglogistic` duration option (`mean`, `tail_index`), per operation type: a power-law tail, with KS, exceedance and Hill-estimate tests. Done. | Mine (approved 2026-10-05) |
| T6 | The complication mixture is documented as a cluster of long cases, not a heavy tail (see B5 and the `Complicated` docstring). | Mine (approved 2026-10-05) |
| T7 | M2c is split: M2c-1 shifts, breaks, absences, between-operation breakdowns; M2c-2 pathway reveal, no-shows, patients leaving, fatigue. Review stop after each, and before both for a background-literature comparison and the reveal design note. | Mine (approved 2026-10-05) |
| T8 | A comparison with background literature on hospital simulation was written for review (kept locally, not part of the repository). Reveal design: `docs/DESIGN_pathway_reveal.md`. Both were proposals; their recommendations were not decisions until approved. | Mine (approved 2026-10-05) |

## Decisions from the review of the literature comparison and reveal note

| # | Decision | Status |
|---|---|---|
| U1 | Room turnover time is added in M2c-1. Reasoning: whole-hospital simulations commonly impose idle time between uses of a room for cleaning, and the model had no setup time at all. | Mine (approved 2026-10-05) |
| U2 | Fatigue is deferred; the duration interface (B9) already supports it. | Mine (approved 2026-10-05) |
| U3 | Breakdowns are implemented only as a thin config on the unavailable-interval machinery. It was a small change (one more interval stream), so it is included in M2c-1. | Mine (approved 2026-10-05) |
| U4 | Class-dependent durations are added in M2c-2. Reasoning: planning models typically fit a separate duration distribution per patient group, whereas ours depended only on operation type, team and the hidden factor. *(M2c-2)* | Mine (approved 2026-10-05) |
| U5 | Reveal: the expected-gated-work prior is a visible feature behind a config switch, so it can be ablated. A gated step's probability depends on the hidden complexity factor, default on. *(M2c-2 / M3)* | Mine (approved 2026-10-05) |
| U6 | Default M3 reward is weighted flow time; makespan for static-batch mode. Rewards stay pluggable. Reasoning: healthcare simulation studies mostly report waiting and time-in-system measures; makespan is a job-shop objective, not a hospital one. *(M3)* | Mine (approved 2026-10-05) |
| U7 | Order of work: M2c-1, then M3, then M2c-2. M3's graph observation must support an operation set that grows during an episode, and per-episode resampling, so M2c-2 does not force a redesign. | Mine (approved 2026-10-05) |
| U8 | The background literature reviewed so far consists of capacity-planning simulations, not scheduling or RL work: nobody in those models decides which patient goes next. It informs realism only. Scheduling and RL references will be supplied separately. | Mine (approved 2026-10-05) |

## Resource availability and turnover (M2c-1)

| # | Decision | Status |
|---|---|---|
| C1 | One mechanism covers everything: each resource has up to three independent streams of unavailable intervals (planned calendar, absences, breakdowns). A resource can start new work only if it is free and no stream has it down. | Mine (approved 2026-10-05) |
| C2 | Lazy removal, no preemption: a resource that is busy when an interval starts finishes its operation, then becomes unavailable. The time worked past the start of the interval is recorded as overtime. | Mine (approved 2026-10-05) |
| C3 | Starting an operation just before a shift ends is allowed, even if it will overrun. Whether to do it is the policy's choice. | Mine |
| C4 | A calendar is a repeating `period` with `shifts` (on duty) and `breaks` (off, inside a shift). Calendar time is simulation time: there is no offset. Staggered breaks need separate calendars. | Mine |
| C5 | Absence: each shift occurrence of each resource is missed with `absence_probability`, independently. | Mine |
| C6 | Breakdown: alternating exponential time-to-failure (`mtbf`) and repair time (`mttr`), running on calendar time regardless of use. | Mine |
| C7 | Absences and breakdowns are drawn lazily by the engine from generators keyed by (instance seed, resource, stream). They do not depend on the policy, so common random numbers still hold, and `reset()` replays them exactly. | Mine |
| C8 | Turnover: an operation type lists `turnover: {role: minutes}`. On completion the resources filling those roles stay unavailable for that long; the rest of the team, and the patient, are released at once. This is the configured exception to "released together" (A5). Turnover time is deterministic. | Mine (approved 2026-10-05) |
| C9 | Utilization is now busy time divided by the time the resource was on duty (plus any overtime) within `[0, makespan]`. Turnover is reported separately, not as busy time. Supersedes E4's denominator. | Mine |
| C10 | Validation rejects a scenario in which some operation type can never be staffed by the planned calendars (checked at every calendar breakpoint over two of the longest periods). Absences and breakdowns are ignored by this check. | Mine |
| C11 | With calendars or breakdowns there is always a future event. The engine therefore raises `DeadlockError` when nothing can start, no arrival, completion or turnover is pending, and more than `stall_limit` (default 10,000) availability events pass. | Mine |
| C12 | The realized end of a breakdown and the fact of an absence are hidden from policies until they happen; the planned calendar is public. `Simulator.downtime_log` holds the truth and is for analysis after the episode only. *(enforced in M3)* | Mine |

## Decisions from the M2c-1 review

| # | Decision | Status |
|---|---|---|
| W1 | `example_hospital_dynamic.yaml` is lightened (12-hour nurse shift, six hours of lower-rate arrivals): about 76% of seeds finish within the shift under FIFO. The overloaded version is kept as `example_hospital_stress.yaml`. | Mine (approved 2026-10-05) |
| W2 | Wait cap: a maximum number of consecutive waits and a maximum simulated time per episode. Reaching either ends the episode as truncated, not terminated, with the reason in `info`. | Mine (approved 2026-10-05) |
| W3 | Observations use the planned roster only. A perturbation test covers future absences and breakdowns. `downtime_log` never reaches the env (a static test forbids the env package from reading it). | Mine (approved 2026-10-05) |
| W4 | Commits: one per milestone from M3 on. M1 to M2c-1 are one commit, because the intermediate states were never saved and could not be reconstructed honestly. | Mine (deviation from the request) |

## Environment (M3)

| # | Decision | Status |
|---|---|---|
| N1 | Actions: `slot * n_rules + rule`, plus a final wait action. One rule by default; several rules give the (operation, rule) formulation. | Mine (approved 2026-10-05) |
| N2 | Slots are assigned in the order operations become visible and are never reused, so an index cannot reveal a hidden operation and the operation set can grow mid-episode. | Mine (approved 2026-10-05) |
| N3 | Default rule `flexible`: least loss of future staffing ability (sum over the resource's roles of 1 / idle members), ties broken by speed in speed-governing roles, then least work, then declaration order. The speed tie-break is my addition: without it the rule picked the slow surgeon as lead in the dynamic example and was 17% worse than first-idle. | Mine |
| N4 | **Correction of an M2a claim.** I reported that first-idle "wastes a surgeon" in the default example. That was wrong: with three surgeons and two per surgery only one surgery can run at a time, and all rules give identical results there. The effect is real with four surgeons (mean makespan 486 vs 704 minutes) and with unequal speeds. | Mine (correction) |
| N5 | The wait action's mask uses only public reasons (something running, turnover, rosters or breakdowns exist, arrivals over time exist). It does not depend on whether an arrival is actually still to come, which would leak the future. A wait with nothing scheduled is a no-op that counts towards the cap. | Mine |
| N6 | An invalid action raises by default; `on_invalid_action="fallback"` executes the lowest valid action and flags it. Gymnasium's `check_env` samples unmasked actions, so it is run in fallback mode. | Mine |
| N7 | Rewards are potential differences measured from time zero, so they telescope exactly to the final metric (tested to 1e-9). `reward_scale` defaults to 1 / time_scale. | Mine (approved 2026-10-05) |
| N8 | Instance seed = `(base_seed << 64) | (split << 60) | (episode << 8) | resample`. The bit fields make splits disjoint by construction, not just with high probability. | Mine (approved 2026-10-05) |
| N9 | `reset(seed=s)` restarts the episode sequence; later `reset()` calls advance it. | Mine (approved 2026-10-05) |
| N10 | Resampling also applies to instances that exceed the padding sizes. The acceptance test looks at the true instance size; the resulting conditioning on size is accepted. | Mine |
| N11 | Randomized parameters (rho, CV multiplier, arrival-rate multiplier) are drawn from a generator keyed by the instance seed and hidden. The observation's duration knowledge (expected durations, expected remaining time) is computed from the nominal scenario, so it cannot reveal the drawn CV. | Mine |
| N12 | Default padding in horizon mode: mean plus six standard deviations of the patient count, times the longest pathway. | Mine |
| N13 | The episode ending is itself information (no more patients are coming). It is the one unavoidable reveal and the twin tests treat it as such. | Mine |
| N14 | Turnover's remaining time is deterministic and could be shown; for now the observation only flags `in_turnover`. | Mine |
| N15 | Observation building is the speed bottleneck: about 4,000 env steps per second against 40,000 to 70,000 engine decisions per second. Not optimized yet. | Mine (flagged) |
| N16 | The eval harness refusing a train-split env (V14) and the expected-gated-work feature (U5) are still to do, in M4 and M2c-2. | Mine (approved 2026-10-05) |

## Requirement flexibility (confirmed) and two proposed extensions

Role quantities per operation type and resource counts per role are plain
config. `tests/test_requirement_flexibility.py` edits them in YAML only
(surgery needing 3 of 4 nurses; eight combinations of nurses, quantity and
rooms with hand-computed makespans) and checks validation, schedules and the
environment's requirement features.

| # | Item | Status |
|---|---|---|
| X1 | Either-or staffing with one-for-one substitution ("a radiologist or a senior technician") is already supported: define a role that both belong to. Tested. | Mine (confirmed) |
| X2 | Quantity overrides by patient class or acuity are **not** supported. Proposal: `requirement_overrides` on an operation type, expanded at generation into derived operation types, so the engine and environment need no change. Not implemented. | Proposal, pending |
| X3 | Alternative requirement sets with different structure ("2 nurses, or 1 nurse and 1 technician") are **not** supported. Proposal: `alternatives` (modes) on an operation type; the engine stores the chosen mode per operation and the allocation rule picks across modes. Not implemented. | Proposal, pending |

## Baselines and evaluation (M4)

| # | Decision | Status |
|---|---|---|
| Y1 | Baselines are non-idling dispatching rules acting on the core simulator: random, FIFO, SPT, longest remaining work, most-constrained-first, urgent-first, weight-aware. Each is paired with any allocation rule. | Mine (approved 2026-10-05) |
| Y2 | Longest remaining work and weight-aware use expected durations of the patient's visible, not-yet-started operations. | Mine |
| Y3 | Most-constrained-first: largest team first, then least spare capacity in the tightest role. | Mine |
| Y4 | Weight-aware: highest patient weight per unit of expected remaining work (Smith's rule at patient level). | Mine |
| Y5 | The random policy is seeded from the instance, so it is reproducible. | Mine |
| Y6 | The harness draws instances from the `eval` or `test` split and refuses `train`, for baselines and for env agents alike. Every policy runs on the same instances; comparisons are reported as paired differences with 95% t intervals. | Mine (approved 2026-10-05) |
| Y7 | Per-class metrics (mean flow time and mean operation wait per patient class) are reported; class intervals are over the episodes that contain the class. | Mine (approved 2026-10-05) |
| Y8 | Allocation-gap study: three dispatchers under four allocation rules on five configs, plus a search reference. The reference enumerates every distinct team at every start on 4-patient instances and is a hindsight bound. Results in `docs/ALLOCATION_GAP.md`. | Mine (approved 2026-10-05) |
| Y9 | Finding: `flexible` equals the hindsight reference on two of the three small configs and is 1.6% above it on the third; `first_idle` is up to 12% above. On these configs the cap from formulation A with `flexible` is small, so sequential slot filling is not justified yet. | Mine (finding) |
| Y10 | Two configs in the study (`four_surgeons`, `four_surgeons_speed`) are built in the study script from the default example, not shipped as YAML. | Mine |
| Y11 | Not done in M4 and not in its stated scope: the minimal calibration hook (V15) and the noise-source sensitivity analysis from the audit. They still need a home. | Open |

## Decisions from the M4 review

| # | Decision | Status |
|---|---|---|
| Z1 | Requirement overrides by patient class or acuity are implemented in M2c-2, with class-dependent durations, as derived operation types. Supersedes X2. | Mine (approved 2026-10-05) |
| Z2 | Structural alternatives (X3) are deferred. The either-or pattern via a shared role is documented in `docs/ENV_API.md`. | Mine (approved 2026-10-05) |
| Z3 | Order: M2c-2, then M5. M5 = noise-source sensitivity analysis (including reveal, no-shows and leaving), a minimal calibration hook tested by round trip, a config guide, and the RL integration guide. | Mine (approved 2026-10-05) |
| Z4 | `docs/ALLOCATION_GAP.md` states that the search reference covers 4-patient instances under one dispatcher only. | Mine (approved 2026-10-05) |
| Z5 | `configs/four_surgeons.yaml` and `configs/four_surgeons_speed.yaml` are shipped; the study script loads them. | Mine (approved 2026-10-05) |
| Z6 | One commit per milestone from here; history is not rewritten by me. | Mine (approved 2026-10-05) |
| Z7 | The RL side should use the dynamic, stress, four-surgeon and full configs. On the default example all allocation rules coincide. | Mine (approved 2026-10-05) |

## Overrides, appointments, leaving, reveal (M2c-2)

| # | Decision | Status |
|---|---|---|
| O1 | An operation type lists `overrides`, each with `when: {patient_class, acuity_class}` and a replacement `requirements`, a replacement `duration`, or a `duration_scale`. Each becomes a derived type named like `Surgery[urgent]`; the generator gives each operation the first matching one. The engine is unchanged. | Mine (approved 2026-10-05) |
| O2 | `duration_scale` stretches the distribution in time: mean, shift, bounds and samples scale; CV and tail shape do not. | Mine |
| O3 | Pathways must name base types. A derived type has its own index in `op_type` observations and its own validation. | Mine |
| O4 | `kind: scheduled` arrivals: appointment i at `start + i * interval`; arrival = appointment + normal lateness, clipped so nobody arrives after the no-show timeout; a no-show is declared `no_show_timeout` after the appointment and its operations are cancelled. | Mine (approved 2026-10-05) |
| O5 | The whole booking list is visible from time zero, in booking order, including class, weight, urgency and acuity. That acuity is known at booking is an assumption (pre-assessment). | Mine |
| O6 | Patient rows in observations are in the order the scheduler learned of each patient (bookings, then walk-ins), never in arrival order, which would reveal no-shows. | Mine |
| O7 | Leaving: each patient of a stream with `patience` has a hidden lognormal threshold. A patient whose first operation has not started by then leaves; once any operation has started they stay. | Mine (approved 2026-10-05) |
| O8 | A leaver's waiting time counts as flow time; no-shows count nowhere. Both are reported as counts. | Mine |
| O9 | The reward adds `leave_penalty x weight` per leaver, default twice the expected work of the longest pathway. Without it, letting patients leave would reduce the cost. **The value is a placeholder that needs a clinical view.** | Mine (needs input) |
| O10 | Reveal: `revealed_by` on a step, `repeat: {probability, max}` for repeats. Outcomes are drawn at generation; a gated step is dropped if its trigger does not happen. Repeats unroll into a chain of copies named `step#2`, `step#3`. | Mine (approved 2026-10-05) |
| O11 | `complexity.reveal_correlation` (default 0.5) links a gated optional step to the hidden factor without changing its marginal probability. With it at 0, gating changes visibility only (tested). It does not apply to branch groups or repeats. | Mine (approved 2026-10-05) |
| O12 | Cancelling a trigger (including through a patient leaving) cancels what it would have revealed, unseen. | Mine |
| O13 | The visible precedence graph is the true graph contracted over hidden operations. See the "As built" section of `docs/DESIGN_pathway_reveal.md` for why filtering was not enough. | Mine |
| O14 | `expected_gated_work` counts direct reveals and remaining expected repeats only, uses the template probability (not adjusted for acuity), and can be switched off. | Mine (approved 2026-10-05) |
| O15 | `configs/example_hospital_full.yaml` turns every M2c-2 feature on. | Mine |

## Decisions from the M2c-2 review

| # | Decision | Status |
|---|---|---|
| Q1 | The leave penalty stays a placeholder default and is a config value (`EnvConfig.leave_penalty`). The leaver count is its own metric in the env's final `info` and in every evaluation table. It is a source in the sensitivity analysis. | Mine (approved 2026-10-05) |
| Q2 | Accepted as proposed: a leaver's wait counts as flow time, no-shows count nowhere, acuity is visible at booking, patient rows are in discovery order, the complexity link applies to gated optional steps only. The acuity assumption is spelled out in `docs/ENV_API.md`. | Mine (approved 2026-10-05) |
| Q3 | Git: history is not touched by me; one commit per milestone. Research papers and anything derived from them are never committed; `.gitignore` covers `docs/papers/`, notes, extractions and `.bib` files; staged files are checked before every commit. | Mine (approved 2026-10-05) |
| Q4 | The literature comparison document is not committed. | Mine (approved 2026-10-05) |

## Analysis, calibration and versioning (M5)

| # | Decision | Status |
|---|---|---|
| R6 | Sensitivity is one-at-a-time ablation from `example_hospital_full` with everything on: each source is switched off alone and the same seeds are rerun with all seven baselines. | Mine (approved 2026-10-05) |
| R7 | The cost used for rankings is per unit of patient weight (weighted mean flow time plus leave penalty per unit weight), so it stays comparable when a source changes how many patients come. | Mine |
| R8 | "Arrival variability off" replaces each Poisson stream by evenly spaced, punctual appointments with the same expected count. This also makes those patients a visible booking list, which the baselines ignore. | Mine |
| R9 | "Reveal off" makes optional steps visible on arrival and removes repeats, so the repeat workload disappears too. "Rosters and absences" is one switch. | Mine |
| R10 | The leave penalty is an objective, not a property of the world: its row reuses the same episodes with the penalty set to zero. | Mine |
| R11 | Finding: in `example_hospital_full` almost nobody leaves (about 0.01 patients per episode), so the leaving and leave-penalty rows are uninformative there. A config where patience binds is needed to study them. | Mine (finding) |
| R12 | Finding: the urgent stream is the only source that reorders the policies (Kendall's tau 0.10). The others change how hard the problem is, with tau between 0.8 and 1.0. With seven policies tau moves in steps of about 0.1. | Mine (finding) |
| R13 | Event log: one row per finished operation, columns in `eval/event_log.py`. Calibration reads that format. | Mine (approved 2026-10-05) |
| R14 | `fit_durations` chooses between lognormal and gamma by AIC from maximum-likelihood fits, per operation type, pooling all teams and patients. `fit_arrival_profile` assumes each episode covers one period from time zero. Both are tested by round trip only. | Mine (approved 2026-10-05) |
| R15 | `OBSERVATION_SCHEMA_VERSION = 1`. A test fingerprints the feature names, so the layout cannot change without the version being bumped. | Mine (approved 2026-10-05) |
| R16 | The documentation states in the README, the RL guide and the API reference that the simulator is not validated against real hospital data. | Mine (approved 2026-10-05) |

## Decisions from the M5 review

| # | Decision | Status |
|---|---|---|
| S8 | Mentions of the background literature in this file keep the reasoning and name no source. | Mine (approved 2026-10-05) |
| S9 | `configs/example_hospital_patience.yaml`: the full example with twice the walk-ins, four nurses instead of six and a mean patience of 20 minutes, so about two walk-ins per episode leave under FIFO. Patience alone did not bind (walk-ins are seen quickly when six nurses do blood draws), so load and staffing were changed too. The main example is unchanged. | Mine (approved 2026-10-05) |
| S10 | Tests marked `slow` are skipped by default (`pytest`, about 30 s). `pytest -m ""` runs everything (about 2 min) and is run before each milestone commit. | Mine (approved 2026-10-05) |
| S11 | Sensitivity report: the `arrival_variability` and `reveal` rows are flagged as mixing two effects each, and three narrower switches separate them. No switch removes arrival-time randomness while keeping arrivals unannounced; that would need a new arrival kind. | Mine (approved 2026-10-05) |
| S12 | `acuity_aware` baseline: `weight_aware` with each remaining operation's expected duration multiplied by `exp(sigma * sqrt(rho) * E[complexity given class])`, all from public scenario parameters. It is a first-order approximation. | Mine (approved 2026-10-05) |
| S13 | Rules that use scenario parameters are built from the nominal scenario (`Policy.run(instance, nominal=...)`; the harness passes it), so under randomized noise they cannot see the drawn values. | Mine |
| S14 | No new features until the RL side reports back. | Mine (approved 2026-10-05) |

## Publishing

| # | Decision | Status |
|---|---|---|
| S15 | All PDFs are git-ignored and untracked, together with the literature comparison. The files stay on the author's disk. | Requested 2026-10-05 |
| S16 | `examples/` holds a minimal agent loop and an evaluation template, both covered by tests, as the entry point for someone adding an RL algorithm. | Requested 2026-10-05 |
| S17 | The repository is named `hospital-flow-sim`. The Python package keeps the name `hospital_sim`. | Mine |

## Durations

| # | Decision | Status |
|---|---|---|
| D1 | Lognormal is the default. Lognormal and gamma are parameterised by `mean` and `cv` (coefficient of variation), which is easier to elicit than shape/scale. | Brief / Mine |
| D2 | For `truncated_normal`, `mean` and `cv` describe the normal before truncation to `[low, high]`; the true expected duration differs and will be computed by the sampler. *(M2)* | Mine |
| D3 | `deterministic` exists for hand-checkable tests. `empirical` resamples the listed values; its `mean` is derived. | Mine |
| D5 | Inverse CDFs come from `scipy.special` (new dependency). Quantiles are clipped to `[1e-12, 1 - 1e-12]` so a duration is never zero or infinite by accident. | Mine |
| D6 | Empirical durations use the inverse of the empirical CDF (a quantile picks one of the listed samples), i.e. plain resampling without interpolation. | Mine |
| D4 | Durations in `configs/example_hospital.yaml` are illustrative, not fitted to any data. | Mine |

## Config

| # | Decision | Status |
|---|---|---|
| C1 | Unknown keys are rejected (typos fail loudly). | Mine |
| C2 | A resource entry with `count > 1`, or without `id`, expands to `<base>_1..<base>_<count>`, where base is `id` or the first role in lower case. | Mine |
| C3 | Validation reports all errors at once. Unused roles, operation types and pathways are warnings, not errors. | Mine |

## Agreed for later milestones

| # | Decision | Status |
|---|---|---|
| L1 | Custom `heapq` event engine, not SimPy. Done in M2a. | Accepted |
| L2 | Action formulation A (pick a ready operation; a rule picks the team) is the default. B is an opt-in wrapper; C is not built. *(M3)* | Accepted |
| L3 | Wait action enabled by default, meaning "advance to the next event"; masked when no future event exists. *(M3)* | Accepted |
| L4 | `Discrete(max_ops + 1)` action space over stable operation slots, with `action_masks()`. *(M3)* | Accepted |
| L5 | Makespan, mean flow time (completion minus arrival) and total waiting (ready-to-start gaps) are all reported; default reward is a weighted makespan and flow-time combination. *(M3/M4)* | Accepted |
| L6 | Future arrivals are hidden from the agent until they occur. *(M3)* | Accepted |
| L7 | Team-dependent durations: a hook plus one worked example (per-resource speed multiplier). *(M2)* | Accepted |

## Out of scope (in the theory primer, not in the brief)

The primer in this folder describes a broader whole-hospital model. The brief
governs, so these are not modelled and have no hook yet beyond what A8 lists:
episode-scoped holds (a bed kept across several operations), blocking after
service and boarding, reservation/backfilling against starvation, pathways
that grow mid-run, patient deterioration, fatigue, and warm-up analysis for
steady-state runs. Tell me if any of these should move into scope.

## Plots

| # | Decision | Status |
|---|---|---|
| G1 | The Gantt chart has one row per resource and one bar per (operation, resource), so a multi-resource operation appears on every row it occupies. Colour encodes the operation type and every bar is labelled with its patient. Colour by patient was rejected: it does not scale past about eight patients. | Mine |
| G2 | Operation types beyond the eight palette colours are drawn in grey and identified by their bar label and the legend. | Mine |

## Tooling

`uv` is not installed on this machine, so the project uses a plain `venv`
(`.venv/`) with pip. `pyproject.toml` works unchanged with `uv` later.
