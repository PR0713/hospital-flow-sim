# Simulator audit: hospital-flow-sim (branch `agent-layer`, commit d2998af)

Audit date 2026-11-25. No repository file was modified. Throwaway scripts and raw outputs are in
`_audit/` (`run_audit.py`, `probe_exploits.py`, `probe_space.py`, `probe_nan.py`, `out_*.txt`).
Paths are relative to the repo root; `file:a-b` means lines a to b.

## 1. Summary

The repository is a discrete-event simulator for a stochastic, online, flexible job shop with
multi-resource tasks. It is written for patient flow in a hospital, but the model underneath is
generic: jobs ("patients") are DAGs of tasks ("operations"), and each task needs a whole bundle of
capacity-1 resources, matched to roles by bipartite matching. On top sits a Gymnasium env
(`HospitalEnv`) with padded dict/graph observations, a flat masked `Discrete` action space, a
wait action and potential-based rewards that sum exactly to the episode cost. Stochastic content
is broad: six duration families plus shift/complication/speed wrappers, correlated durations,
Poisson/profiled/batched/scheduled arrivals with no-shows and reneging, optional/branching/
repeating/hidden tasks revealed mid-episode, rosters, absences, breakdowns and turnover. There is
no preemption, no blocking, no due dates, and the agent cannot pick individual resources (only an
allocation rule). Engineering quality is high: 449 tests pass, ruff and strict mypy are clean,
debug invariants exist, CRN-based paired evaluation is built in, and the docs are thorough. This
audit found four real defects (two in evaluation and reward handling, one in the engine, one in
state copying) that matter for RL benchmarking; see §11 and §12. All parameters are illustrative
and unvalidated against real data (`CLAUDE.md:11`, `docs/ENV_API.md:12-14`).

## 2. Architecture

Layers, each depending only on the ones before it (`CLAUDE.md`, "Architecture"):

| Layer | Key code | Role |
|---|---|---|
| `domain/` | `model.py` (`Scenario`, `ProblemInstance`, `HiddenRealization` `model.py:312-332`), `dag.py`, `matching.py`, `validation.py` | Immutable entities; DAG utilities (Kahn, contraction, transitive reduction); team feasibility as bipartite matching (`matching.py:19-48`); semantic validation |
| `generators/` | `config.py` (pydantic schema, `extra="forbid"` `config.py:40-42`), `instance.py` (`generate_instance` `instance.py:243-333`), `pathways.py` | YAML/JSON to `Scenario`; seed to concrete instance with all hidden draws |
| `rng.py` | `make_streams`, `availability_rng` (`rng.py:25-41`) | Independent named RNG streams |
| `sim/` | `simulator.py` (`Simulator`), `resources.py` (`ResourceManager`), `allocation.py` (`RULES`), `durations.py`, `availability.py`, `metrics.py`, `invariants.py`, `runner.py` | Event engine, resource bookkeeping, team choice, distributions, rosters and breakdowns, metrics, debug checks |
| `env/` | `hospital_env.py` (`HospitalEnv`), `observation.py` (`ObservationBuilder`), `rewards.py`, `config.py` (`EnvConfig`, seed splits) | Gymnasium wrapper |
| `baselines/`, `eval/` | `dispatching.py`, `search.py`, `harness.py`, `sensitivity.py`, `calibration.py`, `gantt.py`, `event_log.py` | Heuristics, hindsight allocation search, CRN evaluation, ablations |
| (branch extras) | `src/hospital_sim_mcp/`, `src/hospital_sim_agents/llm_agent.py` | MCP server and LLM policy over the env; not audited in depth |

**Time advance: discrete-event with an event heap.** Events `(time, kind, seq, payload)` live in a
`heapq` (`simulator.py:383-387`). Ties are ordered by a fixed `EventKind` priority and then by
insertion sequence (`simulator.py:44-53`). `advance()` jumps to the next event time and drains
every event at that time (`simulator.py:340-350`, `389-409`). `run_until_decision()` advances
until some READY task can be staffed from idle resources (`simulator.py:352-365`). There is no
fixed time step. Actions take zero time; time moves only when nothing can start or the agent
waits (semi-Markov; `docs/RL_INTEGRATION.md:69`).

**Episode end to end:** `reset()` derives a seed and generates an instance
(`hospital_env.py:159-176`). It builds `Simulator` and `ObservationBuilder`
(`hospital_env.py:202-213`) and runs to the first decision. `step(a)` decodes `(slot, rule)`,
calls `sim.start(op, rule(sim, op))` or `sim.advance()` for wait, runs to the next decision,
computes the reward and refreshes the observation and mask (`hospital_env.py:225-285`).

**Dependencies** (`pyproject.toml`): numpy, scipy, pydantic, pyyaml, gymnasium (installed 1.4.0;
Python 3.12.13 venv). Optional: matplotlib, mcp, anthropic. **No SimPy, NetworkX, PyTorch or
PyTorch Geometric** in the requirements or the venv.

## 3. Gymnasium interface

- **Signatures.** `reset(*, seed=None, options=None) -> (obs, info)` (`hospital_env.py:178-221`).
  `options={"instance": ProblemInstance}` runs a given instance. `reset(seed=s)` restarts a
  deterministic sequence of episodes; later `reset()` calls give the next instances.
  `step(action) -> (obs, reward, terminated, truncated, info)` (`hospital_env.py:225`). No
  `render` (`metadata = {"render_modes": []}`, `hospital_env.py:65`). Not registered with
  `gym.make` (no `register` call anywhere).
- **Observation: `spaces.Dict` of padded arrays** (`hospital_env.py:114-135`, features
  `observation.py:31-83`):
  `ops [max_ops,15]`, `op_valid`, `op_type`, `op_patient`, `op_requirements [max_ops,n_roles]`,
  `patients [max_patients,13]`, `patient_valid`, `resources [n_res,10]`,
  `resource_roles [n_res,n_roles]`, `resource_op [n_res]`, `roles [n_roles,3]`, `global [6]`.
  With `observation="graph"`, `precedence_edges [2,max_edges]` padded with -1 is added
  (`hospital_env.py:131-134`). An unpadded heterogeneous graph (op, patient, resource and role
  nodes; precedes, requires, of, fills and uses edges) is available from `env.graph()`
  (`observation.py:469-497`) but is **not** part of `observation_space`. All float Boxes are
  `(-inf, inf)`. Sizes for the main config: `max_ops=468`, `max_patients=52`, `max_edges=1404`.
  Rows are assigned in order of visibility and never reused (`observation.py:223-235`).
- **Action: `spaces.Discrete(max_ops * n_rules + 1)`** (`hospital_env.py:106-109`). Action
  `slot*n_rules + rule` starts the task in row `slot` with allocation rule `rule`; the last index
  is wait. The agent chooses a task and, optionally, a rule from `RULES` (`allocation.py:113-118`).
  It **cannot choose specific resources**. One task is started per step, so several starts at
  the same instant take several zero-time steps.
- **Masking: yes.** `env.action_masks()` (`hospital_env.py:307-309`, the maskable-PPO
  convention) and `info["action_mask"]`. Start actions are valid iff the task is READY, its job
  is not busy and a full team is idle (`hospital_env.py:296-301`). Wait is valid if anything is
  running, any resource is in turnover, or the scenario has non-static arrivals, rosters or
  breakdowns (`hospital_env.py:100-104`, `302-304`). A masked action raises
  `InvalidActionError` unless `on_invalid_action="fallback"`, which maps it to the lowest valid
  action (`hospital_env.py:232-236`). The mask is very sparse: on average 0.4–0.8 % of actions
  are valid in the dynamic configs (§10).
- **`info`** is whitelisted by `INFO_KEYS` (`hospital_env.py:38-49`). Every step returns `time`,
  `episode_index` and `action_mask`. Reset adds `n_resamples` and `observation_schema`.
  `invalid_action` and `truncation_reason` appear when relevant. At episode end, `episode` holds
  terminated, time, makespan, patients_finished, total_flow_time, total_weighted_flow_time,
  total_wait, patients_left, no_shows, leave_penalty and reward_potential
  (`hospital_env.py:268-284`).
- **Termination and truncation.** `terminated` is true when every job is finished, cancelled,
  left or a no-show (`sim.done`, `simulator.py:243-245`; `hospital_env.py:247`). `truncated` is
  true at `max_consecutive_waits` (default 100) or `max_episode_time` (default `None`)
  (`hospital_env.py:249-256`, `env/config.py:41-43`). An unstaffable state raises
  `DeadlockError` instead (`simulator.py:360-364`).

## 4. Problem modelling checklist

| Feature | Status | Evidence |
|---|---|---|
| Jobs as DAGs with precedence | **Supported** | Steps with `after` (`generators/config.py:287-296`); cycle check (`dag.py:15-37`); `pending_preds` and `_release_successors` (`simulator.py:153`, `556-566`) |
| Parallel branches within a job | **Supported** | Unordered steps run in parallel. By default the job is an implicit unary resource (`requires_patient=True`, `model.py:124-126`, enforced `simulator.py:256`). Set `requires_patient: false` to let branches truly overlap (e.g. `LabAnalysis` in `configs/example_hospital_dynamic.yaml:54-57`) |
| Multiple resource types; multiple units per type | **Supported** | Roles and resources with `count` (`generators/config.py:78-104`). Each unit has capacity 1 (`model.py:71-73`). No fractional or multi-dimensional capacity |
| Simultaneous multi-resource bundles | **Supported** | `requirements: [(role, quantity)…]`, acquired and released atomically (`resources.py:145-169`); e.g. Surgery needs 6 units over 5 roles (`example_hospital_dynamic.yaml:73-83`) |
| Qualifications / skills | **Supported** | A resource can hold several roles; feasibility via bipartite matching (`matching.py:19-48`, `resources.py:93-118`). Heterogeneity is limited to a scalar `speed` attribute on `speed_roles` (`durations.py:393-424`). No per-(task, machine) duration table |
| Stochastic durations | **Supported** | deterministic, lognormal, gamma, truncated normal, empirical, log-logistic (`durations.py:54-241`), parameterised by `mean`/`cv`, `low`/`high`, `samples` or `tail_index`. Wrappers: `shift` (minimum), `complication` mixture, team-speed `Scaled` (`durations.py:247-358`). Gaussian-copula correlation within a job, `rho` (`instance.py:221-230`). Per-episode `cv_scale`/`rho` randomization (`env/config.py:17-30`, `107-131`). Duration quantiles are pre-drawn per task (`instance.py:194`) and revealed only at completion |
| Online stochastic arrivals | **Supported** | `static`, `poisson` (constant rate or piecewise `rate_profile`, geometric batches, `horizon`/`n_patients`), `scheduled` (booking list, lateness, no-shows) (`instance.py:116-159`, `model.py:171-212`). Future arrivals are hidden; booked appointments are public (`simulator.py:178-182`) |
| Stochastic routing / DAG growing mid-episode | **Supported** | Optional steps, `branch_group`s, `revealed_by` gating, geometric `repeat`s (`instance.py:46-113`). Hidden tasks appear when their trigger completes (`simulator.py:529-533`). All draws are fixed at instance generation, so they are policy-independent. A `random` pathway is drawn **once per config** (`generators/config.py:432-442`), not per job |
| Blocking | **Not supported** | On completion the whole team is released (`simulator.py:510-524`). The only post-completion hold is a fixed `turnover` per (task type, role) (`model.py:130-132`). No buffers, no holding a resource until downstream capacity frees |
| Preemption | **Not supported** | No interrupt action. Breakdowns and shift ends wait for the running task to finish ("lazy removal", `simulator.py:438-440`; `resources.py:171-177`). `Simulator.cancel` only works on unstarted tasks (`simulator.py:313-320`) and is not exposed in the env |
| Breakdowns / unavailability / shifts | **Supported (non-preemptive)** | Repeating calendars with shifts and breaks; random whole-shift absences; exponential MTBF/MTTR breakdowns on calendar time (`availability.py:25-90`); turnover; overtime tracked (`simulator.py:431-455`). The agent sees "unavailable now" and planned roster times, never the realized end of a breakdown or absence (`observation.py:402-424`) |
| Due dates / time windows / priorities | **Partial** | Priorities: per-class `weight` and `urgent` (`model.py:196-198`). Patience (reneging before the first start) acts as a soft deadline (`simulator.py:411-424`). Appointment times work as release times. **No due dates, time windows or tardiness** |
| Deliberate wait / idle | **Supported** | Wait action jumps to the next event (`hospital_env.py:238-240`). There is no "wait Δt" action. Wait can be an unmasked no-op (§11) |
| Elapsed time of running tasks in obs | **Supported** | `ops[:, elapsed]` and `ops[:, expected_remaining]` (conditional expectation under the nominal model and the actual team) (`observation.py:268-282`, `302-308`); `resources[:, expected_remaining]` (`observation.py:419-422`) |

## 5. Reward and objectives

- **Computation** (`hospital_env.py:258-260`, `rewards.py:1-57`):
  `Φ = reward.potential(sim) + leave_penalty · Σ weight(left patients)`, and
  `r_t = −(Φ_t − Φ_{t−1}) · reward_scale` with default `reward_scale = 1/time_scale`.
  For the default `FlowTime`, `Φ = Σ_jobs w_j · (min(now, C_j) − a_j)`, kept as O(1) running
  sums (`simulator.py:278-285`). Rewards telescope: in every episode I ran, the return equalled
  `−Φ_final · scale` to printed precision (§10). The default `leave_penalty` is twice the
  expected work of the longest pathway (`hospital_env.py:91-93`); its value is an explicit
  placeholder (`CLAUDE.md:11`).
- **Built-in objectives** (`rewards.py:53-57`): `weighted_flow_time` (default), `flow_time`,
  `makespan` (potential = `now`), plus `Combined` weighted sums.
- **Not available:** tardiness and lateness (no due dates), utilization, overtime, number of
  leavers as a reward term (separate from the penalty), and throughput. Utilization and overtime
  are computed as metrics (`metrics.py:54-111`) but not as rewards.
- **Swapping:** easy. Pass a name or any object with `potential(sim) -> float`
  (`env/config.py:44-45`, `hospital_env.py:72-74`). The potential must be computed from sim state.
  The leave penalty is always added on top, even for `makespan`.
- **Truncated episodes get no terminal cost** for jobs still in the system. Their future flow
  time is simply absent from the return.

## 6. Configurability and generality

- **Data-driven.** Scenarios are YAML/JSON validated by a strict pydantic schema
  (`generators/config.py:414-511`) plus semantic checks (`domain/validation.py`). Seven configs
  ship in `configs/`; `docs/CONFIG_GUIDE.md` documents them. Nothing scenario-specific is
  hard-coded in the engine.
- **Instance generator.** `generate_instance(scenario, seed, arrivals=None)`
  (`instance.py:243-333`) is configured by the scenario's knobs: arrival process, pathway mix,
  optional/branch/reveal/repeat probabilities, the `complexity` block (`rho`, `acuity_classes`,
  `acuity_correlation`, `reveal_correlation`), duration specs, rosters and breakdowns. Env-level
  `Randomization` adds `rho`, `cv_scale` and `arrival_rate_scale` ranges per episode
  (`env/config.py:17-30`). Random DAG templates take `n_ops`, `density`, `op_types` and `seed`
  (`generators/config.py:299-305`, `pathways.py:13-47`).
- **Hospital scenario:** native; see `configs/example_hospital_dynamic.yaml`.
- **Compute-cluster scenario:** expressible only in a limited way. Machines or GPU pools become
  roles with capacity-1 units, a task needing k GPUs is `quantity: k`, and
  `requires_patient: false` allows parallel stages. Missing pieces:
  - multi-dimensional or fractional capacity (CPU and memory on one node);
  - co-location ("all k units from the same node");
  - machine-specific durations beyond a scalar speed;
  - per-job random DAG shapes (random templates are fixed at config load, `config.py:432-442`).

  The vocabulary (patient, pathway, `requires_patient`) is hospital-specific throughout.
- **Classic JSSP/FJSP benchmarks:** a fixed instance needs one pathway and one `static`
  arrival stream with `n_patients: 1` per job. There is no importer for standard formats
  (Taillard, Brandimarte, Decima traces).
- **Not pluggable through the env:** a custom `DurationModel`. `Simulator` accepts one
  (`simulator.py:89`), but `HospitalEnv` never passes it (`hospital_env.py:202`).
  `EnvConfig.extra` exists (`env/config.py:71`) but nothing reads it.

## 7. Reproducibility and RL-algorithm support

- **Seeding: separate streams.** One instance seed is split with `SeedSequence` into arrivals,
  pathways, durations and latent streams, with a separate family per arrival class
  (`rng.py:34-41`). Each resource has its own absence and breakdown streams (`rng.py:25-31`), and
  randomization has its own stream (`env/config.py:111`). Train, eval and test seeds occupy
  disjoint bit ranges (`env/config.py:86-94`). All job-side randomness is drawn up front
  (`instance.py:243-333`), and availability is drawn on calendar time independently of the policy.
  The trajectory is therefore a deterministic function of (instance, actions).
- **Replay under different policies: yes.** `reset(options={"instance": inst})` replays an
  instance exactly (§10 shows an identical initial observation under three policies). The harness
  does this for every policy (`eval/harness.py:208-255`).
- **Save and restore state: Not supported.** There is no `get_state`/`set_state`. `copy.deepcopy`
  and `pickle` work only for scenarios **without** rosters or breakdowns. Otherwise they fail with
  `TypeError: cannot pickle 'generator' object`, because availability streams are Python
  generators (`simulator.py:209`, `214-233`; `availability.py:61-90`). That covers every dynamic
  config. On `example_hospital` a deep-copied env continued identically. The only workaround is
  re-simulating from the instance with the action prefix.
- **Vectorised envs:** `gymnasium.vector.SyncVectorEnv` and `AsyncVectorEnv` both work (tested
  with 2 envs). Masks are available through `venv.call("action_masks")` or `info["action_mask"]`.
  There is no built-in batched simulator.

## 8. Baselines and evaluation

- **Dispatching rules** (`baselines/dispatching.py:159-170`, all non-idling): FIFO, SPT,
  longest-remaining-work, most-constrained-first, urgent-first, weight-aware (WSPT/Smith on the
  job), acuity-aware, and seeded random. **EDD is not implemented** (there are no due dates).
- **Allocation rules:** `flexible`, `fastest`, `least_used`, `first_idle` (`allocation.py:77-118`),
  combinable with any dispatcher via `Policy` (`dispatching.py:173-206`).
- **Exact or offline solver: Not supported.** The only reference is `best_allocation`: a hindsight
  enumeration of team choices under a *fixed* dispatcher, for tiny instances, capped at 20,000
  leaves (`baselines/search.py:78-118`). There is no MILP/CP solver, no full hindsight optimum and
  no lower bound.
- **Metrics:** makespan, mean and weighted flow time, mean and total task wait, overtime,
  leavers, no-shows, per-role utilization and per-class breakdowns (`eval/harness.py:27-102`,
  `sim/metrics.py`). Also Gantt charts, event logs, a sensitivity/ablation study
  (`eval/sensitivity.py`) and duration/arrival calibration fits (`eval/calibration.py`).
- **Protocol:** fixed `eval`/`test` splits (the `train` split is refused, `eval/harness.py:115`,
  `209-210`). All policies run on identical instances, with paired differences and t-based 95% CIs
  (`eval/harness.py:130-163`). Published results use 200 episodes per config
  (`docs/BASELINE_RESULTS.md`).

## 9. Testing and validation

- 449 tests (427 in the default quick run, about 42 s; 22 more marked `slow`). **All 449 pass**
  (`pytest -m ""`, exit 0). `ruff check src tests` reports no issues; `mypy src` (strict) reports
  no issues in 41 files.
- Coverage includes engine invariants and the simulator, matching, DAGs, validation,
  availability, durations (KS tests against the configured distributions,
  `tests/test_realism_durations.py:143-170`), and hidden-information leak tests
  (`tests/test_env_integrity.py:76-226`). Gymnasium `check_env` is run, but only with
  `on_invalid_action="fallback"` (`tests/test_env.py:72-80`).
- **Validation against known results:** M/M/1 and M/M/c mean waits and utilization against
  closed forms (`tests/test_queueing.py:64-80`). There is **no validation against real data**
  (`docs/ENV_API.md:12-14`) and none against published scheduling optima.
- Side effect: `tests/test_performance.py:23-78` appends to the tracked
  `benchmarks/history.csv`, so the full suite dirties the working tree. I restored the file after
  running it.

## 10. Run results (from `_audit/`)

**Random masked policy** (`run_audit.py`, `debug=True` so invariants were checked every step):

| Config | Episodes: steps / waits / return | All terminated | Obs in space |
|---|---|---|---|
| example_hospital_dynamic (graph) | 75/23/−181.6; 42/14/−75.4; 110/39/−184.6 | yes | yes |
| example_hospital_full (graph) | 97/34/−126.7; 111/36/−86.6 | yes | yes |
| example_hospital_patience (graph) | 104/28/−311.6 (2 left); 131/33/−186.8 (3 left) | yes | **no** (NaN, §11) |
| example_hospital (dict) | 54/10/−76.0; 66/15/−105.4 | yes | yes |

There were no errors or warnings during episodes, simulation time never decreased, and returns
equalled `−Φ_final · scale`.

**`check_env`** (gymnasium 1.4.0):
- With the default `EnvConfig` it **fails**: `InvalidActionError: action N is masked`, on all three
  configs tested and in both obs modes. `check_env` samples unmasked random actions.
- It passes with `on_invalid_action="fallback"`; the repo's own tests use that setting.
- Both settings emit `WARN: A Box observation space minimum/maximum value is -infinity/infinity`,
  6× each per check.

**Throughput** (random masked policy, 10,000 steps, Apple-silicon Mac, single process):

| Setting | env steps/s | start decisions/s |
|---|---|---|
| dynamic / dict | 2,342 | 1,477 |
| dynamic / graph | 2,648 | 1,670 |
| full / graph | 2,247 | 1,437 |
| stress / graph | 2,399 | 1,563 |
| bare `Simulator` (FIFO, no env, incl. instance generation), dynamic / stress | — | 14,939 / 17,598 |

The repo's own benchmark on `example_hospital` gave 55.8k / 23.6k / 12.3k decisions/s for 10 / 50 /
200 patients and 2,388 env steps/s. Building the observation, not the engine, is the bottleneck
(about 6–7× slower).

**Determinism:**
- The same seed and the same policy seed gave identical trajectories (action, reward, SHA-256 of
  every observation, time) on dynamic (93 records) and full (243 records).
- The digests were also identical across two processes with different `PYTHONHASHSEED` values
  (dynamic, full, patience).
- Different seeds gave different trajectories and different first observations.
- `generate_instance(sc, 42)` twice gave equal patients and equal hidden data.

**Other probes:**
- `deepcopy`/`pickle` results are in §7.
- An always-wait policy on dynamic returned −7,659 vs −76.6 for greedy, truncated after 100 waits
  at t≈20,430 min. Waiting is not a reward exploit in the shipped configs.

## 11. Code quality

Strengths: a clear layering; frozen dataclasses for the domain; incremental O(1) cost
accounting; an O(n) invariant checker (`sim/invariants.py`); strict typing and lint-clean code;
docstrings that state intent; hidden information is whitelisted and tested; the assumptions log
records who decided what (`docs/ASSUMPTIONS.md`).

Defects and suspicious logic found:

1. **Engine bug: a departed patient's pure-delay task starts after they leave.**
   - Cause: `_leave` cancels the patient's unstarted tasks (`simulator.py:411-424`). Cancelling a
     READY task releases its successors (`simulator.py:322-327`, `556-566`). A successor that is
     a pure delay is auto-started by `_make_ready` (`simulator.py:488-493`, `113`).
   - Reproduced (`_audit/probe_nan.py`, patience config, seed 0): a patient left unseen at
     t=310.4, and `ResultTurnaround` started at 310.4 and ran until 328.7.
   - Effect 1: while it runs, `completion_time` is NaN, so `obs["patients"][:, time_in_system]`
     is **NaN** (`observation.py:365`) and the observation is outside `observation_space`
     (149 of 2,583 steps in my probe).
   - Effect 2: the recorded flow time and the reward cost are inflated (34.2 vs ~15.9 min).
   - Debug invariants do not catch it.
2. **The eval harness scores truncated episodes as if they were complete.**
   - Cause: `_run_env_agent` returns the sim on `truncated` (`eval/harness.py:214-219`), and
     `compute_metrics` covers only finished patients (`sim/metrics.py:60-63`).
   - Measured (`probe_exploits.py`, 20 eval episodes): an EnvAgent that always waits gets
     mean_flow_time 0.0, total_weighted_flow_time 0.0 and makespan 0.0. `weight_aware` gets
     174.2 / 2,304.7 / 631.1. The bad agent "wins" every flow metric.
   - `info["episode"]` has the same blind spot (`hospital_env.py:269-284`).
3. **The wait action can be an unmasked no-op.**
   - Cause: in a scenario with Poisson or scheduled arrivals but no rosters or breakdowns, wait
     stays valid once the event queue is empty (`hospital_env.py:100-104`, `302-304`), and
     `sim.advance()` then does nothing (`hospital_env.py:238-240`).
   - Reproduced: the shipped dynamic config with rosters and breakdowns removed, seed 18,
     t=327.0. Time did not move, the reward was 0, and after 100 such waits the episode
     truncated with a patient unfinished. Their remaining cost never enters the return.
   - Masking on queue emptiness would leak "no more arrivals", so the fix needs care (§12).
   - None of the shipped configs reach this state, because their infinite roster streams always
     keep an event pending.
4. **State cannot be copied** in scenarios with rosters or breakdowns (generator objects, §7).
5. **`check_env` fails with the default config** (§10). Infinite Box bounds trigger warnings.
6. **Minor issues:**
   - When an instance exceeds the padding sizes, it is redrawn (`hospital_env.py:151-176`). This
     truncates the tail of the episode distribution; the counts are exposed as `n_resamples`.
   - After 255 resamples the seed becomes `base_seed + resample` (`hospital_env.py:165-168`),
     which can collide with another base seed's episodes.
   - `EnvAgent.config.scenario` is never checked against the `scenario` passed to `evaluate`
     (`eval/harness.py:208-212`).
   - `EnvConfig.extra` is unused (`env/config.py:71`).
   - Observation building loops in Python over running tasks and resources at every step
     (`observation.py:302-303`, `402-424`). This is the throughput limit.

Checked and fine:
- Resources are always freed: invariants hold, and release is atomic (`resources.py:154-169`).
- Time is monotone (checked every step).
- No action is free in shipped configs, since every start takes a real resource.
- Rewards telescope exactly.
- Feasibility is handled correctly when one resource holds several roles (matching), and duplicate
  roles within one requirement are rejected (`validation.py:84`).

## 12. Gaps ranked by how much they block RL research

| # | Gap | Why it blocks | Suggested fix (one line) |
|---|---|---|---|
| 1 | Harness and `info["episode"]` score truncated episodes over finished jobs only (§11.2) | Any learned policy that learns to truncate looks optimal; invalidates comparisons | Refuse or flag truncated episodes in `evaluate`, and report accrued flow of unfinished jobs (`sim.flow_accrued`) plus their count |
| 2 | No state save/restore; deepcopy/pickle fail with rosters or breakdowns (§7) | Blocks MCTS, rollout-based planning, lookahead baselines and branching evaluation on every realistic config | Replace generator streams with explicit `(rng, next interval)` state and add `get_state()`/`set_state()` (or make `Simulator` deep-copyable), with a round-trip test |
| 3 | Ghost pure-delay task after a patient leaves, causing NaN observations (§11.1) | NaN in inputs poisons gradient-based training; inflates the cost of leaving | In `_cancel`/`_release_successors`, do not ready or auto-start successors of a cancelled job; add an invariant "LEFT ⇒ no RUNNING tasks"; `nan_to_num` the patient times |
| 4 | Agent cannot choose individual resources, only an allocation rule (§3) | FJSP-style RL, where machine assignment is part of the policy, cannot be studied directly | Add an optional action mode: (task, resource-per-slot) as a two-stage or autoregressive action, with a mask from `idle_members` |
| 5 | No per-job random DAGs and no importers for standard benchmarks (§6) | Decima-style variable DAG workloads and JSSP/FJSP literature comparisons are hard to set up | Draw `random_pathway` per job inside `generate_instance`; add loaders for Taillard/Brandimarte and DAG-trace formats |
| 6 | No due dates/tardiness, preemption or blocking (§4) | Excludes common objectives (tardiness, EDD baselines) and problem classes (flow shops with blocking, preemptive scheduling) | Add `due_date` to `ArrivalSpec`/`Patient` with a `Tardiness` potential; optional `preemptible` task types; a "hold until successor starts" flag |
| 7 | Huge padded action space with <1% valid; graph only as padded edge list; no PyG adapter | Wasted output logits, harder credit assignment; GNN users must write converters | Provide a pointer-style wrapper (action = index into startable list) and a `torch_geometric.HeteroData` converter from `env.graph()` |
| 8 | No exact/offline reference (§8) | Cannot report optimality gaps | Add a CP-SAT/MILP model for small static, deterministic instances, and a hindsight (clairvoyant) solver for stochastic instances |
| 9 | Wait can be a no-op; no "wait Δt"; truncation carries no terminal cost (§5, §11.3) | Agents can burn steps or truncate to avoid cost, depending on the learner's truncation handling | Add a fixed-quantum or next-own-event wait; optionally add the accrued-but-unfinished cost or a penalty at truncation |
| 10 | Throughput of about 2.3–2.6k env steps/s, single-threaded (§10) | 10⁷-step PPO runs take hours per env copy | Build observations incrementally (update only changed rows) and vectorize the per-resource and per-running-task loops |
| 11 | `check_env` fails with defaults; infinite Box bounds; no `gym.make` registration or `render` | Friction with standard tooling and wrappers | Make `check_env` pass by default (or document it), set finite bounds where known, register an env id |
| 12 | Custom `DurationModel` not reachable from the env; hospital-specific naming | Limits domain transfer (e.g. machine-specific durations, fatigue) | Add `EnvConfig.duration_model` and pass it to `Simulator` at `hospital_env.py:202`; alias generic names (job/task) in the schema |
