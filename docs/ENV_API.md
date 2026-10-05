# Environment API

For whoever writes the RL agent. The environment is a standard Gymnasium
`Env`; this page covers what is specific to it. No training code ships with
this project. For a shorter starting point see `docs/RL_INTEGRATION.md`; for
scenario files see `docs/CONFIG_GUIDE.md`.

**Observation schema version: 1** (`hospital_sim.env.OBSERVATION_SCHEMA_VERSION`,
also `info["observation_schema"]` at reset). It changes whenever a field is
added, removed, reordered or changes meaning.

**Not validated against real data.** The simulator's mechanics are tested;
its parameters are illustrative. Nothing here shows that a policy trained in
it transfers to a real hospital.

## Quick start

```python
import numpy as np
from hospital_sim.generators import load_scenario
from hospital_sim.env import EnvConfig, HospitalEnv

scenario = load_scenario("configs/example_hospital_dynamic.yaml")
env = HospitalEnv(EnvConfig(scenario, split="train", observation="graph"))

obs, info = env.reset(seed=0)
while True:
    mask = env.action_masks()  # bool array over actions
    action = np.random.choice(np.flatnonzero(mask))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()  # next episode, new instance
```

`env.action_masks()` follows the convention used by maskable-PPO
implementations. The same mask is in `info["action_mask"]`.

## What a step is

A decision is requested whenever at least one ready operation could start
with the resources that are idle now. Time does not pass while the agent
starts operations; it passes when nothing more can start, or when the agent
waits. One episode is one generated instance, run until every patient is
finished.

## Actions

`Discrete(max_ops * n_rules + 1)` (the `+ 1` only if waiting is enabled).

| Action | Meaning |
|---|---|
| `slot * n_rules + rule` | Start the operation in row `slot` of `obs["ops"]`, forming its team with `config.allocation_rules[rule]` |
| `env.wait_action` (the last index) | Start nothing; jump to the next event |

- **Slots.** An operation gets the next free row when it becomes visible
  (when its patient arrives, or later for operations revealed mid-stay) and
  keeps that row for the whole episode. Rows are never reused.
- **Rules.** With the default `allocation_rules=("flexible",)` the agent only
  picks operations. Passing several rules, for example
  `("flexible", "fastest", "least_used")`, lets it pick the rule as well.

| Rule | Team it forms |
|---|---|
| `flexible` | Keeps versatile and scarce resources free; among equals, the faster one for speed-governing roles |
| `fastest` | Fastest resources in the roles that govern the operation's speed; `flexible` elsewhere |
| `least_used` | Resources with the least cumulative work |
| `first_idle` | First idle in declaration order |

### Mask

`mask[a]` is true exactly when action `a` can be executed now. Start actions
are valid for operations that are ready, whose patient is free and for which
a full team is idle; all rules of a valid operation are valid.

Wait is valid when waiting could change something *for a reason the agent can
know*: an operation is running, a room is in turnover, the scenario has
rosters or breakdowns, or the scenario has arrivals over time. In a static
batch with nothing running, wait is masked.

A masked action raises `InvalidActionError`. With
`on_invalid_action="fallback"` the lowest valid action is executed instead
and `info["invalid_action"]` is set.

### Waiting limits

| Limit | Config | Default |
|---|---|---|
| Consecutive waits | `max_consecutive_waits` | 100 |
| Simulated time | `max_episode_time` | none |

Reaching either ends the episode with `truncated=True` (not `terminated`)
and `info["truncation_reason"]` set to the limit's name.

## Observation

A `Dict` of fixed-shape arrays. Rows beyond the valid ones are zero (or `-1`
for index arrays). All times are divided by `env.time_scale` (default: the
scenario's mean expected operation duration).

| Key | Shape | Content |
|---|---|---|
| `ops` | `[max_ops, 15]` float32 | Operation features, one row per slot |
| `op_valid` | `[max_ops]` int8 | 1 for rows in use |
| `op_type` | `[max_ops]` int32 | Operation type index, `-1` if unused |
| `op_patient` | `[max_ops]` int32 | Row of the patient in `patients` (rows are in the order the scheduler learned of each patient) |
| `op_requirements` | `[max_ops, n_roles]` float32 | Requirement vector (quantity per role) |
| `patients` | `[max_patients, 13]` float32 | Patient features; booked patients are listed from the start, walk-ins when they arrive |
| `patient_valid` | `[max_patients]` int8 | 1 for rows in use |
| `resources` | `[n_resources, 10]` float32 | Resource features |
| `resource_roles` | `[n_resources, n_roles]` int8 | Role membership (constant) |
| `resource_op` | `[n_resources]` int32 | Slot of the operation the resource is working on, `-1` if none |
| `roles` | `[n_roles, 3]` float32 | idle count, member count, demand from ready operations |
| `global` | `[6]` float32 | time, patients in system, ready, startable, running, consecutive waits (fraction of the limit) |
| `precedence_edges` | `[2, max_edges]` int32 | Only with `observation="graph"`: `(predecessor slot, successor slot)`, padded with `-1` |

Feature names, in column order, are exported as `OP_FEATURES`,
`PATIENT_FEATURES`, `RESOURCE_FEATURES`, `ROLE_FEATURES`, `GLOBAL_FEATURES`
from `hospital_sim.env.observation`.

**Operation features.** `is_waiting`, `is_ready`, `is_running`, `is_done`,
`is_cancelled`, `startable`, `expected_duration`, `waited`, `elapsed`,
`expected_remaining`, `realized_duration`, `open_predecessors`, `successors`,
`requires_patient`, `expected_gated_work`.

- `expected_remaining` of a running operation is the conditional expectation
  given how long it has run, for the team actually assigned. It is never the
  true remaining time.
- `realized_duration` is filled in only once the operation is done. With
  correlated durations it lets an agent infer that a patient is slow.

- `expected_gated_work` is the expected duration of work that this
  operation's completion may reveal (see "Operations revealed mid-stay"). It
  comes from the pathway template, is the same whether or not anything will
  actually be revealed, and drops to zero once the operation is done. Set
  `gated_work_prior=False` to remove it.

**Patient features.** `in_system`, `finished`, `expected`, `left`,
`no_show`, `weight`, `urgent`, `acuity` (class scaled to `[0, 1]`, or `-1` if
the scenario has no acuity classes), `time_in_system`, `scheduled_in`,
`open_operations`, `expected_remaining_work`, `busy`.

- Exactly one of the first five flags is set. `expected` means a booked
  appointment that has not arrived; `scheduled_in` is the time until the
  appointment (negative once the patient is late).
- `left`: gave up waiting before being seen. `no_show`: declared missing
  after the scenario's timeout.
- **Assumption: acuity is known at booking.** A booked patient's row shows
  `weight`, `urgent` and `acuity` from time zero, before the patient arrives,
  as if a pre-assessment had been done. Walk-ins show theirs on arrival. If
  acuity is really only assessed on arrival, an agent trained here has
  information it would not have, and results that rely on reordering booked
  patients by acuity would not carry over. This has not been confirmed with
  anyone clinical.

**Resource features.** `available`, `busy`, `unavailable`, `rostered_on`,
`unplanned_down`, `in_turnover`, `time_to_roster_change`,
`expected_remaining` (of its current operation), `work_done`, `speed`.

- `rostered_on` and `time_to_roster_change` come from the published calendar.
- An absence or breakdown appears as `unplanned_down` while it lasts. When it
  will end is not shown.

### Graph form

`env.graph()` returns the current observation without padding:

```python
{
    "nodes": {"op": [k, 15], "patient": [m, 13], "resource": [R, 10], "role": [L, 3]},
    "op_type": [k],
    "edges": {  # each an edge_index array [2, E]
        "op__precedes__op": ...,
        "op__requires__role": ...,  # quantity in edge_attr
        "op__of__patient": ...,
        "resource__fills__role": ...,
        "op__uses__resource": ...,  # running operations only
    },
    "edge_attr": {"op__requires__role": [E]},
    "global": [6],
}
```

The keys split on `__` into (source type, relation, target type), which maps
directly onto a heterogeneous graph in PyTorch Geometric or DGL.

**The graph changes size during an episode.** Operations and patients are
added as they become visible, and `k` and `m` differ between episodes. Do not
assume a fixed number of nodes.

### Operations revealed mid-stay

A scenario can mark a pathway step as `revealed_by` another step, or as
repeatable. Such operations do not exist for the agent until the trigger
completes: no row, no edges, no effect on counts. At that moment the new
operation gets the next free row and the precedence edges are rewired, for
example `surgery -> recovery` becomes `surgery -> intensive care -> recovery`.
A successor never becomes ready in between.

So within one episode the number of operation nodes grows not only when
patients arrive but also when an operation finishes. An edge can disappear
when a revealed operation takes its place.

### What the agent never sees

Realized durations of unfinished operations, patients who have not arrived,
operations not yet revealed (or whether any will be), a patient's true
complexity (only the noisy `acuity` class), whether a booked patient will
turn up, how long a patient will wait before leaving, the end of a breakdown,
a future absence, and the noise parameters drawn for the episode. Tests
enforce each of these.

## Rewards

Each reward is minus the increase of a cost since the previous decision, so
an episode's rewards sum exactly to minus the final cost (times
`reward_scale`).

| `reward=` | Episode total (before scaling) | Use for |
|---|---|---|
| `"weighted_flow_time"` (default) | Sum over patients of weight x (completion - arrival) | Everything with arrivals over time; urgent classes count more |
| `"flow_time"` | The same without weights | |
| `"makespan"` | Time the last patient finishes | Static batches |
| `Combined([(c1, r1), (c2, r2)])` | Weighted sum | Mixed objectives |

`reward_scale` defaults to `1 / env.time_scale`. A custom reward is any
object with `potential(sim) -> float` that is zero at time zero and never
decreases.

For a truncated episode the total covers only the cost accrued up to the
truncation.

**Patients who leave.** A patient who gives up stops accruing flow time,
which on its own would reward ignoring them. So every reward adds
`leave_penalty x weight` per patient who leaves unseen. The default penalty
is twice the expected work of the scenario's longest pathway; set
`EnvConfig(leave_penalty=...)` to change it. The final `info["episode"]`
reports `patients_left`, `no_shows` and the total `leave_penalty`; no-shows
cost nothing.

## Episodes, seeds and splits

- `reset(seed=s)` restarts the episode sequence: the same `s` always gives
  the same first instance, and each later `reset()` gives the next one.
  Without any seed the sequence starts from fresh entropy.
- `EnvConfig(split=...)` takes `"train"`, `"eval"` or `"test"`. The three
  draw from disjoint families of instance seeds, whatever seed you pass.
  **Train on `train`, tune on `eval`, report on `test`.**
- `reset(options={"instance": instance})` runs a specific instance, for
  example to compare policies on identical episodes.
- In horizon mode an instance can have too few patients. The environment
  resamples until it has at least `min_patients` (default 1) and fits the
  padding. `info["n_resamples"]` reports the count for that reset and
  `env.total_resamples` the running total.

### Randomized noise parameters

```python
EnvConfig(
    scenario,
    randomize=Randomization(rho=(0.0, 0.6), cv_scale=(0.7, 1.3), arrival_rate_scale=(0.8, 1.2)),
)
```

Each episode draws the within-patient correlation, a multiplier on all
duration CVs and a multiplier on all Poisson arrival rates. The draws are
hidden. Expected durations in the observation stay at their nominal values.

## Either-or staffing in a scenario

"A scan run by a radiologist or a senior technician" needs no special
feature. Define a role that both belong to and require that role:

```yaml
roles: [{name: ScanOperator}, {name: Radiologist}]
resources:
  - {id: radiologist, roles: [Radiologist, ScanOperator]}
  - {id: senior_tech, roles: [ScanOperator]}
operation_types:
  - name: Scan
    requirements: [{role: ScanOperator}, {role: Scanner}]
```

This covers one-for-one substitution. Alternatives with a different shape
("two nurses, or one nurse and one technician") are not supported.

## Sizes

`max_patients`, `max_ops` and `max_edges` set the padding. Left at `None`
they are derived from the scenario: exact for fixed-size batches, and a
generous bound (mean plus six standard deviations) for horizon mode.
Instances that do not fit are resampled, so set them explicitly if you want
tighter arrays.

## `info`

Only these keys ever appear: `time`, `episode_index`, `action_mask`,
`n_resamples` and `observation_schema` (from `reset`), `invalid_action`, `truncation_reason`, and, on
the last step, `episode` with `terminated`, `time`, `makespan`,
`patients_finished`, `total_flow_time`, `total_weighted_flow_time`,
`total_wait`, `patients_left`, `no_shows`, `leave_penalty`,
`reward_potential`.

## Speed and observation profile

About 4,300 environment steps per second on the example hospitals (one core,
Apple silicon, Python 3.12). The engine alone makes 40,000 to 67,000
decisions per second, so the environment layer is the bottleneck.

Profile of a random masked policy on `example_hospital_dynamic` (60 episodes,
4,342 steps, `cProfile`):

| Part of a step | Share of time |
|---|---|
| `ObservationBuilder.build` in total | 71% |
| ...of which its own Python (per-resource loop, feature assembly) | 29% |
| ...of which small NumPy calls (`np.array` on engine lists, `column_stack`, `nan_to_num`, `bincount`) | about 27% |
| ...of which roster lookups (now cached per calendar) | 6% |
| Mask, reward, `info` | 5% |
| Engine (start, advance, allocation rule) | about 12% |
| Policy and loop overhead in the benchmark itself | about 12% |

Nothing here has been optimized. The obvious steps, in order of expected
gain: keep engine state in NumPy arrays instead of converting lists every
step; update operation rows incrementally instead of rebuilding them; replace
the per-resource Python loop with array operations. The core throughput is
logged on every test run in `benchmarks/history.csv`, including an
`env-10` row for full environment steps.
