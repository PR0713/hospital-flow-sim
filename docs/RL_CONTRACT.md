# RL contract (v0.1)

What an algorithm developer can rely on. Package `hospital_sim` 0.1.0,
observation schema 1 (`info["observation_schema_version"]` at reset),
environment id `SchedSim-v0`. Details of every field are in
`docs/ENV_API.md`; modelling decisions in `docs/ASSUMPTIONS.md`.

The simulator is **not validated against real hospital data**. Results say
how algorithms compare on this model, not how they would do in a hospital.

## 1. Decision epochs

The environment is event-driven (semi-Markov).

- A step is requested when at least one ready operation can start with the
  resources idle now. Starting an operation takes **zero time**.
- After a start, if something else can still start, the next step comes at
  the **same timestamp**. Several decisions often share one timestamp (one per
  operation started then).
- Time moves only when nothing more can start (the simulator runs to the next
  event that makes a start possible) or when the agent chooses **wait**, which
  jumps to the next event (see §4).
- The clock is `info["time"]` (also `obs["global"][0] * env.time_scale`), in
  the scenario's `time_unit` (minutes in all shipped scenarios). The step
  duration `Δt_k = time_{k+1} - time_k` is often 0 and otherwise irregular.

## 2. Reward and return

For a cost potential `Φ` (cost accrued from time 0 to now; default: weighted
flow time `Σ_j w_j (min(now, C_j) - a_j)` plus `leave_penalty x weight` per
patient who left unseen):

    r_k = -c · (Φ(s_{k+1}) - Φ(s_k)),    c = reward_scale (default 1 / env.time_scale)

so the **undiscounted return equals minus the scaled episode cost**:
`Σ_k r_k = -c · Φ_final` (verified to 1e-9 in the tests). Rewards are ≤ 0.
A zero-time step has reward 0; the cost of an interval is charged on the step
that spans it, so it does not depend on how many decisions share a timestamp.
Other objectives (`flow_time`, `makespan`, `Combined`, any object with
`potential(sim)`) follow the same rule (`EnvConfig.reward`).

Watch `info["episode"]["patients_left"]`: the leave penalty is a placeholder
default, and a policy can trade flow time for leavers.

## 3. Discounting

Use **γ = 1**. Episodes are finite (every patient is eventually done), and
the undiscounted return is exactly the objective. Pair it with GAE λ < 1 to
control variance.

A **per-step γ < 1 is wrong here**. Steps are not equal time intervals, so
`γ^k` discounts by how many *decisions* happened, not by how much *time*
passed. Starting three operations at the same instant would discount
everything after them by `γ³`, while one wait over two hours would discount
by `γ` only. The effective objective then depends on decision counts, and
the policy is biased (for example towards waiting, which compresses many
time units into one step).

If you need discounting (an infinite-horizon view, or variance reduction),
discount **by time**:

    G_k = Σ_{i≥k} γ^{(t_i - t_k)/τ} r_i,      TD target  y_k = r_k + γ^{Δt_k/τ} V(s_{k+1})

with `t_i = info["time"]` at step `i` and a time constant `τ` (for example
`env.time_scale`). This is an approximation: `r_k` accrues over `[t_k, t_{k+1}]`
and is not discounted within the interval. Most libraries (for example
SB3 PPO) only support a fixed per-step γ, so with them use γ = 1.

## 4. Actions and masking

- **Base action:** `Discrete(max_ops · n_rules + 1)`. Action
  `slot · n_rules + rule` starts the operation in row `slot` of `obs["ops"]`
  with allocation rule `config.allocation_rules[rule]`. The last index
  (`env.wait_action`) waits. Rows keep their slot for the whole episode and
  are never reused.
- **Mask:** `env.action_masks()` (the maskable-PPO convention) and
  `info["action_mask"]`, a boolean over all actions. It is valid exactly when
  the action can be executed now. Under `gym.make` wrappers use
  `env.unwrapped.action_masks()` or `info["action_mask"]`.
- **Invalid actions** raise `InvalidActionError` (default
  `on_invalid_action="raise"`). The exception is an action equal to what
  `env.action_space.sample()` returned for an *earlier* state: it runs the
  lowest valid action and sets `info["invalid_action"]`. `"fallback"` runs the
  lowest valid action for every invalid one.
- **`env.action_space.sample()`** draws a currently valid action.
- **Wait** is offered only when waiting can change something, for a public
  reason: an operation is running, a resource is in turnover, the scenario has
  rosters or breakdowns, or an arrival stream is not provably over. A valid
  wait always lets time pass. Whether an arrival is actually still to come is
  hidden, and neither the mask nor the wait depends on it.
- **Pointer actions (optional):** `PointerActionWrapper` makes the action an
  index into the current valid (operation, rule) pairs, with candidate
  features in the observation. Candidates are shuffled per step by default
  (seeded).

## 5. Termination and truncation

- `terminated=True`: every patient is finished, left or a no-show. This is a
  true terminal state, so do not bootstrap.
- `truncated=True`: a limit was hit, either `max_consecutive_waits` (default
  100; the agent can cause this) or `max_episode_time` (default none).
  `info["truncation_reason"]` names the limit.
  **Bootstrap on truncation:** target `r + V(s_T)` with the final observation.
  The episode did not really end; patients are still in the system. If you
  treat truncation as terminal with the default `truncation_cost="none"`, the
  agent is rewarded for cutting episodes short, because their future cost
  disappears.
- `EnvConfig(truncation_cost="expected_remaining")` adds one final reward
  term at truncation: minus the scaled weight times a public lower bound on
  each present patient's expected remaining time in system.
  It is reported in `info["episode"]["truncation_cost"]`. **Use it only with a
  learner that treats truncation as terminal.** With bootstrapping it counts
  the future twice.
- Evaluation scores truncated episodes with the cost accrued so far,
  including patients still in the system (`info["episode"]["unfinished_*"]`).
  `evaluate()` reports the truncation rate per policy.
- `SimulationError`/`DeadlockError` are bugs or impossible scenarios, not
  episode ends.

## 6. Observation

`spaces.Dict` of padded arrays; rows past `op_valid`/`patient_valid` are 0
(`-1` in index arrays). **Times are in units of `env.time_scale`**, which
defaults to the scenario's mean expected operation duration (about 30–50
minutes in the shipped scenarios). Weights and counts are unitless.

| Key | Shape | Content and units |
|---|---|---|
| `ops` | `[max_ops, 15]` | Flags `is_waiting/ready/running/done/cancelled`, `startable`; times `expected_duration`, `waited` (ready to now or to start), `elapsed` (running), `expected_remaining` (conditional on elapsed, actual team), `realized_duration` (done only), `expected_gated_work`; counts `open_predecessors`, `successors`; flag `requires_patient` |
| `op_valid` | `[max_ops]` | 1 for rows in use |
| `op_type`, `op_patient` | `[max_ops]` | Type index; patient row |
| `op_requirements` | `[max_ops, n_roles]` | Units of each role needed |
| `patients` | `[max_patients, 13]` | Flags `in_system/finished/expected/left/no_show/urgent/busy`; `weight`; `acuity` (class in [0, 1], -1 if none); times `time_in_system`, `scheduled_in` (booked, may be negative), `expected_remaining_work`; count `open_operations` |
| `patient_valid` | `[max_patients]` | 1 for rows in use |
| `resources` | `[n_res, 10]` | Flags `available/busy/unavailable/rostered_on/unplanned_down/in_turnover`; times `time_to_roster_change` (planned roster only, capped), `expected_remaining`, `work_done`; `speed` |
| `resource_roles` | `[n_res, n_roles]` | Role membership |
| `resource_op` | `[n_res]` | Row of the operation the resource works on, -1 if none |
| `roles` | `[n_roles, 3]` | Counts `idle`, `members`, `demand` (units needed by ready ops) |
| `global` | `[6]` | `time`; counts `patients_in_system`, `ready`, `startable`, `running`; `consecutive_waits` as a fraction of the limit |
| `precedence_edges` | `[2, max_edges]` | `observation="graph"` only: (predecessor row, successor row), -1 padding |

The graph changes size within an episode: patients arrive, and operations
can be revealed mid-stay. `env.graph()` gives the unpadded heterogeneous graph,
and `hospital_sim.env.pyg.to_hetero_data(env.graph())` converts it to a
PyTorch Geometric `HeteroData` (extra `[gnn]`). Bounds: flags in [0, 1],
counts up to the padding sizes, times in [0, inf).
`info["nonfinite_obs_count"]` must stay 0.

**Never observed** (tested): realized durations before completion, latent
patient complexity, future arrivals, future absences and breakdowns (only
"down now"), when patience runs out, steps not yet revealed.

## 7. Seeds, splits and replay

- An instance is a deterministic function of its seed. The trajectory is a
  deterministic function of (instance, actions), including across processes.
- `reset(seed=s)` restarts a sequence of episodes. Episode `e` uses instance
  seed `instance_seed(s, split, e, resample)`. The base seed, split and
  episode sit in disjoint bit ranges, so **train, eval and test instances
  never coincide**. Train with `EnvConfig(split="train")`.
- **Evaluate only through `hospital_sim.eval.evaluate`.** It refuses the train
  split, runs every policy on the same instances (common random numbers) and
  reports paired differences with 95% CIs. An RL agent enters as
  `EnvAgent(name, act(obs, mask) -> action, config, wrapper=None)`; pass
  `wrapper=lambda env: PointerActionWrapper(env)` for an agent trained on the
  wrapper. `benchmarks/ppo_smoke.py` is a complete MaskablePPO example.
- **Benchmark tiers** `T1`–`T4` (`benchmarks/tiers/`) ship fixed eval (100)
  and test (200) seed lists. Use `load_tier("T2")` and
  `evaluate(..., seeds=tier.test_seeds)`. Tune on eval, report on test.
- **Replay a fixed instance across policies:**

  ```python
  from hospital_sim.generators import generate_instance
  inst = generate_instance(scenario, seed)
  obs, info = env.reset(options={"instance": inst})   # same patients, same hidden luck
  ```

  Durations are drawn from per-operation quantiles fixed in the instance, so
  any two policies face the same luck. A duration still depends on the speed
  of the team chosen.

## 8. Versions

`hospital_sim.__version__` is `"0.1.0"`. A change to the meaning, order or
number of observation fields bumps `OBSERVATION_SCHEMA_VERSION`. Check
`info["observation_schema_version"]` at reset if you load a trained policy.
Changes are listed in `CHANGELOG.md`.
