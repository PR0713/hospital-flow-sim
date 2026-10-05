# hospital-flow-sim

A discrete-event simulator of patient flow in a small hospital unit, with a
Gymnasium environment on top, built for developing and benchmarking
scheduling policies. Version 0.1.

The scheduling problem is a stochastic flexible job shop: patients follow
pathways that are partial orders of operations; each operation needs a whole
team at once (room, surgeons, anaesthetist, nurses); durations are random and
unknown until an operation ends.

This project contains the simulator, the environment, baseline policies and
an evaluation harness. It contains no learning algorithm.

## What it does not claim

**The simulator has not been validated against real hospital data.** All
parameters in `configs/` are illustrative. Its internal mechanics are tested
(over 430 tests, including checks against queueing formulas and tests that
hidden information cannot reach a policy), but that is correctness of the
model as specified, not evidence that the model matches a hospital. Results
obtained here are statements about these configs.

Three modelling choices are placeholders pending a domain decision: the
objective (weighted flow time), the penalty for a patient who leaves unseen,
and the assumption that acuity is known at booking.

## Get it running

Needs Python 3.11 or newer.

```bash
git clone https://github.com/PR0713/hospital-flow-sim.git
cd hospital-flow-sim
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # on Windows: .venv\Scripts\pip
.venv/bin/pytest                      # quick tests, about 30 seconds
```

Then try the two example scripts:

```bash
.venv/bin/python examples/random_agent.py         # the bare environment loop
.venv/bin/python examples/evaluate_my_agent.py    # an agent compared with the baselines
```

## Add your own RL algorithm

The environment is a standard Gymnasium `Env` with action masking. Nothing in
this repository trains anything; you bring the algorithm.

1. **Copy `examples/evaluate_my_agent.py`.** It is a working template.
2. **Write your policy in `MyAgent.act(observation, mask)`.** It gets the
   observation (a dict of arrays, or a graph via `env.graph()`) and a boolean
   mask over actions, and returns the index of a valid action.
3. **Train on the train split.** `make_env("train")` gives you the
   environment; call `reset()`, `step(action)` and `action_masks()` as usual.
   Any masked-action method works (for example maskable PPO, which uses the
   same `action_masks()` convention).
4. **Run the script.** It evaluates your agent on the `test` split, on exactly
   the same instances as the baseline rules, and prints the paired difference
   with a confidence interval.

Things that commonly trip people up, all covered in
`docs/RL_INTEGRATION.md`:

- Most action indices are invalid at any moment. Always use the mask.
- The number of operations and patients changes during an episode. Use
  `op_valid` / `patient_valid`, or the unpadded `env.graph()`.
- Steps are not equal time intervals; the clock is in the observation.
- Use `example_hospital_dynamic.yaml` or a harder config. On
  `example_hospital.yaml` every team-allocation rule gives the same result.
- The baselines worth beating are `urgent_first`, `weight_aware` and
  `acuity_aware`. FIFO is no better than random on the weighted objective.

To describe a different hospital, write a new YAML file; see
`docs/CONFIG_GUIDE.md`. No code changes are needed for different roles,
team sizes, staffing levels, pathways or arrival patterns.

## Tests

```bash
.venv/bin/pytest            # quick tests, about 30 seconds
.venv/bin/pytest -m ""      # everything, about 2 minutes
```

The default run skips tests marked `slow` (queueing-formula checks, long
smoke runs, the sensitivity study, benchmarks). The benchmark tests append to
`benchmarks/history.csv`.

## Documentation

| Document | Content |
|---|---|
| `docs/RL_INTEGRATION.md` | Start here to plug in an agent |
| `docs/ENV_API.md` | Reference: actions, masks, observation fields, rewards, seeds |
| `docs/CONFIG_GUIDE.md` | Every key of a scenario file |
| `docs/BASELINE_RESULTS.md` | Baseline policies on the shipped configs |
| `docs/ALLOCATION_GAP.md` | How much team choice matters |
| `docs/SENSITIVITY.md` | Which noise sources change outcomes and rankings |
| `docs/DESIGN_pathway_reveal.md` | How operations discovered mid-stay are kept hidden |
| `docs/ASSUMPTIONS.md` | Every modelling decision, with who decided it |

## Layout

```
src/hospital_sim/
  domain/      entities, validation, team matching
  generators/  config schema, loader, instance generation
  sim/         event engine, resources, durations, allocation rules
  env/         Gymnasium environment, observations, rewards
  baselines/   dispatching rules, allocation search
  eval/        harness, sensitivity, calibration, plots
configs/       example scenarios
examples/      a minimal agent loop and an evaluation template
benchmarks/    scripts that regenerate the result documents
```

## Known limits

- No preemption: nothing is ever interrupted.
- No fatigue, no beds held across several operations, no blocking between
  units, no alternative requirement sets of different shape.
- Outcomes of optional steps are drawn when a patient is generated; they do
  not depend on how the patient was scheduled.
- About 4,000 environment steps per second on one core.
