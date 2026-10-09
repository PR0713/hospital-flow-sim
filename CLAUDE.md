# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A discrete-event simulator of patient flow in a small hospital unit, with a Gymnasium environment on top, for developing and benchmarking scheduling policies. The problem is a stochastic flexible job shop: patients follow pathways (partial orders of operations), each operation needs a whole team at once (room, surgeons, anaesthetist, nurses), and durations are random and hidden until an operation ends.

The repo ships the simulator, environment, baseline policies and evaluation harness. It contains **no learning algorithm**; agents (RL or LLM) are meant to be built on top without changing the core.

**Not validated against real hospital data.** All config numbers are illustrative. Do not write claims that results transfer to real hospitals. Three modelling choices are explicit placeholders awaiting the owner's decision: the objective (weighted flow time), the leave penalty, and whether acuity is known at booking. `docs/ASSUMPTIONS.md` logs every modelling decision with who made it; check it before changing model behaviour and add an entry for any new decision.

## Commands

Python 3.11+. The venv is `.venv`; install with `.venv/bin/pip install -e ".[dev]"`.

```bash
.venv/bin/pytest                          # quick suite (~30s); skips tests marked slow
.venv/bin/pytest -m ""                    # everything (~2 min); run before a milestone commit
.venv/bin/pytest tests/test_env.py::test_name   # single test
.venv/bin/ruff check src tests            # lint (line length 100; E,F,I,UP,B,SIM)
.venv/bin/mypy src                        # strict mode, pydantic plugin
.venv/bin/python examples/random_agent.py         # bare env loop
.venv/bin/python examples/evaluate_my_agent.py    # agent vs baselines, paired CIs
```

Benchmark tests append to `benchmarks/history.csv`. `benchmarks/run_m4_studies.py` and `run_m5_studies.py` regenerate the result docs in `docs/`.

## Architecture

Layers, bottom to top (each depends only on the layers listed before it):

1. `domain/`: entities (`Scenario`, `ProblemInstance`), validation, DAG of precedence, team matching. Team feasibility is a **bipartite matching**, not a per-role count, because a resource can hold several roles.
2. `generators/`: YAML config schema (pydantic), `load_scenario`, and `generate_instance`, which draws one concrete instance (arrivals, pathways, hidden durations) from a seed.
3. `sim/`: `Simulator` is the event engine. `run_until_decision()` advances to a decision epoch (some READY operation whose team can be formed from idle resources); `startable_ops()` lists candidates; `start(op, team?)` commits; `advance()` jumps to the next event. Allocation rules (how to choose the team) are in `allocation.py` (`RULES`). `invariants.py` holds engine consistency checks.
4. `env/`: `HospitalEnv`, a Gymnasium `Env` over the simulator. `EnvConfig` selects the split, observation form (`"graph"` or flat), allocation rules, reward and whether waiting is allowed.
5. `baselines/` (dispatching rules, allocation search) and `eval/` (harness, sensitivity, calibration, Gantt, event log) sit on top of both `sim` and `env`.

### Things that need several files to understand

- **Action space is padded and almost entirely invalid.** `Discrete(max_ops * n_rules + 1)`; action `slot * n_rules + rule` starts the op in observation row `slot` using allocation rule `rule`; the last index is "wait". Always use `env.action_masks()`. A masked action raises `InvalidActionError` unless `on_invalid_action="fallback"`.
- **Steps are not equal time intervals** (semi-Markov): time passes only when nothing can start or the agent waits. The clock is `obs["global"][0]`.
- **Graph size changes within an episode.** Patients arrive, and in `example_hospital_full` operations appear mid-stay. Use `op_valid` / `patient_valid` or the unpadded `env.graph()`. Rows keep their slot for the whole episode and are never reused.
- **Hidden information is deliberate and tested.** True durations, complexity, future arrivals, no-shows, absences and mid-stay revealed steps must never reach a policy (`INFO_KEYS` in `hospital_env.py` whitelists `info`; see `docs/DESIGN_pathway_reveal.md`). Any new observation or info field must be checked against this. Changing observation layout requires bumping `OBSERVATION_SCHEMA_VERSION`.
- **Seeds are split into train / eval / test.** `instance_seed(base, split, episode, resample)` gives disjoint bit ranges per split. The harness refuses the `train` split, evaluates every policy on identical instances (common random numbers), and reports paired differences with confidence intervals. Compare policies through `eval.harness.evaluate`, not ad hoc loops.
- **Rewards are negative costs** that telescope to the episode metric. A policy can look good by letting patients leave unseen, so watch `info["episode"]["patients_left"]`.

### Configs

`configs/example_hospital_dynamic.yaml` is the main development target. `example_hospital.yaml` is a smoke-test only: every allocation rule gives identical results there, so it cannot separate good policies from bad. Baselines worth beating: `urgent_first`, `weight_aware`, `acuity_aware` (FIFO is no better than random on the weighted objective). New hospitals are YAML only; see `docs/CONFIG_GUIDE.md`.

## Agent layer (branch `agent-layer`)

`src/hospital_sim_mcp/` wraps `HospitalEnv` as an MCP server (`.mcp.json` registers it as `hospital-sim`; run manually with `.venv/bin/python -m hospital_sim_mcp.server`). `session.py` holds all logic with no MCP imports (tests: `tests/test_mcp_session.py`); `server.py` is a thin FastMCP wrapper. Everything shown to the LLM is built from the public observation arrays, never from `sim.instance.hidden`. The core packages (`domain`, `generators`, `sim`, `env`, `baselines`, `eval`) are deliberately left unmodified on this branch; keep it that way unless asked.

## Docs

`docs/RL_INTEGRATION.md` (start here to plug in an agent), `docs/ENV_API.md` (actions, masks, observation fields, seeds), `docs/CONFIG_GUIDE.md`, `docs/BASELINE_RESULTS.md`, `docs/ASSUMPTIONS.md`.
