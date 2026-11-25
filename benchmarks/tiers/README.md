# Benchmark tiers (frozen for v0.1)

Four scenarios of increasing difficulty with fixed instance seeds:
100 for tuning (`eval` split) and 200 for reporting (`test` split). Train
with `EnvConfig(split="train")`; train seeds can never coincide with these.

| Tier | Adds | Based on |
|---|---|---|
| T1 | Static batch of 10 patients, one role per resource, lognormal durations with cv 0.1, no rosters or breakdowns | `configs/example_hospital.yaml` |
| T2 | Poisson arrivals over 6 h (profiled, in clumps) plus an urgent stream (weight 5); surgeons qualified for two roles and of different speed; plain lognormal durations; a pure-delay step | `configs/example_hospital_dynamic.yaml` without its realism layer |
| T3 | T2 + nurse rosters with breaks, random absences, a scanner that breaks down, correlated durations within a patient (rho 0.4) with a visible noisy acuity class | `example_hospital_dynamic` without turnover and heavy tails |
| T4 | T3 + steps revealed mid-stay, repeated scans, branching imaging, booked electives with lateness and no-shows, walk-ins who give up (patience mean 20 min), turnover, heavy-tailed surgery and ICU, overrides; offered load about 1.6 on the scanner and nurses during arrivals (T2/T3: 0.64) | `configs/example_hospital_patience.yaml` |

```python
from hospital_sim.eval import evaluate, load_tier
tier = load_tier("T2")                       # scenario, eval_seeds, test_seeds
result = evaluate(tier.scenario, policies, seeds=tier.test_seeds)
env = gym.make("SchedSim-v0", scenario="T2") # same scenario as an env
```

**Frozen.** `seeds.yaml` records the SHA-256 of `config.yaml`, and
`load_tier` refuses a tier whose config changed. To change a benchmark, add
a new tier (for example `T5`) and freeze its seeds with
`.venv/bin/python benchmarks/tiers/make_seeds.py T5`. Seed `i` of a split is
`instance_seed(2026, split, i, r)`, with the smallest `r` whose instance has
a patient and fits the environment's default padding.

All numbers are illustrative and not validated against real data.
