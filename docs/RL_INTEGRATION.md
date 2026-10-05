# Plugging an RL agent into the environment

`docs/ENV_API.md` is the reference for spaces, masks, fields and rewards.
This page is the shorter "how do I start, and what will trip me up".

## What this is, and what it does not claim

This is a simulator of a small, made-up hospital unit. Its mechanics are
tested extensively: precedence, multi-resource teams, rosters, hidden
information, reproducibility.

**It has not been validated against any real hospital data.** Every number
in the shipped configs is illustrative. Nothing here supports a claim that a
policy trained in this simulator will perform well in a real hospital. Use it
to develop and compare scheduling methods under controlled conditions; treat
any result as a statement about these configs.

Open points that change results and still need a domain decision: the
objective (weighted flow time is a placeholder), the leave penalty, and
whether acuity is known at booking.

## Minimal loop

```python
import numpy as np
from hospital_sim.generators import load_scenario
from hospital_sim.env import EnvConfig, HospitalEnv, OBSERVATION_SCHEMA_VERSION

assert OBSERVATION_SCHEMA_VERSION == 1  # the layout your agent was built for
scenario = load_scenario("configs/example_hospital_dynamic.yaml")
env = HospitalEnv(EnvConfig(scenario, split="train", observation="graph"))

obs, info = env.reset(seed=0)
done = False
while not done:
    mask = env.action_masks()
    action = my_agent.act(obs, mask)  # must pick an index where mask is True
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Which config

| Config | Use |
|---|---|
| `example_hospital_dynamic.yaml` | Main development target |
| `example_hospital_stress.yaml` | Behaviour under overload |
| `four_surgeons.yaml`, `four_surgeons_speed.yaml` | Checking that team choice is handled |
| `example_hospital_full.yaml` | Everything, including operations that appear mid-stay |
| `example_hospital_patience.yaml` | Leaving and the leave penalty: here patients really do give up |
| `example_hospital.yaml` | Smoke tests only: every allocation rule gives the same result there, so it cannot tell good policies from bad |

## Things an agent must handle

1. **Masks are mandatory.** A masked action raises `InvalidActionError`. Use
   a masked policy (the `action_masks()` method follows the maskable-PPO
   convention), or set `on_invalid_action="fallback"` while debugging.
2. **The action space is padded.** Most action indices are invalid at any
   moment. Do not sample uniformly from `action_space`.
3. **The graph changes size within an episode.** Patients arrive, and in
   `example_hospital_full` operations appear when another operation finishes
   and precedence edges are rewired. Use `op_valid` and `patient_valid`, or
   `env.graph()` for the unpadded graph. Never assume a fixed node count.
4. **Row indices are stable but arbitrary.** An operation keeps its row for
   the episode. Row order carries no meaning across episodes, so use a
   permutation-invariant model or the graph form.
5. **Steps are not equal time intervals.** Time passes only when nothing can
   start or when the agent waits. `obs["global"][0]` is the clock. This is a
   semi-Markov decision process; think about discounting accordingly.
6. **Rewards are negative costs** and telescope to the episode metric. The
   scale differs a lot between configs; `reward_scale` is yours to set.
7. **Waiting is a real action.** It can be right to hold a resource for an
   urgent arrival. Episodes that wait too long are truncated
   (`info["truncation_reason"]`).
8. **Leaving.** In configs with `patience`, patients give up. The reward
   charges a penalty per leaver; watch `info["episode"]["patients_left"]`,
   since a policy can otherwise look good by shedding patients.
9. **It is partially observable on purpose.** Durations, complexity, future
   arrivals, no-shows, absences and revealed steps are hidden. Finished
   operations show their realized duration, which is evidence about a
   patient's complexity. A memoryless policy sees that evidence in the
   current observation; nothing else is carried between steps for you.

## Seeds and splits

- Train with `EnvConfig(split="train")`. Tune on `"eval"`. Report on `"test"`
  once. The three never share an instance.
- `reset(seed=s)` once at the start; then plain `reset()` for each episode.
- For robustness, train with `randomize=Randomization(...)`, which redraws
  the noise parameters every episode.

## Evaluating against the baselines

Use the harness. It runs every policy on the same instances and reports
paired differences, and it refuses a train-split environment.

```python
from hospital_sim.baselines import Policy
from hospital_sim.eval import EnvAgent, evaluate

agent = EnvAgent(
    "my_agent",
    lambda obs, mask: my_agent.act(obs, mask),
    EnvConfig(scenario, split="test", observation="graph"),
)
result = evaluate(
    scenario, [Policy("urgent_first"), Policy("weight_aware"), agent], n_episodes=200, split="test"
)
print(result.table(("weighted_mean_flow_time", "mean_flow_time", "makespan", "left")))
print(result.paired_difference("weighted_mean_flow_time", "my_agent", "urgent_first+flexible"))
```

**Baselines worth beating are `urgent_first`, `weight_aware` and
`acuity_aware`.** On the configs with an urgent class, FIFO and SPT are no
better than a random dispatcher on the weighted objective
(`docs/BASELINE_RESULTS.md`), so beating them shows little.

`acuity_aware` is `weight_aware` with remaining work predicted from the
visible acuity class. The paired difference between the two is what the
observed class is worth to a simple rule; an agent that uses the class well
should gain at least that much.

Report, at least: the config, the split, the number of episodes, the paired
difference with its confidence interval, the leaver count, and the per-class
flow times (`result.by_class`).

## Team selection

By default the agent picks operations and the `flexible` rule forms teams.
On the shipped configs that rule is within 2% of a hindsight-best allocation,
measured on 4-patient instances only (`docs/ALLOCATION_GAP.md`). To let the
agent choose, pass `allocation_rules=("flexible", "fastest", "least_used")`;
the action becomes (operation, rule).

## Speed

About 4,000 steps per second on one core; observation building dominates.
Run several environments in parallel processes if you need more. See the
profile in `docs/ENV_API.md`.

## If the layout changes

`OBSERVATION_SCHEMA_VERSION` is bumped whenever a field is added, removed,
reordered or changes meaning. Check it when loading a trained model.
