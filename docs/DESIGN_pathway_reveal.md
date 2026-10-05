# Design note: pathway reveal (M2c-2)

Status: implemented in M2c-2 as described, with the differences listed at the end.

## Goal

Some operations should not be known to the scheduler until an earlier
operation finishes: a repeat test after an abnormal result, an ICU stay after
a complicated surgery. The scheduler must not be able to tell in advance
whether such an operation will appear.

## How revealed operations enter the instance

- **Config.** A pathway step gains `revealed_by: <step id>`. It combines with
  the existing `probability` and `branch_group`.
- **Generation is unchanged.** Whether the step happens is still drawn when
  the patient is generated, from the same random stream. Switching reveal on
  or off therefore leaves patients, pathways and durations identical; only
  the moment of visibility moves. This lets M4 measure the value of that
  information.
- **Two cases.** A gated step that was *not* drawn does not exist in the
  instance at all. One that *was* drawn exists as an ordinary operation that
  starts `HIDDEN` and stays so after the patient arrives.
- **Where the secret lives.** `instance.hidden.reveal_trigger` maps each gated
  operation to its trigger. The engine reads it; nothing policy-facing does.
- **Engine.** On arrival only ungated operations are revealed. When a trigger
  completes, its gated operations become `WAITING` or `READY` in the same
  event batch. A successor of a gated operation counts it as an unfinished
  predecessor from the start, so it can never run early.
- **Cancelled trigger.** Its gated operations are cancelled without ever
  being shown.
- **Repeats.** `repeat: {probability, max}` unrolls into a chain of copies,
  each revealed by the previous one. The number of repeats is drawn up front
  and hidden.
- **Out of scope.** Outcomes that depend on the schedule (a complication made
  likelier by a long wait or a junior surgeon).

## Leak channels and how each is closed

| Channel | Fix |
|---|---|
| Operation ids are consecutive per patient, so a gap betrays a hidden operation | The policy never sees engine ids. Action slots are assigned when an operation becomes visible |
| Precedence edges into or out of a hidden operation | Edges are emitted only between visible operations |
| Counts: operations per patient, remaining work, progress fractions | Computed over visible operations only |
| Array sizes and padding | Sized from the template's maximum, never from the realized count |
| A successor "should" become ready when its visible predecessors finish, but does not | The reveal happens in the same event batch as the trigger's completion, so no observation exists in between |
| Duration quantiles of hidden operations | Already in `instance.hidden` |
| `info` | Key whitelist; ground truth may appear only in the terminal `info` |

## What the observation shows

| | Before the trigger completes | After |
|---|---|---|
| Gated operation | Absent | Present, like any other |
| Its edges | Absent; a visible successor shows only its visible predecessors | Present |
| Trigger operation | One extra feature: expected gated work, the sum over its gated template steps of probability times mean duration | Feature drops to zero |
| Patient progress | Over visible operations | Recomputed with the new operations |

The "expected gated work" feature is prior knowledge from the template (a
clinician knows one surgery in ten needs intensive care). It is the same
number whether or not the step was drawn, so it leaks nothing.

## The tests that enforce it

1. **Twin perturbation.** Build two instances from one seed that differ only
   in hidden outcomes: in one the gated step was drawn, in the other it was
   not. Replay the same actions on both. Observation, mask, reward and `info`
   must be byte-identical at every step up to the trigger's completion, and
   must differ right after it (so the test is known to have teeth).
2. **Truncation twin, over many seeds.** At every decision, the observation
   of the real instance must equal the observation of a copy in which every
   currently hidden operation has been deleted.
3. **Poison.** Overwrite every field of every hidden operation with sentinel
   values (and the hidden quantiles with NaN) and check that observations do
   not change. This catches a builder that reads hidden data without it
   showing up in a particular seed.
4. **Shape.** Observation and mask shapes are equal across instances with
   different numbers of hidden operations.

Tests 2 to 4 are written against the observation builder, so they land in M3
with the environment. In M2c-2 the same four checks run against the engine's
visible view (`ready_ops`, `startable_ops`, visible operation list).

## Questions for you

1. Is "expected gated work" acceptable as a visible feature, or should the
   agent get no prior at all?
2. Should a gated step's probability be allowed to depend on the patient's
   hidden complexity factor (complex patients get complications)? It keeps
   everything pre-drawn, and I would default it on with a configurable
   strength.

## As built (M2c-2)

- Both questions were answered: the prior is a feature behind
  `EnvConfig(gated_work_prior=...)`, and a gated optional step depends on the
  hidden complexity through `complexity.reveal_correlation` (default 0.5).
- The visible precedence graph is the true graph *contracted* over hidden
  operations, not merely filtered. Filtering would have leaked: with
  `surgery -> intensive care -> recovery`, the redundant edge
  `surgery -> recovery` is not stored, so a filtered graph would show recovery
  waiting on nothing.
- A gated step whose trigger does not happen for a patient does not happen
  either.
- The prior counts only what an operation reveals directly, not chains.
- All four tests run against the environment's observations
  (`tests/test_m2c2.py`, `test_leak_1` to `test_leak_4`). The twin is built
  with `without_operations`, which also renumbers engine ids, so the tests
  check that nothing depends on them.
