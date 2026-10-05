# Config guide

A scenario is one YAML (or JSON) file. This guide lists every key. Working
examples are in `configs/`; the loader rejects unknown keys and reports all
semantic errors at once.

```python
from hospital_sim.generators import load_config, load_scenario

scenario = load_scenario("configs/example_hospital_full.yaml")
scenario, warnings = load_config("my.yaml").to_scenario()  # to see warnings
```

Times are plain numbers in one unit of your choice (the examples use
minutes). `time_unit` is only a label.

## Shipped examples

| File | What it is for |
|---|---|
| `example_hospital.yaml` | Smallest case: a static batch of 10 patients, no realism features. All allocation rules coincide here, so do not use it to compare policies |
| `four_surgeons.yaml`, `four_surgeons_speed.yaml` | Static batch where team choice matters (overlapping roles; unequal speeds) |
| `example_hospital_dynamic.yaml` | Arrivals over a day, an urgent class, rosters, breakdowns, turnover |
| `example_hospital_stress.yaml` | The same, overloaded: most days spill into the next morning |
| `example_hospital_full.yaml` | Everything on: appointments, no-shows, leaving, reveal, overrides |
| `example_hospital_patience.yaml` | The full example with impatient walk-ins and fewer nurses, so that patients really do leave |

All numbers in them are illustrative. None is fitted to data.

## Top level

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Label used in reports |
| `time_unit` | no | Label only (default `minute`) |
| `roles` | yes | List of `{name, kind}`; `kind` is `staff`, `room` or `equipment` |
| `resources` | yes | Who and what exists |
| `calendars` | no | Rosters that resources can follow |
| `operation_types` | yes | Kinds of step, with requirements and durations |
| `pathways` | yes | What a patient goes through |
| `arrivals` | yes | One stream, or a list of streams (one per patient class) |
| `complexity` | no | Hidden per-patient complexity and its visible signal |

## Resources

```yaml
resources:
  - {roles: [Nurse], count: 6}                      # nurse_1 .. nurse_6
  - {id: surgeon_senior, roles: [LeadSurgeon, AssistantSurgeon], count: 2,
     attributes: {speed: 1.2}, calendar: day_shift}
  - {id: scan_room, roles: [ScanRoom], breakdown: {mtbf: 2000, mttr: 45}}
```

| Key | Meaning |
|---|---|
| `roles` | Every role this resource can fill. One person fills at most one slot of an operation |
| `id`, `count` | With `count > 1` or no `id`, ids are `<id or first role>_1..n` |
| `attributes.speed` | Default 1. Used by operation types that list `speed_roles` |
| `calendar` | Name of a roster; omitted means always on duty |
| `breakdown` | Mean time between failures and mean time to repair (both exponential) |

Every resource has capacity one. A ward with five beds is five resources.

**Either-or staffing.** "A radiologist or a senior technician" is a role that
both belong to. See `docs/ENV_API.md`.

## Calendars

```yaml
calendars:
  - name: day_shift
    period: 1440
    shifts: [{start: 0, end: 720}]
    breaks: [{start: 240, duration: 30}]
    absence_probability: 0.05
```

- The pattern repeats every `period`. Calendar time is simulation time.
- A resource busy when a shift or break starts finishes its operation first;
  the extra time is recorded as overtime. Nothing is interrupted.
- Everyone on one calendar breaks together. Use several calendars to stagger.
- Each shift of each resource is missed with `absence_probability`.
- The loader rejects a scenario in which some operation type can never have a
  full team on duty at once.

## Operation types

```yaml
operation_types:
  - name: Surgery
    requirements:
      - {role: OperatingRoom}
      - {role: LeadSurgeon}
      - {role: Nurse, quantity: 2}
    duration: {mean: 80, cv: 0.4, shift: 30,
               complication: {probability: 0.08, multiplier: 2.5}}
    speed_roles: [LeadSurgeon]
    turnover: {OperatingRoom: 15}
    overrides:
      - when: {patient_class: urgent}
        requirements: [{role: OperatingRoom}, {role: LeadSurgeon}, {role: Nurse, quantity: 3}]
      - when: {acuity_class: 2}
        duration_scale: 1.25
```

| Key | Meaning |
|---|---|
| `requirements` | `(role, quantity)` pairs, all needed at once and held for the whole duration. Empty means a pure delay |
| `duration` | See below |
| `requires_patient` | Default true: the patient must be present, so two such operations of one patient cannot overlap. Set false for work on a sample |
| `speed_roles` | Duration is divided by the geometric mean `speed` of the resources filling these roles |
| `turnover` | `role: time`. Those resources stay unavailable that long after completion; the rest of the team is released at once |
| `overrides` | Different requirements or duration for some patients; first match wins |

A pure delay with `requires_patient: false` starts by itself when ready. With
`requires_patient: true` it is still a decision, because it occupies the
patient.

An override takes `when` (a `patient_class`, an `acuity_class`, or both) and
any of: `requirements` (replaces the whole vector), `duration` (replaces it),
`duration_scale` (stretches it). Internally each becomes its own operation
type, named like `Surgery[urgent]`.

### Durations

| `distribution` | Parameters | Notes |
|---|---|---|
| `lognormal` (default) | `mean`, `cv` | |
| `gamma` | `mean`, `cv` | `cv: 1` is exponential |
| `truncated_normal` | `mean`, `cv`, `low`, `high` | `mean` and `cv` describe the normal before truncation |
| `loglogistic` | `mean`, `tail_index` (> 1) | Power-law tail; variance is infinite at 2 or below |
| `empirical` | `samples` | Resamples the listed values |
| `deterministic` | `mean` | |

Extras: `shift` (lognormal and gamma) is a minimum duration, with `mean` and
`cv` still describing the total. `complication: {probability, multiplier}`
adds a cluster of long cases; it is not a heavy tail, use `loglogistic` for
that.

The duration of an operation is fixed when it starts and is hidden until it
completes. What a scheduler knows is the distribution.

## Pathways

```yaml
pathways:
  - name: surgical
    steps:
      - {id: blood_draw, op_type: BloodDraw}
      - {id: lab, op_type: LabAnalysis, after: [blood_draw]}
      - {id: xray, op_type: XRay, branch_group: imaging, branch_weight: 0.7,
         repeat: {probability: 0.15, max: 2}}
      - {id: ct, op_type: CTScan, branch_group: imaging, branch_weight: 0.3}
      - {id: review, op_type: AnaestheticReview, after: [lab], probability: 0.6}
      - {id: surgery, op_type: Surgery, after: [review, xray, ct]}
      - {id: icu, op_type: IntensiveCare, after: [surgery],
         probability: 0.125, revealed_by: surgery}
      - {id: recovery, op_type: Recovery, after: [surgery, icu]}
```

| Key | Meaning |
|---|---|
| `after` | Steps that must finish first. Steps with no ordering between them may run in parallel |
| `probability` | The step happens with this probability |
| `branch_group`, `branch_weight` | Exactly one step of a group happens |
| `revealed_by` | The step is unknown to the scheduler until that other step completes |
| `repeat` | After each execution, repeat with `probability`, at most `max` more times; each repeat is discovered when the previous one ends |

- When a step does not happen, whatever followed it follows its predecessors
  instead. Order is never lost.
- A step whose `revealed_by` step does not happen does not happen either.
- Whether a step happens is decided when the patient is generated, not during
  the episode.
- A pathway can instead be generated: `{name: x, random: {n_ops: 6, density:
  0.4, seed: 1}}`, with density 0 fully parallel and 1 a chain.

## Arrivals

One block per patient class. Three kinds:

```yaml
arrivals:
  - name: elective
    kind: scheduled
    n_patients: 6
    appointments: {start: 0, interval: 40, punctuality_sd: 8,
                   no_show_probability: 0.08, no_show_timeout: 30}
    pathway_mix: {surgical: 1.0}
  - name: walk_in
    kind: poisson
    horizon: 360
    rate_profile: {period: 360, rates: [0.03, 0.03, 0.025, 0.02, 0.015, 0.01]}
    batch_mean: 1.5
    patience: {mean: 90, cv: 0.6}
    pathway_mix: {diagnostic: 1.0}
  - name: urgent
    kind: poisson
    horizon: 360
    rate: 0.003
    weight: 5.0
    urgent: true
    pathway_mix: {emergency_surgery: 1.0}
```

| Key | Kinds | Meaning |
|---|---|---|
| `kind` | all | `static` (everyone at time zero), `poisson`, `scheduled` |
| `n_patients` | all | Fixed count. Required for `static` and `scheduled` |
| `horizon` | poisson | Arrivals stop at this time, then the episode drains. With `n_patients` too, whichever comes first |
| `rate` or `rate_profile` | poisson | Patients per time unit; the profile is piecewise constant over equal bins of `period` and repeats |
| `batch_mean` | poisson | Mean patients per arrival event (bursts); the mean rate is unchanged |
| `appointments` | scheduled | Booking times, lateness, no-shows |
| `patience` | all | A patient not yet seen after this long leaves |
| `pathway_mix` | all | Weights over pathways; omitted means uniform |
| `weight`, `urgent` | all | Importance in weighted objectives; a visible flag |

`urgent` gives no priority by itself. It is a label and a weight; acting on
it is the policy's job.

## Complexity

```yaml
complexity:
  rho: 0.4
  acuity_classes: 3
  acuity_correlation: 0.7
  reveal_correlation: 0.5
```

| Key | Meaning |
|---|---|
| `rho` | Correlation between the durations of one patient's operations. 0 is independent |
| `acuity_classes` | Number of visible classes (0 = none); higher is more complex |
| `acuity_correlation` | How well the visible class tracks the hidden truth |
| `reveal_correlation` | How strongly hidden complexity drives revealed optional steps |

The true complexity is never visible; only the class is.

## Fitting parameters from data

`hospital_sim.eval.calibration` has two minimal helpers that read the event
log format of `hospital_sim.eval.event_log`:

```python
from hospital_sim.eval.calibration import fit_durations, fit_arrival_profile
from hospital_sim.eval.event_log import read_event_log

rows = read_event_log("log.csv")
for op_type, fit in fit_durations(rows).items():
    print(op_type, fit.to_config())  # a ready-made `duration:` block
print(fit_arrival_profile(rows, period=1440, n_bins=24))
```

They have been tested only by simulating with known parameters and
recovering them. They have never seen real hospital data.
