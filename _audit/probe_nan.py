"""Locate non-finite values in obs['patients'] (example_hospital_patience)."""
import numpy as np
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.env.observation import PATIENT_FEATURES
from hospital_sim.generators import load_scenario
from hospital_sim.sim.simulator import PatientStatus

env = HospitalEnv(EnvConfig(load_scenario("configs/example_hospital_patience.yaml"), split="train"))
rng = np.random.default_rng(0)
obs, _ = env.reset(seed=0)
for step in range(5000):
    p = obs["patients"]
    bad = np.argwhere(~np.isfinite(p))
    if len(bad):
        row, col = bad[0]
        sim, b = env.sim, env._builder
        pid = int(np.flatnonzero(b.patient_slot == row)[0])
        pat = sim.instance.patients[pid]
        print(f"step {step}: t={sim.now:.1f} row {row} feature {PATIENT_FEATURES[col]}={p[row, col]}"
              f" status={PatientStatus(sim.patient_status[pid]).name} arrival={pat.arrival_time}"
              f" scheduled={pat.scheduled_time} completion={sim.completion_time[pid]}"
              f" n_bad_cells={len(bad)} cols={sorted({PATIENT_FEATURES[c] for c in bad[:,1]})}")
        break
    obs, r, t, u, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    if t or u:
        obs, _ = env.reset()

# --- detail of the offending patient's operations
from hospital_sim.sim.simulator import OpState
sim = env.sim
for op in pat.operations:
    t = sim.op_types[sim.type_of[op.id]]
    print(f"  op {op.id} {op.step_id:10s} type={t.name:18s} reqs={len(t.requirements)} "
          f"requires_patient={t.requires_patient} state={OpState(sim.state[op.id]).name} "
          f"start={sim.start_time[op.id]:.1f}")
# run on until the patient closes
while np.isnan(sim.completion_time[pid]):
    obs, r, t_, u, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
print(f"  patient closes at t={sim.completion_time[pid]:.1f}; recorded flow time = "
      f"{sim.completion_time[pid] - pat.arrival_time:.1f} (left at ~{310.4:.1f})")
