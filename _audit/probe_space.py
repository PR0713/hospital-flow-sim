"""Which observation keys leave observation_space, and why."""
import numpy as np
from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import load_scenario

for name in ["example_hospital_patience", "example_hospital_full", "example_hospital_dynamic"]:
    env = HospitalEnv(EnvConfig(load_scenario(f"configs/{name}.yaml"), split="train"))
    rng = np.random.default_rng(0)
    bad: dict[str, int] = {}
    example = {}
    steps = 0
    obs, _ = env.reset(seed=0)
    for ep in range(20):
        while True:
            for k, space in env.observation_space.spaces.items():
                if not space.contains(obs[k]):
                    bad[k] = bad.get(k, 0) + 1
                    if k not in example:
                        arr = obs[k]
                        lo, hi = space.low, space.high
                        idx = np.argwhere((arr < lo) | (arr > hi))
                        example[k] = (idx[:3].tolist(), arr[tuple(idx[0])] if len(idx) else None,
                                      float(np.min(lo)), float(np.max(hi)), str(arr.dtype), str(space.dtype))
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, r, t, u, info = env.step(a)
            steps += 1
            if t or u:
                obs, _ = env.reset()
                break
    print(name, "steps", steps, "violations per key:", bad, "examples:", example)
