"""Throwaway audit script. Run from the repo root:

    .venv/bin/python _audit/run_audit.py

Does not modify any repository file.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.generators import load_scenario
from hospital_sim.generators.instance import generate_instance
from hospital_sim.sim.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def scenario(name: str):
    return load_scenario(CONFIGS / f"{name}.yaml")


def header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def obs_digest(obs: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(obs):
        arr = np.ascontiguousarray(obs[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
def random_episodes(config_name: str, n: int, observation: str = "graph", debug: bool = True):
    header(f"random masked policy: {config_name} ({observation}, debug={debug})")
    env = HospitalEnv(
        EnvConfig(scenario(config_name), split="train", observation=observation, debug=debug)
    )
    print(
        f"action_space={env.action_space}, max_ops={env.max_ops}, "
        f"max_patients={env.max_patients}, max_edges={env.max_edges}, "
        f"time_scale={env.time_scale:.2f}, leave_penalty={env.leave_penalty:.1f}"
    )
    rng = np.random.default_rng(123)
    obs, info = env.reset(seed=1)
    for ep in range(n):
        total, steps, waits, contained, mono = 0.0, 0, 0, True, True
        last_t = info["time"]
        mask_density = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            while True:
                mask = env.action_masks()
                assert (mask == info["action_mask"]).all()
                mask_density.append(mask.mean())
                if not env.observation_space.contains(obs):
                    contained = False
                a = int(rng.choice(np.flatnonzero(mask)))
                waits += a == env.wait_action
                obs, r, term, trunc, info = env.step(a)
                total += r
                steps += 1
                if info["time"] < last_t:
                    mono = False
                last_t = info["time"]
                if term or trunc:
                    break
        e = info["episode"]
        print(
            f"ep {ep}: steps={steps} waits={waits} return={total:.2f} "
            f"(-potential*scale={-e['reward_potential'] / env.time_scale:.2f}) "
            f"terminated={term} truncated={trunc} reason={info.get('truncation_reason')} "
            f"patients_finished={e['patients_finished']} left={e['patients_left']} "
            f"no_shows={e['no_shows']} makespan={e['makespan']:.0f} "
            f"obs_in_space={contained} time_monotone={mono} "
            f"mean_mask_density={np.mean(mask_density):.4f} "
            f"warnings={[str(w.message) for w in caught]}"
        )
        obs, info = env.reset()


# ---------------------------------------------------------------------------
def run_check_env():
    from gymnasium.utils.env_checker import check_env

    for name, observation in [
        ("example_hospital_dynamic", "dict"),
        ("example_hospital_dynamic", "graph"),
        ("example_hospital_full", "graph"),
    ]:
        header(f"check_env: {name} / {observation}")
        env = HospitalEnv(EnvConfig(scenario(name), split="train", observation=observation))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                check_env(env, skip_render_check=True)
                print("check_env (skip_render_check=True): passed")
            except Exception as exc:  # noqa: BLE001
                print(f"check_env raised {type(exc).__name__}: {exc}")
        for w in caught:
            print(f"  warning: {w.category.__name__}: {w.message}")
        # Without skipping the render check
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                check_env(HospitalEnv(EnvConfig(scenario(name), split="train")))
                print("check_env (default args): passed")
            except Exception as exc:  # noqa: BLE001
                print(f"check_env (default args) raised {type(exc).__name__}: {exc}")
        for w in caught:
            print(f"  warning: {w.category.__name__}: {w.message}")


# ---------------------------------------------------------------------------
def trajectory(config_name: str, seed: int, policy_seed: int, episodes: int = 2):
    env = HospitalEnv(EnvConfig(scenario(config_name), split="train", observation="graph"))
    rng = np.random.default_rng(policy_seed)
    obs, info = env.reset(seed=seed)
    log = [("reset", obs_digest(obs), info["time"])]
    for _ in range(episodes):
        while True:
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, r, term, trunc, info = env.step(a)
            log.append((a, round(r, 12), obs_digest(obs), info["time"]))
            if term or trunc:
                break
        obs, info = env.reset()
        log.append(("reset", obs_digest(obs), info["time"]))
    return log


def determinism():
    header("determinism")
    for name in ["example_hospital_dynamic", "example_hospital_full"]:
        a = trajectory(name, seed=7, policy_seed=0)
        b = trajectory(name, seed=7, policy_seed=0)
        c = trajectory(name, seed=8, policy_seed=0)
        d = hashlib.sha256(repr(a).encode()).hexdigest()[:16]
        print(
            f"{name}: same seed identical={a == b} (len {len(a)}, digest {d}); "
            f"different seed differs={a != c} (len {len(c)}); "
            f"first reset obs equal across seeds={a[0][1] == c[0][1]}"
        )
    # Same instance, two different policies: identical instance & hidden draws?
    sc = scenario("example_hospital_dynamic")
    i1 = generate_instance(sc, 42)
    i2 = generate_instance(sc, 42)
    print(
        "generate_instance(seed=42) twice: patients equal="
        f"{i1.patients == i2.patients}, hidden equal={i1.hidden == i2.hidden}"
    )


# ---------------------------------------------------------------------------
def throughput():
    header("throughput")
    for name, observation in [
        ("example_hospital_dynamic", "dict"),
        ("example_hospital_dynamic", "graph"),
        ("example_hospital_full", "graph"),
        ("example_hospital_stress", "graph"),
    ]:
        env = HospitalEnv(EnvConfig(scenario(name), split="train", observation=observation))
        rng = np.random.default_rng(0)
        env.reset(seed=0)
        steps = starts = episodes = 0
        sim_time = 0.0
        t0 = time.perf_counter()
        while steps < 10_000:
            mask = env.action_masks()
            a = int(rng.choice(np.flatnonzero(mask)))
            starts += a != env.wait_action
            *_, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                sim_time += info["time"]
                episodes += 1
                env.reset()
        dt = time.perf_counter() - t0
        print(
            f"env {name}/{observation}: {steps} steps in {dt:.2f}s -> {steps / dt:.0f} steps/s, "
            f"{starts / dt:.0f} start-decisions/s, {episodes} episodes finished, "
            f"obs ops-table shape={env.observation_space['ops'].shape}"
        )

    # Raw simulator without the env (FIFO dispatch, first-idle team)
    for name in ["example_hospital_dynamic", "example_hospital_stress"]:
        sc = scenario(name)
        decisions = 0
        t0 = time.perf_counter()
        seed = 0
        while decisions < 10_000:
            sim = Simulator(generate_instance(sc, seed))
            while sim.run_until_decision():
                sim.start(sim.startable_ops()[0])
                decisions += 1
            seed += 1
        dt = time.perf_counter() - t0
        print(
            f"bare Simulator {name}: {decisions} decisions in {dt:.2f}s -> "
            f"{decisions / dt:.0f} decisions/s ({seed} episodes, incl. instance generation)"
        )


# ---------------------------------------------------------------------------
def state_copy():
    header("state save/restore probes")
    for name in ["example_hospital", "example_hospital_dynamic"]:
        env = HospitalEnv(EnvConfig(scenario(name), split="train"))
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(10):
            env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        for label, fn in [
            ("copy.deepcopy(env)", lambda e=env: copy.deepcopy(e)),
            ("pickle.dumps(env)", lambda e=env: pickle.dumps(e)),
            ("copy.deepcopy(env.sim)", lambda e=env: copy.deepcopy(e.sim)),
        ]:
            try:
                fn()
                print(f"{name}: {label}: OK")
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: {label}: FAILS with {type(exc).__name__}: {exc}")
        # If deepcopy works, does the copy continue identically?
        try:
            clone = copy.deepcopy(env)
        except Exception:  # noqa: BLE001
            continue
        r1 = np.random.default_rng(5)
        r2 = np.random.default_rng(5)
        same = True
        while True:
            a1 = int(r1.choice(np.flatnonzero(env.action_masks())))
            a2 = int(r2.choice(np.flatnonzero(clone.action_masks())))
            o1, x1, t1, u1, _ = env.step(a1)
            o2, x2, t2, u2, _ = clone.step(a2)
            same &= a1 == a2 and x1 == x2 and obs_digest(o1) == obs_digest(o2)
            if t1 or u1 or t2 or u2:
                same &= (t1, u1) == (t2, u2)
                break
        print(f"{name}: deepcopy clone continues identically under the same actions: {same}")


def replay_fixed_instance():
    header("replay a fixed instance under different policies")
    sc = scenario("example_hospital_dynamic")
    inst = generate_instance(sc, 99)
    env = HospitalEnv(EnvConfig(sc, split="eval"))
    results = {}
    for pol in ["first", "last", "random"]:
        obs, info = env.reset(options={"instance": inst})
        rng = np.random.default_rng(0)
        d0 = obs_digest(obs)
        total = 0.0
        while True:
            valid = np.flatnonzero(env.action_masks())
            valid_start = [v for v in valid if v != env.wait_action] or list(valid)
            a = {"first": valid_start[0], "last": valid_start[-1]}.get(
                pol, int(rng.choice(valid_start))
            )
            obs, r, term, trunc, info = env.step(int(a))
            total += r
            if term or trunc:
                break
        results[pol] = (d0, round(total, 3), info["episode"]["total_weighted_flow_time"])
    print(results)


def vector_env():
    header("gymnasium vector env")
    import gymnasium as gym

    sc = scenario("example_hospital_dynamic")
    for cls in [gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv]:
        try:
            venv = cls([lambda: HospitalEnv(EnvConfig(sc, split="train")) for _ in range(2)])
            obs, info = venv.reset(seed=0)
            masks = np.stack(venv.call("action_masks"))
            acts = [int(np.flatnonzero(m)[0]) for m in masks]
            obs, r, term, trunc, info = venv.step(np.array(acts))
            print(
                f"{cls.__name__}: reset+step OK; obs['ops'] shape {obs['ops'].shape}; "
                f"call('action_masks') shape {masks.shape}; info keys {sorted(info)[:6]}..."
            )
            venv.close()
        except Exception as exc:  # noqa: BLE001
            print(f"{cls.__name__}: FAILS with {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)


if __name__ == "__main__":
    which = sys.argv[1:] or [
        "episodes",
        "check_env",
        "determinism",
        "throughput",
        "state",
        "replay",
        "vector",
    ]
    if "episodes" in which:
        random_episodes("example_hospital_dynamic", 3)
        random_episodes("example_hospital_full", 2)
        random_episodes("example_hospital_patience", 2)
        random_episodes("example_hospital", 2, observation="dict")
    if "check_env" in which:
        run_check_env()
    if "determinism" in which:
        determinism()
    if "throughput" in which:
        throughput()
    if "state" in which:
        state_copy()
    if "replay" in which:
        replay_fixed_instance()
    if "vector" in which:
        vector_env()
