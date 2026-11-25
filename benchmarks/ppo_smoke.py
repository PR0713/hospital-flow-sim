"""Learnability smoke test: MaskablePPO on a benchmark tier (needs the [rl] extra).

    caffeinate -ims .venv/bin/python benchmarks/ppo_smoke.py T1 --timesteps 1000000

Not a tuned agent. It checks that the environment is learnable (the policy
beats random) and that the learner finds no way around the cost
(truncation, waiting, leavers, reward bookkeeping), then evaluates on the
tier's 200 test seeds through the harness against random and the best
heuristic. Writes ``benchmarks/results/v0.1/ppo_<tier>.json``, a learning
curve in ``docs/figures/v0.1/ppo_<tier>.png``, and the model under
``benchmarks/results/v0.1/models/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from hospital_sim.baselines import Policy
from hospital_sim.env import EnvConfig, HospitalEnv, PointerActionWrapper
from hospital_sim.eval import EnvAgent, Evaluation, evaluate, load_tier

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "v0.1"
FIGURES = ROOT / "docs" / "figures" / "v0.1"
MAX_CANDIDATES = 32  # the most simultaneous valid pairs seen in the shipped configs is 30
STATS = ("ep_cost", "ep_truncated", "ep_waits", "ep_left", "ep_nonfinite", "ep_return_gap")


class EpisodeStats(gym.Wrapper[Any, Any, Any, Any]):
    """Adds flat end-of-episode statistics to ``info`` (Monitor replaces
    ``info["episode"]`` with its own record, so they are copied out first)."""

    def __init__(self, env: PointerActionWrapper) -> None:
        super().__init__(env)
        self.pointer = env
        self._waits = self._steps = 0
        self._return = 0.0

    def reset(self, **kwargs: Any) -> Any:
        self._waits = self._steps = 0
        self._return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._waits += int(action) == self.pointer.wait_index
        self._steps += 1
        self._return += float(reward)
        if terminated or truncated:
            episode = info["episode"]
            base = self.pointer.base
            cost = episode["total_weighted_flow_time"] + episode["leave_penalty"]
            info["ep_cost"] = cost
            info["ep_truncated"] = float(truncated)
            info["ep_waits"] = self._waits / self._steps
            info["ep_left"] = float(episode["patients_left"])
            info["ep_nonfinite"] = float(info["nonfinite_obs_count"])
            # Reward bookkeeping: the undiscounted return is minus the scaled
            # cost (including any truncation cost). Should be ~0.
            scale = 1.0 / base.time_scale
            info["ep_return_gap"] = self._return + scale * (
                episode["reward_potential"] + episode["truncation_cost"]
            )
        return obs, reward, terminated, truncated, info


def make_env(tier: str, shuffle: bool = True) -> gym.Env[Any, Any]:
    scenario = load_tier(tier).scenario
    base = HospitalEnv(EnvConfig(scenario, split="train"))
    pointer = PointerActionWrapper(
        base,
        max_candidates=MAX_CANDIDATES,
        shuffle_candidates=shuffle,
        keep_base_observation=False,
    )
    return Monitor(EpisodeStats(pointer), info_keywords=STATS)


def ppo_agent(
    model: MaskablePPO,
    vecnorm: VecNormalize,
    tier: str,
    counter: dict[str, int],
    shuffle: bool = True,
) -> EnvAgent:
    def act(obs: dict[str, Any], mask: np.ndarray) -> int:
        action, _ = model.predict(vecnorm.normalize_obs(obs), action_masks=mask, deterministic=True)
        action = int(action)
        counter["steps"] += 1
        counter["waits"] += action == len(mask) - 1
        return action

    return EnvAgent(
        "ppo",
        act,
        EnvConfig(load_tier(tier).scenario, split="eval"),
        wrapper=lambda env: PointerActionWrapper(
            env,
            max_candidates=MAX_CANDIDATES,
            shuffle_candidates=shuffle,
            keep_base_observation=False,
        ),
    )


def cost(result: Evaluation, policy: str, penalty: float) -> np.ndarray:
    return result.series(policy, "total_weighted_flow_time") + penalty * result.series(
        policy, "left_weight"
    )


def ci(x: np.ndarray) -> list[float]:
    from scipy import stats

    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return [float(x.mean()), half]


class Curve(BaseCallback):
    """Every ``every`` steps: training statistics and a quick eval on eval seeds."""

    def __init__(
        self, tier: str, vecnorm: VecNormalize, every: int, n_eval: int, shuffle: bool
    ) -> None:
        super().__init__()
        self.tier, self.vecnorm, self.every, self.n_eval = tier, vecnorm, every, n_eval
        self.shuffle = shuffle
        self.points: list[dict[str, float]] = []
        self._next = 0
        t = load_tier(tier)
        self.seeds = t.eval_seeds[:n_eval]
        self.penalty = HospitalEnv(EnvConfig(t.scenario)).leave_penalty

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.points.append(self._measure())
            p = self.points[-1]
            print(
                f"[{self.tier}] {self.num_timesteps:>9,} steps | eval cost {p['eval_cost']:8.0f} | "
                f"train cost {p['train_cost']:8.0f} | trunc {p['train_truncated']:.2f} | "
                f"wait {p['train_wait_share']:.2f} | {p['minutes']:.1f} min",
                flush=True,
            )
        return True

    def _measure(self) -> dict[str, float]:
        model: Any = self.model
        counter = {"steps": 0, "waits": 0}
        result = evaluate(
            load_tier(self.tier).scenario,
            [ppo_agent(model, self.vecnorm, self.tier, counter, self.shuffle)],
            seeds=self.seeds,
        )
        buffer = list(model.ep_info_buffer)

        def mean(key: str) -> float:
            return float(np.mean([e[key] for e in buffer])) if buffer else float("nan")

        return {
            "timesteps": float(self.num_timesteps),
            "minutes": (time.perf_counter() - STARTED) / 60,
            "eval_cost": float(cost(result, "ppo", self.penalty).mean()),
            "eval_truncated": result.truncation_rate("ppo"),
            "train_cost": mean("ep_cost"),
            "train_return": mean("r"),
            "train_truncated": mean("ep_truncated"),
            "train_wait_share": mean("ep_waits"),
            "train_left": mean("ep_left"),
            "train_return_gap": float(np.max(np.abs([e["ep_return_gap"] for e in buffer])))
            if buffer
            else float("nan"),
            "train_nonfinite": mean("ep_nonfinite"),
        }


STARTED = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="diagnostic: candidates in oldest-ready order instead of shuffled",
    )
    args = parser.parse_args()
    tier = args.tier
    shuffle = not args.no_shuffle
    tag = tier if shuffle else f"{tier}_noshuffle"
    torch.set_num_threads(2)

    venv = SubprocVecEnv([lambda t=tier: make_env(t, shuffle) for _ in range(args.n_envs)])
    venv.seed(args.seed)  # train split: each worker resets with seed + rank
    vecnorm = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        norm_obs_keys=["candidate_features", "global"],
        gamma=1.0,
        clip_obs=10.0,
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        vecnorm,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        device="cpu",
        seed=args.seed,
        verbose=0,
    )
    curve = Curve(tier, vecnorm, args.eval_every, args.n_eval, shuffle)
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=curve)
    train_minutes = (time.perf_counter() - started) / 60
    curve.points.append(curve._measure())
    models = RESULTS / "models"
    models.mkdir(parents=True, exist_ok=True)
    model.save(models / f"ppo_{tag}")
    vecnorm.save(str(models / f"vecnormalize_{tag}.pkl"))
    vecnorm.training = False
    venv_steps = model.num_timesteps

    # --- final evaluation on the test seeds, paired with the baselines ---------
    t = load_tier(tier)
    baselines = json.loads((RESULTS / f"baselines_{tier}.json").read_text())
    best_name = baselines["best"]
    dispatcher, allocation = best_name.split("+")
    counter = {"steps": 0, "waits": 0}
    result = evaluate(
        t.scenario,
        [
            ppo_agent(model, vecnorm, tier, counter, shuffle),
            Policy("random", "flexible"),
            Policy(dispatcher, allocation),
        ],
        seeds=t.test_seeds,
    )
    penalty = curve.penalty
    costs = {name: cost(result, name, penalty) for name in result.records}

    def block(name: str) -> dict[str, Any]:
        return {
            "cost": ci(costs[name]),
            "weighted_mean_flow_time": ci(result.series(name, "weighted_mean_flow_time")),
            "left": ci(result.series(name, "left")),
            "truncation_rate": result.truncation_rate(name),
        }

    ppo = block("ppo") | {"wait_share": counter["waits"] / max(counter["steps"], 1)}
    out = {
        "tier": tier,
        "variant": "shuffled candidates (default)" if shuffle else "unshuffled candidates",
        "timesteps": venv_steps,
        "wall_minutes": train_minutes,
        "setup": [
            f"sb3-contrib MaskablePPO, `MultiInputPolicy` MLP [256, 256], {args.n_envs} "
            "subprocess envs on the `train` split, `PointerActionWrapper` "
            f"(K = {MAX_CANDIDATES}, {'shuffled' if shuffle else 'unshuffled'} candidates, "
            "compact observation: candidate "
            "features + global), VecNormalize on observations and rewards.",
            "gamma = 1 (see docs/RL_CONTRACT.md), GAE lambda 0.95, n_steps 512 per env, "
            "batch 512, 10 epochs, lr 3e-4, entropy 0.01, clip 0.2, seed 0; default "
            "`truncation_cost='none'`, SB3 bootstraps on `truncated`.",
            "Budget fixed in environment steps; wall time measured on this machine "
            "(Apple M-series laptop, CPU only). Evaluation is deterministic (argmax).",
        ],
        "ppo": ppo,
        "random": block("random+flexible"),
        "best_heuristic": block(best_name) | {"name": best_name},
        "paired": {
            "ppo_minus_random": ci(costs["ppo"] - costs["random+flexible"]),
            "ppo_minus_best": ci(costs["ppo"] - costs[best_name]),
        },
        "curve_points": curve.points,
        "curve": str((FIGURES / f"ppo_{tag}.png").relative_to(ROOT)),
    }
    out["notes"] = notes(out)
    plot(out, result, penalty)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"ppo_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("ppo", "random", "best_heuristic", "paired")}, indent=1))
    for note in out["notes"]:
        print("NOTE:", note)


def notes(out: dict[str, Any]) -> list[str]:
    points = out["curve_points"]
    logged = [p for p in points if not math.isnan(p["train_return_gap"])]
    gaps = [p["train_return_gap"] for p in logged]
    trunc = max(p["train_truncated"] for p in logged)
    nonfinite = max(p["train_nonfinite"] for p in logged)
    ppo, best = out["ppo"], out["best_heuristic"]
    found = [
        f"Reward bookkeeping: largest |return + scaled cost| over logged training episodes "
        f"{max(gaps):.1e} (the undiscounted return equals minus the scaled episode cost).",
        f"Truncation: at most {100 * trunc:.1f}% of training episodes in any logging window; "
        f"{100 * ppo['truncation_rate']:.0f}% of test episodes.",
        f"Wait action: {100 * ppo['wait_share']:.1f}% of the trained policy's test decisions.",
        f"Non-finite observation cells: {nonfinite:.0f} per training episode (worst window).",
    ]
    if ppo["cost"][0] < best["cost"][0] - best["cost"][1]:
        found.append(
            "The trained policy beats the best heuristic; check its episodes for an exploit "
            "before trusting this."
        )
    return found


def plot(out: dict[str, Any], result: Evaluation, penalty: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = out["curve_points"]
    x = [p["timesteps"] for p in points]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    top.plot(x, [p["eval_cost"] for p in points], color="#3266cc", label="PPO (eval seeds)")
    top.plot(x, [p["train_cost"] for p in points], color="#8fb0e6", label="PPO (training episodes)")
    n_eval = len(load_tier(out["tier"]).eval_seeds[:20])
    t = load_tier(out["tier"])
    reference = evaluate(
        t.scenario,
        [Policy("random", "flexible"), Policy(*out["best_heuristic"]["name"].split("+"))],
        seeds=t.eval_seeds[:n_eval],
    )
    for name, colour, label in (
        ("random+flexible", "#c0392b", "random (eval seeds)"),
        (out["best_heuristic"]["name"], "#27ae60", f"{out['best_heuristic']['name']} (eval seeds)"),
    ):
        top.axhline(cost(reference, name, penalty).mean(), color=colour, ls="--", label=label)
    top.set_ylabel("episode cost (min x weight)")
    top.set_title(f"{out['tier']}: MaskablePPO learning curve ({out['variant']})")
    top.legend(fontsize=8)
    bottom.plot(x, [p["train_wait_share"] for p in points], label="wait share (training)")
    bottom.plot(x, [p["train_truncated"] for p in points], label="truncated share (training)")
    bottom.set_xlabel("environment steps")
    bottom.set_ylabel("share")
    bottom.legend(fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(ROOT / out["curve"], dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    main()
