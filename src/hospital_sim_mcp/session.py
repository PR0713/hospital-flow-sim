"""One interactive episode of the hospital environment, shaped for an LLM.

No MCP imports here, so the logic is testable on its own. Everything shown to
the caller is built from the environment's public observation arrays; nothing
reads the simulator's hidden state (true durations, future arrivals, no-shows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hospital_sim.env import EnvConfig, HospitalEnv, InvalidActionError
from hospital_sim.env.config import SPLITS
from hospital_sim.env.observation import OP_FEATURES, PATIENT_FEATURES
from hospital_sim.generators import load_scenario

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_OP = {name: i for i, name in enumerate(OP_FEATURES)}
_PATIENT = {name: i for i, name in enumerate(PATIENT_FEATURES)}


class SessionError(Exception):
    """A caller mistake (unknown config, no episode running). The message is
    written for the LLM that will read it."""


class EpisodeSession:
    def __init__(self, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        self.configs_dir = configs_dir
        self.env: HospitalEnv | None = None
        self.config_name = ""
        self._obs: dict[str, Any] = {}
        self._info: dict[str, Any] = {}
        self._over = True
        self._total_reward = 0.0
        self._steps = 0

    # --- helpers -------------------------------------------------------------

    def available_configs(self) -> list[str]:
        return sorted(p.stem for p in self.configs_dir.glob("*.yaml"))

    def _need_env(self) -> HospitalEnv:
        if self.env is None:
            raise SessionError("No episode yet. Call reset(config, seed) first.")
        return self.env

    # --- episode control -----------------------------------------------------

    def reset(self, config: str, seed: int = 0, split: str = "train") -> dict[str, Any]:
        # Only names from the configs directory: the caller cannot pass a path.
        if config not in self.available_configs():
            raise SessionError(
                f"Unknown config {config!r}. Choose one of: {self.available_configs()}"
            )
        if split not in SPLITS:
            raise SessionError(f"split must be one of {sorted(SPLITS)}")
        scenario = load_scenario(self.configs_dir / f"{config}.yaml")
        self.env = HospitalEnv(EnvConfig(scenario, split=split))
        self.config_name = config
        self._obs, self._info = self.env.reset(seed=seed)
        self._over = False
        self._total_reward, self._steps = 0.0, 0
        return {"config": config, "seed": seed, "split": split, "state": self.state()}

    def step(self, action: int) -> dict[str, Any]:
        env = self._need_env()
        if self._over:
            raise SessionError("The episode is over. Call get_metrics() or reset().")
        try:
            self._obs, reward, terminated, truncated, self._info = env.step(action)
        except InvalidActionError:
            # Recoverable: tell the model what is valid instead of failing the call.
            return {
                "ok": False,
                "error": f"Action {action} is not valid right now.",
                "valid_actions": self.valid_actions(),
            }
        self._steps += 1
        self._total_reward += reward
        self._over = terminated or truncated
        result: dict[str, Any] = {
            "ok": True,
            "reward": round(reward, 4),
            "episode_over": self._over,
        }
        if truncated:
            result["truncated"] = self._info.get("truncation_reason")
        result["state"] = self.state()
        return result

    # --- views ---------------------------------------------------------------

    def state(self) -> str:
        env = self._need_env()
        obs, scale = self._obs, env.time_scale
        unit = env.config.scenario.time_unit
        g = obs["global"]
        lines = [
            f"Clock: {g[0] * scale:.1f} {unit}s. Patients in system: {int(g[1])}. "
            f"Operations: {int(g[2])} ready, {int(g[3])} startable, {int(g[4])} running."
        ]
        if self._over:
            lines.append("The episode is over.")
            return "\n".join(lines)
        scenario = env.config.scenario
        for role, (idle, members, demand) in zip(scenario.roles, obs["roles"], strict=True):
            lines.append(
                f"Role {role.name}: {int(idle)}/{int(members)} idle, "
                f"{demand:.0f} wanted by ready ops."
            )
        running = [s for s in range(env.max_ops) if obs["ops"][s, _OP["is_running"]] > 0]
        for slot in running:
            lines.append(f"Running: {self._describe_op(slot)}")
        return "\n".join(lines)

    def _describe_op(self, slot: int) -> str:
        env = self._need_env()
        obs, scale = self._obs, env.time_scale
        op, p = obs["ops"][slot], obs["patients"][int(obs["op_patient"][slot])]
        name = env.config.scenario.operation_types[int(obs["op_type"][slot])].name
        flags = []
        if p[_PATIENT["urgent"]] > 0:
            flags.append("URGENT")
        if p[_PATIENT["acuity"]] > 0:
            flags.append(f"acuity {p[_PATIENT['acuity']]:g}")
        flags.append(f"weight {p[_PATIENT['weight']]:g}")
        detail = (
            f"waited {op[_OP['waited']] * scale:.1f}, expected duration "
            f"{op[_OP['expected_duration']] * scale:.1f}"
        )
        if op[_OP["is_running"]] > 0:
            detail = (
                f"running {op[_OP['elapsed']] * scale:.1f}, "
                f"expected remaining {op[_OP['expected_remaining']] * scale:.1f}"
            )
        patient = int(obs["op_patient"][slot])
        in_system = p[_PATIENT["time_in_system"]] * scale
        return (
            f"[{name}] patient {patient} ({', '.join(flags)}; in system {in_system:.1f}) - {detail}"
        )

    def valid_actions(self) -> list[dict[str, Any]]:
        env = self._need_env()
        if self._over:
            return []
        mask = env.action_masks()
        n_rules = len(env.config.allocation_rules)
        actions: list[dict[str, Any]] = []
        for raw in np.flatnonzero(mask):
            index = int(raw)
            if index == env.wait_action:
                actions.append(
                    {"action": index, "kind": "wait", "description": "Wait until the next event."}
                )
            else:
                slot = index // n_rules
                description = "Start " + self._describe_op(slot)
                actions.append({"action": index, "kind": "start", "description": description})
        return actions

    def metrics(self) -> dict[str, Any]:
        self._need_env()
        result: dict[str, Any] = {
            "config": self.config_name,
            "steps": self._steps,
            "total_reward": round(self._total_reward, 4),
            "episode_over": self._over,
        }
        if "episode" in self._info:
            result["episode"] = self._info["episode"]
        return result
