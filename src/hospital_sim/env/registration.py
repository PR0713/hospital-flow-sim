"""``gymnasium.make`` support: ``gym.make("SchedSim-v0", scenario=..., **EnvConfig fields)``.

Registered when ``hospital_sim`` is imported (or use the id
``"hospital_sim:SchedSim-v0"``, which imports it for you).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hospital_sim.domain.model import Scenario
from hospital_sim.env.config import EnvConfig
from hospital_sim.env.hospital_env import HospitalEnv
from hospital_sim.generators.config import load_scenario

ENV_ID = "SchedSim-v0"
DEFAULT_SCENARIO = "example_hospital_dynamic"
# Named scenarios resolve against a source checkout of the repository.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_scenario(scenario: str | Path | Scenario) -> Scenario:
    """A ``Scenario`` as is, a path to a YAML/JSON file, or a name: a benchmark
    tier (``"T1"`` .. ``"T4"``, ``benchmarks/tiers/<name>/config.yaml``) or a
    file in ``configs/`` without its extension (``"example_hospital"``)."""
    if isinstance(scenario, Scenario):
        return scenario
    path = Path(scenario)
    if path.suffix in {".yaml", ".yml", ".json"} or path.exists():
        return load_scenario(path)
    for candidate in (
        REPO_ROOT / "benchmarks" / "tiers" / str(scenario) / "config.yaml",
        REPO_ROOT / "configs" / f"{scenario}.yaml",
    ):
        if candidate.exists():
            return load_scenario(candidate)
    raise FileNotFoundError(
        f"no scenario named {scenario!r}: give a Scenario, a path, a tier name or a "
        f"config name from {REPO_ROOT / 'configs'} (named scenarios need a source checkout)"
    )


def make_env(
    scenario: str | Path | Scenario = DEFAULT_SCENARIO,
    render_mode: str | None = None,
    **config: Any,
) -> HospitalEnv:
    """Entry point of ``SchedSim-v0``. ``config`` holds ``EnvConfig`` fields."""
    if render_mode is not None:
        raise ValueError("this environment has no render modes")
    return HospitalEnv(EnvConfig(resolve_scenario(scenario), **config))


def register() -> None:
    """Register ``SchedSim-v0`` with gymnasium (idempotent)."""
    from gymnasium.envs.registration import register as gym_register
    from gymnasium.envs.registration import registry

    if ENV_ID not in registry:
        gym_register(id=ENV_ID, entry_point="hospital_sim.env.registration:make_env")
