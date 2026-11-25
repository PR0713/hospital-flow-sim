"""v0.1, task 8: version and observation schema are reported consistently."""

from __future__ import annotations

import tomllib
from pathlib import Path

import hospital_sim
from hospital_sim.env import OBSERVATION_SCHEMA_VERSION, EnvConfig, HospitalEnv, make_env

ROOT = Path(__file__).parent.parent


def test_package_version_matches_pyproject_and_changelog() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert hospital_sim.__version__ == project["version"] == "0.1.0"
    assert f"## [{hospital_sim.__version__}]" in (ROOT / "CHANGELOG.md").read_text()


def test_reset_reports_the_observation_schema_version() -> None:
    env = make_env("example_hospital")
    _, info = env.reset(seed=0)
    assert info["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION == 1
    assert info["observation_schema"] == OBSERVATION_SCHEMA_VERSION  # kept for old callers
    *_, info = env.step(int(env.action_space.sample()))
    assert "observation_schema_version" not in info  # reset only
    assert isinstance(env, HospitalEnv) and isinstance(env.config, EnvConfig)
