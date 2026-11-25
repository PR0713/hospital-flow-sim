"""Frozen benchmark tiers T1-T4 (``benchmarks/tiers/<name>/``).

Each tier is a scenario (``config.yaml``) plus fixed instance seeds
(``seeds.yaml``): 100 for tuning (``eval`` split) and 200 for reporting
(``test`` split). The seeds come from ``instance_seed(TIER_BASE_SEED, split,
episode, resample)``, so they can never coincide with training instances.
``seeds.yaml`` records the SHA-256 of ``config.yaml``; ``load_tier`` refuses
a tier whose config was edited after its seeds were frozen.

Tiers live in the source checkout, not in the installed package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from hospital_sim.domain.model import Scenario
from hospital_sim.generators.config import load_scenario

TIERS = ("T1", "T2", "T3", "T4")
TIER_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "tiers"
TIER_BASE_SEED = 2026
N_EVAL, N_TEST = 100, 200


@dataclass(frozen=True)
class Tier:
    name: str
    scenario: Scenario
    eval_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    path: Path


def config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tier(name: str, root: Path = TIER_ROOT) -> Tier:
    """Scenario and frozen seed lists of tier ``name`` (``"T1"`` .. ``"T4"``)."""
    folder = root / name
    config, seeds_file = folder / "config.yaml", folder / "seeds.yaml"
    if not seeds_file.exists():
        raise FileNotFoundError(f"no benchmark tier at {folder} (needs a source checkout)")
    seeds = yaml.safe_load(seeds_file.read_text())
    if seeds["config_sha256"] != config_digest(config):
        raise ValueError(
            f"{config} changed after its seeds were frozen; tiers are frozen for v0.1 - "
            "add a new tier instead of editing one"
        )
    return Tier(
        name=name,
        scenario=load_scenario(config),
        eval_seeds=tuple(int(s) for s in seeds["eval"]),
        test_seeds=tuple(int(s) for s in seeds["test"]),
        path=folder,
    )
