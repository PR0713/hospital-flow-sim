"""v0.1, task 6: ``to_hetero_data``. Skipped unless torch_geometric is installed."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.env.pyg import EDGE_TYPES, to_hetero_data
from hospital_sim.generators import load_scenario

CONFIGS = Path(__file__).parent.parent / "configs"


def _env(name: str = "example_hospital_full") -> HospitalEnv:
    return HospitalEnv(EnvConfig(load_scenario(CONFIGS / f"{name}.yaml"), observation="graph"))


def test_core_import_does_not_need_torch() -> None:
    import subprocess

    code = (
        "import sys, hospital_sim.env, hospital_sim.env.pyg;"
        "assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_torch_geometric_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch_geometric", None)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", None)
    env = _env()
    env.reset(seed=0)
    with pytest.raises(ImportError, match=r"\.\[gnn\]"):
        to_hetero_data(env.graph())


def test_hetero_data_matches_the_graph_view() -> None:
    pytest.importorskip("torch_geometric")
    import torch

    env = _env()
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(120):
        graph = env.graph()
        data = to_hetero_data(graph)
        data.validate(raise_on_error=True)
        k = int(obs["op_valid"].sum())
        assert data["op"].num_nodes == k
        assert torch.equal(data["op"].x, torch.as_tensor(obs["ops"][:k]))
        assert data["patient"].num_nodes == int(obs["patient_valid"].sum())
        assert data["resource"].num_nodes == obs["resources"].shape[0]
        assert set(data.edge_types) == set(EDGE_TYPES)
        precedes = data["op", "precedes", "op"].edge_index
        assert precedes.shape[1] == int((obs["precedence_edges"][0] >= 0).sum())
        requires = data["op", "requires", "role"]
        assert requires.edge_attr.shape == (requires.edge_index.shape[1], 1)
        assert data.global_x.shape == (1, obs["global"].shape[0])
        obs, *_, terminated, truncated, _ = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        if terminated or truncated:
            obs, _ = env.reset()


def test_a_hetero_gnn_runs_on_it() -> None:
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.nn import HeteroConv, SAGEConv

    env = _env()
    env.reset(seed=1)
    data: Any = to_hetero_data(env.graph())
    conv = HeteroConv(
        {edge: SAGEConv((-1, -1), 8) for edge in EDGE_TYPES}, aggr="sum"
    )
    out = conv(data.x_dict, data.edge_index_dict)
    assert out["role"].shape == (data["role"].num_nodes, 8)
    assert torch.isfinite(out["role"]).all()
