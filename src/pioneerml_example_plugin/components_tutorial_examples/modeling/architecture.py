from __future__ import annotations

"""Model architecture plugin for the sensor-health tutorial."""

from pathlib import Path
from collections.abc import Sequence

import torch
import torch.nn as nn

from pioneerml.integration.pytorch.models.architectures.factory.registry import REGISTRY as ARCHITECTURE_REGISTRY
from pioneerml.integration.pytorch.models.architectures.graph import BaseGraphModel


@ARCHITECTURE_REGISTRY.register("sensor_health_mlp")
class SensorHealthMLP(BaseGraphModel):
    """Tiny graph model for one-node sensor-health graphs.

    The model still implements the graph-model contract: it receives a PyG
    batch, pools node features by graph id, and returns one logit per graph.
    """

    def __init__(
        self,
        node_dim: int = 3,
        edge_dim: int = 1,
        graph_dim: int = 0,
        hidden: int = 16,
        dropout: float = 0.0,
        output_dim: int = 1,
        feature_mean: Sequence[float] = (68.0, 0.85, 3.30),
        feature_std: Sequence[float] = (9.0, 0.65, 0.18),
    ) -> None:
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            graph_dim=graph_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.output_dim = int(output_dim)
        self.register_buffer("feature_mean", torch.tensor(tuple(feature_mean), dtype=torch.float32).view(1, -1))
        self.register_buffer("feature_std", torch.tensor(tuple(feature_std), dtype=torch.float32).view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.output_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_features(data).float()
        mean = self.feature_mean.to(dtype=x.dtype, device=x.device)
        std = self.feature_std.to(dtype=x.dtype, device=x.device).clamp_min(1e-6)
        x = (x - mean) / std
        graph_id = self.node_graph_id(data).to(dtype=torch.long, device=x.device)
        num_graphs = int(getattr(data, "num_graphs", 0))
        if num_graphs <= 0:
            num_graphs = int(graph_id.max().item() + 1) if graph_id.numel() else 0
        if num_graphs <= 0:
            return x.new_zeros((0, self.output_dim))

        pooled = x.new_zeros((num_graphs, x.shape[1]))
        counts = x.new_zeros((num_graphs, 1))
        # Mean-pooling is enough for a one-node graph and keeps the example
        # independent of attention/convolution layers.
        pooled.index_add_(0, graph_id, x)
        counts.index_add_(0, graph_id, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device))
        pooled = pooled / counts.clamp_min(1.0)
        return self.net(pooled)

    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        _ = strict
        scripted = torch.jit.script(self)
        if path is not None:
            scripted.save(str(Path(path).expanduser().resolve()))
        return scripted

    def tutorial_config(self) -> dict[str, int | float | list[float]]:
        """Return constructor settings needed by the state-dict model handle."""

        return {
            "node_dim": int(self.node_dim),
            "edge_dim": int(self.edge_dim),
            "graph_dim": int(self.graph_dim),
            "hidden": int(self.hidden),
            "dropout": float(self.dropout),
            "output_dim": int(self.output_dim),
            "feature_mean": [float(v) for v in self.feature_mean.detach().cpu().reshape(-1)],
            "feature_std": [float(v) for v in self.feature_std.detach().cpu().reshape(-1)],
        }
